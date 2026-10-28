"""Build a non-loco multi-task training manifest for SIMPLE G1/Dex3.

The script reads one or more LeRobot v2.1 repositories, measures each episode's
base command from action dimensions 32:35, and keeps only episodes with:

    abs(vx) < threshold, abs(vy) < threshold, turning < threshold

The kept episodes are grouped by ``environment_config`` and split into train/val
without splitting a scene group.  The resulting JSON is consumed by
``LeRobotSimpleG1Dex3DataConfig.dataset_manifest``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _episode_path(root: pathlib.Path, episode: int, chunks_size: int) -> pathlib.Path:
    chunk = episode // chunks_size
    return root / f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet"


def _group_key(environment_config: str) -> str:
    payload = json.dumps(json.loads(environment_config), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _digest(datasets: list[dict]) -> str:
    payload = json.dumps(
        [{"repo_id": d["repo_id"], "train": sorted(d["train"]), "val": sorted(d["val"])} for d in datasets],
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def audit_repo(
    repo_id: str,
    root: pathlib.Path,
    *,
    max_vx: float,
    max_vy: float,
    max_turning: float,
) -> tuple[list[int], dict[int, int], list[dict]]:
    info = json.loads((root / "meta/info.json").read_text())
    episodes = _read_jsonl(root / "meta/episodes.jsonl")
    chunks_size = int(info.get("chunks_size", 1000))
    selected: list[int] = []
    lengths: dict[int, int] = {}
    skipped: list[dict] = []

    for record in episodes:
        episode = int(record["episode_index"])
        parquet = _episode_path(root, episode, chunks_size)
        action = np.asarray(pq.read_table(parquet, columns=["action"]).to_pydict()["action"], dtype=np.float32)
        metrics = {
            "episode": episode,
            "max_abs_vx": float(np.abs(action[:, 32]).max()),
            "max_abs_vy": float(np.abs(action[:, 33]).max()),
            "max_abs_turning": float(np.abs(action[:, 34]).max()),
        }
        is_nonloco = (
            metrics["max_abs_vx"] <= max_vx
            and metrics["max_abs_vy"] <= max_vy
            and metrics["max_abs_turning"] <= max_turning
        )
        if is_nonloco:
            selected.append(episode)
            lengths[episode] = int(record["length"])
        else:
            skipped.append(metrics)

    print(
        f"{repo_id}: kept {len(selected)}/{len(episodes)} non-loco episodes, "
        f"skipped {len(skipped)}"
    )
    return selected, lengths, skipped


def split_episodes(
    root: pathlib.Path,
    episodes: list[int],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    records = {int(r["episode_index"]): r for r in _read_jsonl(root / "meta/episodes.jsonl")}
    groups: dict[str, list[int]] = defaultdict(list)
    for episode in episodes:
        groups[_group_key(records[episode]["environment_config"])].append(episode)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target_val = max(1, round(len(episodes) * val_fraction))
    val: list[int] = []
    for key in keys:
        if len(val) >= target_val:
            break
        val.extend(groups[key])
    val_set = set(val)
    train = [episode for episode in episodes if episode not in val_set]
    return sorted(train), sorted(val)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", action="append", required=True, help="LeRobot repo_id; repeat for each task")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--max-abs-vx", type=float, default=0.02)
    ap.add_argument("--max-abs-vy", type=float, default=0.02)
    ap.add_argument("--max-abs-turning", type=float, default=0.5)
    args = ap.parse_args()

    lerobot_home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
    datasets = []
    total_train_frames = 0
    total_val_frames = 0
    for repo_id in args.repo:
        root = lerobot_home / repo_id
        if not (root / "meta/info.json").exists():
            raise FileNotFoundError(f"{repo_id}: {root} is not a LeRobot dataset")
        episodes, lengths, skipped = audit_repo(
            repo_id,
            root,
            max_vx=args.max_abs_vx,
            max_vy=args.max_abs_vy,
            max_turning=args.max_abs_turning,
        )
        if not episodes:
            raise ValueError(f"{repo_id}: no non-loco episodes selected")
        train, val = split_episodes(
            root,
            episodes,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        train_frames = sum(lengths[episode] for episode in train)
        val_frames = sum(lengths[episode] for episode in val)
        total_train_frames += train_frames
        total_val_frames += val_frames
        datasets.append(
            {
                "repo_id": repo_id,
                "train": train,
                "val": val,
                "frames_train": train_frames,
                "frames_val": val_frames,
                "skipped_loco_episodes": skipped,
            }
        )

    payload = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "selection": {
            "max_abs_vx": args.max_abs_vx,
            "max_abs_vy": args.max_abs_vy,
            "max_abs_turning": args.max_abs_turning,
        },
        "datasets": datasets,
        "frames_train": total_train_frames,
        "frames_val": total_val_frames,
    }
    payload["sha256"] = _digest(datasets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(
        f"wrote {args.out}: {len(datasets)} tasks, "
        f"{sum(len(d['train']) for d in datasets)} train episodes / {total_train_frames} frames, "
        f"{sum(len(d['val']) for d in datasets)} val episodes / {total_val_frames} frames"
    )
    print(f"sha256: {payload['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
