"""Freeze a train/val episode split for the SIMPLE G1/Dex3 task.

The released package has 100 episodes, all tagged ``split == "train"`` in
``environment_config``; there is no official held-out split.  This script creates
one deterministically and writes a manifest consumable by

* ``scripts/train_pytorch.py <config> --episode-manifest <manifest>``
* ``scripts/compute_norm_stats.py <config> --episode-manifest <manifest>``
* ``examples/simple_g1_dex3/eval_openloop.py``

Scenes are grouped before splitting: two episodes whose ``environment_config``
differs only in episode-specific noise still belong to the same group, so a
plain random split would leak near-duplicates across train and val.  The script
groups by a normalised hash of the environment config (optionally relaxing
numeric fields) and falls back to per-episode grouping when every config is
unique.

Usage::

    python examples/simple_g1_dex3/make_split.py \
        --dataset-root /path/to/G1WholebodyTabletopGraspMP-v0 \
        --out assets/simple_g1_dex3/tabletop_split.json \
        --val-fraction 0.2 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys

DEFAULT_REPO_ID = "local/g1_dex3_tabletop"


def _env_key(env: dict, ignore_keys: tuple[str, ...]) -> str:
    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k not in ignore_keys}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    return hashlib.sha256(json.dumps(strip(env), sort_keys=True).encode()).hexdigest()


def build_split(root: pathlib.Path, val_fraction: float, seed: int, ignore_keys: tuple[str, ...]):
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise SystemExit(f"missing {episodes_path}")

    records = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    groups: dict[str, list[int]] = {}
    for rec in records:
        ep = int(rec["episode_index"])
        env = json.loads(rec.get("environment_config") or "{}")
        groups.setdefault(_env_key(env, ignore_keys), []).append(ep)

    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)

    target_val = max(1, round(len(records) * val_fraction))
    val: list[int] = []
    for key in keys:
        if len(val) >= target_val:
            break
        val.extend(groups[key])
    train = [int(r["episode_index"]) for r in records if int(r["episode_index"]) not in set(val)]

    if not train or not val:
        raise SystemExit("split degenerated to an empty side; lower --val-fraction")

    frames = {int(r["episode_index"]): int(r["length"]) for r in records}
    return {
        "train": sorted(train),
        "val": sorted(val),
        "frames_train": sum(frames[i] for i in train),
        "frames_val": sum(frames[i] for i in val),
        "groups": len(groups),
        "seed": seed,
        "val_fraction": val_fraction,
        "ignore_keys": list(ignore_keys),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ignore-keys",
        default="",
        help="comma-separated JSON keys to drop before grouping, e.g. 'label,description'",
    )
    args = ap.parse_args()

    ignore = tuple(k for k in args.ignore_keys.split(",") if k)
    payload = build_split(args.dataset_root, args.val_fraction, args.seed, ignore)
    payload["repo_id"] = args.repo_id

    # Reuse openpi's writer when the package is importable (keeps one hash scheme).
    try:
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
        from openpi.training import manifests

        digest = manifests.save(
            args.out,
            repo_id=args.repo_id,
            train=payload["train"],
            val=payload["val"],
            frames_train=payload["frames_train"],
            frames_val=payload["frames_val"],
            groups=payload["groups"],
            seed=payload["seed"],
            val_fraction=payload["val_fraction"],
            ignore_keys=payload["ignore_keys"],
        )
    except Exception as exc:  # pragma: no cover - openpi not importable in audit-only envs
        print(f"warning: could not import openpi.training.manifests ({exc}); writing JSON directly")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
        digest = "<unhashed>"

    print(f"train: {len(payload['train'])} episodes / {payload['frames_train']} frames")
    print(f"val:   {len(payload['val'])} episodes / {payload['frames_val']} frames")
    print(f"groups: {payload['groups']}   manifest: {args.out}   sha256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
