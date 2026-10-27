"""Open-loop evaluation of a trained SIMPLE G1/Dex3 policy.

For each frame of a few episodes: feed the recorded image + 32-d state to the
policy, compare the predicted 36-d action chunk against the recorded action.

Reports per-block MAE (left/right hand, left/right arm, torso rpy, height, base
commands) rather than one mixed MSE, as required by the plan (section 8.2).

Usage (on the training box, after ``compute_norm_stats.py`` and training)::

    python examples/simple_g1_dex3/eval_openloop.py \
        --checkpoint checkpoints/pi05_simple_g1_dex3_tabletop/run/3000 \
        --dataset-root /data/psi_simple_tabletop/G1WholebodyTabletopGraspMP-v0 \
        --episodes 80 81 82 --num-steps 10 --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.training import config as _config  # noqa: E402

BLOCKS = {
    "left_hand": slice(0, 7),
    "right_hand": slice(7, 14),
    "left_arm": slice(14, 21),
    "right_arm": slice(21, 28),
    "torso_rpy": slice(28, 31),
    "height": slice(31, 32),
    "base_cmd": slice(32, 36),
}


def read_episode(dataset_root: pathlib.Path, episode: int) -> dict:
    import pyarrow.parquet as pq

    shard = episode // 1000
    path = dataset_root / f"data/chunk-{shard:03d}/episode_{episode:06d}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pydict()


def read_video_frames(dataset_root: pathlib.Path, episode: int) -> np.ndarray:
    """Decode one episode's egocentric video to (T, H, W, 3) uint8 RGB."""
    import cv2

    shard = episode // 1000
    path = dataset_root / f"videos/chunk-{shard:03d}/egocentric/episode_{episode:06d}.mp4"
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--dataset-root", required=True, type=pathlib.Path)
    ap.add_argument("--config-name", default="pi05_simple_g1_dex3_tabletop")
    ap.add_argument("--episodes", nargs="+", type=int, required=True)
    ap.add_argument("--episode-manifest", default=None)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--stride", type=int, default=5, help="evaluate every Nth frame")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, type=pathlib.Path)
    args = ap.parse_args()

    train_config = _config.get_config(args.config_name)
    if args.episode_manifest:
        train_config = _config.resolve_manifest(
            dataclasses.replace(train_config, episode_manifest=args.episode_manifest)
        )
    # No repack transforms here: ``SimpleG1Dex3Inputs`` also accepts the raw
    # dataset keys, and ``RepackTransform`` would require an ``actions`` key that
    # a single-frame observation does not have.
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device=args.device,
    )

    errors: dict[str, list[float]] = {k: [] for k in BLOCKS}
    per_episode: dict[str, dict] = {}

    for episode in args.episodes:
        cols = read_episode(args.dataset_root, episode)
        states = np.asarray(cols["states"], dtype=np.float32)
        actions = np.asarray(cols["action"], dtype=np.float32)
        frames = read_video_frames(args.dataset_root, episode)
        n = min(len(states), len(actions), len(frames))
        if n == 0:
            print(f"episode {episode}: no aligned frames, skipped")
            continue

        ep_errors: dict[str, list[float]] = {k: [] for k in BLOCKS}
        for t in range(0, n, args.stride):
            result = policy.infer(
                {
                    "observation.images.egocentric": frames[t],
                    "states": states[t],
                }
            )
            pred = np.asarray(result["actions"], dtype=np.float32)
            gt = actions[t : t + len(pred)]
            m = min(len(pred), len(gt))
            if m == 0:
                continue
            err = np.abs(pred[:m] - gt[:m]).mean(axis=0)  # (36,)
            for name, sl in BLOCKS.items():
                ep_errors[name].append(float(err[sl].mean()))
                errors[name].append(float(err[sl].mean()))

        per_episode[str(episode)] = {k: float(np.mean(v)) if v else float("nan") for k, v in ep_errors.items()}
        print(f"episode {episode}: " + "  ".join(f"{k}={v:.4f}" for k, v in per_episode[str(episode)].items()))

    summary = {k: float(np.mean(v)) if v else float("nan") for k, v in errors.items()}
    print("\noverall MAE by block:")
    for name, value in summary.items():
        print(f"  {name:10s} {value:.5f}")

    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "episodes": per_episode}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
