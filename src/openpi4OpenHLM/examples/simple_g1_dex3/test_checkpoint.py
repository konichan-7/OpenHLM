"""Load a trained checkpoint and run one real policy inference.

This is the checkpoint-level gate before starting the WebSocket server: it
checks that the saved model, the checkpoint's norm stats, and the 36-d output
transform all work together on a real dataset frame.

    HF_LEROBOT_HOME=/data/psi_simple_tabletop/lerobot \
      PYTHONPATH=src python examples/simple_g1_dex3/test_checkpoint.py \
        --checkpoint checkpoints/pi05_simple_g1_dex3_tabletop/smoke/100 \
        --dataset-root $HF_LEROBOT_HOME/local/g1_dex3_tabletop \
        --episode-manifest assets/simple_g1_dex3/tabletop_split.json \
        --episode 1 --frame 30
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--dataset-root", required=True, type=pathlib.Path)
    ap.add_argument("--episode-manifest", default=None)
    ap.add_argument("--config-name", default="pi05_simple_g1_dex3_tabletop")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--frame", type=int, default=30)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import cv2
    import polars as pl

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
    from openpi.policies import policy_config as _policy_config
    from openpi.policies import simple_g1_dex3_policy as P
    from openpi.training import config as _config

    train_config = _config.get_config(args.config_name)
    if args.episode_manifest:
        train_config = _config.resolve_manifest(
            dataclasses.replace(train_config, episode_manifest=args.episode_manifest)
        )

    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device=args.device,
    )

    shard = args.episode // 1000
    parquet = args.dataset_root / f"data/chunk-{shard:03d}/episode_{args.episode:06d}.parquet"
    video = args.dataset_root / f"videos/chunk-{shard:03d}/egocentric/episode_{args.episode:06d}.mp4"
    df = pl.read_parquet(parquet)
    states = np.stack(df["states"].to_list()).astype(np.float32)
    actions = np.stack(df["action"].to_list()).astype(np.float32)
    if not (0 <= args.frame < len(states)):
        raise SystemExit(f"frame {args.frame} outside episode length {len(states)}")

    cap = cv2.VideoCapture(str(video))
    frame_index = 0
    image = None
    while True:
        ok, bgr = cap.read()
        if not ok:
            raise SystemExit(f"video ended before frame {args.frame}")
        if frame_index == args.frame:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            break
        frame_index += 1
    cap.release()

    request = {
        "observation/image": image,
        "observation/state": states[args.frame],
        "prompt": "grasp the object on the tabletop.",
    }
    # First call includes torch.compile/max-autotune for the sampling path.
    out = policy.infer(request)
    first_timing = out.get("policy_timing")
    out = policy.infer(request)
    predicted = np.asarray(out["actions"], dtype=np.float32)
    print(f"checkpoint : {args.checkpoint}")
    print(f"request    : state {states[args.frame].shape}, image {image.shape}")
    print(f"response   : actions {predicted.shape} dtype={predicted.dtype}")
    print(f"timing     : first={first_timing}, warm={out.get('policy_timing')}")
    assert predicted.shape == (train_config.model.action_horizon, P.ACTION_DIM), predicted.shape
    assert np.isfinite(predicted).all()
    print(f"first row  : {np.round(predicted[0, :8], 4).tolist()} ...")
    print(f"height     : {predicted[:, 31].min():.6f} .. {predicted[:, 31].max():.6f}")
    print(f"base cmd   : {np.abs(predicted[:, 32:36]).max():.6f}")
    np.testing.assert_allclose(predicted[:, 31], P.NOMINAL_HEIGHT, atol=1e-6)
    np.testing.assert_allclose(predicted[:, 32:36], 0.0, atol=1e-6)
    print("\nCHECKPOINT INFERENCE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
