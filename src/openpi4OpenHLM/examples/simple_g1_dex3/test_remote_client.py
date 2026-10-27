"""Send one real observation to a running OpenHLM policy server.

    python examples/simple_g1_dex3/test_remote_client.py \
      --host 127.0.0.1 --port 8000 \
      --dataset-root $HF_LEROBOT_HOME/local/g1_dex3_tabletop \
      --episode 1 --frame 30
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dataset-root", required=True, type=pathlib.Path)
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--frame", type=int, default=30)
    args = ap.parse_args()

    import cv2
    import polars as pl
    from openpi_client import websocket_client_policy

    shard = args.episode // 1000
    parquet = args.dataset_root / f"data/chunk-{shard:03d}/episode_{args.episode:06d}.parquet"
    video = args.dataset_root / f"videos/chunk-{shard:03d}/egocentric/episode_{args.episode:06d}.mp4"
    states = np.stack(pl.read_parquet(parquet)["states"].to_list()).astype(np.float32)

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not decode frame {args.frame} from {video}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port, api_key=None)
    metadata = client.get_server_metadata()
    print(f"metadata: {metadata}")
    assert metadata.get("action_protocol") == "simple_g1_dex3", metadata
    assert int(metadata.get("action_dim", -1)) == 36, metadata

    out = client.infer(
        {
            "observation/image": image,
            "observation/state": states[args.frame],
            "prompt": "grasp the object on the tabletop.",
            "reset": True,
        }
    )
    actions = np.asarray(out["actions"], dtype=np.float32)
    print(f"actions: {actions.shape} {actions.dtype}")
    print(f"timing : {out.get('policy_timing')}")
    assert actions.shape == (30, 36), actions.shape
    assert np.isfinite(actions).all()
    np.testing.assert_allclose(actions[:, 31], 0.75, atol=1e-6)
    np.testing.assert_allclose(actions[:, 32:36], 0.0, atol=1e-6)
    print("\nREMOTE CLIENT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
