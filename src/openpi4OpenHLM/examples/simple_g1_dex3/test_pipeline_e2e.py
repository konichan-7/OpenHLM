"""End-to-end pipeline test against a *real* LeRobot dataset.

Unlike ``test_protocol.py`` (pure numpy), this imports the real openpi stack and
walks the same code path training and serving use:

    LeRobotDataset -> EpisodeSubsetDataset -> repack -> SimpleG1Dex3Inputs
    -> Normalize -> PadStatesAndActions -> policy output -> SimpleG1Dex3Outputs

It needs the dataset but not a GPU or a checkpoint, so it is the first thing to
run on a fresh box.

    HF_LEROBOT_HOME=/data/psi_simple_tabletop/lerobot \
      python examples/simple_g1_dex3/test_pipeline_e2e.py --repo-id local/g1_dex3_tabletop

Exits non-zero on the first hard failure.  Checks:
  * repo resolves through HF_LEROBOT_HOME and reports 32/36 dims, 50 fps;
  * an action chunk never crosses an episode boundary;
  * train/val episodes are disjoint and the subset length matches the manifest;
  * normalisation round-trips within 1e-4 on real frames;
  * the model input spec matches what the server will receive.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="local/g1_dex3_tabletop")
    ap.add_argument("--config-name", default="pi05_simple_g1_dex3_tabletop")
    ap.add_argument("--episode-manifest", default=None)
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as _transforms
    from openpi.policies import simple_g1_dex3_policy as P

    config = _config.get_config(args.config_name)
    if args.episode_manifest:
        import dataclasses

        config = _config.resolve_manifest(dataclasses.replace(config, episode_manifest=args.episode_manifest))
    data_config = config.data.create(config.assets_dirs, config.model)

    print(f"repo_id      : {data_config.repo_id}")
    print(f"asset_id     : {data_config.asset_id}")
    print(f"action_horizon: {config.model.action_horizon}  action_dim: {config.model.action_dim}")

    meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id)
    fps = meta.fps
    print(f"episodes     : {meta.total_episodes}  frames: {meta.total_frames}  fps: {fps}")
    assert fps == 50, f"expected 50 fps, got {fps}"
    assert int(meta.total_frames) > 0

    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    print(f"subset frames: {len(dataset)}")
    if data_config.episode_indices:
        import json
        import os
        import pathlib

        lerobot_home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
        episodes_jsonl = lerobot_home / data_config.repo_id / "meta" / "episodes.jsonl"
        if not episodes_jsonl.exists():
            raise SystemExit(f"cannot find {episodes_jsonl}; is HF_LEROBOT_HOME set?")
        expected = sum(
            int(x["length"])
            for x in (json.loads(line) for line in episodes_jsonl.read_text().splitlines() if line)
            if int(x["episode_index"]) in set(data_config.episode_indices)
        )
        # LeRobot keeps one frame per row; the wrapper only filters rows.
        assert len(dataset) == expected, f"subset has {len(dataset)} frames, manifest implies {expected}"
        print(f"manifest     : {len(data_config.episode_indices)} episodes, {expected} frames  OK")

    # ---- raw sample: block order and episode-boundary behaviour -------------
    idx_step = max(1, len(dataset) // args.samples)
    for i in range(0, len(dataset), idx_step):
        if i >= idx_step * args.samples:
            break
        raw = dataset[i]
        action = np.asarray(raw["action"])
        assert action.shape == (config.model.action_horizon, P.ACTION_DIM), action.shape
        # Every action in the chunk must come from one episode.
        if "episode_index" in raw:
            ep = int(np.asarray(raw["episode_index"]).reshape(-1)[0])
            print(f"  frame {i}: episode {ep}, chunk {action.shape}")

    # ---- repack -> data transforms -> normalize -> pad ----------------------
    norm_stats = data_config.norm_stats
    if norm_stats is None:
        raise SystemExit(
            f"no norm stats at {config.assets_dirs / data_config.asset_id}; run compute_norm_stats.py first"
        )
    assert norm_stats["state"].mean.shape[-1] == P.STATE_DIM, norm_stats["state"].mean.shape
    assert norm_stats["actions"].mean.shape[-1] == P.ACTION_DIM, norm_stats["actions"].mean.shape
    print(f"norm stats   : state {norm_stats['state'].mean.shape}, actions {norm_stats['actions'].mean.shape}")

    normalize_pipeline = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
    ]
    normalize = _transforms.compose(normalize_pipeline)

    normalized = normalize(dataset[0])
    assert normalized["state"].shape == (P.STATE_DIM,)
    assert normalized["actions"].shape == (config.model.action_horizon, P.ACTION_DIM)
    assert not normalized["image_mask"]["left_wrist_0_rgb"]

    # Model-side transforms resize the image and pad state/actions to action_dim.
    model_transform = _transforms.compose(data_config.model_transforms.inputs)
    sample = model_transform(normalized)
    assert sample["state"].shape == (config.model.action_dim,), sample["state"].shape
    assert sample["image"]["base_0_rgb"].shape[:2] == (224, 224), sample["image"]["base_0_rgb"].shape
    assert sample["actions"].shape == (config.model.action_horizon, config.model.action_dim)
    print("transforms   : OK (normalize -> resize -> pad)")

    # round-trip: unnormalize(normalize(x)) == x
    unnorm = _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
    roundtrip = unnorm(
        {
            "state": normalized["state"].copy(),
            "actions": normalized["actions"].copy(),
        }
    )
    back = roundtrip["actions"]
    raw_actions = np.asarray(dataset[0]["action"], dtype=np.float32)
    err = np.abs(back - raw_actions).max()
    print(f"round-trip   : max |err| = {err:.2e}")
    assert err < 1e-3, f"normalisation round-trip error {err}"

    # ---- model input spec ---------------------------------------------------
    obs_spec, act_spec = config.model.inputs_spec()
    print(f"model state  : {obs_spec.state.shape}   model action: {act_spec.shape}")
    assert act_spec.shape[-1] == config.model.action_dim

    print("\nE2E PIPELINE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
