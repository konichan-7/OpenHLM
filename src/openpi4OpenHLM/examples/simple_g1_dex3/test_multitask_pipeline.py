"""Verify the curated non-loco multi-task dataset before training.

Checks all repositories in ``dataset_manifest``, their prompt routing, frame
counts, action-chunk shapes, and the model-side transform stack.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data_loader  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402
from openpi.policies import simple_g1_dex3_policy as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", default="pi05_simple_g1_dex3_multitask")
    args = ap.parse_args()

    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.dataset_manifest is None:
        raise SystemExit("config has no dataset_manifest")
    manifest = json.loads(pathlib.Path(data_config.dataset_manifest).read_text())
    expected_frames = int(manifest["frames_train"])

    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    print(f"dataset frames: {len(dataset)} (manifest {expected_frames})")
    assert len(dataset) == expected_frames
    assert isinstance(dataset, _data_loader.ConcatDataset)
    assert len(dataset._datasets) == len(manifest["datasets"]), dataset._datasets

    prompts = set()
    for spec, subdataset in zip(manifest["datasets"], dataset._datasets, strict=True):
        sample = subdataset[0]
        state = np.asarray(sample["states"])
        action = np.asarray(sample["action"])
        assert state.shape == (P.STATE_DIM,), state.shape
        assert action.shape == (config.model.action_horizon, P.ACTION_DIM), action.shape
        assert np.isfinite(state).all() and np.isfinite(action).all()
        prompts.add(str(sample["prompt"]))
        print(f"{spec['repo_id']}: {len(subdataset)} frames, prompt={sample['prompt']!r}")

    print(f"unique task prompts: {len(prompts)}")
    assert len(prompts) == len(manifest["datasets"])

    if data_config.norm_stats is None:
        raise SystemExit(
            f"norm stats missing at {config.assets_dirs / data_config.asset_id}; "
            "run compute_norm_stats.py first"
        )
    sample = next(iter(dataset))
    pipeline = _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    model_sample = pipeline(sample)
    assert model_sample["state"].shape == (config.model.action_dim,)
    assert model_sample["actions"].shape == (config.model.action_horizon, config.model.action_dim)
    assert model_sample["image"]["base_0_rgb"].shape[:2] == (224, 224)
    assert not model_sample["image_mask"]["left_wrist_0_rgb"]
    print("MULTITASK PIPELINE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
