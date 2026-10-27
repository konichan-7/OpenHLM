"""Episode split manifests for the SIMPLE G1/Dex3 experiments.

A manifest is a small JSON file produced by ``examples/simple_g1_dex3/make_split.py``::

    {
      "repo_id": "local/g1_dex3_tabletop",
      "seed": 0,
      "train": [0, 1, 2, ...],
      "val": [80, 81, ...],
      "notes": "...",
      "sha256": "<hash of the sorted train/val lists>"
    }

The loader validates the hash so a manifest cannot silently drift from the runs
that used it, and ``apply_to_config`` injects ``train`` into the data config.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid a circular import: config imports this module
    import openpi.training.config as _config


def _digest(train: list[int], val: list[int]) -> str:
    payload = json.dumps({"train": sorted(train), "val": sorted(val)}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def load(path: str | pathlib.Path) -> dict[str, Any]:
    data = json.loads(pathlib.Path(path).read_text())
    for key in ("train", "val"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"manifest {path} is missing a '{key}' list")
    train, val = [int(i) for i in data["train"]], [int(i) for i in data["val"]]
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"train/val overlap in {path}: {sorted(overlap)[:10]}")
    if not train:
        raise ValueError(f"manifest {path} has an empty train split")
    digest = data.get("sha256") or data.get("digest")
    if digest is not None and digest != _digest(train, val):
        raise ValueError(f"manifest {path} hash does not match its train/val lists")
    data["train"], data["val"] = train, val
    return data


def save(path: str | pathlib.Path, *, repo_id: str, train: list[int], val: list[int], **extra: Any) -> str:
    payload = {
        "repo_id": repo_id,
        "train": sorted(int(i) for i in train),
        "val": sorted(int(i) for i in val),
        **extra,
    }
    payload["sha256"] = _digest(payload["train"], payload["val"])
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload["sha256"]


def apply_to_config(config: "_config.TrainConfig", manifest: dict[str, Any]) -> "_config.TrainConfig":
    """Return ``config`` with ``data.episode_indices`` set to the manifest train split."""
    data = config.data
    if not hasattr(data, "episode_indices"):
        raise TypeError(
            f"{type(data).__name__} does not support episode subsetting; "
            "use LeRobotSimpleG1Dex3DataConfig or add the field first"
        )
    expected = manifest.get("repo_id")
    if expected and getattr(data, "repo_id", None) != expected:
        raise ValueError(f"manifest is for repo_id={expected!r}, config uses {data.repo_id!r}")
    return dataclasses.replace(config, data=dataclasses.replace(data, episode_indices=tuple(manifest["train"])))
