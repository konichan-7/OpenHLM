"""Random-access reader for the public merged PSI0 LeRobot v3 dataset.

The released 14-task archive stores each split in one parquet file and one
continuous video.  The LeRobot revision pinned by this repository only reads
v2.1 repositories, so this adapter implements the small subset of v3 needed by
the G1/Dex3 training pipeline without rewriting or re-encoding the dataset.
"""

import json
import pathlib
from typing import SupportsIndex

from lerobot.common.datasets.video_utils import decode_video_frames
import numpy as np
import pyarrow.parquet as pq


class Psi0V3Dataset:
    """Read one PSI0 v3 split and construct episode-clamped action chunks."""

    _IMAGE_KEY = "observation.images.egocentric"

    def __init__(self, root: pathlib.Path | str, action_horizon: int, *, include_images: bool = True):
        self.root = pathlib.Path(root)
        self.action_horizon = int(action_horizon)
        self.include_images = include_images
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive")

        info_path = self.root / "meta/info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"PSI0 v3 metadata not found: {info_path}")
        self.info = json.loads(info_path.read_text())
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError(f"expected LeRobot v3.0 at {self.root}, got {self.info.get('codebase_version')!r}")
        self.fps = float(self.info["fps"])
        self._length = int(self.info["total_frames"])

        state_shape = self.info["features"]["states"]["shape"]
        action_shape = self.info["features"]["action"]["shape"]
        if state_shape != [32] or action_shape != [36]:
            raise ValueError(f"expected PSI0 32-state/36-action protocol, got {state_shape}/{action_shape}")

        tasks_table = pq.read_table(self.root / "meta/tasks.parquet")
        self.tasks = dict(
            zip(
                np.asarray(tasks_table["task_index"], dtype=np.int64).tolist(),
                tasks_table["task"].to_pylist(),
                strict=True,
            )
        )

        episodes = pq.read_table(self.root / "meta/episodes/chunk-000/file-000.parquet")
        episode_indices = np.asarray(episodes["episode_index"], dtype=np.int64)
        starts = np.asarray(episodes["dataset_from_index"], dtype=np.int64)
        ends = np.asarray(episodes["dataset_to_index"], dtype=np.int64)
        lengths = np.asarray(episodes["length"], dtype=np.int64)
        if not (
            np.array_equal(starts[1:], ends[:-1])
            and starts[0] == 0
            and ends[-1] == self._length
            and np.array_equal(ends - starts, lengths)
        ):
            raise ValueError("PSI0 episode metadata is not a contiguous partition of the data parquet")

        self._frame_ends = np.repeat(ends, lengths)
        self._frame_episode = np.repeat(episode_indices, lengths)
        video_starts = np.asarray(
            episodes[f"videos/{self._IMAGE_KEY}/from_timestamp"],
            dtype=np.float64,
        )
        max_episode = int(episode_indices.max(initial=-1))
        self._video_starts = np.full(max_episode + 1, np.nan, dtype=np.float64)
        self._video_starts[episode_indices] = video_starts

        self._data_path = self.root / "data/chunk-000/file-000.parquet"
        self._video_path = self.root / f"videos/{self._IMAGE_KEY}/chunk-000/file-000.mp4"
        self._states: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._timestamps: np.ndarray | None = None
        self._task_indices: np.ndarray | None = None

    def __len__(self) -> int:
        return self._length

    def __getstate__(self):
        # DataLoader uses spawn.  Let every worker memory-map/load the compact
        # numeric parquet columns instead of pickling hundreds of MB from the parent.
        state = self.__dict__.copy()
        for key in ("_states", "_actions", "_timestamps", "_task_indices"):
            state[key] = None
        return state

    def _load_columns(self) -> None:
        if self._states is not None:
            return
        table = pq.read_table(
            self._data_path,
            columns=["states", "action", "timestamp", "task_index"],
            memory_map=True,
        )
        if table.num_rows != self._length:
            raise ValueError(f"info.json declares {self._length} frames, parquet has {table.num_rows}")
        self._states = np.asarray(table["states"].combine_chunks().to_pylist(), dtype=np.float32)
        self._actions = np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
        self._timestamps = np.asarray(table["timestamp"], dtype=np.float64)
        self._task_indices = np.asarray(table["task_index"], dtype=np.int64)

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = int(index.__index__())
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        self._load_columns()
        assert self._states is not None
        assert self._actions is not None
        assert self._timestamps is not None
        assert self._task_indices is not None

        end = int(self._frame_ends[idx])
        action_indices = np.minimum(np.arange(idx, idx + self.action_horizon), end - 1)
        if self.include_images:
            episode = int(self._frame_episode[idx])
            timestamp = float(self._video_starts[episode] + self._timestamps[idx])
            image = decode_video_frames(
                self._video_path,
                [timestamp],
                tolerance_s=0.5 / self.fps + 1e-4,
            )[0]
        else:
            # Normalisation only consumes state/actions.  Avoid decoding 432k
            # video frames while retaining the regular transform contract.
            image = np.zeros((3, 1, 1), dtype=np.float32)

        return {
            self._IMAGE_KEY: image,
            "states": self._states[idx],
            "action": self._actions[action_indices],
            "task_index": self._task_indices[idx],
        }
