import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from openpi.training import psi0_v3_dataset


def _write_dataset(root):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos/observation.images.egocentric/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_frames": 5,
                "fps": 50,
                "features": {
                    "states": {"shape": [32]},
                    "action": {"shape": [36]},
                },
            }
        )
    )
    pq.write_table(
        pa.table({"task_index": [0, 1], "task": ["task zero", "task one"]}),
        root / "meta/tasks.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "dataset_from_index": [0, 2],
                "dataset_to_index": [2, 5],
                "length": [2, 3],
                "videos/observation.images.egocentric/from_timestamp": [0.0, 10.0],
            }
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    states = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)
    actions = np.repeat(np.arange(5, dtype=np.float32)[:, None], 36, axis=1)
    pq.write_table(
        pa.table(
            {
                "states": states.tolist(),
                "action": actions.tolist(),
                "timestamp": np.array([0.0, 0.02, 0.0, 0.02, 0.04]),
                "task_index": [0, 0, 1, 1, 1],
            }
        ),
        root / "data/chunk-000/file-000.parquet",
    )
    (root / "videos/observation.images.egocentric/chunk-000/file-000.mp4").touch()


def test_reads_v3_and_clamps_action_chunks_to_episode(monkeypatch, tmp_path):
    _write_dataset(tmp_path)
    decoded_timestamps = []

    def fake_decode(path, timestamps, tolerance_s):
        del path, tolerance_s
        decoded_timestamps.extend(timestamps)
        return torch.ones(1, 3, 4, 5)

    monkeypatch.setattr(psi0_v3_dataset, "decode_video_frames", fake_decode)
    dataset = psi0_v3_dataset.Psi0V3Dataset(tmp_path, action_horizon=3)

    assert len(dataset) == 5
    assert dataset.tasks == {0: "task zero", 1: "task one"}
    first_tail = dataset[1]
    np.testing.assert_array_equal(first_tail["action"][:, 0], [1, 1, 1])
    second_start = dataset[2]
    np.testing.assert_array_equal(second_start["action"][:, 0], [2, 3, 4])
    assert second_start["task_index"] == 1
    assert tuple(second_start["observation.images.egocentric"].shape) == (3, 4, 5)
    np.testing.assert_allclose(decoded_timestamps, [0.02, 10.0])


def test_norm_stats_mode_skips_video_decode(monkeypatch, tmp_path):
    _write_dataset(tmp_path)

    def fail_decode(*args, **kwargs):
        raise AssertionError("video must not be decoded for norm stats")

    monkeypatch.setattr(psi0_v3_dataset, "decode_video_frames", fail_decode)
    dataset = psi0_v3_dataset.Psi0V3Dataset(tmp_path, action_horizon=2, include_images=False)
    item = dataset[0]
    assert item["observation.images.egocentric"].shape == (3, 1, 1)
    np.testing.assert_array_equal(item["action"][:, 0], [0, 1])
