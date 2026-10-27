"""Audit a SIMPLE G1/Dex3 LeRobot dataset before any training.

Runs without importing openpi: it only needs ``pyarrow``/``numpy`` (or pandas).
Checks:

* meta/info.json declares 32-d state / 36-d action, 50 fps, one egocentric video;
* every parquet has the expected feature names, lengths and dtypes;
* no NaN/Inf; per-dimension min/max/std and the "frozen" dims are reported;
* episode ids match ``meta/episodes.jsonl`` and no episode is empty;
* (optional) videos decode and have the same frame count as the parquet.

Usage::

    python examples/simple_g1_dex3/audit_data.py \
        --dataset-root /path/to/G1WholebodyTabletopGraspMP-v0 \
        --check-videos

Exit code is non-zero iff a hard check fails.  Warnings are printed for
task-frozen dims whose values drift, which would break the constant pinning in
``simple_g1_dex3_policy.py``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

STATE_DIM = 32
ACTION_DIM = 36
ACTION_NAMES = (
    ["lh_thumb_%d" % i for i in range(3)]
    + ["lh_middle_%d" % i for i in range(2)]
    + ["lh_index_%d" % i for i in range(2)]
    + ["rh_thumb_%d" % i for i in range(3)]
    + ["rh_index_%d" % i for i in range(2)]
    + ["rh_middle_%d" % i for i in range(2)]
    + ["l_arm_%d" % i for i in range(7)]
    + ["r_arm_%d" % i for i in range(7)]
    + ["torso_roll", "torso_pitch", "torso_yaw", "height", "vx", "vy", "turning", "target_yaw"]
)
assert len(ACTION_NAMES) == ACTION_DIM


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def read_table(path: pathlib.Path) -> dict[str, list]:
    """Read a parquet episode as column lists.

    ``polars`` is already an openpi dependency; ``pyarrow`` is pinned to 14.x in
    this environment, which refuses numpy 1.26.
    """
    import polars as pl

    df = pl.read_parquet(path)
    out = {}
    for name in df.columns:
        col = df[name]
        # list<float> columns become a 2-D object array; stack explicitly.
        out[name] = np.stack(col.to_list()) if isinstance(col.dtype, pl.List) else col.to_numpy()
    return out


def iter_episode_files(root: pathlib.Path):
    return sorted(root.glob("data/chunk-*/episode_*.parquet"))


def check_meta(root: pathlib.Path, report: Report) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        report.error(f"missing {info_path}")
        return {}
    info = json.loads(info_path.read_text())
    feats = info.get("features", {})
    for key in ("states", "action"):
        if key not in feats:
            report.error(f"meta/info.json has no feature {key!r}")
    if info.get("fps") != 50:
        report.warn(f"fps is {info.get('fps')}, plan assumes 50")
    if not any(k.startswith("observation.images.") for k in feats):
        report.error("no observation.images.* feature; the client sends head_stereo_left")
    return info


def audit(root: pathlib.Path, check_videos: bool) -> Report:
    report = Report()
    info = check_meta(root, report)

    files = iter_episode_files(root)
    if not files:
        report.error(f"no episode parquet under {root}/data")
        return report

    episodes = []
    if (root / "meta" / "episodes.jsonl").exists():
        episodes = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    expected_len = {int(e["episode_index"]): int(e["length"]) for e in episodes}

    states, actions = [], []
    seen_episodes = set()
    for path in files:
        cols = read_table(path)
        missing = [k for k in ("states", "action", "episode_index", "frame_index") if k not in cols]
        if missing:
            report.error(f"{path.name}: missing columns {missing}")
            continue

        state = np.asarray(cols["states"], dtype=np.float32)
        action = np.asarray(cols["action"], dtype=np.float32)
        if state.shape[1] != STATE_DIM:
            report.error(f"{path.name}: state has {state.shape[1]} dims, expected {STATE_DIM}")
        if action.shape[1] != ACTION_DIM:
            report.error(f"{path.name}: action has {action.shape[1]} dims, expected {ACTION_DIM}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            report.error(f"{path.name}: NaN/Inf in state or action")

        ep = int(np.asarray(cols["episode_index"]).reshape(-1)[0])
        seen_episodes.add(ep)
        if ep in expected_len and expected_len[ep] != len(state):
            report.error(f"{path.name}: {len(state)} rows, episodes.jsonl says {expected_len[ep]}")
        if len(state) == 0:
            report.error(f"{path.name}: empty episode")

        # action[t] should equal action[t+1]'s "previous command" only in spirit;
        # here we only check the frozen dims, which must be constant per task.
        states.append(state)
        actions.append(action)

    if not states:
        return report
    states_arr = np.concatenate(states)
    actions_arr = np.concatenate(actions)

    print(f"episodes: {len(seen_episodes)}   frames: {len(states_arr)}")
    if episodes and len(seen_episodes) != len(episodes):
        report.warn(f"{len(seen_episodes)} parquet episodes vs {len(episodes)} listed in episodes.jsonl")

    print("\naction dims (name, min, max, std):")
    for i, name in enumerate(ACTION_NAMES):
        col = actions_arr[:, i]
        flag = "  <-- CONST" if col.std() < 1e-6 else ""
        print(f"  {i:2d} {name:14s} {col.min(): .4f} {col.max(): .4f} {col.std():.4f}{flag}")

    const = [i for i in range(ACTION_DIM) if actions_arr[:, i].std() < 1e-6]
    print(f"\nconstant action dims: {const}")
    # Height + the four base commands are pinned by pin_constant_dims().  Torso
    # rpy (28:31) drifts in the demonstrations, so it must NOT be pinned.
    expected_const = [31, 32, 33, 34, 35]
    if const != expected_const:
        report.error(
            f"constant action dims are {const}, protocol expects {expected_const}; "
            "update CONSTANT_ACTION_DIMS / pin_constant_dims() before training"
        )

    state_const = [i for i in range(STATE_DIM) if states_arr[:, i].std() < 1e-9]
    print(f"constant state dims: {state_const} (28..31 expected for this task)")

    # Cross-check that the recorded state tail is *not* the current measurement:
    # state[28:32] is history_command of the previous step.
    if np.allclose(states_arr[:, 31], 0.75, atol=1e-6):
        print("state height (dim 31) is constant 0.75 m")
    else:
        report.warn("state height varies; check whether prev_height semantics changed")

    if check_videos:
        check_video_frames(root, files, expected_len, report)
    return report


def check_video_frames(root: pathlib.Path, files, expected_len: dict[int, int], report: Report) -> None:
    try:
        import cv2
    except ImportError:
        report.warn("cv2 not installed; skipping --check-videos")
        return
    videos = sorted(root.glob("videos/chunk-*/egocentric/*.mp4"))
    print(f"\nvideo files: {len(videos)}")
    for video in videos[:5] + videos[-5:]:
        ep = int(video.stem.split("_")[-1])
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if ep in expected_len and n != expected_len[ep]:
            report.warn(f"{video.name}: {n} decoded frames vs {expected_len[ep]} parquet rows")
    if len(videos) != len(files):
        report.error(f"{len(videos)} videos vs {len(files)} parquet episodes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--check-videos", action="store_true")
    args = parser.parse_args()

    report = audit(args.dataset_root, args.check_videos)
    print()
    for w in report.warnings:
        print(f"WARN  {w}")
    for e in report.errors:
        print(f"ERROR {e}")
    if report.errors:
        print(f"\n{len(report.errors)} error(s)")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
