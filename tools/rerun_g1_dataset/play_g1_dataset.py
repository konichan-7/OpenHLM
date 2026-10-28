#!/usr/bin/env python3
"""Play SIMPLE G1/Dex3 episodes with the real G1 URDF in Rerun.

The released LeRobot episode stores:

* 15 leg/waist joint positions
* 14 arm joint positions
* 14 Dex3 hand joint positions
* one egocentric 640x360 video

This script runs forward kinematics with ``yourdfpy`` and logs the URDF's 49
visual meshes plus the camera stream to Rerun.  The generated ``.rrd`` can be
opened with the local ``rerun`` viewer.

Examples:

    /home/pc/.venvs/rerun-g1/bin/python tools/rerun_g1_dataset/play_g1_dataset.py \
      --episode 0 \
      --out artifacts/simple_g1_dex3/rerun/episode_000000.rrd

    /home/pc/.venvs/rerun-g1/bin/python tools/rerun_g1_dataset/play_g1_dataset.py \
      --all \
      --out artifacts/simple_g1_dex3/rerun/full_dataset.rrd

    rerun artifacts/simple_g1_dex3/rerun/episode_000000.rrd
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import av
import numpy as np
import pyarrow.parquet as pq
import rerun as rr
import trimesh
from yourdfpy import URDF


def _resolve_mesh_path(urdf_dir: pathlib.Path, filename: str) -> pathlib.Path:
    if filename.startswith("package://"):
        raise ValueError(f"package:// meshes are not supported: {filename}")
    return (urdf_dir / filename).resolve()


def _link_color(link_name: str) -> tuple[int, int, int, int]:
    if "hand" in link_name:
        return (235, 127, 42, 255)
    if "shoulder" in link_name or "elbow" in link_name or "wrist" in link_name:
        return (73, 150, 235, 255)
    if "hip" in link_name or "knee" in link_name or "ankle" in link_name:
        return (70, 190, 120, 255)
    if "head" in link_name or "d435" in link_name:
        return (240, 200, 55, 255)
    return (180, 185, 195, 255)


def _load_joint_episode(parquet_path: pathlib.Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    columns = [
        "observation.leg_joints",
        "observation.arm_joints",
        "observation.hand_joints",
        "observation.prev_height",
    ]
    table = pq.read_table(parquet_path, columns=columns).to_pydict()
    legs = np.asarray(table["observation.leg_joints"], dtype=np.float32)
    arms = np.asarray(table["observation.arm_joints"], dtype=np.float32)
    hands = np.asarray(table["observation.hand_joints"], dtype=np.float32)
    heights = np.asarray(table["observation.prev_height"], dtype=np.float32).reshape(-1)
    if legs.shape[1] != 15 or arms.shape[1] != 14 or hands.shape[1] != 14:
        raise ValueError(f"unexpected joint shapes: {legs.shape}, {arms.shape}, {hands.shape}")

    # Right-hand data order is thumb, index, middle.  The URDF order is thumb,
    # middle, index, so only the right hand needs a permutation.
    right_hand_to_urdf = (0, 1, 2, 5, 6, 3, 4)
    joints = {}
    for i in range(len(legs)):
        joints[str(i)] = np.concatenate(
            [
                legs[i],
                arms[i],
                hands[i, :7],
                hands[i, 7:][list(right_hand_to_urdf)],
            ]
        )
    return joints, heights


def _urdf_joint_names(urdf: URDF) -> list[str]:
    names = [joint.name for joint in urdf.actuated_joints]
    if len(names) != 43:
        raise ValueError(f"expected 43 actuated joints, got {len(names)}")
    return names


def _log_static_meshes(urdf: URDF, urdf_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Log each URDF visual mesh once and return link->entity mapping."""
    entities: dict[str, pathlib.Path] = {}
    for link_name, link in urdf.link_map.items():
        for visual_idx, visual in enumerate(link.visuals):
            mesh = visual.geometry.mesh
            if mesh is None:
                continue
            mesh_path = _resolve_mesh_path(urdf_dir, mesh.filename)
            loaded = trimesh.load(mesh_path, force="mesh", process=False)
            vertices = np.asarray(loaded.vertices, dtype=np.float32)
            if mesh.scale is not None:
                vertices = vertices * np.asarray(mesh.scale, dtype=np.float32)
            entity = f"world/g1/{link_name}/visual_{visual_idx}"
            rr.log(
                entity,
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=np.asarray(loaded.faces, dtype=np.uint32),
                    albedo_factor=_link_color(link_name),
                ),
                static=True,
            )
            entities[link_name] = entity
    if not entities:
        raise RuntimeError("URDF contains no visual meshes")
    return entities


def _four_by_four_matrix_to_transform(matrix: np.ndarray) -> rr.Transform3D:
    return rr.Transform3D(
        translation=matrix[:3, 3].astype(np.float32),
        mat3x3=matrix[:3, :3].astype(np.float32),
    )


def play_episodes(
    dataset_root: pathlib.Path,
    urdf_path: pathlib.Path,
    episodes: list[int],
    out_path: pathlib.Path,
) -> None:
    if not episodes:
        raise ValueError("no episodes selected")
    urdf = URDF.load(
        urdf_path,
        load_meshes=False,
        build_scene_graph=True,
    )
    joint_names = _urdf_joint_names(urdf)

    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Horizontal(
            rr.blueprint.Spatial3DView(origin="world", name="G1 URDF"),
            rr.blueprint.Spatial2DView(origin="camera/head", name="Egocentric camera"),
            column_shares=[2.0, 1.0],
        ),
    )
    app_id = "openhlm_g1_full_dataset" if len(episodes) > 1 else f"openhlm_g1_episode_{episodes[0]:06d}"
    rr.init(app_id, spawn=False, default_blueprint=blueprint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(out_path)

    _log_static_meshes(urdf, urdf_path.parent)
    rr.log(
        "dataset",
        rr.TextLog(
            f"G1 + Dex3 URDF | {len(episodes)} episode(s) | "
            f"{episodes[0]:06d}..{episodes[-1]:06d} | 50 Hz"
        ),
        static=True,
    )

    total_frames = 0
    total_seconds = 0.0
    for episode in episodes:
        parquet_path = dataset_root / f"data/chunk-000/episode_{episode:06d}.parquet"
        video_path = dataset_root / f"videos/chunk-000/egocentric/episode_{episode:06d}.mp4"
        if not parquet_path.exists() or not video_path.exists():
            raise FileNotFoundError(f"episode {episode} is missing parquet or video")

        joint_frames, heights = _load_joint_episode(parquet_path)
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 50)
        video_iter = container.decode(video=0)
        print(f"episode {episode:06d}: {len(joint_frames)} frames")

        for frame_idx in range(len(joint_frames)):
            try:
                video_frame = next(video_iter)
            except StopIteration:
                print(f"warning: episode {episode:06d} video ended at frame {frame_idx}", file=sys.stderr)
                break

            cfg = {
                name: float(value)
                for name, value in zip(joint_names, joint_frames[str(frame_idx)], strict=True)
            }
            urdf.update_cfg(cfg)
            base = np.eye(4, dtype=np.float32)
            base[2, 3] = float(heights[frame_idx])

            rr.set_time("episode", sequence=episode)
            rr.set_time("frame", sequence=total_frames)
            rr.set_time("time", duration=total_seconds)
            rr.log("metrics/episode", rr.Scalars(float(episode)))
            rr.log("metrics/height", rr.Scalars(float(heights[frame_idx])))

            for link_name in urdf.link_map:
                link_transform = urdf.get_transform(link_name)
                for visual_idx, visual in enumerate(urdf.link_map[link_name].visuals):
                    if visual.geometry.mesh is None:
                        continue
                    visual_entity = f"world/g1/{link_name}/visual_{visual_idx}"
                    transform = base @ link_transform @ visual.origin
                    rr.log(visual_entity, _four_by_four_matrix_to_transform(transform))

            image = video_frame.to_ndarray(format="rgb24")
            rr.log("camera/head", rr.Image(image))
            total_frames += 1
            total_seconds += 1.0 / fps

        container.close()

    rr.disconnect()
    print(f"wrote {out_path} ({len(episodes)} episodes, {total_frames} frames, {total_seconds:.2f}s)")


def play(
    dataset_root: pathlib.Path,
    urdf_path: pathlib.Path,
    episode: int,
    out_path: pathlib.Path,
) -> None:
    play_episodes(dataset_root, urdf_path, [episode], out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/pc/.cache/openhlm/rerun_g1_dataset/G1WholebodyTabletopGraspMP-v0"),
    )
    ap.add_argument(
        "--urdf",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/pc/桌面/reappear/OpenHLM/src/GR00T-WholeBodyControl4OpenHLM/"
            "gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf"
        ),
    )
    ap.add_argument("--episode", type=int, default=None, help="single episode index")
    ap.add_argument("--all", action="store_true", help="play every episode in order")
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
    )
    args = ap.parse_args()
    if args.all:
        episodes = sorted(
            int(path.stem.split("_")[-1])
            for path in args.dataset_root.glob("data/chunk-*/episode_*.parquet")
        )
        default_out = pathlib.Path("artifacts/simple_g1_dex3/rerun/full_dataset.rrd")
    else:
        episodes = [0 if args.episode is None else args.episode]
        default_out = pathlib.Path(
            f"artifacts/simple_g1_dex3/rerun/episode_{episodes[0]:06d}.rrd"
        )
    out = args.out or default_out
    play_episodes(args.dataset_root, args.urdf, episodes, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
