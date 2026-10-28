# Rerun G1 Dataset Playback

Play one episode from the released SIMPLE G1/Dex3 LeRobot dataset with the real
G1 URDF in Rerun.

## Install

```bash
uv tool install rerun-sdk
uv venv ~/.venvs/rerun-g1 --python 3.12
uv pip install --python ~/.venvs/rerun-g1/bin/python \
  rerun-sdk numpy pyarrow yourdfpy trimesh scipy av
```

## Generate a recording

```bash
cd /home/pc/桌面/reappear/OpenHLM
~/.venvs/rerun-g1/bin/python tools/rerun_g1_dataset/play_g1_dataset.py \
  --episode 0 \
  --out artifacts/simple_g1_dex3/rerun/episode_000000.rrd
```

Play the complete 100-episode dataset continuously:

```bash
cd /home/pc/桌面/reappear/OpenHLM
~/.venvs/rerun-g1/bin/python tools/rerun_g1_dataset/play_g1_dataset.py \
  --all \
  --out artifacts/simple_g1_dex3/rerun/full_dataset.rrd
```

## Open the viewer

```bash
rerun artifacts/simple_g1_dex3/rerun/episode_000000.rrd
```

The recording contains:

- real 43-DoF G1 + Dex3 URDF robot with forward kinematics
- 15 leg/waist joints, 14 arm joints, 14 hand joints
- egocentric 640x360 camera stream
- `episode`, global `frame`, and 50 Hz `time` timelines

The complete dataset recording is 12,226 frames and 244.52 seconds. Use the
`episode` timeline to jump between demonstrations, or the `time` timeline to
play the full 4-minute sequence.

## Dataset facts

- `meta/info.json`: `total_tasks = 1`
- only task: `grasp the object on the tabletop.`
- only training camera: `observation.images.egocentric`
- no wrist camera in the training dataset

The right-hand dataset order is thumb/index/middle; the URDF order is
thumb/middle/index, so the playback script applies that single permutation.
