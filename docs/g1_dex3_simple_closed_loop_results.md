# G1 + Dex3 Tabletop Grasp: SIMPLE Closed-Loop Results

- Date: 2026-10-28
- Policy: `pi05_simple_g1_dex3_tabletop`, checkpoint `run1/3000`
- Simulator: SIMPLE / Isaac Sim 4.5 + MuJoCo 3.3.6, headless
- GPU: RTX 4090 48 GB
- Client: `examples/simple_g1_dex3/simple_client/pi05_openhlm.py`
- Protocol: 32-d state, 36-d action, `action_horizon=30`
- Success criterion: official task reward `>= 0.9`

## Evaluation protocol

The official `simple-eval/G1WholebodyTabletopGraspMP-v0` package was evaluated at all three shipped DR levels:

- 10 episodes per level
- one environment worker
- no manual episode-step override; the task metadata limit of 800 steps applies
- all camera streams recorded

## Results

| Split | Success | Episodes | Rate |
|---|---:|---:|---:|
| `dr-level-0` | 7 | 10 | 70% |
| `dr-level-1` | 7 | 10 | 70% |
| `dr-level-2` | 4 | 10 | 40% |
| **Total** | **18** | **30** | **60%** |

Episode outcomes:

```text
dr-level-0: T T F T F F T T T T
dr-level-1: T T T T F T F T T F
dr-level-2: F F F T F T F T T F
```

The result is the official binary success rate, not an open-loop action MAE.

## Training-time RTC checkpoint (synchronous execution)

On 2026-10-30, checkpoint `rtc_run_20261030/3000` was evaluated with the same
official 30-episode protocol, the synchronous `pi05_openhlm` client, and
`SIMPLE_CHUNK_BLEND_STEPS=6`:

- `dr-level-0`: 7/10 (70%)
- `dr-level-1`: 6/10 (60%)
- `dr-level-2`: 3/10 (30%)
- total: 16/30 (53.3%)

```text
dr-level-0: T T T T T T F F F T
dr-level-1: T T F T T F F T T F
dr-level-2: T F F F F T F F F T
```

This evaluates the weights learned with training-time RTC, but it is **not an
RTC execution benchmark**: the current SIMPLE client waits for each complete
action chunk, does not send `action_prefix` / `inference_delay`, and only applies
the existing six-frame boundary blend. The checkpoint is therefore 2 successes
below `run1/3000` under synchronous execution; this result alone does not measure
whether real-time overlapping chunk execution improves continuity or latency.

## Recorded video

A successful DR-level-0 episode was composited with ffmpeg into a single 1280x720,
50 fps, H.264 video with four synchronized views:

- head camera
- front camera
- right wrist camera
- side camera

Local artifact:

```text
artifacts/simple_g1_dex3/closed_loop_3000/episode_0_dr0_success_2x2.mp4
```

The video and evaluation logs are generated artifacts and are intentionally not committed.

## Deployment issues found and fixed

- Isaac Sim's default `base.python` experience crashed in a headless viewport; `isaacsim.exp.base.zero_delay.kit` starts reliably.
- The AMO submodule and `src/simple/robots/policy/` initially contained Git LFS pointers; the real `amo_jit.pt` and adapter files were materialized.
- Robot, GraspNet, HSSD scene, and material archives were required before environment construction.
- The policy connection is now opened on first inference and closed at reset, avoiding the bundled websockets keepalive timeout during the 60-step stand warmup.
