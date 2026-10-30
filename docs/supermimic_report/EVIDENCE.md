# SuperMimic audit evidence

Audit date: 2026-10-30 (Asia/Shanghai).

## Fixed sources

- Paper: *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*, Qiayuan Liao et al., arXiv:2610.28480v1, 2026-10-20, 19 pages.
- Retrieved from the user's Zotero library using the Zotero local API. Item key `IQ8SWVH7`; PDF attachment key `TQR99565`; exported BibTeX key `liao_supermimic_2026`. These are distinct identifiers.
- Repository: https://github.com/zita-ch/supermimic_vibed
- Commit: `8bfc1158bd968f729fcc65d5e897a05b05ea3d36` (2026-10-28).
- The repository describes itself as an **unofficial reimplementation**, targeting LAFAN1 Fig. 4(a), not a complete implementation of every paper experiment. Its historical claim about official code availability is not treated as a verified current fact.
- `source_manifest.json` records the PDF checksum and reviewed source-file checksums. The source checkout and paper PDF are not redistributed.

## Validation performed

Command (against an external clone pinned to the commit above):

```sh
python verify_alignment.py /absolute/path/to/supermimic_vibed
```

Environment: Python 3.11.18; PyTorch 2.15.0+cpu; NumPy 2.4.6. Result: **16/16 passed** (8 upstream unit tests, 8 additional checks). All Python sources parsed with `ast.parse`.

An explicit `supermimic.kinematics` import shim bypasses the unavailable Isaac Lab import chain. It is not a replacement implementation, does not supply FK, and is never used to make a physics claim. The selected upstream motion tests construct synthetic in-memory reference arrays; buffer and reset-sampling tests use actual upstream modules.

Additional checks cover dimensions/state round-trip, yaw/xy feature invariance, actual world-model local-increment integration using a small hidden width and controlled outputs, input gradients through frozen model weights, reset/window/history boundaries, survival indexing, the lambda-return recurrence, and soft-termination gradients. The survival/GAE check is an independent reference calculation, not execution of the Isaac-dependent training loop. No full-size trained model was evaluated.

Not run: Isaac Lab/Newton/MuJoCo-Warp simulation, asset FK matching, CUDA/bf16/torch.compile execution, full GPU training, checkpoints, LAFAN1 returns, BONES-SEED scaling, multi-GPU or real-robot deployment. README/MILESTONES performance values are repository-author claims, not independently reproduced here.

## Primary alignment anchors

| Paper | Source at fixed commit | Assessment |
|---|---|---|
| Fig. 2, Algorithm A.1 | `train.py:259-352` | collect / fit dynamics / imagine actor-critic |
| Eq. (1), §3.2 | `train.py:90-108,147-156`; `networks.py:67-134` | k=8 autoregression; standardized local-state error; analytic heads have no learned losses |
| Eq. (2), §3.3 | `train.py:175-222` | H=32, survival, critic bootstrap; freeze weights without detaching state gradients |
| Eq. (3) | `train.py:223-252` | fixed detached lambda-return targets, 5 x 4 critic optimization |
| App. A, Tables A.1-A.3 | `mdp.py:18-51,94-161`; `env.py:72-90,142-208` | same main reward/DR thresholds; extra 6D orientation observation and soft termination are explicit choices |
| App. B/C, Table A.4 | `networks.py:137-165` | motion encoder/decoder/critic only; downstream interfaces absent |
| App. D.1 | `env.py:131-140,225-234`; `reset_sampling.py` | same reset mixture; EMA and timeout heuristic are extra implementation details |
| App. F.1 | `evaluate.py:52-96` | deterministic first-frame evaluation; clip-mean and frame-weighted returns |

## Important distinctions for the report

- State storage is 71-dimensional, invariant features are also 71-dimensional but contain different quantities, and dynamics outputs are 70-dimensional. These are not interchangeable.
- The paper's single observation has 93 dimensions; the repository has 99, giving 990 history inputs. Goal size is 73 x 10 = 730; world-model input 71+29+33 = 133; critic input 914.
- Dropping a self-collision **reward term** in the paper does not imply disabling self-collision **physics**. The repository disables the latter separately.
- Simulator reward is computed on `S1, f1` and stored with executed noisy action. `done` includes timeout for buffer masking; the analytic imagined termination only uses height/tilt.
- Root/body reward frames must be read literally: body pose targets are xy/yaw-reanchored, root reward remains global, reference body velocities are used in world coordinates.
- The reported ~1.47 h / 0.829 / 0 failures applies to 40 LAFAN1 clips; it does not establish the 129,514-clip BONES-SEED or hardware claims.
- `wm_err8` uses adaptive output-standardization and component floors, not a proven match to the paper's Fig. 3(c) nominal-error normalization.
- Resume restores networks/optimizers/counters but not replay, sampling statistics, simulation state or RNG; it is not bitwise continuation.
- Training wall time begins after initialization and warmup collection; build/data/evaluation are excluded. Timing boundaries and hardware differ from the paper.
