#!/usr/bin/env python3
"""CPU-only checks against a pinned, external checkout; no simulator or training.

The kinematics module is replaced ONLY to bypass the unavailable Isaac Lab import
chain. No FK, asset, actuator, contact, or physics behavior is tested by this shim.
Run: python verify_alignment.py /absolute/path/to/supermimic_vibed
Dependencies: torch, numpy. The external checkout is never modified.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import subprocess
import sys
import types
import unittest

PIN = "8bfc1158bd968f729fcc65d5e897a05b05ea3d36"
ROOT = pathlib.Path(sys.argv[1]).resolve()
sys.argv = sys.argv[:1]
sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
if sha != PIN:
    raise SystemExit(f"Expected {PIN}, found {sha}")
for path in sorted(ROOT.rglob("*.py")):
    if ".git" not in path.parts:
        ast.parse(path.read_text(), filename=str(path))
print(f"Source pin: {sha}; all Python files parsed", flush=True)
sys.path.insert(0, str(ROOT))
shim = types.ModuleType("supermimic.kinematics")
shim.G1Kinematics = type("UnavailableKinematics", (), {})
sys.modules[shim.__name__] = shim

import torch
from supermimic.buffer import FIELDS, ReplayBuffer
from supermimic.mdp import INV_DIM, OBS_DIM, MDP, invariant_features, flatten_state, unflatten_state
from supermimic.networks import WorldModel
from supermimic.rotations import so3_exp, yaw_mat


def state(n=3):
    return (
        torch.randn(n, 3), so3_exp(torch.randn(n, 3) * 0.2),
        torch.randn(n, 3), torch.randn(n, 3),
        torch.randn(n, 29), torch.randn(n, 29),
    )


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_dimensions_and_state_round_trip(self):
        s = state()
        x = flatten_state(s)
        self.assertEqual(x.shape, (3, 71))
        self.assertEqual(invariant_features(s).shape, (3, 71))
        self.assertEqual((OBS_DIM, INV_DIM), (99, 71))
        for a, b in zip(s, unflatten_state(x)):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_heading_and_xy_translation_invariance(self):
        p, r, v, w, q, qd = s = state()
        z = so3_exp(torch.tensor([[0.0, 0.0, 0.7]])).expand(3, -1, -1)
        rotate = lambda x: (z @ x[..., None])[..., 0]
        moved = (rotate(p) + torch.tensor([2.0, -3.0, 0.0]), z @ r, rotate(v), rotate(w), q, qd)
        torch.testing.assert_close(invariant_features(s), invariant_features(moved), atol=2e-6, rtol=2e-6)

    def test_world_model_local_increment_integration(self):
        s, target = state(), state()
        wm = WorldModel(hidden=(16,))
        delta = wm.targets(s, target)
        self.assertEqual(delta.shape[-1], 70)
        # Make the actual network output the exact local-increment target for each row.
        # A per-row constant model avoids replacing WorldModel.step itself.
        for row in range(3):
            with torch.no_grad():
                wm.net[-1].bias.copy_(delta[row] / wm.out_norm.std())
            actual = wm.step(tuple(x[row:row+1] for x in s), torch.zeros(1, 29), torch.zeros(1, 33))
            for x, y in zip(actual, target):
                torch.testing.assert_close(x, y[row:row+1], atol=4e-6, rtol=4e-6)

    def test_frozen_world_model_retains_action_gradient(self):
        wm = WorldModel(hidden=(16,))
        with torch.no_grad():
            wm.net[-1].weight.normal_(std=0.01)
        for p in wm.parameters():
            p.requires_grad_(False)
        a = torch.randn(3, 29, requires_grad=True)
        out = wm.step(state(), a, torch.zeros(3, 33))
        sum(x.square().sum() for x in out).backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(a.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in wm.parameters()))

    def test_done_transition_is_included_then_window_is_masked(self):
        b = ReplayBuffer(4, 1, "cpu")
        for t in range(4):
            tr = {k: torch.zeros((1,) + ((sh,) if isinstance(sh, int) else sh), dtype=dt)
                  for k, (sh, dt) in FIELDS.items()}
            tr["done"][:] = t == 1
            b.add(tr)
        actual = b.sample_windows(2, 4)["mask"]
        torch.testing.assert_close(actual, torch.tensor([[1., 1., 0., 0.]]).expand(2, -1))

    def test_history_does_not_cross_episode_or_ring_boundary(self):
        b = ReplayBuffer(4, 1, "cpu")
        for t in range(7):
            tr = {k: torch.zeros((1,) + ((sh,) if isinstance(sh, int) else sh), dtype=dt)
                  for k, (sh, dt) in FIELDS.items()}
            tr["s"][:] = t
            tr["ep_step"][:] = (t - 4) if t >= 4 else t
            b.add(tr)
        s, h = b.sample_starts(256, 5)
        for current, seq in zip(s["s"][:, 0].tolist(), h["s"][:, :, 0].tolist()):
            earliest = 4 if current >= 4 else 3
            expected = [max(earliest, current - k) for k in range(4, -1, -1)]
            self.assertEqual(seq, expected)

    def test_survival_indexing_and_lambda_target_reference(self):
        d = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
        r = torch.tensor([1., 2., 3.], dtype=torch.float64)
        gamma, lam = 0.99, 0.95
        alive = torch.cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, dtype=d.dtype), alive[:-1]))
        torch.testing.assert_close(c_t, torch.tensor([1., .8, .56], dtype=d.dtype))
        v = torch.tensor([.5, .6, .7, .8], dtype=d.dtype)
        R, targets = v[-1], []
        for t in range(2, -1, -1):
            R = r[t] + gamma * (1-d[t]) * ((1-lam)*v[t+1] + lam*R)
            targets.append(R)
        # Independently expand the lambda recurrence, not a simulator return.
        r2 = 3 + gamma * .6 * .8
        r1 = 2 + gamma * .7 * (.05 * .7 + .95 * r2)
        r0 = 1 + gamma * .8 * (.05 * .6 + .95 * r1)
        torch.testing.assert_close(torch.stack(targets[::-1]), torch.tensor([r0, r1, r2], dtype=d.dtype))

    def test_soft_termination_value_and_gradient(self):
        mdp = MDP.__new__(MDP)
        mdp.cfg = types.SimpleNamespace(term_height=.25, term_tilt=.8)
        mdp.motion = types.SimpleNamespace(root_p=torch.zeros(1,3), root_R=torch.eye(3)[None])
        p = torch.tensor([[0., 0., .25]], requires_grad=True)
        r = torch.eye(3)[None]
        s = (p, r, torch.zeros(1,3), torch.zeros(1,3), torch.zeros(1,29), torch.zeros(1,29))
        d = mdp.soft_termination(s, torch.tensor([0]))
        torch.testing.assert_close(d, torch.tensor([.5]))
        d.sum().backward()
        self.assertGreater(p.grad[0,2].item(), 0)


suite = unittest.TestSuite()
loader = unittest.TestLoader()
for name in ("test_buffer", "test_reset_sampling", "test_motion_goal"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite.addTests(loader.loadTestsFromModule(module))
suite.addTests(loader.loadTestsFromTestCase(AlignmentTests))
print(f"Runtime: Python {sys.version.split()[0]}, torch {torch.__version__}; CPU only", flush=True)
print("Kinematics import shim active: NO physics/FK/training validation", flush=True)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
