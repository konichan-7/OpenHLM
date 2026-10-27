"""Offline tests for ``simple_client/pi05_openhlm.py``.

The SIMPLE checkout has a different dependency stack (Isaac Sim, MuJoCo), so
this loads the client with minimal stand-ins for its imports.  It verifies the
two mappings that matter at the robot boundary:

* dataset hand order -> PD order (left hand only)
* 36-d model action -> ``eval_move_actuators`` payloads and AMO command order
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import numpy as np


def _load_client():
    simple = types.ModuleType("simple")
    simple.__path__ = []
    agents = types.ModuleType("simple.agents")
    agents.__path__ = []
    primitive = types.ModuleType("simple.agents.primitive_agent")
    baselines = types.ModuleType("simple.baselines")
    baselines.__path__ = []
    baseline_pi05 = types.ModuleType("simple.baselines.pi05")
    core = types.ModuleType("simple.core")
    core.__path__ = []
    action = types.ModuleType("simple.core.action")
    openpi_client = types.ModuleType("openpi_client")
    openpi_client.__path__ = []
    websocket = types.ModuleType("openpi_client.websocket_client_policy")

    class PrimitiveAgent:
        def __init__(self, robot, **kwargs):
            self.robot = robot

        def reset(self):
            pass

    class ActionCmd:
        def __init__(self, type, **parameters):
            self.type = type
            self.parameters = parameters

        def __getitem__(self, key):
            return self.parameters[key]

    def from_psi0_upper_joints(value):
        return np.concatenate([value[14:28], value[0:3], value[5:7], value[3:5], value[7:14]])

    primitive.PrimitiveAgent = PrimitiveAgent
    action.ActionCmd = ActionCmd
    baseline_pi05.STATE_SLICES = [
        ("left_hand_thumb", 29, 32),
        ("left_hand_middle", 34, 36),
        ("left_hand_index", 32, 34),
        ("right_hand", 36, 43),
        ("left_arm", 15, 22),
        ("right_arm", 22, 29),
    ]
    baseline_pi05.from_psi0_upper_joints = from_psi0_upper_joints
    websocket.WebsocketClientPolicy = object

    for name, module in {
        "simple": simple,
        "simple.agents": agents,
        "simple.agents.primitive_agent": primitive,
        "simple.baselines": baselines,
        "simple.baselines.pi05": baseline_pi05,
        "simple.core": core,
        "simple.core.action": action,
        "openpi_client": openpi_client,
        "openpi_client.websocket_client_policy": websocket,
    }.items():
        sys.modules[name] = module

    path = pathlib.Path(__file__).with_name("simple_client") / "pi05_openhlm.py"
    spec = importlib.util.spec_from_file_location("pi05_openhlm_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def main() -> int:
    client = _load_client()

    row = np.arange(36, dtype=np.float32)
    upper, waist, command = client.split_action(row)
    expected_upper = np.concatenate(
        [
            row[14:28],
            row[0:7][[0, 1, 2, 5, 6, 3, 4]],  # left hand: middle/index -> index/middle
            row[7:14],                          # right hand already PD order
        ]
    )
    np.testing.assert_array_equal(upper, expected_upper)
    assert upper.shape == (28,)

    assert waist == {
        "waist_yaw_joint": 30.0,
        "waist_roll_joint": 28.0,
        "waist_pitch_joint": 29.0,
    }, waist
    # [vx, target_yaw, vy, d_height, torso_yaw, torso_pitch, torso_roll, turning]
    assert command == [0.0, 0.0, 0.0, 0.0, 30.0, 29.0, 28.0, 0.0], command

    proprio = np.arange(43, dtype=np.float32)
    history = np.array([0.0, -0.15, 0.0, 0.75], dtype=np.float32)
    state = client.build_state(proprio, history)[0]
    expected_state = np.concatenate(
        [
            proprio[29:32],
            proprio[34:36],
            proprio[32:34],
            proprio[36:43],
            proprio[15:22],
            proprio[22:29],
            history,
        ]
    )
    np.testing.assert_array_equal(state, expected_state)
    assert state.shape == (32,)

    print("SIMPLE CLIENT MAPPING OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
