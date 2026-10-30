"""Offline tests for ``simple_client/pi05_openhlm.py``.

The SIMPLE checkout has a different dependency stack (Isaac Sim, MuJoCo), so
this loads the client with minimal stand-ins for its imports.  It verifies the
two mappings that matter at the robot boundary:

* dataset hand order -> PD order (left hand only)
* 36-d model action -> ``eval_move_actuators`` payloads and AMO command order
"""

from __future__ import annotations

from collections import deque
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
            self._action_queue = deque()

        def queue_action(self, command):
            self._action_queue.append(command)

        def get_action(self, observation, instruction=None, **kwargs):
            return self._action_queue.popleft()

        def reset(self):
            self._action_queue.clear()

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
    _, _, mobile_command = client.split_action(row, pin_constants=False)
    assert mobile_command == [32.0, 35.0, 33.0, 30.25, 30.0, 29.0, 28.0, 34.0]

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

    raw = np.ones((30, 36), dtype=np.float32)
    previous = np.zeros(36, dtype=np.float32)
    bridge = client.blend_chunk_start(raw, previous, 6)
    assert bridge.shape == raw.shape
    np.testing.assert_array_equal(bridge[0, :31], previous[:31])
    assert np.all(np.diff(bridge[:6, 14]) > 0)
    np.testing.assert_array_equal(bridge[5:], raw[5:])
    np.testing.assert_array_equal(bridge[:, 31:], raw[:, 31:])
    np.testing.assert_array_equal(raw, 1.0)
    np.testing.assert_array_equal(client.blend_chunk_start(raw, previous, 0), raw)
    np.testing.assert_array_equal(client.blend_chunk_start(raw, None, 6), raw)
    assert np.isfinite(client.blend_chunk_start(raw[:2], previous, 6)).all()
    for bad in (np.zeros((0, 36)), np.zeros((30, 35)), np.full((30, 36), np.nan)):
        try:
            client.blend_chunk_start(bad, previous, 6)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chunk was accepted")

    class FakeClient:
        def get_server_metadata(self):
            return {"action_protocol": "simple_g1_dex3", "action_dim": 36}

    robot = types.SimpleNamespace(joint_names=[f"joint_{i}" for i in range(43)])
    agent = client.Pi05OpenhlmAgent(robot, "localhost", 8000, client=FakeClient(), upsample_factor=2)
    first = np.zeros((3, 36), dtype=np.float32)
    agent._enqueue_chunk(first)
    assert len(agent._action_queue) == len(agent._dataset_action_queue) == 6
    obs = {"joint_qpos": np.zeros(43)}
    for _ in range(6):
        agent.get_action(obs)
    np.testing.assert_array_equal(agent._last_executed_action, first[-1])
    agent._enqueue_chunk(raw)
    assert agent.get_last_chunk()[0, 14] < 0.1
    agent.get_action(obs)
    np.testing.assert_array_equal(agent._last_executed_action, agent.get_last_chunk()[0])
    np.testing.assert_allclose(agent._last_cmd_torso_rpyh[:3], agent.get_last_chunk()[0, 28:31])
    agent.reset()
    assert not agent._action_queue and not agent._dataset_action_queue
    assert agent._last_executed_action is None
    agent._enqueue_chunk(raw)
    np.testing.assert_array_equal(agent.get_last_chunk(), raw)

    class FakeRtcClient:
        def __init__(self):
            self.calls = []

        def get_server_metadata(self):
            return {
                "action_protocol": "simple_g1_dex3",
                "action_dim": 36,
                "rtc_mode": "training_time",
                "rtc_max_delay": 2,
            }

        def infer(self, observation):
            self.calls.append(observation)
            if "rtc/action_prefix" not in observation:
                return {"actions": np.repeat(np.arange(6, dtype=np.float32)[:, None], 36, axis=1)}
            prefix = np.asarray(observation["rtc/action_prefix"])
            actions = np.full((6, 36), 100.0, dtype=np.float32)
            actions[: len(prefix)] = prefix
            return {"actions": actions}

    rtc_client = FakeRtcClient()
    rtc_agent = client.Pi05OpenhlmAgent(
        robot,
        "localhost",
        8000,
        client=rtc_client,
        rtc_inference_delay=2,
        chunk_blend_steps=0,
        pin_constant_actions=False,
    )
    rtc_obs = {
        "joint_qpos": np.zeros(43, dtype=np.float32),
        "head_stereo_left": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    for _ in range(7):
        rtc_agent.get_action(rtc_obs)
    assert len(rtc_client.calls) == 2
    np.testing.assert_array_equal(
        rtc_client.calls[1]["rtc/action_prefix"],
        np.repeat(np.array([4, 5], dtype=np.float32)[:, None], 36, axis=1),
    )
    assert rtc_client.calls[1]["rtc/inference_delay"] == 2
    assert len(rtc_agent._dataset_action_queue) == 3
    assert rtc_agent._last_executed_action[0] == 100.0
    assert rtc_agent._rtc_prefix_max_error == 0.0
    rtc_agent.reset()

    print("SIMPLE CLIENT MAPPING + CHUNK BRIDGE + RTC OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
