# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT
"""SIMPLE client for the OpenHLM ``pi05_simple_g1_dex3_tabletop`` policy.

This is a *separate* client, not a replacement for ``simple.baselines.pi05``:
the original baseline is left untouched so results stay comparable.

Differences from ``pi05.py``:

* sends the full 32-d state instead of ``states[0, :28]`` (the baseline silently
  drops the previous command's torso rpy + height, which are present in training);
* never truncates the policy output to 28 dims before the joint/AMO conversion;
* checks shape, dtype and finiteness of every returned chunk;
* resets the action queue, the command history and the episode bookkeeping
  together, so a new episode cannot inherit the previous height/rpy;
* uses the dataset prompt by default (``grasp the object on the tabletop.``).

Protocol reference: ``OpenHLM/src/openpi4OpenHLM/src/openpi/policies/simple_g1_dex3_policy.py``.
The action layout (36 dims) is::

     0: 7  left  Dex3   thumb0/1/2, middle0/1, index0/1
     7:14  right Dex3   thumb0/1/2, index0/1, middle0/1
    14:21  left  arm
    21:28  right arm
    28:31  torso roll, pitch, yaw
    31     absolute height (m)
    32     vx
    33     vy
    34     turning flag
    35     target yaw

Hand sub-order: the dataset stores the left hand as thumb / middle / index and
the right hand as thumb / index / middle.  The PD controller wants
thumb / index / middle on both hands, so only the left hand is re-ordered here.
Do NOT reuse ``from_psi0_upper_joints`` on model output: it applies the same
left-hand swap again, which double-permutes and silently mis-drives the fingers.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from simple.agents.primitive_agent import PrimitiveAgent
from simple.baselines.pi05 import STATE_SLICES
from simple.core.action import ActionCmd

logger = logging.getLogger(__name__)

# Nominal standing height used by SIMPLE; the action's dim 31 is absolute metres
# and the AMO command wants ``height - 0.75``.
NOMINAL_HEIGHT = 0.75
STATE_DIM = 32
ACTION_DIM = 36
ACTION_HORIZON = 30
DEFAULT_PROMPT = "grasp the object on the tabletop."

# Task-frozen dims, verified over all 12,226 frames of the released package:
# height is exactly 0.75 m and vx / vy / turning flag / target yaw are all 0.
# Torso rpy is NOT frozen (it drifts by ~0.05 rad), so it is taken from the model.
CONST_HEIGHT = NOMINAL_HEIGHT
CONST_VX = 0.0
CONST_VY = 0.0
CONST_TURNING = 0.0
CONST_TARGET_YAW = 0.0

# Dataset hand columns -> Dex3 PD order (thumb 0/1/2, index 0/1, middle 0/1).
# Left hand only: it is stored as thumb/middle/index.  The right hand is already
# thumb/index/middle in the dataset.
LEFT_HAND_TO_PD = (0, 1, 2, 5, 6, 3, 4)


def blend_chunk_start(actions: np.ndarray, previous: np.ndarray | None, steps: int) -> np.ndarray:
    """Fade a boundary offset out over ``steps`` model frames, in dataset order.

    This is a synchronous boundary bridge, NOT asynchronous overlap/RTC. Preserve
    the new chunk's relative motion and horizon; only correct its first 31 joint
    and torso coordinates. Height and discrete/base commands are never blended.
    """
    result = np.array(actions, dtype=np.float32, copy=True)
    if result.ndim != 2 or result.shape[1] != ACTION_DIM or not len(result):
        raise ValueError(f"expected nonempty (H, {ACTION_DIM}), got {result.shape}")
    if not np.isfinite(result).all() or steps < 0:
        raise ValueError("actions must be finite and blend steps must be nonnegative")
    if previous is None or steps == 0:
        return result
    previous = np.asarray(previous, dtype=np.float32)
    if previous.shape != (ACTION_DIM,) or not np.isfinite(previous).all():
        raise ValueError("previous action must be a finite 36-d vector")
    count = min(steps, len(result))
    if count == 1:
        return result
    phase = np.linspace(0.0, 1.0, count, dtype=np.float32)
    weight = phase * phase * (3.0 - 2.0 * phase)
    offset = previous[:31] - result[0, :31]
    result[:count, :31] += (1.0 - weight[:, None]) * offset
    return result


def build_state(proprio11, last_cmd_torso_rpyh: np.ndarray) -> np.ndarray:
    """Assemble the 32-d state exactly like the training data pipeline."""
    proprio = np.asarray(proprio11, dtype=np.float32)[None]
    state = np.concatenate(
        [proprio[:, s:e] for _, s, e in STATE_SLICES] + [np.asarray(last_cmd_torso_rpyh, dtype=np.float32)[None]],
        axis=1,
    )
    if state.shape != (1, STATE_DIM):
        raise ValueError(f"state must be (1, {STATE_DIM}), got {state.shape}")
    return state


def split_action(
    action: np.ndarray,
    *,
    pin_constants: bool = True,
) -> tuple[np.ndarray, dict[str, float], list[float]]:
    """Convert one 36-d row into the three payloads ``eval_move_actuators`` wants.

    Returns ``(target_qpos, waist_qpos, action_command)`` where ``action_command``
    is the 8-d AMO vector ``[vx, target_yaw, vy, d_height, torso_yaw, torso_pitch,
    torso_roll, turning_flag]``.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"action must be ({ACTION_DIM},), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("policy returned non-finite action")

    # 28 upper-body joint targets.  ``robot.joint_names[15:]`` is
    # [left arm(7), right arm(7), left hand(7), right hand(7)] -- arms first,
    # then hands -- while the model output is hands first, then arms.
    left_hand = action[0:7][list(LEFT_HAND_TO_PD)]
    right_hand = action[7:14]
    left_arm = action[14:21]
    right_arm = action[21:28]
    upper = np.concatenate([left_arm, right_arm, left_hand, right_hand]).astype(np.float32)

    # Torso rpy comes from the model: dataset action[28:31] is (roll, pitch, yaw)
    # and SIMPLE sends it as (roll, pitch, yaw) to the PD targets but
    # (yaw, pitch, roll) to the AMO command -- the two orders below are not a typo.
    roll, pitch, yaw = (float(v) for v in action[28:31])

    if pin_constants:
        # Safe compatibility mode for the tabletop-only dataset.
        height = CONST_HEIGHT
        vx, vy, turning, target_yaw = CONST_VX, CONST_VY, CONST_TURNING, CONST_TARGET_YAW
    else:
        # Full PSI0 includes mobile manipulation; preserve all high-level
        # commands learned from its 14 tasks.
        height = float(action[31])
        vx, vy, turning, target_yaw = (float(v) for v in action[32:36])

    waist_qpos = {
        "waist_yaw_joint": float(yaw),
        "waist_roll_joint": float(roll),
        "waist_pitch_joint": float(pitch),
    }
    command = [
        vx,
        target_yaw,
        vy,
        float(height - NOMINAL_HEIGHT),  # AMO wants a delta from the nominal 0.75 m
        float(yaw),
        float(pitch),
        float(roll),
        turning,
    ]

    # ``upper`` is in robot.joint_names[15:] order: 14 arm + 14 hand.
    return upper, waist_qpos, command


class Pi05OpenhlmAgent(PrimitiveAgent):
    """SIMPLE agent that talks to an OpenHLM pi0.5 WebSocket policy server.

    Name matters: ``simple.cli.eval`` derives the class from the module name via
    ``snake_to_pascal("pi05_openhlm") -> "Pi05Openhlm"``, so ``--policy pi05_openhlm``
    resolves to this class.  ``Pi05OpenHLMAgent`` is kept as an alias for code
    that spells it the other way.
    """

    def __init__(
        self,
        robot,
        host: str,
        port: int,
        *,
        instruction: str | None = None,
        upsample_factor: int = 1,
        max_chunk: int | None = None,
        chunk_blend_steps: int | None = None,
        rtc_inference_delay: int | None = None,
        pin_constant_actions: bool | None = None,
        client=None,
        **kwargs,
    ):
        super().__init__(robot, **kwargs)
        if upsample_factor < 1:
            raise ValueError("upsample_factor must be >= 1")
        if max_chunk is not None and max_chunk < 1:
            raise ValueError("max_chunk must be >= 1")
        if chunk_blend_steps is None:
            chunk_blend_steps = int(os.environ.get("SIMPLE_CHUNK_BLEND_STEPS", "6"))
        if chunk_blend_steps < 0:
            raise ValueError("chunk_blend_steps must be >= 0")
        if rtc_inference_delay is None:
            rtc_inference_delay = int(os.environ.get("SIMPLE_RTC_INFERENCE_DELAY", "0"))
        if not 0 <= rtc_inference_delay < ACTION_HORIZON:
            raise ValueError(f"rtc_inference_delay must be in [0, {ACTION_HORIZON})")
        if rtc_inference_delay and upsample_factor != 1:
            raise ValueError("RTC currently requires upsample_factor=1 so delay is measured in model frames")
        if max_chunk is not None and rtc_inference_delay and max_chunk <= rtc_inference_delay:
            raise ValueError("max_chunk must exceed rtc_inference_delay")
        if pin_constant_actions is None:
            pin_constant_actions = os.environ.get("SIMPLE_PIN_CONSTANT_ACTIONS", "1") != "0"

        self.server_ip = host
        self.server_port = port
        self.instruction = instruction or DEFAULT_PROMPT
        self.upsample_factor = int(upsample_factor)
        self.max_chunk = max_chunk
        self.chunk_blend_steps = chunk_blend_steps
        self.rtc_inference_delay = int(rtc_inference_delay)
        self.pin_constant_actions = bool(pin_constant_actions)
        self._dataset_action_queue = deque()
        self._last_executed_action: np.ndarray | None = None
        # SIMPLE creates the agent before its 60-step stand warmup.  Connecting
        # here leaves the socket idle long enough for the bundled websockets
        # client to drop it, so connect lazily on the first inference instead.
        self.client = client
        if client is not None:
            self._check_server_metadata()

        self._global_step_idx = 0
        self._last_cmd_torso_rpyh = np.array([0.0, -0.15, 0.0, NOMINAL_HEIGHT], dtype=np.float32)
        self._reset_history = True
        self._last_chunk: np.ndarray | None = None
        self._rtc_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openhlm-rtc")
        self._rtc_future: Future | None = None
        self._rtc_prefix: np.ndarray | None = None
        self._rtc_requests = 0
        self._rtc_blocked_ms = 0.0
        self._rtc_prefix_max_error = 0.0

    # --------------------------------------------------------------- handshake
    def _ensure_client(self) -> None:
        if self.client is None:
            self.client = _websocket_client_policy.WebsocketClientPolicy(
                host=self.server_ip,
                port=self.server_port,
                api_key=None,
            )
            self._check_server_metadata()

    def _check_server_metadata(self) -> None:
        """Fail fast if the server is not the matching 32/36-d policy.

        The OpenHLM server sends ``policy_metadata`` during the WebSocket
        handshake.  Training configs that predate this field send nothing, so an
        absent protocol tag is tolerated but logged.
        """
        getter = getattr(self.client, "get_server_metadata", None)
        metadata = getter() if callable(getter) else None
        if not metadata:
            if self.rtc_inference_delay:
                raise RuntimeError("RTC execution requires server metadata")
            logger.warning("policy server sent no metadata; cannot verify the action protocol")
            return
        protocol = metadata.get("action_protocol") or metadata.get("policy_metadata", {}).get("action_protocol")
        if protocol is None:
            if self.rtc_inference_delay:
                raise RuntimeError("RTC server metadata has no action_protocol tag")
            logger.warning("policy server metadata has no action_protocol tag: %s", metadata)
            return
        if protocol != "simple_g1_dex3":
            raise RuntimeError(f"server speaks {protocol!r}, client expects 'simple_g1_dex3'")
        action_dim = metadata.get("action_dim") or metadata.get("policy_metadata", {}).get("action_dim")
        if action_dim is not None and int(action_dim) != ACTION_DIM:
            raise RuntimeError(f"server action_dim={action_dim}, client expects {ACTION_DIM}")
        if self.rtc_inference_delay:
            policy_metadata = metadata.get("policy_metadata", metadata)
            if policy_metadata.get("rtc_mode") != "training_time":
                raise RuntimeError("RTC execution requires a checkpoint trained with rtc_mode='training_time'")
            max_delay = int(policy_metadata.get("rtc_max_delay", -1))
            if self.rtc_inference_delay > max_delay:
                raise RuntimeError(
                    f"client RTC delay {self.rtc_inference_delay} exceeds the trained maximum {max_delay}"
                )

    # ------------------------------------------------------------------ input
    def _build_observation(self, observation) -> dict:
        state = build_state(observation["joint_qpos"], self._last_cmd_torso_rpyh)
        image = observation["head_stereo_left"]
        obs = {
            "observation/image": np.asarray(image),
            "observation/state": state[0],
            "prompt": self.instruction,
            "reset": bool(self._reset_history),
        }
        self._reset_history = False
        return obs

    # ----------------------------------------------------------------- output
    def _enqueue_chunk(self, actions: np.ndarray, *, blend: bool = True) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"expected (H, {ACTION_DIM}) actions, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("policy returned non-finite actions")
        if self.max_chunk is not None:
            actions = actions[: self.max_chunk]
        if self._action_queue or self._dataset_action_queue:
            raise RuntimeError("cannot replace a chunk before its queue is exhausted")
        if blend:
            actions = blend_chunk_start(actions, self._last_executed_action, self.chunk_blend_steps)
        self._last_chunk = actions

        for row in actions:
            upper, waist_qpos, command = split_action(row, pin_constants=self.pin_constant_actions)
            target_qpos = dict(zip(self.robot.joint_names[15:], upper, strict=True))
            for _ in range(self.upsample_factor):
                self.queue_action(
                    ActionCmd(
                        "eval_move_actuators",
                        target_qpos=target_qpos,
                        action_command=command,
                        waist_qpos=waist_qpos,
                    )
                )
                self._dataset_action_queue.append(row.copy())

    def _infer(self, obs: dict) -> dict:
        self._ensure_client()
        assert self.client is not None
        response = self.client.infer(obs)
        if not isinstance(response, dict) or "actions" not in response:
            raise RuntimeError(f"policy server returned unexpected payload: {type(response)}")
        return response

    def _start_rtc_request(self, observation) -> None:
        delay = self.rtc_inference_delay
        if delay == 0 or self._rtc_future is not None:
            return
        if len(self._dataset_action_queue) != delay:
            raise RuntimeError(f"RTC request must start with exactly {delay} queued actions")
        prefix = np.stack(list(self._dataset_action_queue), axis=0)
        obs = self._build_observation(observation)
        obs["rtc/action_prefix"] = prefix
        obs["rtc/inference_delay"] = delay
        self._ensure_client()
        self._rtc_prefix = prefix
        self._rtc_requests += 1
        self._rtc_future = self._rtc_executor.submit(self._infer, obs)

    def _consume_rtc_response(self) -> None:
        if self._rtc_future is None or self._rtc_prefix is None:
            raise RuntimeError("RTC queue exhausted without an in-flight request")
        wait_started = time.monotonic()
        response = self._rtc_future.result()
        self._rtc_blocked_ms += (time.monotonic() - wait_started) * 1000.0
        actions = np.asarray(response["actions"], dtype=np.float32)
        delay = self.rtc_inference_delay
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(actions) <= delay:
            raise ValueError(f"RTC response must be (H, {ACTION_DIM}) with H>{delay}, got {actions.shape}")
        prefix_error = float(np.max(np.abs(actions[:delay] - self._rtc_prefix)))
        self._rtc_prefix_max_error = max(self._rtc_prefix_max_error, prefix_error)
        if prefix_error > 1e-3:
            raise RuntimeError(f"RTC server did not preserve the action prefix (max error {prefix_error:.6f})")

        # The old chunk executed the complete prefix while inference ran.
        # Continue with only the newly generated suffix; no boundary blend is
        # needed because prefix conditioning supplies the continuity constraint.
        self._enqueue_chunk(actions[delay:], blend=False)
        self._rtc_future = None
        self._rtc_prefix = None

    def get_action(self, observation, instruction=None, info=None, conditions=None, **kwargs):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        if len(self._action_queue) == 0:
            if self._rtc_future is not None:
                self._consume_rtc_response()
            else:
                response = self._infer(self._build_observation(observation))
                self._enqueue_chunk(response["actions"])

        if (
            self.rtc_inference_delay
            and self._rtc_future is None
            and len(self._action_queue) == self.rtc_inference_delay
        ):
            self._start_rtc_request(observation)

        action_cmd = super().get_action(observation, instruction, **kwargs)
        if action_cmd.type == "eval_move_actuators":
            self._last_executed_action = self._dataset_action_queue.popleft()
            # History for the next observation: the command we just issued.
            self._last_cmd_torso_rpyh = np.array(
                [
                    action_cmd["action_command"][6],  # torso roll
                    action_cmd["action_command"][5],  # torso pitch
                    action_cmd["action_command"][4],  # torso yaw
                    action_cmd["action_command"][3] + NOMINAL_HEIGHT,  # absolute height
                ],
                dtype=np.float32,
            )
        self._last_pred_action = action_cmd
        self._global_step_idx += 1
        return action_cmd

    # ------------------------------------------------------------------ reset
    def reset(self):
        # Clear the queued actions *before* touching the history so a partial
        # episode cannot leak targets into the next one.
        self._action_queue.clear()
        super().reset()
        if self.rtc_inference_delay:
            logger.info(
                "RTC episode stats: requests=%d blocked_ms=%.1f prefix_max_error=%.6f",
                self._rtc_requests,
                self._rtc_blocked_ms,
                self._rtc_prefix_max_error,
            )
        ws = getattr(self.client, "_ws", None) if self.client is not None else None
        if ws is not None:
            ws.close()
        if self._rtc_future is not None:
            self._rtc_future.cancel()
        self._rtc_executor.shutdown(wait=True, cancel_futures=True)
        self._rtc_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openhlm-rtc")
        self._rtc_future = None
        self._rtc_prefix = None
        self._rtc_requests = 0
        self._rtc_blocked_ms = 0.0
        self._rtc_prefix_max_error = 0.0
        self.client = None
        self._global_step_idx = 0
        self._reset_history = True
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._last_chunk = None
        self._dataset_action_queue.clear()
        self._last_executed_action = None
        self._last_cmd_torso_rpyh = np.array([0.0, -0.15, 0.0, NOMINAL_HEIGHT], dtype=np.float32)

    # ------------------------------------------------------------- inspection
    def get_last_chunk(self) -> np.ndarray | None:
        return self._last_chunk


# Alias for callers that write the acronym in upper case.
Pi05OpenHLMAgent = Pi05OpenhlmAgent
