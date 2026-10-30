"""Policy transforms for the SIMPLE G1 + dual Dex3-1 tabletop grasp task.

This module defines the frozen wire protocol between the SIMPLE simulation
environment and the pi0.5 policy server. Everything here mirrors
``scripts/postprocess_psi0.py`` in the SIMPLE repository; when that file changes,
this module must change with it (and the tests in
``examples/simple_g1_dex3/`` must be re-run).

Protocol (fixed for the first phase):

* one first-person RGB image (``observation.images.egocentric``), 640x360 -> 224x224
* 32-dim state, built from raw robot qpos + the previous high-level command
* 36-dim absolute action (NOT the OpenHLM 34-dim G1 protocol)

State layout (32) -- hands first, in *dataset* order::

    [ 0: 3] left  Dex3 thumb 0/1/2
    [ 3: 5] left  Dex3 middle 0/1
    [ 5: 7] left  Dex3 index  0/1
    [ 7:10] right Dex3 thumb 0/1/2
    [10:12] right Dex3 index 0/1
    [12:14] right Dex3 middle 0/1
    [14:21] left  arm         shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
    [21:28] right arm         shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
    [28:31] previous command torso roll, pitch, yaw
    [31]    previous command height (absolute, metres)

Action layout (36) -- hands first, then arms (same block order as the state)::

    [ 0: 7] left  Dex3   thumb 0/1/2, middle 0/1, index 0/1   (dataset order)
    [ 7:14] right Dex3   thumb 0/1/2, index 0/1, middle 0/1   (dataset order)
    [14:21] left  arm    (hands come first; arms are 14:28)
    [21:28] right arm
    [28:31] torso roll, pitch, yaw
    [31]    absolute height (metres)
    [32]    torso vx
    [33]    torso vy
    [34]    turning flag (bool-like, not a yaw rate)
    [35]    target yaw

Hand sub-order.  ``postprocess_psi0.py`` reads the raw qpos as
``[29:32] + [34:36] + [32:34]``, so the left hand is stored as
thumb / *middle* / index; the right hand is stored as thumb / index / middle
(``[36:39] + [41:43] + [39:41]`` on raw qpos 36:43).

The PD controller (``LEFT_HAND_JOINTS`` / ``RIGHT_HAND_JOINTS`` in SIMPLE's
``g1_wholebody.py``) wants thumb / index / middle on both hands.  So the client
maps each hand with ``HAND_TO_PD_ORDER = (0, 1, 2, 5, 6, 3, 4)``: the left hand
gets swapped, the right hand does not.

Note the block order: hands (0:14) come *before* arms (14:28).  The OpenHLM 34-d
G1 protocol has arms first; copying that layout here would silently command the
hands with arm targets.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

STATE_DIM = 32
ACTION_DIM = 36

# Dims that are *exactly* constant across all 12,226 frames of the released
# package: height is 0.75 m and the four base-command dims are zero.  Torso rpy
# (28:31) is NOT constant -- it drifts a few hundredths of a radian during the
# rollout -- so it is left to the model and to quantile normalisation.
CONSTANT_ACTION_DIMS = (31, 32, 33, 34, 35)
NOMINAL_HEIGHT = 0.75

# Dataset hand order -> Dex3 PD order (thumb 0/1/2, index 0/1, middle 0/1).
#
#   left  columns 3:5 hold middle0/1 and 5:7 hold index0/1 -> swap them
#   right columns 10:12 hold index0/1 and 12:14 hold middle0/1 -> already PD order
#
# Verified against observation.hand_joints and the SIMPLE joint-name lists;
# test_protocol.py cross-checks this against from_psi0_upper_joints.
HAND_TO_PD_ORDER = (0, 1, 2, 5, 6, 3, 4)

# Action -> client units.  SIMPLE sends ``height - 0.75`` (a delta) and expects
# torso rpy in (roll, pitch, yaw) for the PD targets, (yaw, pitch, roll) for the
# AMO command.  These conversions live in the client, not here.
DEFAULT_PROMPT = "grasp the object on the tabletop."


def make_simple_g1_dex3_example() -> dict:
    """One random training example in the *dataset* format (pre-repack)."""
    return {
        "observation.images.egocentric": np.random.randint(
            256, size=(360, 640, 3), dtype=np.uint8
        ),
        "states": np.random.rand(STATE_DIM).astype(np.float32) * 2.0 - 1.0,
        "action": np.random.rand(ACTION_DIM).astype(np.float32) * 2.0 - 1.0,
        "prompt": DEFAULT_PROMPT,
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SimpleG1Dex3Inputs(transforms.DataTransformFn):
    """Map the SIMPLE observation/action sample to the pi0.5 model input.

    Accepts either the raw dataset keys (``observation.images.egocentric``,
    ``states``, ``action``) or the repacked keys (``observation/image``,
    ``observation/state``, ``actions``).  The training data config uses
    ``RepackTransform`` first, so training hits the repacked path; the policy
    server receives the repacked path directly from the client.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        image = data.get("observation/image", data.get("observation.images.egocentric"))
        state = data.get("observation/state", data.get("states"))
        if image is None:
            raise KeyError("expected 'observation/image' (or 'observation.images.egocentric')")
        if state is None:
            raise KeyError("expected 'observation/state' (or 'states')")

        state = np.asarray(state, dtype=np.float32)
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"state must be {STATE_DIM}-d, got shape {state.shape}")

        base_image = _parse_image(image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                # No wrist cameras in this task.  These are placeholders that are
                # explicitly masked out; they must never be interpreted as real.
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.bool_(False),
                "right_wrist_0_rgb": np.bool_(False),
            },
        }

        actions = data.get("actions", data.get("action"))
        if actions is not None:
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"action must be {ACTION_DIM}-d, got shape {actions.shape}")
            inputs["actions"] = actions

        prompt = data.get("prompt", DEFAULT_PROMPT)
        if prompt is not None:
            if isinstance(prompt, np.ndarray):
                prompt = prompt.item()
            inputs["prompt"] = str(prompt)

        return inputs


@dataclasses.dataclass(frozen=True)
class SimpleG1Dex3Outputs(transforms.DataTransformFn):
    """Return the first 36 action dims, un-normalised, for the SIMPLE client."""

    pin_constants: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < ACTION_DIM:
            raise ValueError(
                f"model produced {actions.shape[-1]} action dims, expected >= {ACTION_DIM}"
            )
        actions = actions[..., :ACTION_DIM]
        if not np.isfinite(actions).all():
            raise ValueError("model produced non-finite actions")
        if self.pin_constants:
            # Zero-IQR dims are numerical noise after un-normalisation in the
            # fixed-base tabletop dataset.  Full PSI0 contains locomotion and
            # deliberately disables this override.
            actions = pin_constant_dims(actions)
        return {"actions": actions}


def split_action(action36: np.ndarray) -> dict[str, np.ndarray]:
    """Split a 36-d action row into named blocks (numpy only, no torch)."""
    a = np.asarray(action36)
    return {
        "left_hand": a[..., 0:7],
        "right_hand": a[..., 7:14],
        "left_arm": a[..., 14:21],
        "right_arm": a[..., 21:28],
        "torso_rpy": a[..., 28:31],
        "height": a[..., 31],
        "vx": a[..., 32],
        "vy": a[..., 33],
        "turning_flag": a[..., 34],
        "target_yaw": a[..., 35],
    }


def pin_constant_dims(actions: np.ndarray) -> np.ndarray:
    """Force the task-constant action dims to their dataset values.

    These dims have (near) zero quantile range in the training data, so the
    un-normalised model output there is numerical noise amplified by a
    ~1e-6 denominator.  Writing the constants back is both safer and closer to
    what the demonstrations actually contain.
    """
    actions = np.array(actions, copy=True)
    # Height plus the four base commands; torso rpy (28:31) is left untouched.
    actions[..., 31] = NOMINAL_HEIGHT
    actions[..., 32:36] = 0.0
    return actions
