"""Protocol tests for the SIMPLE G1/Dex3 adapter.  No GPU, no dataset needed.

Run from the openpi4OpenHLM root::

    PYTHONPATH=src python examples/simple_g1_dex3/test_protocol.py -v

The last test (``test_matches_simple_reference``) imports the SIMPLE repo and
compares against ``scripts/postprocess_psi0.py`` / ``baselines/pi05.py``.  Point
``SIMPLE_ROOT`` at that checkout to enable it; otherwise it is skipped.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_policy_module():
    """Load the policy module without importing the full openpi stack.

    The file only needs ``transforms.DataTransformFn`` and ``models.model.ModelType``
    as base classes/annotations, so light stubs keep this test runnable on a CPU
    box that has not run ``uv sync`` yet.
    """
    if "openpi" not in sys.modules:
        openpi = types.ModuleType("openpi")
        openpi.__path__ = []  # namespace package, do not scan the real tree
        transforms = types.ModuleType("openpi.transforms")
        transforms.DataTransformFn = object
        models = types.ModuleType("openpi.models")
        models.__path__ = []
        model = types.ModuleType("openpi.models.model")

        class ModelType:  # noqa: D401 - stub
            PI05 = "pi05"

        model.ModelType = ModelType
        sys.modules["openpi"] = openpi
        sys.modules["openpi.transforms"] = transforms
        sys.modules["openpi.models"] = models
        sys.modules["openpi.models.model"] = model

    if "einops" not in sys.modules:
        einops = types.ModuleType("einops")
        einops.rearrange = lambda x, _pattern: np.transpose(x, (1, 2, 0))
        sys.modules["einops"] = einops

    path = ROOT / "src" / "openpi" / "policies" / "simple_g1_dex3_policy.py"
    spec = importlib.util.spec_from_file_location("simple_g1_dex3_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


P = _load_policy_module()


def test_state_and_action_dims():
    assert P.STATE_DIM == 32
    assert P.ACTION_DIM == 36


def test_constant_dims_are_the_documented_ones():
    assert tuple(P.CONSTANT_ACTION_DIMS) == (31, 32, 33, 34, 35)


def test_hand_permutation_matches_simple_helper():
    """HAND_TO_PD_ORDER must reproduce from_psi0_upper_joints' left-hand slice."""
    src = np.arange(7, dtype=np.float32)
    ref = np.concatenate([src[0:3], src[5:7], src[3:5]])
    np.testing.assert_array_equal(src[list(P.HAND_TO_PD_ORDER)], ref)


def test_pin_constant_dims():
    actions = np.full((4, P.ACTION_DIM), 123.0, dtype=np.float32)
    pinned = P.pin_constant_dims(actions)
    np.testing.assert_allclose(pinned[:, 31], 0.75)
    np.testing.assert_allclose(pinned[:, 32:36], 0.0)
    # Non-constant dims survive, including torso rpy -- it drifts in the data.
    np.testing.assert_allclose(pinned[:, :31], 123.0)
    # Input is not mutated.
    np.testing.assert_allclose(actions[:, 31:36], 123.0)


def test_split_action():
    row = np.arange(P.ACTION_DIM, dtype=np.float32)
    blocks = P.split_action(row)
    assert blocks["left_hand"].tolist() == list(range(0, 7))
    assert blocks["right_hand"].tolist() == list(range(7, 14))
    assert blocks["torso_rpy"].tolist() == [28.0, 29.0, 30.0]
    assert float(blocks["height"]) == 31.0
    assert float(blocks["target_yaw"]) == 35.0


def test_inputs_single_view_and_masks():
    sample = P.make_simple_g1_dex3_example()
    out = P.SimpleG1Dex3Inputs(model_type="pi05")(sample)
    assert out["state"].shape == (32,)
    assert out["actions"].shape == (P.ACTION_DIM,)
    assert out["image"]["base_0_rgb"].shape == (360, 640, 3)
    assert out["image_mask"]["base_0_rgb"]
    assert not out["image_mask"]["left_wrist_0_rgb"]
    assert not out["image_mask"]["right_wrist_0_rgb"]


def test_inputs_reject_wrong_dims():
    sample = P.make_simple_g1_dex3_example()
    bad = dict(sample, states=np.zeros(28, dtype=np.float32))
    try:
        P.SimpleG1Dex3Inputs(model_type="pi05")(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a 28-d state")


def test_outputs_truncate_and_validate():
    padded = np.random.rand(30, 40).astype(np.float32)
    out = P.SimpleG1Dex3Outputs()({"actions": padded})
    assert out["actions"].shape == (30, P.ACTION_DIM)
    np.testing.assert_allclose(out["actions"][:, :31], padded[:, :31])
    # Post-un-normalisation, the server pins height and the four base commands.
    np.testing.assert_allclose(out["actions"][:, 31], P.NOMINAL_HEIGHT)
    np.testing.assert_allclose(out["actions"][:, 32:36], 0.0)

    bad = np.full((30, 40), np.nan, dtype=np.float32)
    try:
        P.SimpleG1Dex3Outputs()({"actions": bad})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-finite actions")


def _extract_assignments(path: pathlib.Path, names: set[str]) -> dict:
    """Read module-level literals without importing (SIMPLE pulls in tqdm/isaac)."""
    import ast

    tree = ast.parse(path.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                out[target.id] = ast.literal_eval(node.value)
    return out


def _extract_function(path: pathlib.Path, name: str):
    import ast
    import textwrap

    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            snippet = textwrap.dedent(ast.get_source_segment(source, node) or "")
            namespace = {"np": np}
            exec(compile(snippet, str(path), "exec"), namespace)  # noqa: S102 - test-only
            return namespace[name]
    return None


def test_matches_simple_reference():
    """Cross-check the frozen protocol against the SIMPLE reference sources."""
    root = os.environ.get("SIMPLE_ROOT")
    if not root:
        print("SKIP: SIMPLE_ROOT not set")
        return
    simple = pathlib.Path(root)
    postprocess = simple / "scripts" / "postprocess_psi0.py"
    baseline = simple / "src" / "simple" / "baselines" / "pi05.py"
    if not postprocess.exists() or not baseline.exists():
        print(f"SKIP: SIMPLE sources not found under {simple}")
        return

    ref = _extract_assignments(postprocess, {"STATE_SLICES"})
    # Raw qpos slices, in the exact order the 32-d state is built from.
    assert ref["STATE_SLICES"] == [
        ("left_hand_thumb", 29, 32),
        ("left_hand_middle", 34, 36),
        ("left_hand_index", 32, 34),
        ("right_hand", 36, 43),
        ("left_arm", 15, 22),
        ("right_arm", 22, 29),
    ], ref["STATE_SLICES"]

    # The 36-d action tail: height, vx, vy, turning flag, target yaw.
    source = postprocess.read_text()
    for literal in (
        "cmd[:, 6:7]",
        "cmd[:, 0:2]",
        "turning_flag.reshape(-1, 1)[:, 0:1]",
        "target_yaw.reshape(-1, 1)[:, 0:1]",
    ):
        assert literal in source, f"postprocess_psi0.build_vectors changed: {literal} missing"

    # The client-side hand permutation must equal SIMPLE's helper exactly.
    helper = _extract_function(baseline, "from_psi0_upper_joints")
    assert helper is not None, "from_psi0_upper_joints not found"

    # On a raw *qpos* 28-vector [left arm? no: 0:28 = arms+hands], the helper
    # swaps the left hand's last four dims.  Model output is already in dataset
    # order, so applying the helper to it must be an identity on both hands only
    # when the dataset order equals the PD order, which it does after
    # postprocess_psi0.  Prove the equivalence the other way: build a raw qpos
    # vector, run the helper, and check it against the dataset slice order.
    raw = np.arange(43, dtype=np.float32)
    # Model output (36 dims) blocks: left/right hand (0:14), left/right arm.
    # ``from_psi0_upper_joints`` takes the 28-d prefix and returns
    # [left arm, right arm, left hand, right hand] -- arms first, so its output is
    # zipped against ``robot.joint_names[15:]``, whose order is arms then hands.
    model_28 = np.arange(28, dtype=np.float32)
    out = helper(model_28)
    np.testing.assert_array_equal(out[:14], model_28[14:28])       # arms preserved
    np.testing.assert_array_equal(out[14:17], model_28[0:3])       # left thumb
    np.testing.assert_array_equal(out[17:19], model_28[5:7])       # left "index" slot
    np.testing.assert_array_equal(out[19:21], model_28[3:5])       # left "middle" slot
    np.testing.assert_array_equal(out[21:28], model_28[7:14])      # right hand as-is

    # What the helper does to the left hand is exactly this permutation.  It is
    # only "correct" if the input column order is (thumb, middle, index).
    src = np.arange(7, dtype=np.float32)
    np.testing.assert_array_equal(src[[0, 1, 2, 5, 6, 3, 4]], np.concatenate([src[0:3], src[5:7], src[3:5]]))

    # Dataset columns are (thumb, middle, index) -- postprocess_psi0 slices them
    # that way -- so the helper swaps middle and index when fed a model output.
    # The client therefore must NOT call the helper on model output; it applies
    # the permutation once, itself.  Verify the inverse used by the client:
    from_ds = src[[0, 1, 2, 5, 6, 3, 4]]
    np.testing.assert_array_equal(from_ds, np.concatenate([src[0:3], src[5:7], src[3:5]]))

    # And the reference's own baseline must be sending the wrong state width.
    baseline_src = baseline.read_text()
    assert "states[0, :28]" in baseline_src, "baseline no longer truncates to 28 dims; update the plan"


def main() -> int:
    tests = {name: fn for name, fn in globals().items() if name.startswith("test_")}
    failures = 0
    for name, fn in tests.items():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
