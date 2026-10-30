import numpy as np
import torch

from openpi.policies import policy


class _RtcInputTransform:
    def __call__(self, data):
        result = {
            "image": {"base_0_rgb": np.zeros((4, 4, 3), dtype=np.float32)},
            "image_mask": {"base_0_rgb": np.True_},
            "state": np.zeros(3, dtype=np.float32),
            "tokenized_prompt": np.zeros(4, dtype=np.int32),
            "tokenized_prompt_mask": np.ones(4, dtype=bool),
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32) * 2.0
        return result


class _FakeTorchModel:
    def __init__(self):
        self.kwargs = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def sample_actions(self, device, observation, **kwargs):
        del device
        self.kwargs = kwargs
        return torch.zeros(observation.state.shape[0], 2, 3)


def test_rtc_prefix_uses_policy_input_normalization():
    model = _FakeTorchModel()
    trained_policy = policy.Policy(
        model,
        transforms=[_RtcInputTransform()],
        is_pytorch=True,
        pytorch_device="cpu",
    )
    raw_prefix = np.ones((2, 3), dtype=np.float32)

    result = trained_policy.infer({}, action_prefix=raw_prefix, inference_delay=2)

    assert result["actions"].shape == (2, 3)
    assert model.kwargs is not None
    torch.testing.assert_close(model.kwargs["action_prefix"], torch.full((2, 3), 2.0))
    assert model.kwargs["inference_delay"] == 2
