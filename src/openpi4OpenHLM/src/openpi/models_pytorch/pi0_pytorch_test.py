import types

import pytest
import torch

from openpi.models import pi0_config

check = pytest.importorskip("transformers.models.siglip.check", reason="transformers_replace is not installed")
if not check.check_whether_transformers_replace_is_installed_correctly():
    pytest.skip("transformers_replace is not installed correctly", allow_module_level=True)

from openpi.models_pytorch import pi0_pytorch  # noqa: E402

BATCH = 2
HORIZON = 8
ACTION_DIM = 36
NUM_IMAGE_TOKENS = 4
NUM_LANG_TOKENS = 6


def _make_model(**overrides) -> pi0_pytorch.PI0Pytorch:
    kwargs = {
        "pi05": True,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "action_dim": ACTION_DIM,
        "action_horizon": HORIZON,
        "dtype": "float32",
        "use_training_time_rtc": True,
        "rtc_max_delay": 3,
    }
    kwargs.update(overrides)
    torch.manual_seed(0)
    model = pi0_pytorch.PI0Pytorch(pi0_config.Pi0Config(**kwargs))
    width = pi0_config._gemma.get_config("dummy").width  # noqa: SLF001
    # Replace SigLIP with a cheap deterministic embedding; the RTC logic only concerns the suffix.
    model.paligemma_with_expert.embed_image = lambda img: (
        img.flatten(1)[:, :NUM_IMAGE_TOKENS, None].expand(-1, -1, width).contiguous()
    )
    return model


def _make_observation():
    generator = torch.Generator().manual_seed(1)
    images = {
        key: torch.rand(BATCH, 3, 224, 224, generator=generator) * 2 - 1
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    return types.SimpleNamespace(
        images=images,
        image_masks={key: torch.ones(BATCH, dtype=torch.bool) for key in images},
        state=torch.zeros(BATCH, ACTION_DIM),
        tokenized_prompt=torch.randint(1, 100, (BATCH, NUM_LANG_TOKENS), generator=generator),
        tokenized_prompt_mask=torch.ones(BATCH, NUM_LANG_TOKENS, dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )


def test_rtc_config_validation():
    with pytest.raises(ValueError, match="pi05"):
        _make_model(pi05=False)
    with pytest.raises(ValueError, match="rtc_max_delay"):
        _make_model(rtc_max_delay=HORIZON)


def test_per_token_timestep_matches_per_sample_timestep():
    model = _make_model().eval()
    state = torch.zeros(BATCH, ACTION_DIM)
    noisy = torch.randn(BATCH, HORIZON, ACTION_DIM)
    time = torch.tensor([0.3, 0.8])

    embs, _, _, cond = model.embed_suffix(state, noisy, time)
    embs_tok, _, _, cond_tok = model.embed_suffix(state, noisy, time[:, None].expand(BATCH, HORIZON))

    torch.testing.assert_close(embs_tok, embs)
    torch.testing.assert_close(cond_tok, cond[:, None, :].expand_as(cond_tok))


def test_forward_gives_prefix_clean_actions_and_masks_loss():
    model = _make_model().train()
    observation = _make_observation()
    actions = torch.randn(BATCH, HORIZON, ACTION_DIM)
    delay = torch.tensor([0, 3])

    captured = {}
    original_embed_suffix = model.embed_suffix

    def spy(state, noisy_actions, timestep):
        captured["x_t"], captured["t"] = noisy_actions, timestep
        return original_embed_suffix(state, noisy_actions, timestep)

    model.embed_suffix = spy
    noise = torch.randn(BATCH, HORIZON, ACTION_DIM)
    time = torch.tensor([0.5, 0.5])
    torch.manual_seed(2)
    losses = model(observation, actions, noise=noise, time=time, rtc_delay=delay)

    assert losses.shape == (BATCH, HORIZON, ACTION_DIM)
    assert captured["t"].shape == (BATCH, HORIZON)
    assert torch.all(captured["t"][1, :3] == 0)
    assert torch.all(captured["t"][1, 3:] == 0.5)
    assert torch.all(captured["t"][0] == 0.5)
    torch.testing.assert_close(captured["x_t"][1, :3], actions[1, :3])

    assert torch.all(losses[1, :3] == 0)
    assert torch.all(losses[1, 3:] > 0)
    assert torch.all(losses[0] > 0)

    model.embed_suffix = original_embed_suffix
    unmasked = _make_model(rtc_mask_loss=False).train()
    torch.manual_seed(2)
    raw = unmasked(observation, actions, noise=noise, time=time, rtc_delay=delay)
    valid = torch.ones(BATCH, HORIZON, dtype=torch.bool)
    valid[1, :3] = False
    torch.testing.assert_close(losses.mean(), raw[valid].mean())


def test_forward_backward_with_sampled_delay():
    model = _make_model().train()
    losses = model(_make_observation(), torch.randn(BATCH, HORIZON, ACTION_DIM))
    losses.mean().backward()
    assert model.time_mlp_in.weight.grad is not None
    assert torch.isfinite(model.action_out_proj.weight.grad).all()


def test_sample_actions_holds_prefix():
    model = _make_model().eval()
    observation = _make_observation()
    noise = torch.randn(BATCH, HORIZON, ACTION_DIM)
    prefix = torch.randn(BATCH, 3, ACTION_DIM)

    out = model.sample_actions("cpu", observation, noise=noise, num_steps=4, action_prefix=prefix, inference_delay=3)

    assert out.shape == (BATCH, HORIZON, ACTION_DIM)
    torch.testing.assert_close(out[:, :3], prefix)
    assert torch.isfinite(out).all()


def test_sample_actions_zero_delay_matches_plain_sampling():
    model = _make_model().eval()
    observation = _make_observation()
    noise = torch.randn(BATCH, HORIZON, ACTION_DIM)

    plain = model.sample_actions("cpu", observation, noise=noise, num_steps=4)
    zero_delay = model.sample_actions(
        "cpu", observation, noise=noise, num_steps=4, action_prefix=torch.zeros(0, ACTION_DIM), inference_delay=0
    )

    torch.testing.assert_close(zero_delay, plain)


def test_sample_actions_prefix_validation():
    observation = _make_observation()
    noise = torch.randn(BATCH, HORIZON, ACTION_DIM)

    model = _make_model().eval()
    with pytest.raises(ValueError, match="together"):
        model.sample_actions("cpu", observation, noise=noise, num_steps=2, inference_delay=2)
    with pytest.raises(ValueError, match="fewer than"):
        model.sample_actions(
            "cpu", observation, noise=noise, num_steps=2, action_prefix=torch.zeros(1, ACTION_DIM), inference_delay=2
        )

    plain_model = _make_model(use_training_time_rtc=False).eval()
    with pytest.raises(ValueError, match="use_training_time_rtc"):
        plain_model.sample_actions(
            "cpu", observation, noise=noise, num_steps=2, action_prefix=torch.zeros(2, ACTION_DIM), inference_delay=2
        )
