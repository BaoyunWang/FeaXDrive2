import copy
import pytest
import torch
from feaxdrive2 import build_planner, token_noise
from feaxdrive2.planner import initialize_from_base
from feaxdrive2.checkpoints import save_planner, load_planner, validate_state, load_alignment
from feaxdrive2.training.api import configure_rcf
from feaxdrive2.training.alignment_parameters import dit_parameters
from .helpers import features


def test_token_noise_is_order_independent():
    joint = token_noise(["scene_a", "scene_b"])
    assert torch.equal(joint[1], token_noise(["scene_b"])[0])
    assert not torch.equal(joint[0], joint[1])


def test_sampler_trace_and_condition_gradient():
    model = build_planner().eval().requires_grad_(False)
    # Fresh DiT AdaLN gates are zero. A gradient-path check needs nonzero gates,
    # as in a trained checkpoint; this fixture does not change model defaults.
    with torch.no_grad():
        for block in model.action_head.model.transformer_blocks:
            block.adaLN_modulation[-1].bias.fill_(.1)
    data = features()
    data["last_hidden_state"].requires_grad_(True)
    model.action_head.model.camera_grounded_query.retain_scene_grad = True
    trace = model(data, token_noise(["scene_a"]), return_trace=True)
    assert trace["pred_traj"].shape == (1, 8, 3)
    assert trace["flow_trace_states"].shape == (1, 6, 8, 3)
    assert not trace["flow_trace_previous"].requires_grad
    assert torch.isfinite(trace["pred_traj"]).all()
    assert torch.equal(trace["flow_trace_previous"][:, 0], torch.zeros(1, 8, 3))
    trace["flow_trace_states"][:, -1].square().mean().backward()
    assert data["last_hidden_state"].grad is not None
    assert torch.isfinite(data["last_hidden_state"].grad).all()
    assert data["last_hidden_state"].grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("variant", ["body_scene", "rcf_no_spatial", "fixed_query"])
def test_rcf_real_backward_and_parameter_ownership(variant, monkeypatch):
    base = build_planner("base").eval().requires_grad_(False)
    model = build_planner(variant)
    initialize_from_base(model, base)
    groups = configure_rcf(model)
    assert len(dit_parameters(model.action_head)) == 44
    assert sum(p.numel() for p in dit_parameters(model.action_head).values()) == 6505152
    data = features(batch=3)
    data.update(flow_noise=token_noise(["risk", "hard", "normal"]), flow_time=torch.tensor([.2, .4, .8]))
    calls = []
    predict = model.action_head.predict_flow_velocity

    def capture(*args, **kwargs):
        if "previous_metric" in kwargs:
            calls.append(kwargs["previous_metric"].detach().clone())
        return predict(*args, **kwargs)

    monkeypatch.setattr(model.action_head, "predict_flow_velocity", capture)
    target = torch.zeros(3, 8, 3)
    target[..., 0] = torch.arange(1, 9)
    result = model.rcf_loss(data, target, torch.tensor([0, 1, 2]), base)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    selected = {id(p) for group in groups for p in group["params"]}
    assert selected == {id(p) for p in model.parameters() if p.requires_grad}
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert all(p.grad is None for p in base.parameters())
    if variant == "fixed_query":
        # Five actual Flow evaluations, then the local auxiliary evaluation.
        assert len(calls) == 6
        assert torch.equal(calls[0], torch.zeros_like(calls[0]))
        for query in calls[2:]:
            # All three local times are after the first clean estimate.
            assert torch.equal(query, calls[1])


def test_base_training_and_normalization():
    base = build_planner("base")
    target = torch.randn(1, 8, 3)
    restored = base.action_head.denorm_odo(base.action_head.norm_odo(target))
    torch.testing.assert_close(restored, target, atol=3e-6, rtol=1e-5)
    result = base.flow_matching_loss(features(), target, noise=token_noise(["a"]), time=torch.tensor([.5]))
    result["loss"].backward()
    assert base.action_head.action_decoder.fc2.weight.grad is not None


def test_strict_checkpoint_roundtrip(tmp_path):
    model = build_planner("base").eval()
    path = tmp_path / "base.pt"
    save_planner(model, path)
    with pytest.raises(FileExistsError):
        save_planner(model, path)
    expected = model.action_head.action_decoder.fc2.weight.clone()
    with torch.no_grad():
        model.action_head.action_decoder.fc2.weight.zero_()
    load_planner(model, path)
    assert torch.equal(expected, model.action_head.action_decoder.fc2.weight)
    state = dict(model.action_head.state_dict())
    state.pop(next(iter(state)))
    with pytest.raises(ValueError, match="keys differ"):
        validate_state(model.action_head, state)
    with pytest.raises(ValueError, match="format/variant"):
        load_planner(build_planner(), path)


def test_alignment_rejects_incomplete_artifact(tmp_path):
    path = tmp_path / "incomplete.pt"
    torch.save({"lora": {}}, path)
    with pytest.raises(ValueError):
        load_alignment(path)


def test_feature_conditioning_must_match_alignment():
    from feaxdrive2.data import validate_conditioning_source
    validate_conditioning_source([{"token": "a"}], None)
    validate_conditioning_source([{"token": "a", "alignment_sha256": "matching"}], "matching")
    with pytest.raises(ValueError, match="checkpoint differ"):
        validate_conditioning_source([{"token": "a"}], "matching")
