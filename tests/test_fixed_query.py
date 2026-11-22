"""Formal fixed-query semantics and release checkpoint safeguards.

The analytic velocity fixture checks Euler timing and gradients independently
of network weights. The real RC-F backward is covered in test_planner.py.
"""
from types import SimpleNamespace

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from feaxdrive2 import build_planner, token_noise
from feaxdrive2.checkpoints import load_planner, save_planner
from feaxdrive2.model.flow import FlowPlanner
from examples.convert_checkpoint import convert_planner
from .helpers import features


class AnalyticHead(torch.nn.Module):
    get_action = FlowPlanner.get_action
    denorm_odo = FlowPlanner.denorm_odo

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(sampling_method="flow", num_inference_steps=5,
            action_horizon=8, action_dim=3)
        self.model = torch.nn.Module()
        self.model.camera_grounded_query = torch.nn.Identity()
        self.his_traj_encoder = self.ego_status_encoder = torch.nn.Identity()
        self.bias = torch.nn.Parameter(torch.tensor(.05))

    def _prepare_vl_conditioning(self, vl, inputs):
        return vl, None

    def _prepare_status_feature(self, status):
        return status

    def predict_flow_velocity(self, x, t, vl, history, ego, mask, geometry,
                              camera_inputs=None, previous_metric=None):
        return .125 * x + .01 * previous_metric + self.bias + .1 * t[:, None, None]


def sample(head, **kwargs):
    noise = torch.linspace(-2., 2., 48).reshape(2, 8, 3)
    inputs = BatchFeature(data=dict(his_traj=torch.zeros(2, 12),
                                   status_feature=torch.zeros(2, 8)))
    return head.get_action(torch.zeros(2, 32, 32), inputs,
                           init_actions=noise, return_trace=True, **kwargs)


def test_fixed_query_training_preserves_euler_gradients_and_default_dynamic():
    head = AnalyticHead().train()
    dynamic = sample(head)
    explicit = sample(head, query_update="dynamic")
    for key in dynamic:
        assert torch.equal(dynamic[key], explicit[key])
    head.spatial_query_update = "fixed_first_estimate"
    fixed = sample(head)
    states = fixed.flow_trace_states
    queries = fixed.flow_trace_previous
    velocities = fixed.flow_trace_velocity
    assert states.requires_grad and not queries.requires_grad
    assert torch.equal(queries[:, 0], torch.zeros_like(queries[:, 0]))
    # r1 comes from the pre-Euler clean estimate, not from the updated state.
    clean0 = states[:, 0] + velocities[:, 0]
    assert clean0.min() < -1 and clean0.max() > 1
    r1 = head.denorm_odo(clean0.clamp(-1, 1)).detach()
    assert torch.equal(queries[:, 1:], r1[:, None].expand(-1, 4, -1, -1))
    assert not torch.equal(r1, head.denorm_odo(states[:, 1].clamp(-1, 1)))
    assert torch.equal(velocities[:, :2], dynamic.flow_trace_velocity[:, :2])
    assert not torch.equal(velocities[:, 2], dynamic.flow_trace_velocity[:, 2])
    for k in range(5):
        assert torch.equal(states[:, k + 1], states[:, k] + .2 * velocities[:, k])
        assert not torch.equal(states[:, k + 1], states[:, k])
    states[:, -1].sum().backward()
    # Query detach makes dx/dbias = .2 * sum(1.025**k), for each of 48 entries.
    assert float(head.bias.grad) == pytest.approx(48 * sum(.2 * 1.025 ** k for k in range(5)), abs=1e-4)
    with torch.no_grad():
        inferred = sample(head.eval())
    for key in fixed:
        assert torch.equal(fixed[key], inferred[key])


def test_fixed_query_requires_five_steps_and_excludes_reference_query_override():
    head = AnalyticHead()
    head.spatial_query_update = "fixed_first_estimate"
    head.config.num_inference_steps = 4
    with pytest.raises(ValueError, match="five-step Camera Flow"):
        sample(head)
    head.config.num_inference_steps = 5
    with pytest.raises(ValueError, match="cannot be combined"):
        sample(head, query_prefixes=torch.zeros(2, 5, 8, 3))


def test_fixed_query_roundtrip_and_research_conversion(tmp_path):
    model = build_planner("fixed_query").eval()
    data, noise = features(), token_noise(["fixed-roundtrip"])
    with torch.no_grad():
        expected = model(data, noise, return_trace=True)
    path = tmp_path / "fixed.pt"
    save_planner(model, path)
    # A stale in-memory override must not survive a fixed-query checkpoint load.
    model.action_head.spatial_query_update = "dynamic"
    with pytest.raises(ValueError, match="Query-update policy"):
        save_planner(model, tmp_path / "wrong.pt")
    load_planner(model, path)
    assert model.action_head.spatial_query_update == "fixed_first_estimate"
    with torch.no_grad():
        restored = model(data, noise, return_trace=True)
    for key in expected:
        assert torch.equal(expected[key], restored[key])
    with pytest.raises(ValueError, match="format/variant"):
        load_planner(build_planner("body_scene"), path)

    prefix = "agent.action_head."
    saved = dict(state_dict={prefix + k: v for k, v in model.action_head.state_dict().items()},
                 e2=dict(camera_anchor_layout="body_side_context", spatial_query_update="fixed_first_estimate"))
    converted = convert_planner(saved, "fixed_query", prefix)
    assert converted.action_head.spatial_query_update == "fixed_first_estimate"
    for name, value in model.action_head.state_dict().items():
        assert torch.equal(value, converted.action_head.state_dict()[name])
    with pytest.raises(ValueError, match="query-update policy"):
        convert_planner(saved, "body_scene", prefix)
    saved["e2"].pop("spatial_query_update")
    with pytest.raises(ValueError, match="query-update policy"):
        convert_planner(saved, "fixed_query", prefix)
    assert convert_planner(saved, "body_scene", prefix).spatial_query_update == "dynamic"
