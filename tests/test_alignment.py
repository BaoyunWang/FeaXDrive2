"""Real planner/loss integration with a small differentiable language test double.

Native Alpamayo replay additionally requires licensed model assets and a GPU.
"""
import torch
from torch import nn
from feaxdrive2 import build_planner, token_noise
from feaxdrive2.training.alignment import PlanningGuidedLoRA
from feaxdrive2.training.alignment_parameters import configuration, dit_parameters
from feaxdrive2.training.api import alignment_optimizer
from feaxdrive2.vlm.planning_lora import PLANNER_KEYS
from .helpers import features


class LanguageFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})

    def forward(self, boundary):
        hidden = boundary["llm_" + PLANNER_KEYS[0]]
        # A nonconstant channel modulation ensures LayerNorm cannot erase it.
        delta = self.lora_B["default"](self.lora_A["default"](hidden[..., :1]))
        channel = torch.linspace(-1., 1., 4096, device=hidden.device)
        return (hidden + delta * channel, *(boundary["llm_" + k] for k in PLANNER_KEYS[1:]))


def test_joint_alignment_updates_only_lora_and_selected_dit():
    planner = build_planner()
    with torch.no_grad():
        for block in planner.action_head.model.transformer_blocks:
            block.adaLN_modulation[-1].bias.fill_(.1)
    model = PlanningGuidedLoRA(planner, LanguageFixture(), bp_multiplier=.2,
                              unfreeze_spec=configuration())
    optimizer = alignment_optimizer(model)
    data = features(batch=2)
    ids = torch.tensor([0]*48 + [1]*16 + [2] + [3]*2).expand(2, -1).clone()
    data.update(last_hidden_state=torch.randn(2, 67, 4096), multimodal_type_ids=ids,
                multimodal_position_ids=torch.cat((torch.arange(48), torch.arange(16),
                    torch.zeros(1, dtype=torch.long), torch.arange(2))).expand(2, -1).clone(),
                multimodal_valid_mask=torch.ones(2, 67, dtype=torch.bool))
    data.update({"llm_" + k: data[k].clone() for k in PLANNER_KEYS})
    data.update(llm_context_input_ids=torch.zeros(2, 67, dtype=torch.long),
        llm_visual_pool_size=torch.tensor([2, 2]), llm_history_tokens=torch.tensor([16, 16]),
        llm_max_coc_tokens=torch.tensor([256, 256]),
        flow_noise=token_noise(["align-risk", "align-safe"]), flow_time=torch.tensor([.2, .6]))
    target = torch.zeros(2, 8, 3)
    target[..., 0] = torch.arange(1, 9)
    result = model(data, {"trajectory": target}, torch.tensor([0, 1]), torch.tensor([.5, .5]))
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.language.parameters())
    assert all(p.grad is None for p in model.reference_head.parameters())
    assert all(p.grad is None for p in planner.action_head.model.camera_grounded_query.parameters())
    assert all(p.grad is None for p in planner.action_head.feature_encoder.parameters())
    assert len(dit_parameters(planner.action_head)) == 44
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad}
    assert [g["lr"] for g in optimizer.param_groups] == [3e-6, 2e-6]
