"""Training setup for the three paper objectives; no cluster dependencies."""
import torch
from .parameters import configure_camera_trainables, late_dit_parameter_names


def configure_rcf(planner, *, spatial_lr=1e-4, action_lr=1e-5):
    planner.eval().float()
    if planner.has_spatial_retrieval:
        return configure_camera_trainables(planner.action_head, spatial_lr, action_lr)
    if planner.variant != "rcf_no_spatial":
        raise ValueError("RC-F requires a spatial or explicitly no-spatial planner")
    planner.requires_grad_(False)
    selected = set(late_dit_parameter_names(planner.action_head))
    parameters = []
    for name, value in planner.action_head.named_parameters():
        if name in selected:
            value.requires_grad_(True)
            parameters.append(value)
    return [{"params": parameters, "lr": action_lr, "initial_lr": action_lr}]


def build_alignment(planner, *, model_path, vlm_path, device="cuda"):
    from ..vlm.planning_lora import build_language
    from .alignment import PlanningGuidedLoRA
    from .alignment_parameters import configuration
    from .constants import lora_configuration
    if planner.variant != "body_scene":
        raise ValueError("Joint alignment starts from the Body–Scene Grounded Planner")
    language = build_language(model_path, vlm_path, device,
        gradient_checkpointing=True, lora_config=lora_configuration())
    return PlanningGuidedLoRA(planner.to(device), language, bp_multiplier=.2,
                             unfreeze_spec=configuration())


def alignment_optimizer(model):
    from .alignment_parameters import trainable_groups
    return torch.optim.AdamW(trainable_groups(model, {"lr": 3e-6}),
                             betas=(.9, .95), weight_decay=1e-4)
