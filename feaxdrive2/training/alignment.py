# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Planning-guided LoRA through original A4 and unchanged full RC-F.

Ordinary post-language feature keys hold the cached REPLAY reference. The
external language module consumes llm_* boundary tensors and returns the five
post-language tensors below. Camera F, geometry, history and query prefixes
remain frozen. E1-W-U additionally trains only its explicit DiT whitelist and
keeps an independent original-A4 reference head. Rollout states never alias.
"""

import copy
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from . import rcf
from ..model.camera_grounded_flow import discrete_flow_time
from .rollout import check_sampler
from ..vlm.planning_lora import PLANNER_KEYS, planner_features
from .precision import (
    assert_strict_fp32, floating_to_fp32, strict_fp32,
)
from .constants import PACIFICA


POST_LANGUAGE_FIELDS = PLANNER_KEYS
CAMERA_FIELDS = (
    "camera_features", "camera_projection", "camera_image_size", "camera_valid",
    "camera_intrinsics", "camera_distortion", "camera_distortion_r2_max",
)


def _check_head(head):
    check_sampler(head)
    expected = set(getattr(head, "e1wu_trainable_names", ()))
    actual = {name for name, value in head.named_parameters() if value.requires_grad}
    if expected:
        from .alignment_parameters import allowed_parameter
        if len(expected) != 44 or not all(allowed_parameter(name) for name in expected):
            raise ValueError("Changed E1-W-U DiT parameter set")
    if (any(module.training for module in head.modules())
            or actual != expected
            or not hasattr(head.model, "camera_grounded_query")
            or rcf.mechanism(head) != "full"
            or getattr(head, "rcf_local_only", False)
            or getattr(head, "rcf_no_body", False)
            or getattr(head.model, "rcf_no_body", False)):
        raise ValueError("Planning-Guided LoRA requires frozen eval A4 with full RC-F")
    query = head.model.camera_grounded_query
    if ((query.mode, query.retrieval, query.anchor_layout) !=
            ("geometry", "projected", "body_side_context")
            or not torch.equal(query.pacifica, query.pacifica.new_tensor(PACIFICA))):
        raise ValueError("Planning-Guided LoRA requires original Pacifica A4 BodySideContext")


def _inputs(head, features):
    """Prepare the inherited agent's ordinary head inputs, without CUDA dispatch.

    Explicit routing excludes llm_* boundaries, targets and stale loss fields
    from both canonical samplers. Only post-language hidden states carry grad.
    """
    dtype = next(head.parameters()).dtype
    history = features["history_trajectory"].detach().flatten(1).to(dtype)
    status = features["status_feature"].detach().to(dtype)
    geometry = features["vehicle_geometry"]
    if (geometry.shape != (len(status), 3) or "projection_geometry" in features
            or not torch.equal(geometry, geometry.new_tensor(PACIFICA).expand_as(geometry))):
        raise ValueError("Planning-Guided LoRA requires fixed Pacifica without projection interventions")
    data = {key: features[key].detach() for key in
            (*POST_LANGUAGE_FIELDS[1:], "vehicle_geometry", *CAMERA_FIELDS)}
    data.update(his_traj=history, status_feature=status,
                state=torch.cat((status, history), dim=1))
    return BatchFeature(data=data)


def _detached(features):
    return {key: value.detach() for key, value in features.items()}


def _planner_fp32(features):
    # Never promote language boundary tensors (especially Vision/DeepStack).
    return floating_to_fp32({key: value for key, value in features.items()
                             if not key.startswith("llm_")})


@torch.no_grad()
def reference_trace(head, features, noise):
    """Independent cached-REPLAY A4 chain, with its own causal camera prefixes."""
    if noise.requires_grad:
        raise ValueError("Reference noise must be detached")
    with strict_fp32():
        assert_strict_fp32(head)
        return _reference_trace(head, _planner_fp32(features), noise.float())


def _reference_trace(head, features, noise):
    _check_head(head)
    if (noise.shape != (len(features["last_hidden_state"]), 8, 3)
            or noise.requires_grad or not torch.isfinite(noise).all()):
        raise ValueError("Reference requires detached finite noise [B,8,3]")
    return head.get_action(
        features["last_hidden_state"].detach().to(next(head.parameters()).dtype),
        _inputs(head, features), init_actions=noise.detach().clone(), return_trace=True)


@torch.no_grad()
def reference_fields(head, features, targets, roles, weights):
    """Full RC-F fields plus the shared metric bridge target in ``action``.

    Roles use the existing RC-F encoding: HR=0, BP=1/2. HR retains its teacher
    and needs no reference forward. BP uses this noise's frozen A4 final as
    bridge endpoint. Its local reference velocity is evaluated at that shared
    bridge/time with the REFERENCE chain's own previous-clean query prefix.
    This intentionally does not use ego_body_rcf.reference_fields (bare Base).
    """
    rcf.validate_roles_noise(roles, weights, features["flow_noise"])
    if features["flow_time"].requires_grad:
        raise ValueError("Reference time must be detached")
    with strict_fp32():
        assert_strict_fp32(head)
        return _reference_fields(head, _planner_fp32(features), floating_to_fp32(targets),
                                 roles, weights.float())


def _reference_fields(head, features, targets, roles, weights):
    _check_head(head)
    noise = features["flow_noise"]
    time = discrete_flow_time(features["flow_time"])
    teacher = targets["trajectory"]
    if (time.shape != roles.shape or teacher.shape != noise.shape
            or not torch.isfinite(teacher).all()):
        raise ValueError("Planning-Guided LoRA requires finite [B,8,3] targets and detached time [B]")
    dtype = next(head.parameters()).dtype
    action = teacher.detach().to(dtype).clone()
    preserve = roles != 0
    states = noise.new_zeros((len(noise), 4, 8, 3))
    final = torch.zeros_like(noise)
    velocity = torch.zeros_like(noise)
    if preserve.any():
        selected = {key: value[preserve] for key, value in features.items()
                    if key in (*POST_LANGUAGE_FIELDS, "history_trajectory", "status_feature",
                               "vehicle_geometry", "projection_geometry", *CAMERA_FIELDS)}
        trace = _reference_trace(head, selected, noise[preserve])
        states[preserve] = trace["flow_trace_states"][:, 1:5].to(states)
        final[preserve] = trace["pred_traj"].to(final)
        action[preserve] = final[preserve].to(action)

    # Use exactly the operation order/dtypes in rcf.forward_loss on both paths.
    target = head.norm_odo(action)
    x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * target
    if preserve.any():
        inputs = _inputs(head, selected)
        vl, mask, history, status = rcf.conditioning(head, selected["last_hidden_state"], inputs)
        indices = (time[preserve] * 5).round().long()
        previous = trace["flow_trace_previous"][torch.arange(len(indices), device=indices.device), indices]
        velocity[preserve] = head.predict_flow_velocity(
            x_t[preserve], time[preserve], vl, history, status, mask,
            inputs["vehicle_geometry"], camera_inputs=inputs,
            previous_metric=previous).to(velocity)

    return dict(action=action.detach(),
        flow_loss_weights=(weights * ~preserve).detach(),
        flow_reference_velocity=velocity.detach(), flow_reference_actions=x_t.detach(),
        flow_reference_time=time.detach(), flow_preservation_weights=(weights * preserve).detach(),
        flow_rcf_roles=roles.detach().clone(), flow_rcf_weights=weights.detach().clone(),
        flow_rcf_noise=noise.detach().clone(), flow_rcf_reference_states=states.detach(),
        flow_rcf_reference_final=final.detach())


class PlanningGuidedLoRA(nn.Module):
    """Own A4 and an external language backbone, with explicit optional DiT tuning.

    forward returns the unchanged RC-F BatchFeature (loss and detached metrics).
    predict returns metric [B,8,3] trajectories from the original sampler.
    Callers own device placement, noise/time sampling and loss weights.
    language=None supports reference inference without loading the language
    backbone; forward and student prediction then fail before any model work.
    """

    def __init__(self, agent, language, *, bp_multiplier=1., quality_alignment=False, unfreeze_spec=None):
        super().__init__()
        if type(bp_multiplier) not in (int, float) or bp_multiplier not in (1., .2):
            raise ValueError("E1 BP multiplier must be 1 or 0.2")
        if type(quality_alignment) is not bool or (quality_alignment and bp_multiplier != .2):
            raise ValueError("E1 quality alignment requires weak BP")
        self.bp_multiplier, self.quality_alignment = bp_multiplier, quality_alignment
        lora = []
        if language is not None:
            if {id(p) for p in agent.parameters()} & {id(p) for p in language.parameters()}:
                raise ValueError("Language parameters must not alias the frozen agent")
            lora = [p for name, p in language.named_parameters()
                    if any(part in ("lora_A", "lora_B") for part in name.split("."))]
            if not lora:
                raise ValueError("Language backbone has no LoRA A/B parameters")
        self.agent, self.language = agent, language
        self.agent.requires_grad_(False)
        self.agent.action_head.float()
        self.reference_head = None
        if unfreeze_spec is not None:
            from .alignment_parameters import configure_head
            if quality_alignment or language is None or bp_multiplier != .2:
                raise ValueError("E1-W-U requires the original weak-BP language objective")
            self.reference_head = copy.deepcopy(self.agent.action_head).eval().requires_grad_(False)
            configure_head(self.agent.action_head, unfreeze_spec)
        if self.language is not None:
            self.language.requires_grad_(False)
        for parameter in lora:
            parameter.requires_grad_(True)
        for parameter in self.parameters():
            parameter.grad = None
        self.train()
        _check_head(self.agent.action_head)
        # Plain bool, never a buffer/parameter: existing A4 checkpoints stay exact.
        if self.language is not None:
            self.agent.action_head.model.camera_grounded_query.retain_scene_grad = True
            self.agent.action_head.planning_lora_diagnostics = True

    def train(self, mode=True):
        super().train(mode)
        self.agent.eval()
        if self.reference_head is not None:
            self.reference_head.eval()
        # Preserve the external language module's nonreentrant gradient
        # checkpointing setup; it needs training=True for checkpointed forward.
        if self.language is not None:
            self.language.train(mode)
        return self

    def _require_language(self):
        if self.language is None:
            raise ValueError("language=None is reference-only; student forward/prediction requires a language backbone")

    def _student_features(self, features):
        self._require_language()
        detached = _detached(features)
        # This runs outside strict_fp32: leave language dtype, SDPA and its
        # nonreentrant checkpoint/autocast policy intact. The shared packer
        # preserves canonical FP16 publication -> FP32 with differentiable casts.
        post_language = planner_features(self.language, {
            key: value for key, value in detached.items() if key.startswith("llm_")})
        return _planner_fp32(dict(detached, **post_language))

    def reference_trace(self, features, noise):
        head = self.reference_head if self.reference_head is not None else self.agent.action_head
        return reference_trace(head, features, noise)

    def reference_fields(self, features, targets, roles, weights):
        head = self.reference_head if self.reference_head is not None else self.agent.action_head
        return reference_fields(head, features, targets, roles, weights)

    def forward(self, features, targets, roles, weights):
        self._require_language()
        quality = {}
        if self.quality_alignment:
            if not all(k in targets for k in ("quality_mask", "quality_trajectory")):
                raise ValueError("E1-P requires explicit quality targets and fallback masks")
            quality = dict(quality_mask=targets["quality_mask"], quality_targets=targets["quality_trajectory"])
            rcf.alignment_masks(roles, self.bp_multiplier, **quality)
        elif any(k in targets for k in ("quality_mask", "quality_trajectory")):
            raise ValueError("Quality supervision is restricted to E1-P")
        head = self.agent.action_head
        fields = self.reference_fields(features, targets, roles, weights)
        student = self._student_features(features)
        with strict_fp32():
            assert_strict_fp32(head)
            inputs = _inputs(head, student)
            inputs.update(fields)
            inputs.update(flow_noise=student["flow_noise"], flow_time=student["flow_time"])
            # agent.forward branches on agent.training, so bypass ONLY dispatch.
            result = rcf.forward_loss(head, student["last_hidden_state"], inputs,
                                      bp_multiplier=self.bp_multiplier, **quality)
        # Existing metrics already use disjoint role masks/global scene weights.
        result.update(loss_hr_local=result["loss_flow"], loss_hr_clean=result["loss_rcf_clean"],
                      loss_bp_local=result["loss_preserve"], loss_bp_state=result["loss_rcf_state"])
        if "per_scene_hr_final" in result:
            result["loss_hr_final"] = (result["per_scene_hr_final"] * weights.detach().float()).sum()
            result["loss_bp_final"] = (result["per_scene_bp_final"] * weights.detach().float()).sum()
            result["loss_hr"] = result["loss_hr_local"] + result["loss_hr_clean"] + result["loss_hr_final"]
            result["loss_bp"] = result["loss_bp_local"] + result["loss_bp_state"] + result["loss_bp_final"]
            if self.quality_alignment:
                result["loss_quality"] = (result["per_scene_quality_final"] * weights.detach().float()).sum()
        return result

    @torch.no_grad()
    def predict(self, features, noise, reference=False, use_teacher=False):
        if not reference:
            self._require_language()
        if use_teacher and not reference:
            raise ValueError("Frozen teacher inference requires the supplied original features")
        head = self.reference_head if use_teacher and self.reference_head is not None else self.agent.action_head
        _check_head(head)
        if (noise.shape != (len(features["last_hidden_state"]), 8, 3)
                or noise.requires_grad or not torch.isfinite(noise).all()):
            raise ValueError("Prediction requires detached finite noise [B,8,3]")
        was_training = self.language.training if not reference else None
        try:
            if not reference:
                self.language.eval()
            source = _planner_fp32(_detached(features)) if reference else self._student_features(features)
            with strict_fp32():
                assert_strict_fp32(head)
                return head.get_action(source["last_hidden_state"], _inputs(head, source),
                                       init_actions=noise.detach().float().clone())["pred_traj"]
        finally:
            if not reference:
                self.language.train(was_training)
