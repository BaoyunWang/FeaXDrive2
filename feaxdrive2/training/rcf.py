# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""RC-F: actual five-step supervision, with a single reusable student trace.

Local FM/velocity preservation still uses the original teacher/noise bridge.
By default its detached Query history comes from the student free rollout.
The reference_prefix ablation substitutes frozen Base clean-derived Query
coordinates, while the student still evolves its own live Euler states.
"""
import torch
from torch.nn import functional as F
from .constants import LOCAL_WEIGHT, CLEAN_WEIGHT, STATE_WEIGHT, TERMINAL_WEIGHT, METRIC_SCALES, NATIVE_SCALES, CLEAN_STEP_WEIGHTS
from ..model.camera_grounded_flow import discrete_flow_time
from .preservation import target_aligned_fields, preservation_loss
from .rollout import check_sampler, observations
FIELDS = ('flow_rcf_roles', 'flow_rcf_weights', 'flow_rcf_noise', 'flow_rcf_reference_states', 'flow_rcf_reference_final')
MECHANISMS = ('full', 'repair_no_final', 'local_preserve', 'reference_prefix')
REFERENCE_PREFIXES = 'flow_rcf_reference_prefixes'

def mechanism(head):
    """Training-only loss/query intervention; absent flag is exactly Full."""
    arm = getattr(head, 'rcf_mechanism', 'full')
    if arm not in MECHANISMS:
        raise ValueError(f'Unknown RC-F mechanism: {arm!r}')
    return arm

def validate_roles_noise(roles, weights, noise):
    if roles.dtype != torch.long or roles.ndim != 1 or len(roles) == 0 or (not torch.isin(roles, roles.new_tensor([0, 1, 2])).all()) or (weights.shape != roles.shape) or (noise.shape != (len(roles), 8, 3)) or any((v.requires_grad for v in (weights, noise))) or (not torch.isfinite(weights).all()) or (not torch.isfinite(noise).all()) or (weights <= 0).any():
        raise ValueError('RC-F requires detached noise and positive global scene weights with explicit roles')

@torch.no_grad()
def reference_fields(head, features, targets, roles, weights):
    """One independent Base chain (all roles for C3), plus its local field.

The local field cannot be indexed from this chain: it is evaluated at the
teacher/noise interpolation, not at Base's free-running state.
"""
    check_sampler(head)
    arm = mechanism(head)
    noise = features['flow_noise']
    validate_roles_noise(roles, weights, noise)
    result = target_aligned_fields(head, features, targets, roles, weights)
    states = noise.new_zeros((len(noise), 4, 8, 3))
    final = torch.zeros_like(noise)
    selected = torch.ones_like(roles, dtype=torch.bool) if arm == 'reference_prefix' else roles != 0
    if selected.any():
        dtype = next(head.parameters()).dtype
        history = features['history_trajectory'].flatten(1).to(dtype)
        status = features['status_feature'].to(dtype)
        inputs = observations(dict(features, his_traj=history, status_feature=status, state=torch.cat((status, history), dim=1)), selected)
        trace = head.get_action(features['last_hidden_state'][selected].to(dtype), inputs, init_actions=noise[selected].detach().clone(), return_trace=True)
        preserve = roles[selected] != 0
        states[roles != 0] = trace['flow_trace_states'][preserve, 1:5].to(states)
        final[roles != 0] = trace['pred_traj'][preserve].to(final)
        if arm == 'reference_prefix':
            later = head.denorm_odo(trace['flow_trace_clean'][:, :4].clamp(-1, 1))
            result[REFERENCE_PREFIXES] = torch.cat((torch.zeros_like(later[:, :1]), later), 1).detach()
    result.update(dict(zip(FIELDS, (roles.detach().clone(), weights.detach().clone(), noise.detach().clone(), states.detach(), final.detach()))))
    return result

def conditioning(head, vl_features, inputs):
    vl, mask = head._prepare_vl_conditioning(vl_features, inputs)
    history = head.his_traj_encoder(inputs.his_traj.unsqueeze(1)).repeat(1, 8, 1)
    status = head.ego_status_encoder(head._prepare_status_feature(inputs.status_feature))
    return (vl, mask, history, status)

def scaled_huber(error):
    """SmoothL1(beta=1), averaged over eight waypoints and xyz/yaw, per row."""
    normalized = error.float() / error.new_tensor(METRIC_SCALES, dtype=torch.float32)
    return F.smooth_l1_loss(normalized, torch.zeros_like(normalized), beta=1.0, reduction='none').mean((-2, -1))

def terminal_error(prediction, target):
    delta = prediction.float() - target.float()
    return torch.cat((delta[..., :2], torch.atan2(delta[..., 2:].sin(), delta[..., 2:].cos())), dim=-1)

def alignment_masks(roles, bp_multiplier, quality_mask=None, quality_targets=None):
    """E1 variants retain base sampling roles; Q replaces only safe supervision."""
    if type(bp_multiplier) not in (int, float) or bp_multiplier not in (1.0, 0.2):
        raise ValueError('E1 behavior preservation multiplier must be 1 or 0.2')
    if quality_mask is None:
        if quality_targets is not None:
            raise ValueError('Quality targets require an explicit mask')
        return torch.zeros_like(roles, dtype=torch.bool)
    if quality_mask.dtype != torch.bool or quality_mask.shape != roles.shape or quality_mask.device != roles.device or quality_mask.requires_grad or (quality_mask & (roles == 0)).any():
        raise ValueError('Quality supervision must be detached and restricted to safe roles')
    if quality_targets is None or quality_targets.shape != (len(roles), 8, 3) or quality_targets.dtype != torch.float32 or (quality_targets.device != roles.device) or quality_targets.requires_grad or (not torch.isfinite(quality_targets).all()):
        raise ValueError('Quality targets must be detached finite FP32 [B,8,3]')
    if bp_multiplier != 0.2:
        raise ValueError('Quality alignment requires the weak-BP comparator')
    return quality_mask

def trace_losses(head, trace, inputs, *, diagnostics=None, bp_multiplier=1.0, quality_mask=None, quality_targets=None):
    """Losses do not alter the sampler's unclipped state or detached Query path."""
    roles, weights = (inputs['flow_rcf_roles'], inputs['flow_rcf_weights'])
    quality = alignment_masks(roles, bp_multiplier, quality_mask, quality_targets)
    preserve = (roles != 0) & ~quality
    arm = mechanism(head)
    adapt = roles == 0
    normalized_target = head.norm_odo(inputs.action).detach()
    clean_error = trace['flow_trace_clean'][:, :4].float() - normalized_target[:, None].float()
    clean_error = clean_error * clean_error.new_tensor(NATIVE_SCALES)
    per_clean = (scaled_huber(clean_error) * clean_error.new_tensor(CLEAN_STEP_WEIGHTS)).sum(1)
    state_error = trace['flow_trace_states'][:, 1:5].float() - inputs['flow_rcf_reference_states'].float()
    state_error = state_error * state_error.new_tensor(NATIVE_SCALES)
    per_state = scaled_huber(state_error).mean(1)
    target = torch.where(adapt[:, None, None], inputs.action.detach(), inputs['flow_rcf_reference_final'])
    if quality_targets is not None:
        target = torch.where(quality[:, None, None], quality_targets, target)
    error = terminal_error(trace['pred_traj'], target)
    per_terminal = scaled_huber(error)
    clean = CLEAN_WEIGHT * (per_clean * weights * adapt).sum()
    state = STATE_WEIGHT * (per_state * weights * preserve).sum() * bp_multiplier
    terminal_weights = weights if bp_multiplier == 1.0 else weights * (adapt.to(weights) + quality.to(weights) + bp_multiplier * preserve.to(weights))
    terminal = TERMINAL_WEIGHT * (per_terminal * terminal_weights).sum()
    if arm == 'repair_no_final':
        terminal = TERMINAL_WEIGHT * (per_terminal * weights * ~adapt).sum()
    elif arm == 'local_preserve':
        state = state * 0.0
        terminal = TERMINAL_WEIGHT * (per_terminal * weights * adapt).sum()
    metrics = (error.detach().abs().mean(1) * (weights * preserve)[:, None]).sum(0)
    if diagnostics is not None:
        diagnostics.update(trace_prediction=trace['pred_traj'].detach(), per_scene_hr_clean=(CLEAN_WEIGHT * per_clean * adapt).detach(), per_scene_hr_final=(TERMINAL_WEIGHT * per_terminal * adapt).detach(), per_scene_bp_state=(STATE_WEIGHT * per_state * preserve * bp_multiplier).detach(), per_scene_bp_final=(TERMINAL_WEIGHT * per_terminal * preserve * bp_multiplier).detach(), per_scene_terminal_error_xyz=error.detach().abs().mean(1))
        if quality_targets is not None:
            diagnostics['per_scene_quality_final'] = (TERMINAL_WEIGHT * per_terminal * quality).detach()
    return (clean, state, terminal, metrics)

def forward_loss(head, vl_features, inputs, *, bp_multiplier=1.0, quality_mask=None, quality_targets=None):
    """Entire combined graph stays inside the DDP forward; no second sampler."""
    from transformers.feature_extraction_utils import BatchFeature
    check_sampler(head)
    arm = mechanism(head)
    no_body = getattr(head, 'rcf_no_body', False) is True or getattr(head.model, 'rcf_no_body', False) is True
    if no_body:
        from .parameters import assert_nobody_model, assert_nobody_inputs
        assert_nobody_model(head)
        assert_nobody_inputs(inputs)
    if not no_body and (not hasattr(head.model, 'camera_grounded_query')) or not all((k in inputs for k in FIELDS)) or any((k.startswith(('g3_', 'flow_rollout_')) for k in inputs)):
        raise ValueError('RC-F requires complete own fields and cannot mix G3/E12R objectives')
    if (REFERENCE_PREFIXES in inputs) != (arm == 'reference_prefix'):
        raise ValueError('RC-F reference prefixes are required only for reference_prefix')
    if arm == 'reference_prefix' and (not isinstance(inputs[REFERENCE_PREFIXES], torch.Tensor)):
        raise ValueError('RC-F reference_prefix requires a detached prefix tensor')
    if arm == 'reference_prefix' and no_body:
        raise ValueError('RC-F reference_prefix requires Camera Query')
    roles, weights, noise, ref_states, ref_final = (inputs[k] for k in FIELDS)
    validate_roles_noise(roles, weights, noise)
    quality = alignment_masks(roles, bp_multiplier, quality_mask, quality_targets)
    if (bp_multiplier != 1.0 or quality_targets is not None) and arm != 'full':
        raise ValueError('E1 alignment variants require the original full RC-F mechanism')
    if ref_states.shape != (len(roles), 4, 8, 3) or ref_final.shape != noise.shape or inputs.action.shape != noise.shape or any((v.requires_grad for v in (ref_states, ref_final, inputs.action))) or any((not torch.isfinite(v).all() for v in (ref_states, ref_final, inputs.action))) or (not torch.equal(noise, inputs['flow_noise'])) or (not torch.equal(inputs['flow_loss_weights'], weights * (roles == 0))) or (not torch.equal(inputs['flow_preservation_weights'], weights * (roles != 0))):
        raise ValueError('RC-F target/noise/role coupling differs from its fixed contract')
    time = discrete_flow_time(inputs['flow_time'])
    local_condition = conditioning(head, vl_features, inputs)
    dtype = next(head.parameters()).dtype
    sampler_condition = local_condition if vl_features.dtype == dtype else conditioning(head, vl_features.to(dtype), inputs)
    selected = torch.ones(len(roles), device=noise.device, dtype=torch.bool)
    trace = head.get_action(vl_features.to(dtype), observations(inputs, selected), init_actions=noise.clone(), return_trace=True, prepared_condition=sampler_condition, **{'query_prefixes': inputs[REFERENCE_PREFIXES]} if arm == 'reference_prefix' else {})
    if torch.is_grad_enabled() and (not trace['pred_traj'].requires_grad):
        raise ValueError('RC-F student Euler state graph is detached')
    if not no_body:
        indices = (time * 5).round().long()
        previous = trace['flow_trace_previous'][torch.arange(len(time), device=time.device), indices]
    target = head.norm_odo(inputs.action)
    x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * target
    vl, mask, history, status = local_condition
    if no_body:
        velocity = head.predict_flow_velocity(x_t, time, vl, history, status, mask)
    else:
        velocity = head.predict_flow_velocity(x_t, time, vl, history, status, mask, inputs.get('vehicle_geometry'), camera_inputs=observations(inputs, selected), previous_metric=previous)
    fm = LOCAL_WEIGHT * ((velocity.float() - (target - noise).float()).square().mean((1, 2)) * inputs['flow_loss_weights']).sum()
    preservation_inputs = inputs
    if bp_multiplier != 1.0 or quality_targets is not None:
        preservation_inputs = dict(inputs, flow_preservation_weights=(inputs['flow_preservation_weights'] * ~quality * bp_multiplier).detach())
    preserve = LOCAL_WEIGHT * preservation_loss(velocity, x_t, time, preservation_inputs)
    diagnostics = {} if getattr(head, 'planning_lora_diagnostics', False) else None
    clean, state, terminal, errors = trace_losses(head, trace, inputs, diagnostics=diagnostics, bp_multiplier=bp_multiplier, quality_mask=quality_mask, quality_targets=quality_targets)
    rollout = clean + state + terminal
    return BatchFeature(data=dict(loss=fm + preserve + rollout, loss_base=fm.detach(), loss_flow=fm.detach(), loss_preserve=preserve.detach(), loss_rollout=rollout.detach(), loss_rcf_clean=clean.detach(), loss_rcf_state=state.detach(), loss_rcf_terminal=terminal.detach(), rollout_error_xyz_weighted=errors, flow_t_mean=time.mean().detach(), **diagnostics or {}))
