# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Masked field preservation; no extra inference module or state tensors."""

import torch


FIELDS = ("flow_reference_velocity", "flow_reference_actions", "flow_reference_time",
          "flow_preservation_weights")


def preservation_loss(prediction, actions, time, inputs):
    """Weights already include lambda, replay mask, and DDP global-mean scaling."""
    if not all(k in inputs for k in FIELDS):
        raise ValueError("Incomplete reference-field preservation inputs")
    reference, ref_actions, ref_time, weights = (inputs[k] for k in FIELDS)
    if (reference.shape != prediction.shape or actions.shape != prediction.shape
            or ref_actions.shape != actions.shape or ref_time.shape != time.shape
            or weights.shape != time.shape or time.shape != prediction.shape[:1]):
        raise ValueError("Reference preservation shape mismatch")
    if any(v.requires_grad for v in (reference, ref_actions, ref_time, weights)):
        raise ValueError("Reference preservation targets/weights must be detached")
    if any(not torch.isfinite(v).all() for v in (reference, ref_actions, ref_time, weights)) or (weights < 0).any():
        raise ValueError("Nonfinite or negative reference preservation inputs")
    if not torch.equal(actions.detach(), ref_actions) or not torch.equal(time, ref_time):
        raise ValueError("Actor and reference must use exactly the same x_t and t")
    error = (prediction.float() - reference.float()).square().mean((1, 2))
    return (error * weights).sum()


@torch.no_grad()
def reference_fields(reference_head, features, targets, replay, weights):
    """Independent frozen Base at the actor's FM bridge (not a second rollout).

    Critical rows have zero preservation weight and need no reference forward.
    Called inside the same autocast context as the subsequent actor forward.
    """
    from transformers.feature_extraction_utils import BatchFeature
    from ..model.camera_grounded_flow import discrete_flow_time

    if reference_head.training or any(p.requires_grad for p in reference_head.parameters()):
        raise ValueError("Reference must be frozen and deterministic eval")
    if any(hasattr(reference_head.model, n) for n in ("camera_grounded_query", "footprint_query", "geometry_conditioner")):
        raise ValueError("Preservation requires independent unmodified best-097")
    dtype = next(reference_head.parameters()).dtype
    noise = features["flow_noise"]
    time = discrete_flow_time(features["flow_time"])
    target = reference_head.norm_odo(targets["trajectory"].to(dtype))
    x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * target
    if replay.dtype != torch.bool or replay.shape != time.shape or weights.shape != time.shape:
        raise ValueError("Malformed replay preservation mask/weights")
    velocity = torch.zeros_like(x_t)
    if replay.any():
        selected = {k: v[replay] for k, v in features.items()}
        inputs = BatchFeature(data={**selected,
            "his_traj": selected["history_trajectory"].flatten(1).to(dtype),
            "status_feature": selected["status_feature"].to(dtype)})
        # Match inherited agent.forward: cached hidden-state dtype is retained;
        # casting it only on the reference side can change pre-Linear LayerNorm.
        vl, mask = reference_head._prepare_vl_conditioning(selected["last_hidden_state"], inputs)
        history = reference_head.his_traj_encoder(inputs.his_traj.unsqueeze(1)).repeat(
            1, reference_head.config.action_horizon, 1)
        status = reference_head.ego_status_encoder(reference_head._prepare_status_feature(inputs.status_feature))
        velocity[replay] = reference_head.predict_flow_velocity(
            x_t[replay], time[replay], vl, history, status, mask).to(velocity)
    return dict(zip(FIELDS, (velocity.detach(), x_t.detach(), time.detach(), (weights * replay).detach())))


def target_aligned_fields(reference_head, features, targets, roles, weights):
    """E11 disjoint objectives: Adapt=FM; Boundary/Normal=reference MSE only.

    targets contains the P teacher on Adapt and the saved Base bridge elsewhere.
    Use the original global-mean weights for reference supervision BEFORE masking
    Flow weights. Otherwise pure-preservation rows would silently get zero loss.
    """
    if (roles.dtype != torch.long or roles.shape != weights.shape or roles.ndim != 1
            or not torch.isin(roles, roles.new_tensor([0, 1, 2])).all()
            or weights.requires_grad or not torch.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("E11 requires explicit roles and positive detached global-mean weights")
    fields = reference_fields(reference_head, features, targets, roles != 0, weights)
    fields["flow_loss_weights"] = weights * (roles == 0)
    return fields
