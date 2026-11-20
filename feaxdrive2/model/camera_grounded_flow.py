# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Causal five-step query history; no teacher argument is accepted here.

FM still regresses on a teacher/noise bridge. Its auxiliary body context comes
from a detached free rollout of the current policy, not a teacher bridge prefix.
These two state distributions are deliberately not claimed to be identical.
"""

import torch


def discrete_flow_time(time):
    if time.ndim != 1 or not torch.isfinite(time).all() or ((time < 0) | (time >= 1)).any():
        raise ValueError("Camera Query requires finite Flow time [B] in [0,1)")
    return torch.floor(time * 5).clamp(max=4) / 5


@torch.no_grad()
def rollout_prefix(noise, time, velocity, denormalize):
    """Return previous denoised metric trajectory for each requested FM step.

    velocity(x, t, previous_metric) is the SAME field used during inference.
    x remains unclipped. Only the detached query estimate is clipped. Step zero
    uses literal metric origin, not denormalized zero latent coordinates.
    """
    time = discrete_flow_time(time)
    steps = (time * 5).round().long()
    if (noise.ndim != 3 or noise.shape[0] != len(time) or min(noise.shape[1:]) < 1
            or not torch.isfinite(noise).all()):
        raise ValueError("Expected finite noise [B,action_horizon,action_dim]")
    x = noise.detach().clone()
    previous = torch.zeros_like(x)
    result = torch.zeros_like(x)
    # The caller's AMP scope also contains the subsequent grad-enabled FM pass.
    # Do not cache no_grad casts of trainable weights here: reusing those casts
    # would silently detach Linear parameters from FM (and break DDP reduction).
    # Preserve the caller's precision, autocast enablement and cache state.
    cache_enabled = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        for step in range(int(steps.max().item())):
            t = torch.full_like(time, step / 5)
            v = velocity(x, t, previous)
            # Estimate from pre-update x, NEVER from x + dt*v.
            previous = denormalize((x + (1 - step / 5) * v).clamp(-1, 1)).detach()
            x = x + .2 * v
            result = torch.where((steps > step)[:, None, None], previous, result)
    finally:
        torch.set_autocast_cache_enabled(cache_enabled)
    return result
