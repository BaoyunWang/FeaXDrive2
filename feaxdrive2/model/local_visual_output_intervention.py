# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Capture or disable the complete local visual-attention output for five calls.

Usage::

    with torch.no_grad(), LocalVisualOutputIntervention(branch, disabled=True) as hook:
        prediction = generate()
    diagnostics = hook.steps

The branch must already be frozen and in eval mode. Each context covers exactly
one five-step rollout, including index0. Disabling also removes learned-null
outputs and attention output-projection biases; geometry and scene attention
continue normally. Normal mode returns None from the hook, leaving the original
output tuple untouched. Validity statistics belong to the caller.
"""

from weakref import WeakSet

import torch


_ACTIVE_BRANCHES = WeakSet()


class LocalVisualOutputIntervention:
    """One five-call context; the same instance may be reused sequentially.

    ``steps`` is reset on entry and contains scalar dictionaries with:
    ``flow_index`` (call order 0..4), ``input_output_rms``,
    ``delivered_output_rms``, ``input_output_mean_norm`` and
    ``delivered_output_mean_norm``. RMS averages over all output elements;
    mean norm averages the last-dimension L2 norm over query rows. Diagnostics
    include null-only rows. They retain neither tensors nor autograd graphs.
    """

    def __init__(self, branch, disabled=False):
        if not isinstance(branch, torch.nn.Module) or not isinstance(
                getattr(branch, "visual_attention", None), torch.nn.Module):
            raise TypeError("branch must expose a visual_attention module")
        if type(disabled) is not bool:
            raise TypeError("disabled must be bool")
        self.branch = branch
        self.disabled = disabled
        self.steps = []
        self._handle = None

    def _check_execution(self):
        if torch.is_grad_enabled():
            raise RuntimeError("Local visual intervention requires torch.no_grad()")
        if any(module.training for module in self.branch.modules()):
            raise RuntimeError("Local visual intervention requires an eval() branch")
        if any(parameter.requires_grad for parameter in self.branch.parameters()):
            raise RuntimeError("Local visual intervention requires frozen parameters")

    def __enter__(self):
        if self.branch in _ACTIVE_BRANCHES:
            raise RuntimeError("Branch already has an active local visual intervention")
        self._check_execution()
        self.steps = []
        self._handle = self.branch.visual_attention.register_forward_hook(self._after_attention)
        _ACTIVE_BRANCHES.add(self.branch)
        return self

    def _after_attention(self, module, inputs, output):
        self._check_execution()
        if len(self.steps) >= 5:
            raise RuntimeError("Expected exactly five local visual-attention calls")
        if (not isinstance(output, tuple) or len(output) != 2
                or not isinstance(output[0], torch.Tensor)
                or not output[0].is_floating_point() or output[0].ndim < 2
                or not output[0].numel()):
            raise ValueError("Expected a nonempty (attention_output, weights) tuple")
        values = output[0].detach().double()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Nonfinite local visual-attention output")
        rms = float(values.square().mean().sqrt())
        mean_norm = float(values.norm(dim=-1).mean())
        self.steps.append(dict(
            flow_index=len(self.steps), input_output_rms=rms,
            delivered_output_rms=0. if self.disabled else rms,
            input_output_mean_norm=mean_norm,
            delivered_output_mean_norm=0. if self.disabled else mean_norm))
        if self.disabled:
            return torch.zeros_like(output[0]), output[1]
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            _ACTIVE_BRANCHES.discard(self.branch)
        if exc_type is None and len(self.steps) != 5:
            raise RuntimeError("Expected exactly five local visual-attention calls")
        return False
