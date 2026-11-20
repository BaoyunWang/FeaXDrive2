# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""One explicit precision contract for RC-F actor, reference and deployment.

Cached features are promoted, not regenerated: FP32 execution cannot recover
information already quantized by the frozen VLM. Historical experiments keep
their own precision; this context is opt-in and restores caller backend state.
"""

from contextlib import contextmanager


def precision_contract():
    return dict(schema="RCF-Strict-FP32-v1", parameters="float32", floating_inputs="float32",
                actor="float32", reference="float32", generation="float32", evaluation="float32",
                autocast=False, matmul_tf32=False, cudnn_tf32=False,
                bf16_reduced_reduction=False, flash_sdp=False, mem_efficient_sdp=False,
                math_sdp=True, cache="unchanged-values-promoted-to-float32")


def backend_state():
    import torch
    cuda = torch.backends.cuda
    return dict(matmul_tf32=cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                bf16_reduced_reduction=cuda.matmul.allow_bf16_reduced_precision_reduction,
                flash_sdp=cuda.flash_sdp_enabled(), mem_efficient_sdp=cuda.mem_efficient_sdp_enabled(),
                math_sdp=cuda.math_sdp_enabled())


def _set_backend(state):
    import torch
    cuda = torch.backends.cuda
    cuda.matmul.allow_tf32 = state["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = state["cudnn_tf32"]
    cuda.matmul.allow_bf16_reduced_precision_reduction = state["bf16_reduced_reduction"]
    cuda.enable_flash_sdp(state["flash_sdp"])
    cuda.enable_mem_efficient_sdp(state["mem_efficient_sdp"])
    cuda.enable_math_sdp(state["math_sdp"])


@contextmanager
def strict_fp32():
    import torch
    original = backend_state()
    try:
        _set_backend(precision_contract())
        with torch.autocast("cuda", enabled=False):
            yield precision_contract()
    finally:
        _set_backend(original)


def floating_to_fp32(fields):
    """Preserve bool/int masks and cached floating values; never mutate input."""
    return {name: value.float() if value.is_floating_point() else value for name, value in fields.items()}


def assert_strict_fp32(module=None):
    import torch
    contract = precision_contract()
    if torch.is_autocast_enabled() or any(value != contract[key] for key, value in backend_state().items()):
        raise ValueError("RC-F requires strict FP32 with autocast/TF32 off and math SDPA")
    if module is not None:
        for name, value in list(module.named_parameters()) + list(module.named_buffers()):
            if value.is_floating_point() and value.dtype != torch.float32:
                raise ValueError("RC-F non-FP32 model tensor: " + name)
