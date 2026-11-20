"""Exact RC-F parameter ownership from the evaluated implementation."""
import math
import torch
from .constants import make_late_dit_config
QUERY_PREFIX="model.camera_grounded_query."

def assert_nobody_model(head):
    """Explicit RC-F opt-in, not a fallback for arbitrary non-Camera planners.

    Markers are ordinary attributes: they add neither tensors nor checkpoint
    keys. The No-Body agent constructs the canonical Base head before setting
    them; missing/disabled/alternative geometry branches are never accepted.
    """
    if getattr(head, 'rcf_no_body', False) is not True or getattr(head.model, 'rcf_no_body', False) is not True:
        raise ValueError('No-Body requires explicit head and model RC-F markers')
    for owner in (head, head.model):
        if any((hasattr(owner, key) for key in ('camera_grounded_query', 'footprint_query', 'geometry_conditioner', 'camera_store'))):
            raise ValueError('No-Body cannot contain Camera, Body or geometry branches')

def assert_nobody_inputs(inputs):
    """Require the upstream boundary to remove spatial/vehicle features.

    C32 multimodal metadata stays unchanged. Matched-cohort Camera manifests
    may exist in the data protocol, but are not model inputs in this ablation.
    """
    forbidden = [key for key in inputs if key in ('vehicle_geometry', 'projection_geometry') or key.startswith(('camera_', 'geometry_', 'g3_', 'flow_rollout_'))]
    if forbidden:
        raise ValueError(f'No-Body received incompatible conditions/objectives: {sorted(forbidden)}')

def late_dit_parameter_names(head):
    """Enumerate parameters, never exempt an entire block's state/buffers."""
    model = head.model
    if len(model.transformer_blocks) != 16 or tuple(model.main_cross_attention_layers) != tuple(range(1, 16, 2)):
        raise ValueError('Late-DiT requires the canonical 16-block interleaved Dense8')
    config = make_late_dit_config()
    selected = []
    for key, module_name in (('self_attention_layers', 'attn'), ('ffn_layers', 'ffn'), ('adaln_layers', 'adaLN_modulation')):
        for layer in config[key]:
            if module_name == 'attn' and layer in model.main_cross_attention_layers:
                raise ValueError('Late-DiT must not unfreeze scene Cross-Attention')
            module = getattr(model.transformer_blocks[layer], module_name)
            names = [f'model.transformer_blocks.{layer}.{module_name}.{name}' for name, _ in module.named_parameters()]
            if not names:
                raise ValueError('Empty Late-DiT parameter module')
            selected.extend(names)
    available = dict(head.named_parameters())
    if len(set(selected)) != len(selected) or any((n not in available for n in selected)):
        raise ValueError('Late-DiT parameter ownership is ambiguous')
    return tuple(sorted(selected))

def camera_trainable_names(head, late_dit=False):
    query = tuple((n for n, _ in head.named_parameters() if n.startswith(QUERY_PREFIX)))
    if not query:
        raise ValueError('Camera training requires its complete Query branch')
    return tuple(sorted((*query, *(late_dit_parameter_names(head) if late_dit else ()))))

def configure_camera_trainables(head, query_lr, late_dit_lr=None):
    for lr in (query_lr,) if late_dit_lr is None else (query_lr, late_dit_lr):
        if type(lr) not in (float, int) or not math.isfinite(lr) or lr <= 0:
            raise ValueError('Camera optimizer learning rates must be finite and positive')
    names = camera_trainable_names(head, late_dit=late_dit_lr is not None)
    head.requires_grad_(False)
    parameters = dict(head.named_parameters())
    for name in names:
        parameters[name].requires_grad_(True)
    groups = [{'params': [parameters[n] for n in names if n.startswith(QUERY_PREFIX)], 'lr': query_lr, 'initial_lr': query_lr}]
    if late_dit_lr is not None:
        groups.append({'params': [parameters[n] for n in names if not n.startswith(QUERY_PREFIX)], 'lr': late_dit_lr, 'initial_lr': late_dit_lr})
    return groups
