"""Shared five-step sampler contract and observation routing."""
import torch

def check_sampler(head):
    if head.config.sampling_method != 'flow' or head.config.num_inference_steps != 5 or head.config.action_horizon != 8 or (head.config.action_dim != 3) or (getattr(head, 'final_action_clip_value', 1.0) != 1.0):
        raise ValueError('E12R requires unchanged five-step Flow and final [-1,1] clipping')

def observations(inputs, selected):
    """Only ordinary conditions; no teacher, bridge, reference or loss tensors."""
    from transformers.feature_extraction_utils import BatchFeature
    return BatchFeature(data={k: v[selected] for k, v in inputs.items() if k not in ('action', 'last_hidden_state', 'history_trajectory') and (not k.startswith(('flow_', 'g3_')))})
