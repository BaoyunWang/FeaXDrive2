"""Joint alignment: language LoRA and the original 44 action-generator tensors."""
import re
from .constants import require

def configuration():
    return dict(schema='FeaXDrive2-E1-W-Unfreeze-v1', self_attention_layers=[12, 14], ffn_layers=[12, 13, 14, 15], adaln_layers=[12, 13, 14, 15], dit_lr=2e-06, reference='independent-frozen-original-A4-step1000', quality_teacher=False, base_objective='unchanged-E1-W', checkpoint_steps=[0, 500, 750, 1000, 1250, 1500, 1750, 2000, 2200], core_steps=[500, 1000, 1500, 2200])

def validate(spec):
    require(spec == configuration(), 'Changed E1-W-U intervention')
    return spec

def allowed_parameter(name):
    match = re.fullmatch('model\\.transformer_blocks\\.(\\d+)\\.(attn|ffn|adaLN_modulation)\\..+', name)
    if match is None:
        return False
    layer, component = (int(match[1]), match[2])
    return component == 'attn' and layer in (12, 14) or (component in ('ffn', 'adaLN_modulation') and 12 <= layer <= 15)

def dit_parameters(head):
    require(len(head.model.transformer_blocks) == 16 and (not {12, 14} & set(head.model.main_cross_attention_layers)), 'E1-W-U requires the original 16-layer DiT and self-attention at12/14')
    selected = {name: value for name, value in head.named_parameters() if allowed_parameter(name)}
    modules = {name.rsplit('.', 1)[0] for name in selected}
    require(len(selected) == 44 and modules, 'Missing or additional requested DiT parameter tensors')
    return selected

def configure_head(head, spec):
    validate(spec)
    require(not any((p.requires_grad for p in head.parameters())), 'Configure E1-W-U from frozen A4')
    selected = dit_parameters(head)
    for parameter in selected.values():
        parameter.requires_grad_(True)
    head.e1wu_trainable_names = tuple(selected)
    return selected

def trainable_groups(model, optimizer_config):
    dit = {'agent.action_head.' + name: value for name, value in dit_parameters(model.agent.action_head).items()}
    lora = {name: value for name, value in model.named_parameters() if name.startswith('language.') and ('.lora_A.' in name or '.lora_B.' in name)}
    actual = {name: value for name, value in model.named_parameters() if value.requires_grad}
    require(lora and actual.keys() == (lora | dit).keys(), 'E1-W-U trainable parameter set differs')
    require(all((value.requires_grad for value in (*lora.values(), *dit.values()))), 'Missing requested gradients')
    require(len({id(value) for value in actual.values()}) == len(actual), 'Aliased optimizer parameters')
    return [dict(params=list(lora.values()), name='vlm_lora', lr=optimizer_config['lr'], lr_multiplier=1.0), dict(params=list(dit.values()), name='action_dit', lr=2e-06, lr_multiplier=2e-06 / optimizer_config['lr'])]

def checkpoint_parameters(head):
    return {name: value.detach().cpu().clone() for name, value in dit_parameters(head).items()}

def restore_dit(head, values):
    import torch
    parameters = dit_parameters(head)
    require(parameters.keys() == values.keys(), 'Missing or extra E1-W-U checkpoint tensors')
    with torch.no_grad():
        for name, parameter in parameters.items():
            saved = values[name]
            require(saved.shape == parameter.shape and saved.dtype == torch.float32 and bool(torch.isfinite(saved).all()), 'Invalid E1-W-U DiT tensor: ' + name)
            parameter.copy_(saved.to(parameter))

def inventory(model):
    categories = {name: [] for name in ('VLM', 'ConditionAdapter', 'ActionDiT', 'SpatialRetrieval')}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if name.startswith('language.'):
            category = 'VLM'
        elif 'camera_grounded_query' in name:
            category = 'SpatialRetrieval'
        elif name.startswith('agent.action_head.model.transformer_blocks.'):
            category = 'ActionDiT'
        else:
            category = 'ConditionAdapter'
        categories[category].append(dict(name=name, shape=list(value.shape), parameters=value.numel(), dtype=str(value.dtype)))
    return {name: dict(tensors=len(values), parameters=sum((item['parameters'] for item in values)), trainable=values) for name, values in categories.items()}

def gradients(model, *, require_nonzero=True):
    import torch
    result = {}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            require(value.grad is None, 'Frozen parameter received gradient: ' + name)
            continue
        require(value.grad is not None and bool(torch.isfinite(value.grad).all()), 'Missing or nonfinite gradient: ' + name)
        group = 'VLM' if name.startswith('language.') else 'ActionDiT'
        result[group] = result.get(group, 0.0) + float(value.grad.detach().float().square().sum())
    if require_nonzero:
        require(set(result) == {'VLM', 'ActionDiT'} and all((value > 0 for value in result.values())), 'E1-W-U loss must reach both LoRA and selected DiT parameters')
    return {name: value ** 0.5 for name, value in result.items()}
