"""Adapted autoregressive inference uses the same LoRA names as training replay."""
import torch
from ..training.constants import lora_configuration


def restore_lora(language, values):
    current = {name: p for name, p in language.named_parameters()
               if ".lora_A." in name or ".lora_B." in name}
    if current.keys() != values.keys():
        raise ValueError("Language checkpoint has missing/extra LoRA tensors")
    for name, value in current.items():
        if values[name].shape != value.shape or not torch.isfinite(values[name]).all():
            raise ValueError(f"Invalid LoRA tensor: {name}")
    with torch.no_grad():
        for name, value in current.items():
            value.copy_(values[name].to(value))


def native_readout(*, model_path, vlm_path, processor_path, alignment=None, device="cuda"):
    from .planning_readout import AlpamayoActionExpertReadout
    from .planning_lora import configure_task_alignment_lora
    readout = AlpamayoActionExpertReadout(model_path=model_path, vlm_path=vlm_path,
        processor_path=processor_path, device=str(device), num_probes=1,
        coc_seed=20260720, max_generation_length=256, top_p=.98, temperature=.6)
    if alignment is not None:
        qwen = readout.model.vlm.model
        qwen.language_model = configure_task_alignment_lora(qwen.language_model, lora_configuration())
        for name, parameter in qwen.language_model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                parameter.data = parameter.data.float()
        restore_lora(qwen.language_model, alignment["lora"])
    readout.model.eval().requires_grad_(False)
    return readout
