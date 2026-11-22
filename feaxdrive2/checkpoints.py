"""Strict portable checkpoint I/O. No implicit partial loading or downloads."""
from pathlib import Path
from hashlib import sha256
import torch

FORMAT = "feaxdrive2-planner-v1"
ALIGNMENT_FORMAT = "feaxdrive2-joint-alignment-v1"


def checkpoint_identity(path):
    """Identify the exact adapted feature source, without loading the VLM."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_file(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def validate_state(module, state):
    expected = module.state_dict()
    if expected.keys() != state.keys():
        raise ValueError(f"Checkpoint keys differ: missing={sorted(expected.keys()-state.keys())}, "
                         f"extra={sorted(state.keys()-expected.keys())}")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
            raise ValueError(f"Checkpoint shape/type mismatch: {name}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")


def save_planner(planner, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if planner.action_head.spatial_query_update != planner.spatial_query_update:
        raise ValueError("Query-update policy does not match the planner variant")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "variant": planner.variant,
                "state_dict": {k: v.detach().cpu().clone() for k, v in planner.action_head.state_dict().items()}}, path)


def load_planner(planner, path):
    payload = load_file(path)
    if payload.get("format") != FORMAT or payload.get("variant") != planner.variant:
        raise ValueError("Checkpoint format/variant does not match the requested model")
    validate_state(planner.action_head, payload["state_dict"])
    planner.action_head.load_state_dict(payload["state_dict"], strict=True)
    planner.action_head.spatial_query_update = planner.spatial_query_update
    return planner


def load_alignment(path):
    from .training.constants import lora_configuration
    payload = load_file(path)
    if payload.get("format") != ALIGNMENT_FORMAT or payload.get("lora_config") != lora_configuration():
        raise ValueError("Expected the paper's joint-alignment checkpoint/configuration")
    if not payload.get("lora") or not payload.get("dit"):
        raise ValueError("Joint alignment requires both language LoRA and action-generator tensors")
    return payload


def apply_alignment(planner, payload):
    from .training.alignment_parameters import restore_dit
    if planner.variant != "body_scene":
        raise ValueError("Apply joint alignment to the Body–Scene Grounded Planner")
    restore_dit(planner.action_head, payload["dit"])
    # Language tensors are loaded separately by vlm.runtime.native_readout.


def save_alignment(model, path):
    from .training.constants import lora_configuration
    from .training.alignment_parameters import checkpoint_parameters
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    values = {name: value.detach().cpu().clone()
              for name, value in model.language.language_model.named_parameters()
              if ".lora_A." in name or ".lora_B." in name}
    if len(values) != 32:
        raise ValueError("Expected 32 language LoRA tensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format=ALIGNMENT_FORMAT, lora_config=lora_configuration(),
                    lora=values, dit=checkpoint_parameters(model.agent.action_head)), path)
