# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""A4 language-only LoRA and its shared train/inference cache contract.

The compact cache contains fixed base-CoC IDs/positions, original image paths,
canonical-FP16 frozen reference features, and original planner inputs. E1 schema3
captures native prefill/decode; historical schema2 retains full-sequence replay.
Vision/DeepStack and processed pixels are NEVER persisted. CPU dataset workers
call ``prepare_pixels``; frozen online vision materializes the native BF16 dense
boundary before the existing differentiable language replay. No spatial camera
sidecar or vehicle geometry is owned here.

``collate_boundary`` combines these into the batch consumed by the language
backbone and planner. ``planner_features`` replaces only the five post-language
fields; the caller retains original planner inputs and appends camera/geometry.
Heavy model imports are deferred so cache indexing needs no ML environment.
"""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager, nullcontext
from functools import lru_cache
import gc
import gzip
import json
import pickle
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


SCHEMA_VERSION = 2
NATIVE_SCHEMA_VERSION = 3
SOURCE_PACK = "A4_V48_E16_route_CoC"
PLANNER_KEYS = (
    "last_hidden_state", "multimodal_type_ids", "multimodal_position_ids",
    "multimodal_valid_mask", "coc_valid",
)
INPUT_KEYS = ("history_trajectory", "status_feature", "high_command_one_hot")
BOUNDARY_SCALARS = (
    "llm_spatial_merge_size", "llm_visual_pool_size", "llm_history_tokens",
    "llm_max_coc_tokens", "llm_route_required", "llm_image_pad_token_id",
    "llm_traj_history_start_token_id", "llm_traj_history_end_token_id",
    "llm_route_start_token_id", "llm_route_end_token_id",
    "llm_cot_start_token_id", "llm_cot_end_token_id",
)
BOUNDARY_SEQUENCES = (
    "llm_context_input_ids", "llm_attention_mask", "llm_visual_pos_mask",
    "llm_image_valid_mask", "llm_image_grid_thw",
    "llm_image_grid_valid_mask",
)
DENSE_KEYS = ("llm_image_embeds", "llm_deepstack_visual_embeds")


def cache_path(root: str | Path, token: str) -> Path:
    """One immutable file per token, shared by extraction and all readers."""
    if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{2,}", token):
        raise ValueError("token must be at least two safe filename characters")
    return Path(root) / token[:2] / f"{token}.gz"


def load_payload(path: str | Path) -> dict:
    """Read this pipeline's single dictionary (never the old boundary cache)."""
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected Planning LoRA dictionary: {path}")
    cache_path(".", payload.get("token"))
    metadata = payload.get("metadata", {})
    if metadata.get("schema_version") not in (SCHEMA_VERSION, NATIVE_SCHEMA_VERSION) or metadata.get("source_pack") != SOURCE_PACK:
        raise ValueError(f"Unsupported Planning LoRA payload: {path}")
    for key in ("boundary", "reference_features", "split", "image_paths", *INPUT_KEYS):
        if key not in payload:
            raise ValueError(f"Missing Planning LoRA payload field: {key}")
    if any(key in payload["boundary"] for key in (*DENSE_KEYS, "llm_pixel_values")):
        raise ValueError("Persistent cache must contain no dense vision or processed pixels")
    if metadata["schema_version"] == NATIVE_SCHEMA_VERSION:
        if "llm_prompt_length" not in payload["boundary"] or metadata.get("hidden_source") != "generation_capture":
            raise ValueError("Native E1 cache must capture original prefill/decode boundaries")
    return payload


@lru_cache(maxsize=1)
def _image_processor(processor_path):
    from transformers import AutoProcessor
    from alpamayo1_5 import helper

    return AutoProcessor.from_pretrained(
        processor_path, min_pixels=helper.MIN_PIXELS, max_pixels=helper.MAX_PIXELS,
        local_files_only=True,
    ).image_processor


def _load_camera_tensor(image_paths):
    from .planning_readout import load_camera_tensor
    return load_camera_tensor(image_paths)


def prepare_pixels(payload: dict, processor_path: str) -> dict:
    """Return a NEW payload with CPU ``boundary.llm_pixel_values`` [P,1536].

    Uses the same RGB loading and processor overrides as original extraction.
    No tokenizer, generation, GPU, vision forward, cache write or input mutation.
    The exact cached grid must match; fixed IDs/positions are never regenerated.
    ``collate_payloads`` concatenates P across scenes, with grids retaining order.
    """
    import torch

    boundary = payload["boundary"]
    if any(key in boundary for key in DENSE_KEYS):
        raise ValueError("prepare_pixels expects a compact cache payload")
    validate_boundary(boundary)
    resolved = str(Path(processor_path).expanduser().resolve())
    if resolved != str(Path(payload["metadata"]["processor_path"]).expanduser().resolve()):
        raise ValueError("Processor path differs from compact-cache provenance")
    paths = payload["image_paths"]
    if len(paths) != 3 or any(len(camera) != 4 for camera in paths):
        raise ValueError("A4 CPU preprocessing requires three cameras x four frames")
    frames = _load_camera_tensor(paths)
    processed = _image_processor(resolved)(
        images=list(frames.flatten(0, 1)), return_tensors="pt", device="cpu",
    )
    grid, pixels = processed["image_grid_thw"], processed["pixel_values"]
    if not torch.equal(grid.cpu(), boundary["llm_image_grid_thw"].cpu()):
        raise ValueError("Online processor image grid differs from fixed-token cache")
    expected_patches = int(grid.prod(dim=1).sum())
    if pixels.device.type != "cpu" or pixels.shape != (expected_patches, 1536):
        raise ValueError("Expected original CPU patchified pixels [sum(t*h*w),1536]")
    return dict(payload, boundary=dict(boundary, llm_pixel_values=pixels.contiguous()))


@contextmanager
def _language_precision(device_type="cuda"):
    """Keep language BF16/efficient SDPA even inside a planner FP32 context."""
    import torch

    cuda = torch.backends.cuda
    original = (cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(), cuda.math_sdp_enabled())
    try:
        cuda.enable_flash_sdp(True)
        cuda.enable_mem_efficient_sdp(True)
        cuda.enable_math_sdp(True)
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            yield
    finally:
        cuda.enable_flash_sdp(original[0])
        cuda.enable_mem_efficient_sdp(original[1])
        cuda.enable_math_sdp(original[2])


def _checkpoint_contexts():
    # Non-reentrant backward recomputation occurs after planner_features exits.
    import torch
    return nullcontext(), _language_precision("cuda" if torch.cuda.is_available() else "cpu")


def _load_pretrained(model_path, vlm_path):
    import torch
    from .llm_lora_backbone import Alpamayo1_5, Alpamayo1_5Config

    model_root, vlm_root = Path(model_path).expanduser().resolve(), Path(vlm_path).expanduser().resolve()
    if not (vlm_root / "config.json").is_file():
        raise FileNotFoundError(f"Missing local VLM metadata: {vlm_root}")
    with (model_root / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    config["vlm_name_or_path"] = str(vlm_root)
    return Alpamayo1_5.from_pretrained(
        str(model_root), config=Alpamayo1_5Config(**config), dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True,
    )


def configure_task_alignment_lora(language_model, spec):
    """E1: zero-increment Q/V adapters in the last eight language layers only."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from ..training.constants import lora_configuration

    if spec != lora_configuration():
        raise ValueError("Changed E1 high-layer Q/V LoRA configuration")
    if len(language_model.layers) != spec["language_layers"]:
        raise ValueError("E1 expects the original 36-layer language model")
    expected = {f"layers.{i}.self_attn.{target}" for i in spec["layers"]
                for target in spec["targets"]}
    modules = dict(language_model.named_modules())
    if any(not isinstance(modules.get(name), torch.nn.Linear) for name in expected):
        raise ValueError("Missing E1 high-layer Q/V projections")
    language_model.requires_grad_(False)
    adapted = get_peft_model(language_model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False,
        r=spec["rank"], lora_alpha=spec["alpha"], lora_dropout=spec["dropout"],
        target_modules=sorted(expected), bias="none", init_lora_weights=True,
    ))
    trainable = [name for name, value in adapted.named_parameters() if value.requires_grad]
    found = set()
    for name in trainable:
        match = re.search(r"(layers\.\d+\.self_attn\.(?:q_proj|v_proj))\.lora_([AB])\.", name)
        if match is None or match[1] not in expected:
            raise RuntimeError("Unexpected E1 trainable parameter: " + name)
        found.add((match[1], match[2]))
    if found != {(name, part) for name in expected for part in ("A", "B")}:
        raise RuntimeError("Incomplete E1 LoRA A/B coverage")
    return adapted


def _online_language_class():
    # Keep heavy dependencies lazy, and reuse the existing five-output replay.
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from .llm_lora_backbone import AlpamayoLanguageLoRABackbone, configure_all_layer_qkvo_lora

    class OnlineVisionLanguage(AlpamayoLanguageLoRABackbone):
        def __init__(self, full_model, device, gradient_checkpointing, lora_config=None):
            torch.nn.Module.__init__(self)
            qwen = full_model.vlm.model
            text = qwen.language_model
            self.hidden_size = int(text.config.hidden_size)
            self.num_hidden_layers = len(text.layers)
            self.native_replay = lora_config is not None
            self.language_model = (configure_all_layer_qkvo_lora(
                text, rank=8, alpha=16, dropout=0., expected_layers=36,
            ) if lora_config is None else configure_task_alignment_lora(text, lora_config))
            self.language_model.config.use_cache = self.native_replay
            # Retain Qwen's native get_image_features helper with ONLY its vision
            # module. Remove the language alias BEFORE freezing the vision owner.
            del qwen.language_model
            self.frozen_vision = qwen.eval().requires_grad_(False)
            # Do not rely on PEFT's version-dependent adapter autocasting.
            # Preserve Parameter identities before any optimizer is created.
            for name, parameter in self.language_model.named_parameters():
                if parameter.requires_grad:
                    if "lora_" not in name:
                        raise RuntimeError(f"Non-LoRA trainable language parameter: {name}")
                    parameter.data = parameter.data.float()
            # Checkpointed mutable KV caches cannot be replayed safely during
            # backward. E1 uses eval-mode cached attention with live autograd;
            # only the upper eight language layers require activations.
            if gradient_checkpointing and not self.native_replay:
                self.language_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False, "context_fn": _checkpoint_contexts},
                )
            self.to(torch.device(device))
            for name, parameter in self.named_parameters():
                expected = torch.float32 if parameter.requires_grad else torch.bfloat16
                if parameter.dtype != expected:
                    raise RuntimeError(
                        f"Planning LoRA precision contract: {name} is {parameter.dtype}, expected {expected}"
                    )

        def train(self, mode=True):
            super().train(mode)
            self.frozen_vision.eval()
            if self.native_replay:
                self.language_model.eval()
            return self

        def materialize_boundary(self, features):
            """Transient dense boundary; no vision graph and no disk cache."""
            if "llm_image_embeds" in features:
                if "llm_deepstack_visual_embeds" not in features:
                    raise ValueError("Dense boundary is missing DeepStack")
                return features
            pixels = features["llm_pixel_values"]
            grid = features["llm_image_grid_thw"][features["llm_image_grid_valid_mask"].bool()].long()
            counts = features["llm_image_valid_mask"].sum(dim=1).tolist()
            self.frozen_vision.eval()
            with torch.no_grad(), _language_precision(pixels.device.type):
                image_parts, deepstack = self.frozen_vision.get_image_features(pixels, grid)
                images = torch.cat(list(image_parts), dim=0)
                if len(deepstack) != 3 or any(value.shape != images.shape for value in deepstack):
                    raise ValueError("Expected all three full-resolution DeepStack outputs")
                if images.shape[0] != sum(counts):
                    raise ValueError("Online vision does not match fixed visual token masks")
                if any(value.dtype not in (torch.bfloat16, torch.float32) for value in (images, *deepstack)):
                    raise ValueError("Online vision must preserve native BF16/FP32 precision")
                padded_images = pad_sequence(images.split(counts), batch_first=True)
                padded_deepstack = torch.stack([
                    pad_sequence(value.split(counts), batch_first=True) for value in deepstack
                ], dim=1)
            return dict(features, llm_image_embeds=padded_images,
                        llm_deepstack_visual_embeds=padded_deepstack)

        def forward(self, features):
            features = self.materialize_boundary(features)
            if not self.native_replay:
                return super().forward(features)
            return native_causal_replay(self, features)

    return OnlineVisionLanguage


def native_causal_replay(backbone, features):
    """Differentiate fixed CoC IDs using native B1 prefill + cached decode.

    The entire past-key/value chain stays live. There is no gradient through
    token choices. Eval mode avoids dropout/checkpoint cache mutation; it does
    not disable autograd. Full-sequence replay is deliberately not substituted.
    """
    import torch

    embeds, mask, visual, deepstack = backbone._language_inputs(features)
    if embeds.shape[0] != 1 or not bool(mask.all()):
        raise ValueError("Native E1 replay requires unpadded microbatch1")
    prompt = int(features["llm_prompt_length"].reshape(-1)[0])
    length = embeds.shape[1]
    if not 0 < prompt <= length or bool(visual[:, prompt:].any()):
        raise ValueError("Invalid native E1 prefill/visual boundary")
    backbone.language_model.eval()
    positions = features["llm_position_ids"].long().permute(1, 0, 2)
    cache, hidden = None, []
    for start, end in [(0, prompt)] + [(i, i+1) for i in range(prompt, length)]:
        output = backbone.language_model(input_ids=None, inputs_embeds=embeds[:, start:end],
            attention_mask=mask[:, :end], position_ids=positions[:, :, start:end],
            past_key_values=cache, use_cache=True,
            cache_position=torch.arange(start, end, device=embeds.device),
            visual_pos_masks=visual[:, :prompt] if start == 0 else None,
            deepstack_visual_embeds=list(deepstack) if start == 0 else None)
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("Native E1 text model discarded its KV cache")
        hidden.append(output.last_hidden_state)
    return backbone._pack_post_language(torch.cat(hidden, dim=1), features)


def build_language(
    model_path: str, vlm_path: str, device: str,
    gradient_checkpointing: bool = True,
    *, lora_config=None,
) -> torch.nn.Module:
    """Load frozen online vision plus language LoRA (explicit E1 or historical).

    E1 retains a live KV-cache graph in eval mode; automatic activation
    checkpointing is disabled because it would recompute mutable cache updates.
    Forward accepts pixels or an already materialized dense boundary and returns
    the original five tensors. Only LoRA is trainable, with FP32 parameters for
    FP32 AdamW moments; frozen language/vision parameters must remain BF16.
    Expert/output head are
    discarded; vision stays eval under both language.train() and language.eval().
    """
    import torch

    with torch.no_grad():
        full_model = _load_pretrained(model_path, vlm_path)
        options = {} if lora_config is None else {"lora_config": lora_config}
        language = _online_language_class()(full_model, device, gradient_checkpointing, **options)
        del full_model
        gc.collect()
    return language


def _check_source_pack(features: dict[str, torch.Tensor]) -> None:
    """Reject reduced historical source packs, independently of adapter design."""
    import torch

    hidden = features["last_hidden_state"]
    types = features["multimodal_type_ids"]
    positions = features["multimodal_position_ids"]
    valid = features["multimodal_valid_mask"]
    if hidden.ndim != 3 or types.shape != hidden.shape[:2]:
        raise ValueError("Planner hidden/types must be [B,S,C]/[B,S]")
    if positions.shape != types.shape or valid.shape != types.shape:
        raise ValueError("Planner positions/mask must match type IDs")
    if valid.dtype != torch.bool or not torch.equal(valid, types.ge(0)):
        raise ValueError("Planner valid mask must match nonnegative type IDs")
    for row in range(types.shape[0]):
        row_types = types[row, valid[row]]
        if bool((row_types > 3).any()):
            raise ValueError("Unknown planner modality")
        counts = torch.bincount(row_types.long(), minlength=4).tolist()
        if counts[:2] != [48, 16] or max(counts[2:]) > 256:
            raise ValueError(f"Expected A4 V48/E16/route/CoC, got {counts}")
        expected_types = torch.repeat_interleave(
            torch.arange(4, device=types.device),
            torch.tensor(counts, device=types.device),
        )
        expected_positions = torch.cat([
            torch.arange(count, device=types.device) for count in counts
        ])
        if not torch.equal(row_types, expected_types) or not torch.equal(
            positions[row, valid[row]], expected_positions
        ):
            raise ValueError("Planner pack must have ordered modalities and local positions")
        if bool(features["coc_valid"].reshape(-1)[row]) != bool(counts[3]):
            raise ValueError("CoC validity disagrees with packed tokens")


def planner_features(language, boundary: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Differentiable canonical FP16 publication -> FP32 planner conversion.

    ``boundary`` is a batched dict from ``collate_boundary`` (its reference and
    original planner inputs are harmless extra keys). Never detach the adapted
    output: the two casts must remain on the autograd path.
    """
    import torch

    for key, value in (("llm_visual_pool_size", 2), ("llm_history_tokens", 16),
                       ("llm_max_coc_tokens", 256)):
        if not bool(boundary[key].eq(value).all()):
            raise ValueError(f"Planning LoRA locks {key}={value}")
    with _language_precision(boundary["llm_context_input_ids"].device.type):
        output = language(boundary)
    if len(output) != 5:
        raise ValueError("Language forward must return five tensors")
    result = dict(zip(PLANNER_KEYS, output))
    _check_source_pack(result)
    result["last_hidden_state"] = result["last_hidden_state"].to(torch.float16).float()
    return result


def validate_boundary(boundary: dict[str, torch.Tensor]) -> None:
    """Validate compact fixed tokens, or a transient unpadded dense boundary."""
    import torch

    for key in (*BOUNDARY_SCALARS, *BOUNDARY_SEQUENCES,
                "llm_position_ids", "llm_rope_deltas"):
        if key not in boundary or not isinstance(boundary[key], torch.Tensor):
            raise ValueError(f"Missing boundary tensor: {key}")
    ids = boundary["llm_context_input_ids"]
    if ids.ndim != 1 or not ids.numel():
        raise ValueError("Boundary IDs must be a nonempty sequence")
    length = ids.numel()
    if "llm_prompt_length" in boundary:
        prompt = boundary["llm_prompt_length"]
        if (prompt.ndim != 0 or prompt.is_floating_point() or prompt.dtype == torch.bool
                or not 0 < int(prompt) <= length):
            raise ValueError("Native prefill length must be a positive integer scalar")
    if boundary["llm_position_ids"].shape != (3, length):
        raise ValueError("Boundary position IDs must be [3,L]")
    for key in BOUNDARY_SCALARS:
        if boundary[key].ndim != 0:
            raise ValueError(f"Boundary {key} must be scalar")
    for key, value in (("llm_visual_pool_size", 2), ("llm_history_tokens", 16),
                       ("llm_max_coc_tokens", 256)):
        if int(boundary[key]) != value:
            raise ValueError(f"Planning LoRA locks {key}={value}")
    mask = boundary["llm_attention_mask"]
    visual_mask = boundary["llm_visual_pos_mask"]
    if mask.shape != (length,) or mask.dtype != torch.bool or not bool(mask.all()):
        raise ValueError("Per-token boundary must be unpadded with bool attention")
    if (visual_mask.shape != (length,) or visual_mask.dtype != torch.bool
            or not torch.equal(visual_mask, ids.eq(boundary["llm_image_pad_token_id"]))):
        raise ValueError("Visual mask does not match image token IDs")
    visual_tokens = int(visual_mask.sum())
    if any(key in boundary for key in DENSE_KEYS):
        if not all(key in boundary for key in DENSE_KEYS):
            raise ValueError("Dense boundary must include vision and DeepStack together")
        images, deepstack = (boundary[key] for key in DENSE_KEYS)
        if images.ndim != 2 or images.shape[0] != visual_tokens:
            raise ValueError("Image embeddings must match visual token count")
        if deepstack.ndim != 3 or deepstack.shape[0] != 3 or deepstack.shape[1:] != images.shape:
            raise ValueError("DeepStack must be [3,N,C] and include every visual token")
        for value in (images, deepstack):
            if value.dtype not in (torch.bfloat16, torch.float32) or not bool(torch.isfinite(value).all()):
                raise ValueError("Preserve finite native BF16/FP32 vision; FP16 boundary is forbidden")
    grid = boundary["llm_image_grid_thw"]
    merge = int(boundary["llm_spatial_merge_size"])
    if (grid.shape != (12, 3) or not bool(grid[:, 0].eq(1).all())
            or merge <= 0 or not bool((grid[:, 1:] > 0).all())
            or bool((grid[:, 1:] % merge != 0).any())
            or int((grid.prod(dim=1) // merge**2).sum()) != visual_tokens):
        raise ValueError("A4 boundary requires twelve independent merge-aligned image grids")
    for key, count in (("llm_image_valid_mask", visual_tokens),
                       ("llm_image_grid_valid_mask", 12)):
        value = boundary[key]
        if value.shape != (count,) or value.dtype != torch.bool or not bool(value.all()):
            raise ValueError(f"Unpadded boundary requires all-valid {key}")
    if "llm_pixel_values" in boundary:
        pixels = boundary["llm_pixel_values"]
        if pixels.ndim != 2 or pixels.shape != (int(grid.prod(dim=1).sum()), 1536):
            raise ValueError("Pixel rows/width disagree with the original processor grid")
        if not pixels.is_floating_point() or not bool(torch.isfinite(pixels).all()):
            raise ValueError("Processed pixels must be finite floating tensors")


def collate_payloads(payloads: list[dict], device=None) -> dict[str, torch.Tensor]:
    """Right-pad language and reference axes separately; keep vision dtype.

    Positions: [B,3,L]; pixels: [sum(P),1536]; image grids: [B,G,3].
    For transient dense inputs instead: DeepStack [B,D,N,C], images [B,N,C].
    Sequence/image/grid masks exclude all padding. Stored reference hidden is
    canonical FP16, promoted to [B,S,C] FP32 for the planner without value changes.
    ``device=None`` leaves tensors on their current device (normally CPU).
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if not payloads:
        raise ValueError("Cannot collate an empty boundary batch")
    boundaries = [payload["boundary"] for payload in payloads]
    for payload, boundary in zip(payloads, boundaries):
        metadata = payload["metadata"]
        if metadata.get("schema_version") not in (SCHEMA_VERSION, NATIVE_SCHEMA_VERSION) or metadata.get("source_pack") != SOURCE_PACK:
            raise ValueError("Unsupported Planning LoRA cache schema/source pack")
        validate_boundary(boundary)
    versions = {p["metadata"]["schema_version"] for p in payloads}
    if len(versions) != 1:
        raise ValueError("Cannot mix full-replay and native-causal cache schemas")
    native = next(iter(versions)) == NATIVE_SCHEMA_VERSION
    if native and any("llm_prompt_length" not in b or p["metadata"].get("hidden_source") != "generation_capture"
                      for p, b in zip(payloads, boundaries)):
        raise ValueError("Native cache is missing its prefill/capture contract")
    dense = "llm_image_embeds" in boundaries[0]
    if any(("llm_image_embeds" in boundary) != dense for boundary in boundaries):
        raise ValueError("Cannot mix compact and dense boundaries in one batch")
    result = {
        key: pad_sequence([b[key] for b in boundaries], batch_first=True, padding_value=0)
        for key in BOUNDARY_SEQUENCES
    }
    result["llm_position_ids"] = pad_sequence(
        [b["llm_position_ids"].T for b in boundaries], batch_first=True,
    ).transpose(1, 2).contiguous()
    if dense:
        for key in DENSE_KEYS:
            if any(b[key].dtype != boundaries[0][key].dtype for b in boundaries):
                raise ValueError(f"Mixed native dtypes for {key}")
        if len({tuple(b["llm_deepstack_visual_embeds"].shape[::2]) for b in boundaries}) != 1:
            raise ValueError("Mixed DeepStack layer counts/hidden widths")
        result["llm_image_embeds"] = pad_sequence([b["llm_image_embeds"] for b in boundaries], batch_first=True)
        result["llm_deepstack_visual_embeds"] = pad_sequence(
            [b["llm_deepstack_visual_embeds"].transpose(0, 1) for b in boundaries],
            batch_first=True,
        ).transpose(1, 2).contiguous()
    else:
        if any("llm_pixel_values" not in b for b in boundaries):
            raise ValueError("Call prepare_pixels(payload, processor_path) before collation")
        result["llm_pixel_values"] = torch.cat([b["llm_pixel_values"] for b in boundaries], dim=0)
    for key in (*BOUNDARY_SCALARS, "llm_rope_deltas"):
        result[key] = torch.stack([b[key] for b in boundaries])
    if native:
        result["llm_prompt_length"] = torch.stack([b["llm_prompt_length"] for b in boundaries])
    for key in BOUNDARY_SCALARS:
        if key != "llm_route_required" and not bool(result[key].eq(result[key][0]).all()):
            raise ValueError(f"Mixed locked boundary setting: {key}")
    for key in INPUT_KEYS:
        result[key] = torch.stack([payload[key].float() for payload in payloads])
    for key in PLANNER_KEYS:
        values = [payload["reference_features"][key] for payload in payloads]
        if key == "last_hidden_state":
            if any(value.dtype != torch.float16 for value in values):
                raise ValueError("Stored reference hidden must use canonical FP16 publication")
            values = [value.float() for value in values]
        result[key] = (torch.stack(values) if key == "coc_valid" else pad_sequence(
            values, batch_first=True, padding_value=-1 if key == "multimodal_type_ids" else 0,
        ))
    _check_source_pack(result)
    if device is not None:
        result = {key: value.to(device) for key, value in result.items()}
    return result


# Both public spellings were requested by the boundary and main-runner owners.
collate_boundary = collate_payloads
