# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Deterministic packing helpers for the E3 Alpamayo VLM context cache."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Any

import torch
import torch.nn.functional as F


VISION_MODALITY = 0
HISTORY_MODALITY = 1
ROUTE_MODALITY = 2
COC_MODALITY = 3
NUM_MODALITIES = 4

_ROWWISE_GENERATION_LOCK = threading.Lock()


@contextmanager
def rowwise_multinomial_generators(
    generators: Sequence[torch.Generator],
):
    """Give each row independent RNGs; reject unsupported/concurrent sampling.

    This process-global Torch hook is reserved before processor/hook setup and
    held through capture unpacking. It is not a thread-safe concurrent sampler:
    nested/parallel callers fail immediately instead of sharing random streams.
    """
    if not generators:
        raise ValueError("At least one row generator is required")
    if len({id(generator) for generator in generators}) != len(generators):
        raise ValueError("Rows must not share the same generator object")
    if not _ROWWISE_GENERATION_LOCK.acquire(blocking=False):
        raise RuntimeError("Concurrent or nested batched generation is forbidden")
    original_multinomial = torch.multinomial
    owner_thread = threading.get_ident()
    calls = 0

    def _rowwise_multinomial(
        input: torch.Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: torch.Generator | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonlocal calls
        if threading.get_ident() != owner_thread:
            raise RuntimeError("Sampling from another generation thread is forbidden")
        if (generator is not None or out is not None or input.ndim != 2
                or input.shape[0] != len(generators) or int(num_samples) != 1):
            raise RuntimeError("Unsupported HF sampling call; per-row RNG cannot be guaranteed")
        calls += 1
        return torch.stack(
            [original_multinomial(input[row], num_samples=1, replacement=replacement,
                                  generator=row_generator)
             for row, row_generator in enumerate(generators)], dim=0)

    torch.multinomial = _rowwise_multinomial
    try:
        yield
        if not calls:
            raise RuntimeError("Batched generation bypassed the verified row-wise sampler")
    finally:
        torch.multinomial = original_multinomial
        _ROWWISE_GENERATION_LOCK.release()


@dataclass(frozen=True)
class StructuredVisualGrid:
    """Pre-language visual tokens with explicit geometry metadata."""

    hidden: torch.Tensor
    camera_ids: torch.Tensor
    frame_positions: torch.Tensor
    spatial_coordinates: torch.Tensor
    group_ids: torch.Tensor


def positions_between(
    token_ids: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    *,
    label: str,
    required: bool = True,
    use_last_start: bool = False,
) -> torch.Tensor:
    """Return positions strictly inside one delimited token span."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be one-dimensional")
    starts = torch.nonzero(token_ids.eq(int(start_token_id)), as_tuple=False).flatten()
    if starts.numel() == 0:
        if required:
            raise RuntimeError(f"Sequence has no {label} start token")
        return torch.empty(0, dtype=torch.long, device=token_ids.device)
    start = int(starts[-1 if use_last_start else 0].item())
    positions = torch.arange(token_ids.numel(), device=token_ids.device)
    ends = torch.nonzero(
        token_ids.eq(int(end_token_id)) & positions.gt(start), as_tuple=False
    ).flatten()
    if ends.numel() == 0:
        if required:
            raise RuntimeError(f"Sequence has no closed {label} span")
        return torch.empty(0, dtype=torch.long, device=token_ids.device)
    end = int(ends[0].item())
    return torch.arange(start + 1, end, device=token_ids.device)


def pool_visual_hidden(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    image_token_id: int,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    output_hw: int = 2,
) -> torch.Tensor:
    """Pool each Qwen image-token grid independently to ``output_hw`` square."""
    if hidden.ndim != 2 or token_ids.ndim != 1:
        raise ValueError("Expected hidden [L,D] and token_ids [L]")
    if hidden.shape[0] != token_ids.shape[0]:
        raise ValueError("hidden and token_ids sequence lengths differ")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [N_image,3]")
    merge = int(spatial_merge_size)
    if merge <= 0 or int(output_hw) <= 0:
        raise ValueError("spatial_merge_size and output_hw must be positive")

    image_positions = torch.nonzero(
        token_ids.eq(int(image_token_id)), as_tuple=False
    ).flatten()
    grids: list[tuple[int, int, int]] = []
    expected_tokens = 0
    for grid in image_grid_thw.detach().cpu().tolist():
        temporal, height, width = (int(value) for value in grid)
        if height % merge or width % merge:
            raise ValueError(
                "Image grid (%d,%d) is not divisible by merge size %d"
                % (height, width, merge)
            )
        merged_height, merged_width = height // merge, width // merge
        grids.append((temporal, merged_height, merged_width))
        expected_tokens += temporal * merged_height * merged_width
    if int(image_positions.numel()) != expected_tokens:
        raise ValueError(
            "Found %d image tokens but image_grid_thw describes %d"
            % (image_positions.numel(), expected_tokens)
        )

    pooled_images = []
    cursor = 0
    for temporal, height, width in grids:
        count = temporal * height * width
        positions = image_positions[cursor : cursor + count]
        image_hidden = hidden.index_select(0, positions)
        image_hidden = image_hidden.reshape(
            temporal, height, width, hidden.shape[-1]
        ).mean(dim=0)
        image_hidden = image_hidden.permute(2, 0, 1).unsqueeze(0)
        image_hidden = F.adaptive_avg_pool2d(
            image_hidden.float(), (int(output_hw), int(output_hw))
        ).to(dtype=hidden.dtype)
        pooled_images.append(
            image_hidden.squeeze(0).permute(1, 2, 0).reshape(-1, hidden.shape[-1])
        )
        cursor += count
    if not pooled_images:
        raise ValueError("E3 requires at least one image")
    return torch.cat(pooled_images, dim=0)


def unpack_qwen3vl_pre_merger_grids(
    hidden: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> list[torch.Tensor]:
    """Restore Qwen3-VL pre-merger tokens to per-image ``[t,h,w,d]`` grids.

    Qwen3-VL orders the final Vision-block tokens by spatial-merge cell before
    calling ``visual.merger``.  The inverse permutation here is required before
    any geometry-aware pooling; a direct ``[t,h,w,d]`` reshape would silently
    scramble the two spatial axes inside every merge cell.
    """
    if hidden.ndim != 2:
        raise ValueError("pre-merger hidden must have shape [N,D]")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [N_image,3]")
    merge = int(spatial_merge_size)
    if merge <= 0:
        raise ValueError("spatial_merge_size must be positive")

    image_grids = []
    cursor = 0
    hidden_dim = hidden.shape[-1]
    for grid in image_grid_thw.detach().cpu().tolist():
        temporal, height, width = (int(value) for value in grid)
        if temporal <= 0 or height <= 0 or width <= 0:
            raise ValueError("image_grid_thw entries must be positive")
        if height % merge or width % merge:
            raise ValueError(
                "Image grid (%d,%d) is not divisible by merge size %d"
                % (height, width, merge)
            )
        count = temporal * height * width
        image_hidden = hidden[cursor : cursor + count]
        if image_hidden.shape[0] != count:
            raise ValueError("pre-merger hidden is shorter than image_grid_thw")
        merged_height, merged_width = height // merge, width // merge
        image_hidden = image_hidden.reshape(
            temporal,
            merged_height,
            merged_width,
            merge,
            merge,
            hidden_dim,
        )
        image_hidden = image_hidden.permute(0, 1, 3, 2, 4, 5)
        image_grids.append(
            image_hidden.reshape(
                temporal, height, width, hidden_dim
            ).contiguous()
        )
        cursor += count
    if cursor != hidden.shape[0]:
        raise ValueError(
            "image_grid_thw accounts for %d pre-merger tokens, got %d"
            % (cursor, hidden.shape[0])
        )
    return image_grids


def pack_pre_language_visual_grid(
    hidden: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    camera_indices: Sequence[int] | torch.Tensor,
    num_frames_per_camera: int = 4,
    source_hw: int = 4,
    front_camera_id: int = 1,
    spatial_resampling: str = "adaptive_avg",
) -> StructuredVisualGrid:
    """Build the formal 192-token GeoVision cache without temporal pooling.

    ``hidden`` is the final Qwen3-VL Vision-block output immediately before
    ``visual.merger``.  Each of the formal 3 cameras x 4 frames remains a
    separate group and is spatially summarized to a ``source_hw`` square.  At
    ``source_hw=4`` this produces ``12 * 4 * 4 = 192`` tokens.
    Adaptive averaging preserves existing caches; opt-in bilinear resampling
    places features at regular image-cell centers with align_corners=False.
    """
    if spatial_resampling not in ("adaptive_avg", "bilinear"):
        raise ValueError("Unsupported spatial_resampling: %r" % spatial_resampling)
    if hidden.ndim != 2:
        raise ValueError("pre-language hidden must have shape [N,D]")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [N_image,3]")
    merge = int(spatial_merge_size)
    frames = int(num_frames_per_camera)
    source_hw = int(source_hw)
    if merge <= 0 or frames <= 0 or source_hw <= 0:
        raise ValueError("merge size, frame count, and source_hw must be positive")

    camera_ids = torch.as_tensor(
        camera_indices, dtype=torch.long, device=hidden.device
    ).flatten()
    if camera_ids.numel() != 3 or frames != 4:
        raise ValueError("GeoVision requires exactly 3 cameras x 4 frames")
    if tuple(int(value) for value in camera_ids.tolist()) != (0, 1, 2):
        raise ValueError(
            "Formal GeoVision camera order must be left/front/right = 0/1/2"
        )
    if int((camera_ids == int(front_camera_id)).sum().item()) != 1:
        raise ValueError("GeoVision requires exactly one front camera id")
    expected_images = int(camera_ids.numel()) * frames
    if image_grid_thw.shape[0] != expected_images:
        raise ValueError(
            "Expected %d image grids, got %d"
            % (expected_images, image_grid_thw.shape[0])
        )

    pre_merger_grids = unpack_qwen3vl_pre_merger_grids(
        hidden,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=merge,
    )
    image_grids = []
    for image_hidden in pre_merger_grids:
        if image_hidden.shape[0] != 1:
            raise ValueError(
                "Formal GeoVision inputs must be twelve independent images"
            )
        if (
            image_hidden.shape[1] < source_hw
            or image_hidden.shape[2] < source_hw
        ):
            raise ValueError(
                "GeoVision source grid cannot upsample a smaller Vision grid"
            )
        image_hidden = image_hidden[0].permute(2, 0, 1).unsqueeze(0)
        if spatial_resampling == "adaptive_avg":
            image_hidden = F.adaptive_avg_pool2d(
                image_hidden.float(), (source_hw, source_hw)
            ).to(dtype=hidden.dtype)
        else:
            image_hidden = F.interpolate(
                image_hidden.float(),
                size=(source_hw, source_hw),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            ).to(dtype=hidden.dtype)
        image_grids.append(
            image_hidden.squeeze(0).permute(1, 2, 0).contiguous()
        )
    coordinate_axis = (
        torch.arange(source_hw, device=hidden.device, dtype=torch.float32) + 0.5
    ) / float(source_hw)
    coordinate_axis = coordinate_axis * 2.0 - 1.0
    yy, xx = torch.meshgrid(coordinate_axis, coordinate_axis, indexing="ij")
    spatial_coordinates = torch.stack([yy, xx], dim=-1).reshape(-1, 2)

    group_hidden = []
    group_camera_ids = []
    group_frame_positions = []
    group_coordinates = []
    group_ids = []
    group_index = 0
    for camera_index, camera_id in enumerate(camera_ids.tolist()):
        start = camera_index * frames
        for frame_index, visual_grid in enumerate(
            image_grids[start : start + frames]
        ):
            flattened = visual_grid.reshape(-1, hidden.shape[-1])
            count = flattened.shape[0]
            group_hidden.append(flattened)
            group_camera_ids.append(
                torch.full(
                    (count,), int(camera_id), device=hidden.device, dtype=torch.long
                )
            )
            group_frame_positions.append(
                torch.full(
                    (count,),
                    float(frame_index),
                    device=hidden.device,
                    dtype=torch.float32,
                )
            )
            group_coordinates.append(spatial_coordinates)
            group_ids.append(
                torch.full(
                    (count,), group_index, device=hidden.device, dtype=torch.long
                )
            )
            group_index += 1

    packed = StructuredVisualGrid(
        hidden=torch.cat(group_hidden, dim=0),
        camera_ids=torch.cat(group_camera_ids, dim=0),
        frame_positions=torch.cat(group_frame_positions, dim=0),
        spatial_coordinates=torch.cat(group_coordinates, dim=0),
        group_ids=torch.cat(group_ids, dim=0),
    )
    expected_tokens = expected_images * source_hw * source_hw
    if packed.hidden.shape[0] != expected_tokens:
        raise RuntimeError(
            "GeoVision packing produced %d tokens, expected %d"
            % (packed.hidden.shape[0], expected_tokens)
        )
    return packed


def pool_fixed_groups(hidden: torch.Tensor, output_tokens: int) -> torch.Tensor:
    """Mean-pool an ordered token sequence into equal contiguous groups."""
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [N,D]")
    output_tokens = int(output_tokens)
    if output_tokens <= 0 or hidden.shape[0] % output_tokens:
        raise ValueError(
            "Cannot evenly pool %d source tokens into %d outputs"
            % (hidden.shape[0], output_tokens)
        )
    group_size = hidden.shape[0] // output_tokens
    return hidden.reshape(output_tokens, group_size, hidden.shape[-1]).mean(dim=1)


def assemble_generation_hidden(
    hidden_chunks: Sequence[torch.Tensor],
    context_length: int,
) -> torch.Tensor:
    """Assemble final-layer hidden captured during causal generation.

    A decoder-only ``generate`` call first evaluates the complete prompt and
    then evaluates one previously sampled token per decode step.  The final
    sampled stop token is not evaluated, but E3 excludes ``traj_future_start``
    from its pre-action context.  If generation instead reaches its length
    limit, the returned tensor may therefore be exactly one trailing token
    shorter than ``context_length``.  Callers must prove that such a trailing
    token is not selected into an E3 modality before indexing the tensor.
    """
    context_length = int(context_length)
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if not hidden_chunks:
        raise RuntimeError("Generation produced no captured VLM hidden states")

    hidden_dim = None
    device = None
    dtype = None
    flattened = []
    for index, chunk in enumerate(hidden_chunks):
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[1] <= 0:
            raise ValueError(
                "Captured hidden chunk %d must have shape [1,L,D], got %s"
                % (index, tuple(chunk.shape))
            )
        if index > 0 and chunk.shape[1] != 1:
            raise ValueError(
                "Decode hidden chunk %d must contain exactly one token, got %d"
                % (index, chunk.shape[1])
            )
        if hidden_dim is None:
            hidden_dim = int(chunk.shape[-1])
            device = chunk.device
            dtype = chunk.dtype
        elif (
            int(chunk.shape[-1]) != hidden_dim
            or chunk.device != device
            or chunk.dtype != dtype
        ):
            raise ValueError(
                "All captured hidden chunks must share hidden width, device, and dtype"
            )
        flattened.append(chunk[0])

    hidden = torch.cat(flattened, dim=0)
    if hidden.shape[0] < context_length - 1:
        raise RuntimeError(
            "Generation capture is missing more than the single unevaluated "
            "terminal token: captured=%d context=%d"
            % (hidden.shape[0], context_length)
        )
    return hidden[:context_length].contiguous()


def assemble_batched_generation_hidden(
    hidden_chunks: Sequence[torch.Tensor],
    prompt_attention_mask: torch.Tensor,
    context_lengths: Sequence[int],
) -> list[torch.Tensor]:
    """Assemble per-sample hidden from a left-padded batched generation.

    The first hook output contains the complete padded prompt for every sample;
    later outputs contain one decoded token per sample.  Finished samples may
    continue to receive decode steps until the whole batch stops, so every row
    is independently trimmed to its unpadded pre-action context length.
    """
    if prompt_attention_mask.ndim != 2:
        raise ValueError("prompt_attention_mask must have shape [B,L]")
    batch_size, prompt_width = prompt_attention_mask.shape
    if batch_size <= 0 or prompt_width <= 0:
        raise ValueError("batched generation requires a non-empty prompt batch")
    if len(context_lengths) != batch_size:
        raise ValueError("context_lengths must contain one value per batch row")
    if not hidden_chunks:
        raise RuntimeError("Generation produced no captured VLM hidden states")

    prefill = hidden_chunks[0]
    if prefill.ndim != 3 or tuple(prefill.shape[:2]) != (
        batch_size,
        prompt_width,
    ):
        raise ValueError(
            "Batched prefill hidden must have shape [B,L,D], got %s"
            % (tuple(prefill.shape),)
        )
    hidden_dim = int(prefill.shape[-1])
    device = prefill.device
    dtype = prefill.dtype
    for index, chunk in enumerate(hidden_chunks[1:], start=1):
        if chunk.ndim != 3 or tuple(chunk.shape[:2]) != (batch_size, 1):
            raise ValueError(
                "Batched decode hidden chunk %d must have shape [B,1,D], got %s"
                % (index, tuple(chunk.shape))
            )
        if (
            int(chunk.shape[-1]) != hidden_dim
            or chunk.device != device
            or chunk.dtype != dtype
        ):
            raise ValueError(
                "All captured hidden chunks must share width, device, and dtype"
            )

    prompt_mask = prompt_attention_mask.to(device=device, dtype=torch.bool)
    assembled = []
    for row, requested_length in enumerate(context_lengths):
        context_length = int(requested_length)
        if context_length <= 0:
            raise ValueError("context lengths must be positive")
        row_chunks = [prefill[row][prompt_mask[row]]]
        row_chunks.extend(chunk[row, 0].unsqueeze(0) for chunk in hidden_chunks[1:])
        hidden = torch.cat(row_chunks, dim=0)
        if hidden.shape[0] < context_length - 1:
            raise RuntimeError(
                "Batched generation row %d is missing more than the single "
                "unevaluated terminal token: captured=%d context=%d"
                % (row, hidden.shape[0], context_length)
            )
        assembled.append(hidden[:context_length].contiguous())
    return assembled


def pack_modalities(
    modality_hidden: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack four variable-length modalities with local positions and type ids."""
    if len(modality_hidden) != NUM_MODALITIES:
        raise ValueError("Expected vision, history, route, and CoC tensors")
    nonempty = [tensor for tensor in modality_hidden if tensor.shape[0] > 0]
    if not nonempty:
        raise ValueError("At least one modality must contain tokens")
    hidden_dim = nonempty[0].shape[-1]
    device = nonempty[0].device
    dtype = nonempty[0].dtype
    for tensor in modality_hidden:
        if tensor.ndim != 2 or tensor.shape[-1] != hidden_dim:
            raise ValueError("All modalities must have shape [N,D] with the same D")
        if tensor.device != device or tensor.dtype != dtype:
            raise ValueError("All modalities must share device and dtype")

    packed_hidden = torch.cat(list(modality_hidden), dim=0)
    type_ids = torch.cat(
        [
            torch.full(
                (tensor.shape[0],), modality, dtype=torch.long, device=device
            )
            for modality, tensor in enumerate(modality_hidden)
        ],
        dim=0,
    )
    position_ids = torch.cat(
        [
            torch.arange(tensor.shape[0], dtype=torch.long, device=device)
            for tensor in modality_hidden
        ],
        dim=0,
    )
    return packed_hidden, type_ids, position_ids
