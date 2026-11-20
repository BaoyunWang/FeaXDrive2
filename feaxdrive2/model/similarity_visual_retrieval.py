# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Four extra same-camera cells selected from raw local visual similarity.

This utility leaves the original 3x3 token path untouched. It resamples its
nine raw bilinear descriptors solely to rank extra full-grid cells, before
any learned visual projection, camera/sample embeddings or positional encoder.
There are no parameters, random draws, inference-only guards or model hooks.
The caller owns best097 initialization, paired RCF training and integration.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


EXTRA_CELLS = 4


@dataclass(frozen=True)
class SimilaritySelection:
    """All axes start [B, horizon, 4 probes, cameras, 4 extra cells].

    features: raw input-dtype values, with trailing channel dimension C.
    coordinates: trailing (u,v,u-probe_u,v-probe_v), normalized as grid_sample
        coordinates ([-1,1] image edges, align_corners=False cell centres).
    valid: bool; invalid padding has zero features/coordinates/similarities.
    indices: int64 row-major cell index (y*W+x), or -1 for invalid padding.
    similarities: detached max cosine over nonzero original local descriptors.

    Hard selection is not differentiated. Gathered features retain gradients
    to their source cells; coordinate deltas retain gradients to probe grids.
    A zero similarity can be valid: use valid, never a score threshold.
    """

    features: torch.Tensor
    coordinates: torch.Tensor
    valid: torch.Tensor
    indices: torch.Tensor
    similarities: torch.Tensor


def _validate(camera_features, grid, valid, k):
    if type(k) is not int or k != EXTRA_CELLS:
        raise ValueError("Similarity expansion fixes k=4")
    if (not isinstance(camera_features, torch.Tensor) or camera_features.ndim != 5
            or min(camera_features.shape) < 1 or not camera_features.is_floating_point()):
        raise ValueError("camera_features must be floating [B,Cam,C,H,W] with nonempty axes")
    b, cameras = camera_features.shape[:2]
    if (not isinstance(grid, torch.Tensor) or grid.ndim != 5 or grid.shape[:2] != (b, cameras)
            or grid.shape[2] < 1 or grid.shape[3:] != (4, 2) or not grid.is_floating_point()):
        raise ValueError("grid must be floating [B,Cam,horizon,4,2]")
    if not isinstance(valid, torch.Tensor) or valid.shape != grid.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [B,Cam,horizon,4]")
    if grid.device != camera_features.device or valid.device != camera_features.device:
        raise ValueError("Features, grid and valid must share a device")
    if not torch.isfinite(camera_features).all() or not torch.isfinite(grid).all():
        raise ValueError("Features and grid must be finite, including invalid projections")


def _unit_descriptors(values):
    """True cosine normalization, excluding exactly zero vectors.

    Scaling first prevents overflow/underflow of squared raw magnitudes and
    does not turn small nonzero descriptors into zero or epsilon-scaled ones.
    """
    scale = values.abs().amax(-1, keepdim=True)
    nonzero = scale.squeeze(-1) > 0
    scaled = values / torch.where(scale > 0, scale, torch.ones_like(scale))
    return scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True).clamp_min(1), nonzero


def _local_support(sample_grid, height, width):
    """Union of positive-weight bilinear support, separately per probe/camera.

    Input [B*Cam, horizon*4, 9, 2]; border clipping precedes interpolation,
    just as in grid_sample. Zero-weight neighbours must remain candidates.
    Integer scatter-add handles repeated/clipped indices without overwriting
    a positive contribution with a zero contribution from another neighbour.
    """
    cells = (sample_grid + 1) / 2 * sample_grid.new_tensor((width, height)) - .5
    x, y = cells.unbind(-1)
    x, y = x.clamp(0, width-1), y.clamp(0, height-1)
    x0, y0 = x.floor().long(), y.floor().long()
    dx, dy = x-x0, y-y0
    support = torch.zeros((*sample_grid.shape[:2], height*width), dtype=torch.int32, device=sample_grid.device)
    for xx, wx in ((x0, 1-dx), ((x0+1).clamp_max(width-1), dx)):
        for yy, wy in ((y0, 1-dy), ((y0+1).clamp_max(height-1), dy)):
            support.scatter_add_(2, yy*width+xx, ((wx*wy) > 0).to(torch.int32))
    return support > 0


def select_similar_features(camera_features, grid, valid, k=4):
    """Append-only selection; returns SimilaritySelection, never modifies inputs.

    camera_features [B,Cam,C,H,W], grid [B,Cam,horizon,4,2] and bool valid
    [B,Cam,horizon,4]. k is fixed to 4. Each full-grid cell is ranked by its
    maximum cosine with the nine ORIGINAL bilinear local descriptors from the
    same camera/probe. The stencil offsets are {-1,0,1}*(2/W,2/H), border
    padding, align_corners=False, matching CameraGroundedQuery.

    Exclude the union of ALL nine descriptors' positive-weight source cells,
    even for zero-valued local descriptors. Ignore zero-norm local descriptors
    and zero-norm candidate cells. Invalid projections or no nonzero local
    descriptors produce four invalid slots; insufficient candidates are padded.
    Negative similarities remain eligible. Stable descending argsort breaks
    exact ties by ascending row-major cell index, supported by PyTorch 2.0.

    Sampling/matching uses promote(float32, camera_features.dtype), exactly
    as the original sampler; grid dtype affects stencil construction only.
    Coordinates retain higher grid precision. Autocast is disabled for both.
    Selection/similarity diagnostics are detached; raw feature gather is not.
    """
    _validate(camera_features, grid, valid, k)
    batch, cameras, channels, height, width = camera_features.shape
    horizon = grid.shape[2]
    banks, probes, cell_count = batch*cameras, horizon*4, height*width
    dtype = torch.promote_types(torch.float32, torch.promote_types(camera_features.dtype, grid.dtype))
    sample_dtype = torch.promote_types(torch.float32, camera_features.dtype)
    with torch.no_grad(), torch.autocast(device_type=camera_features.device.type, enabled=False):
        # Construct the stencil in the original grid dtype, as the local path
        # does, before promoting coordinates for the interpolation operation.
        stencil = grid.new_tensor([(x, y) for y in (-1., 0., 1.) for x in (-1., 0., 1.)])
        sample_grid = (grid[..., None, :] + stencil * grid.new_tensor((2./width, 2./height)))
        sample_grid = sample_grid.to(sample_dtype).reshape(banks, probes, 9, 2)
        if not torch.isfinite(sample_grid).all():
            raise ValueError("Nonfinite local sampling grid")
        source = camera_features.to(sample_dtype).reshape(banks, channels, height, width)
        local = F.grid_sample(source, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)
        local = local.permute(0, 2, 3, 1)  # [B*Cam,probes,9,C]
        cells = source.flatten(2).transpose(1, 2)
        if not torch.isfinite(local).all():
            raise ValueError("Nonfinite interpolated local descriptors")
        local_unit, local_nonzero = _unit_descriptors(local)
        cell_unit, cell_nonzero = _unit_descriptors(cells)
        cosine = torch.bmm(local_unit.reshape(banks, probes*9, channels), cell_unit.transpose(1, 2))
        cosine = cosine.reshape(banks, probes, 9, cell_count)
        similarity = cosine.masked_fill(~local_nonzero[..., None], -torch.inf).amax(2)
        allowed = (valid.reshape(banks, probes, 1) & cell_nonzero[:, None, :]
                   & local_nonzero.any(-1, keepdim=True) & ~_local_support(sample_grid, height, width))
        similarity = similarity.masked_fill(~allowed, -torch.inf)
        count = min(k, cell_count)
        indices = torch.argsort(similarity, dim=-1, descending=True, stable=True)[..., :count]
        selected_valid = allowed.gather(-1, indices)
        similarities = similarity.gather(-1, indices)
        # Keep the fixed shape even for feature maps with fewer than four cells.
        indices = F.pad(indices, (0, k-count), value=0)
        selected_valid = F.pad(selected_valid, (0, k-count), value=False)
        similarities = F.pad(similarities, (0, k-count), value=0.)
        indices = indices.masked_fill(~selected_valid, -1)
        similarities = similarities.masked_fill(~selected_valid, 0.)

    # Gather outside no_grad from the ORIGINAL feature tensor/dtype. Padded
    # gathers use index0 for addressing only, then mask their values/gradients.
    cells = camera_features.reshape(banks, channels, cell_count).transpose(1, 2)
    features = cells.gather(1, indices.clamp_min(0).reshape(banks, probes*k, 1).expand(-1, -1, channels))
    features = features.reshape(banks, probes, k, channels).masked_fill(~selected_valid[..., None], 0.)
    with torch.autocast(device_type=grid.device.type, enabled=False):
        safe = indices.clamp_min(0)
        uv = torch.stack(((safe % width).to(dtype), (safe // width).to(dtype)), -1)
        uv = (uv + .5) * uv.new_tensor((2./width, 2./height)) - 1
        delta = uv - grid.to(dtype).reshape(banks, probes, 1, 2)
        coordinates = torch.cat((uv, delta), -1).masked_fill(~selected_valid[..., None], 0.)

    def layout(value):
        shaped = value.reshape(batch, cameras, horizon, 4, k, *value.shape[3:])
        return shaped.permute(0, 2, 3, 1, 4, *range(5, shaped.ndim))

    return SimilaritySelection(features=layout(features), coordinates=layout(coordinates),
        valid=layout(selected_valid), indices=layout(indices), similarities=layout(similarities))
