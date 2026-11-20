# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Camera retrieval at metric footprint anchors, implemented in pure PyTorch.

Geometry is [width, rear-axle-to-front, rear-axle-to-rear] in metres. Camera
order is supplied by the caller and must be fixed across examples/checkpoints.
The four corner identities are always FL, FR, RL, RR (x forward, y left).
z=0 is an explicit footprint plane in the current rear-axle frame; it does
not assert the location of actual ground. Projection validity means positive
optical depth and in-image coordinates, not visibility or absence of occlusion.

The projected arm samples a fixed 3x3 stencil with offsets {-1,0,1} FEATURE
CELLS, using bilinear grid_sample(align_corners=False, padding_mode='border').
The global comparator samples the nine centres of an evenly spaced 3x3 image
partition for every corner/camera. It ignores projected centres AND projection
validity, using camera_valid only. Both arms execute the same projections,
query encoders, sample count, attention and pooling with identical parameters.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.nn import functional as F


__all__ = ["CameraGroundedQuery", "footprint_points", "body_margin_points", "body_side_points", "misaligned_body_points",
           "project_points", "INJECTION_LAYERS", "FixedInputChecks", "fixed_input_checks"]
INJECTION_LAYERS = (9, 11, 13, 15)
BODY_MARGIN_METRES = 0.5
BODY_SIDE_METRES = 1.5

_DEFERRED_CHECKS = ContextVar("camera_query_numeric_checks", default=None)
_FIXED_CHECKS = ContextVar("camera_query_fixed_input_checks", default=None)
_FIXED_INPUT_NAMES = frozenset((
    "geometry", "projection_geometry", "scene", "camera_features", "camera_projection",
    "camera_image_size", "camera_valid", "camera_intrinsics", "camera_distortion",
    "camera_distortion_r2_max",
))


def _fixed_signature(value):
    # Compare StorageImpl identity, never storage contents. The manager retains
    # the storage too, so allocator reuse cannot match an old binding.
    return (value.untyped_storage()._cdata, value.storage_offset(), tuple(value.shape),
            value.stride(), value.dtype, value.device, value._version)


def _fixed_layout(signature):
    storage, offset, shape, strides, dtype, device, version = signature
    # Single-scene expansion only repeats existing values. Ignore singleton
    # strides too, since view()/detach() can change them without moving data.
    strides = tuple(0 if size == 1 else stride for size, stride in zip(shape, strides))
    if shape and shape[0] > 0 and strides[0] == 0:
        shape = (1, *shape[1:])
    return storage, offset, shape, strides, dtype, device, version


class FixedInputChecks:
    """Reusable numeric predicates for explicitly immutable scene inputs.

    Pass camera-query argument names, e.g. geometry=inputs["vehicle_geometry"]
    and optionally scene=condition[0] (only when it does not require gradients).
    Partial/empty mappings are allowed; dynamic names cannot be registered.
    Keep one manager per scene and use ``with fixed_input_checks(manager):``
    around collection, reference/replay and both optimization passes. Checkpoint
    closures must re-enter that context when recomputed during backward.

    Exact layouts, batch expansions and detached aliases share predicates.
    Other layouts/storage, including copies, take the ordinary validation path.
    Shape/dtype/device guards always run. Mutating a registered tensor invalidates
    the manager, including between contexts; create a new manager for new inputs.
    As with autograd, callers must not write through .data, NumPy, raw storage or
    other mechanisms that bypass PyTorch version counters. Versionless inference
    tensors are rejected. The manager retains registered inputs and detached
    predicates, never computed body/query outputs; discard it with the scene record.
    """

    def __init__(self, tensors):
        self._inputs = {}
        self._predicates = {}
        for name, value in tensors.items():
            if name not in _FIXED_INPUT_NAMES:
                raise ValueError(f"{name} is not a fixed camera-query input")
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                raise ValueError(f"Fixed input {name} must be a strided tensor")
            if name == "scene" and value.requires_grad:
                raise ValueError("Fixed input scene must not require gradients")
            try:
                signature = _fixed_signature(value)
            except RuntimeError as exc:
                raise ValueError(f"Fixed input {name} must track a mutation version") from exc
            self._inputs[name] = (value, value.untyped_storage(), signature)

    def _unchanged(self, name):
        value, _, signature = self._inputs[name]
        if (_fixed_signature(value) != signature
                or (name == "scene" and value.requires_grad)):
            raise ValueError(f"Fixed input {name} was mutated; create a new FixedInputChecks")

    def _validate(self):
        for name in self._inputs:
            self._unchanged(name)

    def _predicate(self, key, tensors, condition):
        for name, value in tensors:
            if name not in self._inputs:
                return condition()
            self._unchanged(name)
            try:
                matches = (_fixed_layout(_fixed_signature(value))
                           == _fixed_layout(self._inputs[name][2]))
            except RuntimeError:
                matches = False
            if not matches or (name == "scene" and value.requires_grad):
                return condition()
        cache_key = (key, tuple(name for name, _ in tensors))
        if cache_key not in self._predicates:
            self._predicates[cache_key] = condition().detach()
        return self._predicates[cache_key]


@contextmanager
def fixed_input_checks(manager):
    """Activate a scene's cache; None keeps full checks. Nesting restores scope.

    Composes with deferred_numeric_checks in either order. Cached predicates
    still pass through that context on EVERY use, so an unchecked or invalid
    deferred predicate can never silently become a successful validation.
    """
    if manager is not None:
        if not isinstance(manager, FixedInputChecks):
            raise TypeError("manager must be FixedInputChecks or None")
        manager._validate()
    token = _FIXED_CHECKS.set(manager)
    try:
        yield
        if manager is not None:
            manager._validate()
    finally:
        _FIXED_CHECKS.reset(token)


def _input_numeric_check(key, tensors, condition, message):
    manager = _FIXED_CHECKS.get()
    predicate = condition() if manager is None else manager._predicate(key, tensors, condition)
    _numeric_check(predicate, message)


@contextmanager
def deferred_numeric_checks(checks):
    """Collect detached validity predicates for a caller's pre-update check.

    Ordinary inference remains fail-fast. The GRPO caller checks every collected
    predicate before scoring/stepping, including checkpoint recomputation.
    Shape, dtype and device errors always fail immediately.
    """
    token = _DEFERRED_CHECKS.set(checks)
    try:
        yield
    finally:
        _DEFERRED_CHECKS.reset(token)


def _numeric_check(condition, message):
    checks = _DEFERRED_CHECKS.get()
    if checks is None:
        if not bool(condition):
            raise ValueError(message)
    else:
        checks.append(condition.detach())


def _tensor(name, value, shape=None, *, floating=True):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}")
    if floating and not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    _input_numeric_check("finite", ((name, value),), lambda: torch.isfinite(value).all(),
                         f"{name} must contain only finite values")


def _geometry(name, geometry, batch):
    _tensor(name, geometry, (batch, 3))
    _input_numeric_check("positive_geometry", ((name, geometry),), lambda: (geometry > 0).all(),
                         f"{name} must contain positive width/front/rear distances")


def _same_device(reference, **tensors):
    for name, value in tensors.items():
        if value.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}, got {value.device}")


def footprint_points(trajectory, geometry):
    """Return [B,T,4,3] metric points in FL/FR/RL/RR order, with explicit z=0.

    trajectory [B,T,3] is metric current-rear-axle-frame x/y/yaw (radians).
    geometry [B,3] is [width, rear-axle-to-front, rear-axle-to-rear].
    At each pose, rotate (front,+/-width/2), (-rear,+/-width/2) by yaw,
    then translate by (x,y). T can be any positive horizon count for audits.
    This helper preserves autograd; the module detaches its conditioning inputs.
    Half-precision geometry arithmetic is promoted to float32, with autocast
    disabled for all spatial math (float64 auditing inputs retain precision).
    """
    _tensor("trajectory", trajectory)
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3 or min(trajectory.shape[:2]) < 1:
        raise ValueError("trajectory must be [B,T,3] with nonempty batch and horizons")
    _geometry("geometry", geometry, trajectory.shape[0])
    _same_device(trajectory, geometry=geometry)
    with torch.autocast(device_type=trajectory.device.type, enabled=False):
        dtype = torch.promote_types(torch.float32, torch.promote_types(trajectory.dtype, geometry.dtype))
        trajectory, geometry = trajectory.to(dtype), geometry.to(dtype)
        width, front, rear = geometry.unbind(-1)
        dx = torch.stack((front, front, -rear, -rear), -1)[:, None]
        dy = torch.stack((width / 2, -width / 2, width / 2, -width / 2), -1)[:, None]
        x, y, yaw = trajectory.unbind(-1)
        c, s = yaw.cos()[..., None], yaw.sin()[..., None]
        px = x[..., None] + c * dx - s * dy
        py = y[..., None] + s * dx + c * dy
        points = torch.stack((px, py, torch.zeros_like(px)), -1)
    _tensor("footprint points", points)
    return points


def body_margin_points(trajectory, body_points):
    """Add 0.5m along the yaw-rotated outward lateral normal of each corner.

    body_points is the real FL/FR/RL/RR footprint, never a widened scorer body.
    Keep the body's front/rear coordinates and z; only its visual anchors move.
    """
    _tensor("trajectory", trajectory)
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3 or min(trajectory.shape[:2]) < 1:
        raise ValueError("trajectory must be nonempty [B,T,3]")
    _tensor("body_points", body_points, (*trajectory.shape[:2], 4, 3))
    _same_device(trajectory, body_points=body_points)
    with torch.autocast(device_type=trajectory.device.type, enabled=False):
        dtype = torch.promote_types(torch.float32, torch.promote_types(trajectory.dtype, body_points.dtype))
        yaw = trajectory[..., 2].to(dtype)[..., None]
        lateral = body_points.new_tensor((1., -1., 1., -1.), dtype=dtype) * BODY_MARGIN_METRES
        offset = torch.stack((-yaw.sin() * lateral, yaw.cos() * lateral,
                              torch.zeros_like(yaw * lateral)), -1)
        result = body_points.to(dtype) + offset
    _tensor("body margin points", result)
    return result


def body_side_points(trajectory, geometry):
    """Return [B,T,4,3] visual probes at body-length fractions 3/4 and 1/4.

    In FL/FR/RL/RR slot order, local x is -rear + (3/4 or 1/4)*(front+rear),
    local y is +/-(width/2+1.5m). Rotate by trajectory yaw, translate by xy,
    and set z=0. These are side-context observation points, not body corners,
    measured obstacle clearances, or a replacement scorer footprint.
    Like footprint_points, retain autograd and at least FP32 spatial precision.
    """
    _tensor("trajectory", trajectory)
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3 or min(trajectory.shape[:2]) < 1:
        raise ValueError("trajectory must be [B,T,3] with nonempty batch and horizons")
    _geometry("geometry", geometry, trajectory.shape[0])
    _same_device(trajectory, geometry=geometry)
    with torch.autocast(device_type=trajectory.device.type, enabled=False):
        dtype = torch.promote_types(torch.float32, torch.promote_types(trajectory.dtype, geometry.dtype))
        trajectory, geometry = trajectory.to(dtype), geometry.to(dtype)
        width, front, rear = geometry.unbind(-1)
        length = front + rear
        front_x, rear_x = -rear + .75 * length, -rear + .25 * length
        dx = torch.stack((front_x, front_x, rear_x, rear_x), -1)[:, None]
        lateral = width / 2 + BODY_SIDE_METRES
        dy = torch.stack((lateral, -lateral, lateral, -lateral), -1)[:, None]
        x, y, yaw = trajectory.unbind(-1)
        c, s = yaw.cos()[..., None], yaw.sin()[..., None]
        px = x[..., None] + c * dx - s * dy
        py = y[..., None] + s * dx + c * dy
        points = torch.stack((px, py, torch.zeros_like(px)), -1)
    _tensor("body side points", points)
    return points


def misaligned_body_points(trajectory, geometry):
    """Rotate the actual FL/FR/RL/RR offsets +45 degrees about the body centre.

    Return [B,T,4,3], z=0: xy + R(yaw)[c + R(pi/4)(offset-c)], where
    c=((front-rear)/2, 0). The angle is fixed, not a model setting or buffer.
    Like footprint_points, preserve autograd and use at least FP32 spatial math.
    """
    _tensor("trajectory", trajectory)
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3 or min(trajectory.shape[:2]) < 1:
        raise ValueError("trajectory must be [B,T,3] with nonempty batch and horizons")
    _geometry("geometry", geometry, trajectory.shape[0])
    _same_device(trajectory, geometry=geometry)
    with torch.autocast(device_type=trajectory.device.type, enabled=False):
        dtype = torch.promote_types(torch.float32, torch.promote_types(trajectory.dtype, geometry.dtype))
        trajectory, geometry = trajectory.to(dtype), geometry.to(dtype)
        width, front, rear = geometry.unbind(-1)
        dx = torch.stack((front, front, -rear, -rear), -1)[:, None]
        dy = torch.stack((width / 2, -width / 2, width / 2, -width / 2), -1)[:, None]
        centre = ((front - rear) / 2)[:, None, None]
        rx = centre + ((dx - centre) - dy) * (2. ** -.5)
        ry = ((dx - centre) + dy) * (2. ** -.5)
        x, y, yaw = trajectory.unbind(-1)
        c, s = yaw.cos()[..., None], yaw.sin()[..., None]
        px = x[..., None] + c * rx - s * ry
        py = y[..., None] + s * rx + c * ry
        points = torch.stack((px, py, torch.zeros_like(px)), -1)
    _tensor("misaligned body points", points)
    return points


def project_points(points, camera_projection, camera_image_size, camera_valid,
                   camera_intrinsics=None, camera_distortion=None, camera_distortion_r2_max=None):
    """Project [B,...,3] points; return grid [B,C,...,2], valid [B,C,...].

    camera_projection [B,C,3,4] maps homogeneous current-rear-axle-frame
    points to ideal processed image pixels; its third row gives optical depth.
    With K/D/r2_max supplied, undo K to obtain optical rays, apply Brown-5,
    then K again. r2_max bounds the central monotonic radial branch; rays beyond
    it are invalid (never polynomial-folded into the image). All three fields
    are mandatory together. Omitting them is a pinhole-only mathematical helper,
    not the formal raw-image cache path.
    camera_image_size [B,C,2] contains positive integer-valued H/W (integer
    or floating tensors). camera_valid [B,C] must be bool. C and the point
    dimensions may be any positive sizes for auditing.

    Pixel centres use integer coordinates: grid_x = 2*(u+0.5)/W - 1,
    grid_y = 2*(v+0.5)/H - 1, suitable for align_corners=False. Validity is
    camera_valid & depth>0 & 0<=u<W & 0<=v<H. Invalid grids are finite zero
    sentinels; callers MUST use the returned mask. No occlusion is inferred.
    Nonfinite inputs, including masked cameras, are rejected. Autograd is kept.
    All projection/grid arithmetic runs outside autocast, at least in float32.
    """
    _tensor("points", points)
    if points.ndim < 3 or points.shape[-1] != 3 or min(points.shape[:-1]) < 1:
        raise ValueError("points must be nonempty [B,...,3]")
    _tensor("camera_projection", camera_projection)
    if (camera_projection.ndim != 4 or camera_projection.shape[0] != points.shape[0]
            or camera_projection.shape[1] < 1 or camera_projection.shape[2:] != (3, 4)):
        raise ValueError("camera_projection must be [B,C,3,4]")
    batch, cameras = camera_projection.shape[:2]
    _tensor("camera_image_size", camera_image_size, (batch, cameras, 2), floating=False)
    if camera_image_size.dtype == torch.bool or camera_image_size.is_complex():
        raise ValueError("camera_image_size must contain positive integer-valued H/W")
    def image_valid():
        valid = (camera_image_size > 0).all()
        if camera_image_size.is_floating_point():
            valid = valid & (camera_image_size == camera_image_size.round()).all()
        return valid
    _input_numeric_check("image_size", (("camera_image_size", camera_image_size),), image_valid,
                         "camera_image_size must contain positive integer-valued H/W")
    _tensor("camera_valid", camera_valid, (batch, cameras), floating=False)
    if camera_valid.dtype != torch.bool:
        raise ValueError("camera_valid must be bool")
    _same_device(points, camera_projection=camera_projection,
                 camera_image_size=camera_image_size, camera_valid=camera_valid)
    calibrated = (camera_intrinsics is not None, camera_distortion is not None,
                  camera_distortion_r2_max is not None)
    if any(calibrated) and not all(calibrated):
        raise ValueError("K, distortion and monotonic radius must be supplied together")
    if all(calibrated):
        _tensor("camera_intrinsics", camera_intrinsics, (batch, cameras, 3, 3))
        _tensor("camera_distortion", camera_distortion, (batch, cameras, 5))
        _tensor("camera_distortion_r2_max", camera_distortion_r2_max, (batch, cameras))
        _input_numeric_check("calibration", (("camera_intrinsics", camera_intrinsics),
                ("camera_distortion_r2_max", camera_distortion_r2_max)),
                lambda: (camera_distortion_r2_max > 0).all()
                & (camera_intrinsics[..., 0, 0] > 0).all()
                & (camera_intrinsics[..., 1, 1] > 0).all()
                & (camera_intrinsics[..., 2, :] == camera_intrinsics.new_tensor([0, 0, 1])).all()
                & (camera_intrinsics[..., 1, 0] == 0).all(),
                "Invalid camera intrinsic or distortion domain")
        _same_device(points, camera_intrinsics=camera_intrinsics,
                     camera_distortion=camera_distortion,
                     camera_distortion_r2_max=camera_distortion_r2_max)
    with torch.autocast(device_type=points.device.type, enabled=False):
        dtype = torch.promote_types(torch.float32, torch.promote_types(points.dtype, camera_projection.dtype))
        flat = points.to(dtype).reshape(batch, -1, 3)
        homogeneous = torch.cat((flat, torch.ones_like(flat[..., :1])), -1)
        pixels = torch.einsum("bcij,bnj->bcni", camera_projection.to(dtype), homogeneous)
        _tensor("homogeneous projected pixels", pixels)
        depth = pixels[..., 2]
        positive_depth = depth > 0
        denominator = torch.where(positive_depth, depth, torch.ones_like(depth))
        if all(calibrated):
            intrinsic = camera_intrinsics.to(dtype)
            optical = torch.linalg.solve(intrinsic, pixels.transpose(-1, -2)).transpose(-1, -2)
            xy = optical[..., :2] / denominator[..., None]
            radius2 = xy.square().sum(-1)
            domain = torch.isfinite(radius2) & (radius2 < camera_distortion_r2_max.to(dtype)[..., None])
            positive_depth = positive_depth & domain
            xy = torch.where(positive_depth[..., None], xy, torch.zeros_like(xy))
            x, y = xy.unbind(-1)
            r2 = x.square() + y.square()
            k1, k2, p1, p2, k3 = camera_distortion.to(dtype)[..., None, :].unbind(-1)
            radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            xd = x * radial + 2*p1*x*y + p2*(r2 + 2*x.square())
            yd = y * radial + p1*(r2 + 2*y.square()) + 2*p2*x*y
            distorted = torch.stack((xd, yd, torch.ones_like(xd)), -1)
            uv = torch.einsum("bcij,bcnj->bcni", intrinsic, distorted)[..., :2]
        else:
            uv = pixels[..., :2] / denominator[..., None]
        _tensor("projected pixels", uv)
        hw = camera_image_size.to(dtype)[:, :, None, :]
        wh = hw.flip(-1)
        valid = (camera_valid[:, :, None] & positive_depth
                 & (uv >= 0).all(-1) & (uv < wh).all(-1))
        # Mask before normalization too, so extreme finite out-of-image values
        # cannot overflow into grid_sample. The final sentinel is exactly zero.
        uv = torch.where(valid[..., None], uv, torch.zeros_like(uv))
        grid = 2 * ((uv + .5) / wh) - 1
        grid = torch.where(valid[..., None], grid, torch.zeros_like(grid))
    shape = (batch, cameras, *points.shape[1:-1])
    return grid.reshape(*shape, 2), valid.reshape(shape)


class _BodyActionMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.action_norm = nn.LayerNorm(width)
        self.body_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.SiLU(),
                                 nn.Linear(2 * width, width))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, actions, body):
        fused = torch.cat((self.action_norm(actions.to(self.action_norm.weight.dtype)),
                           self.body_norm(body.to(self.body_norm.weight.dtype))), -1)
        residual = self.mlp(fused).to(actions.dtype)
        return actions + self.gate.tanh().to(actions.dtype) * residual


class CameraGroundedQuery(nn.Module):
    """Compute body [B,8,width] once per DiT call, then reuse with inject().

    Required pacifica_geometry comes from the caller's canonical registry.
    Full constant mode replaces g before BOTH query encoding and anchors.
    projection_geometry [B,3] is an optional diagnostic override of anchors
    only in geometry mode. Constant mode rejects it to keep the full control
    independent of requested/override geometry. It never changes query geometry.
    anchor_layout='centerline' repeats each rear-axle pose four times for BOTH
    query xy and projection anchors. Raw g, yaw, horizon and slot embeddings
    remain unchanged; projection_geometry is forbidden for this layout.
    'geometry_free_centerline' also replaces the three raw g channels with
    exact zeros independent of geometry values/dtype. 'misaligned_body' uses
    actual corners rotated +45 degrees about their geometric centre before
    trajectory yaw. Both retain all slots/parameters and forbid anchor overrides.
    'body_margin' keeps real body queries and adds a second visual stencil at
    each corner's outward lateral offset of 0.5m. It requires projected retrieval.
    'body_side_context' keeps M2's real body queries and single visual stencil,
    but projects side probes at longitudinal fractions 3/4 and 1/4, outward
    by 1.5m. Q stays nine-dimensional. No source embeddings, second stencil,
    new Query encoding, or fallback to body retrieval is used.
    'body_side_no_geometry' uses the same side probes from real geometry, but
    Q is [0,0,0,rear_axle_x,rear_axle_y,sin(yaw),cos(yaw),flow_time,horizon].
    All four slot embeddings and shared parameters remain unchanged.
    All dual-source arms add source_embeddings [2,width], initialized to zero AFTER
    all common modules (identical common parameters and RNG consumption).
    'body_margin_query' additionally encodes metric (margin-body) xy through a
    zero-initialized [width,2] weight added before the first query SiLU. This is
    an 11-input affine map split into the original nine inputs and two extra
    columns, preserving A3's common initialization, retrieval and RNG exactly.

    last_projection_valid is refreshed on each successful forward: detached
    bool [B,3,8,4], reflecting raw positive-depth/in-image projection validity
    even for global retrieval. It is a diagnostic attribute, not a model buffer
    or checkpoint field. A failed forward clears it instead of exposing stale data.
    last_projected_grid [B,3,8,4,2] and last_query_points [B,8,4,3] follow the
    same lifecycle and are detached diagnostics, never inputs to model or loss.

    Forward conditioning is detached by default. The explicit, nonpersistent
    retain_scene_grad flag lets Planning-Guided LoRA differentiate the semantic
    scene only. retain_trajectory_grad separately enables recurrent query
    derivatives for Flow-GRPO likelihood replay; camera features and geometry
    remain detached. Both flags default to False and add no checkpoint state.
    inject() keeps action/body gradients so earlier injections
    can learn through a frozen action backbone. No body is cached across calls.
    Input devices must match the module; floating conditioning is cast to the
    module dtype for learned projections, with at least float32 spatial math.
    Width must be a positive multiple of eight (eight attention heads).
    """

    injection_layers = INJECTION_LAYERS

    def __init__(self, width=384, visual_dim=1152, mode="geometry", retrieval="projected",
                 *, pacifica_geometry, anchor_layout="footprint", action_horizon=8, similarity_k=0):
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0 or width % 8:
            raise ValueError("width must be a positive multiple of eight")
        if isinstance(visual_dim, bool) or not isinstance(visual_dim, int) or visual_dim <= 0:
            raise ValueError("visual_dim must be a positive integer")
        if type(action_horizon) is not int or action_horizon <= 0:
            raise ValueError("action_horizon must be a positive integer")
        if mode not in ("geometry", "constant"):
            raise ValueError("mode must be 'geometry' or 'constant'")
        if retrieval not in ("projected", "global"):
            raise ValueError("retrieval must be 'projected' or 'global'")
        if anchor_layout not in ("footprint", "centerline", "geometry_free_centerline", "misaligned_body", "body_margin", "body_margin_query", "body_side_context", "body_side_no_geometry"):
            raise ValueError("Unknown camera anchor_layout")
        if anchor_layout in ("body_margin", "body_margin_query", "body_side_context", "body_side_no_geometry") and (retrieval != "projected" or mode != "geometry"):
            raise ValueError(f"{anchor_layout} requires projected retrieval and geometry mode")
        if type(similarity_k) is not int or similarity_k not in (0, 4):
            raise ValueError("similarity_k must be the fixed experiment value 0 or 4")
        if similarity_k and (anchor_layout != "body_side_context" or retrieval != "projected" or mode != "geometry"):
            raise ValueError("Similarity expansion requires projected body_side_context geometry")
        pacifica = torch.as_tensor(pacifica_geometry, dtype=torch.float32).detach().clone()
        _geometry("pacifica_geometry", pacifica.unsqueeze(0), 1)
        self.width, self.visual_dim = width, visual_dim
        self.action_horizon = action_horizon
        self.query_count = 4 * action_horizon
        self.mode, self.retrieval = mode, retrieval
        self.anchor_layout = anchor_layout
        self.similarity_k = similarity_k
        self.last_similarity_selection = None
        self.last_projection_valid = None
        self.last_projected_grid = None
        self.last_query_points = None
        self.last_margin_points = self.last_margin_grid = self.last_margin_valid = None
        self.last_margin_query_delta = None
        self.last_feature_hw = self.last_source_attention_mass = None
        self.collect_source_attention = False
        self.retain_scene_grad = False
        # Opt-in likelihood replay differentiates a policy's recurrent query
        # estimate. Standard RC-F and deployed inference keep their behavior.
        self.retain_trajectory_grad = False
        self.register_buffer("pacifica", pacifica)
        self.register_buffer("stencil", torch.tensor([(x, y) for y in (-1., 0., 1.)
                                                       for x in (-1., 0., 1.)]))
        self.register_buffer("global_grid", self.stencil.clone() * (2. / 3.))
        self.register_buffer("horizon", torch.arange(1, action_horizon + 1, dtype=torch.float32) / action_horizon)
        # Raw metric g(3), posed corner xy(2), sin/cos yaw(2), flow time(1),
        # fixed horizon fraction(1). No normalization by the requested vehicle.
        self.query_encoder = nn.Sequential(nn.Linear(9, width), nn.SiLU(), nn.Linear(width, width))
        self.corner_embeddings = nn.Parameter(torch.empty(4, width))
        self.camera_embeddings = nn.Parameter(torch.empty(3, width))
        self.sample_embeddings = nn.Parameter(torch.empty(9, width))
        self.null_visual = nn.Parameter(torch.empty(1, width))
        for parameter in (self.corner_embeddings, self.camera_embeddings,
                          self.sample_embeddings, self.null_visual):
            nn.init.normal_(parameter, std=.02)
        self.visual_projection = nn.Linear(visual_dim, width)
        self.visual_query_norm = nn.LayerNorm(width)
        self.visual_attention = nn.MultiheadAttention(width, 8, dropout=0., batch_first=True)
        self.scene_query_norm = nn.LayerNorm(width)
        self.scene_attention = nn.MultiheadAttention(width, 8, dropout=0., batch_first=True)
        self.corner_pool = nn.Linear(4 * width, width)
        self.body_action = nn.ModuleDict({str(layer): _BodyActionMLP(width)
                                          for layer in INJECTION_LAYERS})
        if anchor_layout in ("body_margin", "body_margin_query"):
            self.source_embeddings = nn.Parameter(torch.zeros(2, width))
        if anchor_layout == "body_margin_query":
            # Parameter rather than nn.Linear: no random initialization is
            # consumed, so all later seeded training draws also remain paired.
            self.margin_query_weight = nn.Parameter(torch.zeros(width, 2))
        if similarity_k:
            # Added only after all shared modules, without consuming RNG. Thus
            # paired arms start with identical shared weights and training draws.
            # These encode actual selected image coordinates, NOT local stencil slots.
            self.similarity_position_weight = nn.Parameter(torch.zeros(width, 4))
            self.similarity_source_embedding = nn.Parameter(torch.zeros(width))

    def _sample_visual_candidates(self, camera_features, grid, valid):
        batch, cameras, channels, height, width = camera_features.shape
        # Keep both grid construction and interpolation outside caller autocast;
        # .float() alone does not stop eligible operations from being recast.
        with torch.autocast(device_type=camera_features.device.type, enabled=False):
            if self.retrieval == "projected":
                cell = grid.new_tensor((2. / width, 2. / height))
                sample_grid = grid[..., None, :] + self.stencil.to(grid.dtype) * cell
            else:
                sample_grid = self.global_grid.to(grid.dtype).expand(batch, cameras, self.action_horizon, 4, 9, 2)
            dtype = torch.promote_types(torch.float32, camera_features.dtype)
            sampled = F.grid_sample(
                camera_features.to(dtype).reshape(batch * cameras, channels, height, width),
                sample_grid.to(dtype).reshape(batch * cameras, self.query_count, 9, 2),
                mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.reshape(batch, cameras, channels, self.action_horizon, 4, 9).permute(0, 3, 4, 1, 5, 2)
        tokens = self.visual_projection(sampled.to(self.visual_projection.weight.dtype))
        tokens = (tokens + self.camera_embeddings[:, None, :]
                  + self.sample_embeddings[None, :, :]).reshape(batch * self.query_count, 27, self.width)
        sample_valid = valid.permute(0, 2, 3, 1)[..., None].expand(batch, self.action_horizon, 4, 3, 9)
        sample_valid = sample_valid.reshape(batch * self.query_count, 27)
        return tokens, sample_valid

    def _sample_visual(self, camera_features, grid, valid, margin_grid=None, margin_valid=None):
        tokens, sample_valid = self._sample_visual_candidates(camera_features, grid, valid)
        if self.similarity_k:
            from .similarity_visual_retrieval import select_similar_features
            selected = select_similar_features(camera_features, grid, valid, k=self.similarity_k)
            extra = self.visual_projection(selected.features.to(self.visual_projection.weight.dtype))
            extra = (extra + self.camera_embeddings[:, None, :]
                     + F.linear(selected.coordinates.to(extra.dtype), self.similarity_position_weight)
                     + self.similarity_source_embedding)
            # The original local 27 tokens remain first, unchanged. Selected
            # cells are appended to the SAME visual attention; no new attention path.
            extra = extra.reshape(camera_features.shape[0] * self.query_count, 3 * self.similarity_k, self.width)
            tokens = torch.cat((tokens, extra), dim=1)
            sample_valid = torch.cat((sample_valid, selected.valid.reshape(sample_valid.shape[0], -1)), dim=1)
            self.last_similarity_selection = dict(indices=selected.indices.detach(),
                coordinates=selected.coordinates.detach(), valid=selected.valid.detach(),
                similarities=selected.similarities.detach())
        if self.anchor_layout in ("body_margin", "body_margin_query"):
            if margin_grid is None or margin_valid is None:
                raise ValueError("Dual-source retrieval requires independent secondary grid and validity")
            margin_tokens, margin_mask = self._sample_visual_candidates(camera_features, margin_grid, margin_valid)
            tokens = torch.cat((tokens + self.source_embeddings[0],
                                margin_tokens + self.source_embeddings[1]), 1)
            sample_valid = torch.cat((sample_valid, margin_mask), 1)
        elif margin_grid is not None or margin_valid is not None:
            raise ValueError("Only dual-source layouts accept secondary visual candidates")
        batch = camera_features.shape[0]
        # Null is available exactly when all local/extra candidates are invalid. No
        # attention row ever has every key masked, including all-invalid batches.
        null_valid = ~sample_valid.any(-1, keepdim=True)
        tokens = torch.cat((tokens, self.null_visual.expand(batch * self.query_count, 1, self.width)), 1)
        padding_mask = ~torch.cat((sample_valid, null_valid), -1)
        return tokens, padding_mask

    @torch.no_grad()
    def _source_attention_mass(self, query, tokens, padding_mask):
        """Detached diagnostic only; leave production need_weights=False unchanged.

        Recompute head-averaged attention probabilities without retaining graphs
        or changing the attention backend used for the model output. The result
        is [B*32,3] (body, margin, null), not evidence of causal contribution.
        """
        attention = self.visual_attention
        with torch.autocast(device_type=query.device.type, enabled=False):
            weight, bias = attention.in_proj_weight.detach().float(), attention.in_proj_bias.detach().float()
            q = F.linear(query.detach().float(), weight[:self.width], bias[:self.width])
            k = F.linear(tokens.detach().float(), weight[self.width:2*self.width], bias[self.width:2*self.width])
            q = q.reshape(-1, 1, 8, self.width // 8).transpose(1, 2)
            k = k.reshape(-1, tokens.shape[1], 8, self.width // 8).transpose(1, 2)
            logits = (q @ k.transpose(-2, -1)) * ((self.width // 8) ** -.5)
            probabilities = logits.masked_fill(padding_mask[:, None, None], float("-inf")).softmax(-1)
            probabilities = probabilities.mean(1).squeeze(1)
            return torch.stack((probabilities[:, :27].sum(-1), probabilities[:, 27:54].sum(-1),
                                probabilities[:, 54]), -1)

    def forward(self, geometry, scene, trajectory, flow_time, camera_features,
                camera_projection, camera_image_size, camera_valid, projection_geometry=None,
                *, camera_intrinsics, camera_distortion, camera_distortion_r2_max):
        self.last_projection_valid = None
        self.last_similarity_selection = None
        self.last_projected_grid = None
        self.last_query_points = None
        self.last_margin_points = self.last_margin_grid = self.last_margin_valid = None
        self.last_margin_query_delta = None
        self.last_feature_hw = self.last_source_attention_mass = None
        if self.anchor_layout != "footprint" and projection_geometry is not None:
            raise ValueError(f"{self.anchor_layout} rejects projection_geometry; layout fixes its visual anchors")
        if self.mode == "constant" and projection_geometry is not None:
            raise ValueError("projection_geometry is an anchor-only diagnostic for geometry mode; "
                             "constant mode requires full Pacifica replacement")
        _tensor("geometry", geometry)
        if geometry.ndim != 2 or geometry.shape[0] < 1:
            raise ValueError("geometry must be nonempty [B,3]")
        batch = geometry.shape[0]
        _geometry("geometry", geometry, batch)
        _tensor("scene", scene, (batch, 32, self.width))
        _tensor("trajectory", trajectory, (batch, self.action_horizon, 3))
        _tensor("flow_time", flow_time, (batch,))
        _tensor("camera_features", camera_features)
        if (camera_features.ndim != 5 or camera_features.shape[:3] != (batch, 3, self.visual_dim)
                or min(camera_features.shape[-2:]) < 1):
            raise ValueError("camera_features must be [B,3,visual_dim,H,W] with positive H/W")
        _tensor("camera_projection", camera_projection, (batch, 3, 3, 4))
        _tensor("camera_image_size", camera_image_size, (batch, 3, 2), floating=False)
        _tensor("camera_valid", camera_valid, (batch, 3), floating=False)
        inputs = dict(geometry=geometry, scene=scene, trajectory=trajectory, flow_time=flow_time,
                      camera_features=camera_features, camera_projection=camera_projection,
                      camera_image_size=camera_image_size, camera_valid=camera_valid,
                      camera_intrinsics=camera_intrinsics, camera_distortion=camera_distortion,
                      camera_distortion_r2_max=camera_distortion_r2_max)
        if projection_geometry is not None:
            _geometry("projection_geometry", projection_geometry, batch)
            inputs["projection_geometry"] = projection_geometry
        _same_device(self.pacifica, **inputs)
        inputs = {name: tensor if ((name == "scene" and self.retain_scene_grad)
                                  or (name == "trajectory" and self.retain_trajectory_grad)) else tensor.detach()
                  for name, tensor in inputs.items()}
        g = inputs["geometry"]
        if self.mode == "constant":
            g = self.pacifica.expand(batch, 3)
        poses = inputs["trajectory"]
        if self.anchor_layout in ("centerline", "geometry_free_centerline", "body_side_no_geometry"):
            # Four ordered slots per pose remain; only their physical offsets
            # collapse. Do not derive even the spatial dtype from raw g.
            with torch.autocast(device_type=poses.device.type, enabled=False):
                xy = poses[..., :2].to(torch.promote_types(torch.float32, poses.dtype))
                query_points = torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)
                query_points = query_points[:, :, None, :].expand(batch, self.action_horizon, 4, 3)
            if self.anchor_layout == "geometry_free_centerline":
                # Shape validation above is the only dependency on supplied g.
                # In particular, its dtype cannot affect query rounding.
                g = query_points.new_zeros((batch, 3))
        elif self.anchor_layout == "misaligned_body":
            query_points = misaligned_body_points(poses, g)
        else:
            query_points = footprint_points(poses, g)
        anchors = (query_points if projection_geometry is None
                   else footprint_points(poses, inputs["projection_geometry"]))
        if self.anchor_layout in ("body_side_context", "body_side_no_geometry"):
            anchors = body_side_points(poses, g)
        grid, projection_valid = project_points(anchors, inputs["camera_projection"],
                                                inputs["camera_image_size"], inputs["camera_valid"],
                                                inputs["camera_intrinsics"], inputs["camera_distortion"],
                                                inputs["camera_distortion_r2_max"])
        retrieval_valid = projection_valid
        margin_points = margin_grid = margin_valid = None
        if self.anchor_layout in ("body_margin", "body_margin_query"):
            margin_points = body_margin_points(poses, query_points)
            margin_grid, margin_valid = project_points(margin_points, inputs["camera_projection"],
                inputs["camera_image_size"], inputs["camera_valid"], inputs["camera_intrinsics"],
                inputs["camera_distortion"], inputs["camera_distortion_r2_max"])
        if self.retrieval == "global":
            retrieval_valid = inputs["camera_valid"][:, :, None, None].expand(batch, 3, self.action_horizon, 4)
        dtype = self.query_encoder[0].weight.dtype
        yaw = poses[..., 2].to(query_points.dtype)
        # G0 masks geometry only in Q; real g still constructs the A4 probes.
        query_geometry = query_points.new_zeros((batch, 3)) if self.anchor_layout == "body_side_no_geometry" else g
        query_input = torch.cat((
            query_geometry[:, None, None, :].expand(batch, self.action_horizon, 4, 3), query_points[..., :2],
            torch.stack((yaw.sin(), yaw.cos()), -1)[:, :, None, :].expand(batch, self.action_horizon, 4, 2),
            inputs["flow_time"][:, None, None, None].expand(batch, self.action_horizon, 4, 1),
            self.horizon[None, :, None, None].expand(batch, self.action_horizon, 4, 1)), -1).to(dtype)
        margin_delta = None
        if self.anchor_layout == "body_margin_query":
            # Use the SAME physical anchors as K/V, never an enlarged scorer
            # body or a GT prefix. Invalid projection masks visual evidence,
            # not the known geometric location of the requested outer point.
            with torch.autocast(device_type=query_points.device.type, enabled=False):
                margin_delta = (margin_points[..., :2] - query_points[..., :2]).to(dtype)
            hidden = self.query_encoder[0](query_input) + F.linear(margin_delta, self.margin_query_weight)
            encoded_queries = self.query_encoder[2](self.query_encoder[1](hidden))
        else:
            encoded_queries = self.query_encoder(query_input)
        queries = encoded_queries + self.corner_embeddings
        if self.anchor_layout in ("body_margin", "body_margin_query"):
            tokens, padding_mask = self._sample_visual(inputs["camera_features"], grid, retrieval_valid,
                                                       margin_grid, margin_valid)
        else:
            tokens, padding_mask = self._sample_visual(inputs["camera_features"], grid, retrieval_valid)
        # PyTorch in navsim-v2 canonicalizes the bool padding mask to the input
        # query dtype before autocasting Q/K/V projections. Keep that dtype equal
        # to the encoder's actual AMP dtype even after FP32 embeddings/LayerNorm.
        visual_queries = self.visual_query_norm(queries.reshape(batch * self.query_count, 1, self.width)).to(encoded_queries.dtype)
        visual_tokens = tokens.to(encoded_queries.dtype)
        visual, _ = self.visual_attention(visual_queries, visual_tokens, visual_tokens,
                                           key_padding_mask=padding_mask, need_weights=False)
        corners = queries.reshape(batch, self.query_count, self.width) + visual.reshape(batch, self.query_count, self.width)
        semantic_scene = inputs["scene"].to(dtype)
        semantic, _ = self.scene_attention(self.scene_query_norm(corners), semantic_scene,
                                            semantic_scene, need_weights=False)
        corners = (corners + semantic).reshape(batch, self.action_horizon, 4, self.width)
        body = self.corner_pool(corners.flatten(-2))
        self.last_projection_valid = projection_valid.detach()
        self.last_projected_grid = grid.detach()
        self.last_query_points = query_points.detach()
        self.last_feature_hw = tuple(camera_features.shape[-2:])
        if self.anchor_layout in ("body_margin", "body_margin_query"):
            self.last_margin_points = margin_points.detach()
            self.last_margin_grid = margin_grid.detach()
            self.last_margin_valid = margin_valid.detach()
            if margin_delta is not None:
                self.last_margin_query_delta = margin_delta.detach()
            if self.collect_source_attention:
                self.last_source_attention_mass = self._source_attention_mass(visual_queries, visual_tokens, padding_mask)
        return body

    def inject(self, layer, actions, body):
        """Return actions + tanh(scalar_gate) * MLP(norm_action, norm_body).

        actions/body are [B,8,width]. Only layers 9/11/13/15 inject; other
        integer layers return the original actions tensor. All four scalar
        gates start at zero; internal projections use ordinary initialization.
        Learned operations use module precision; the residual is cast back to
        actions.dtype, preserving baseline precision even at zero gates.
        """
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError("layer must be an integer")
        _tensor("actions", actions)
        if actions.ndim != 3 or actions.shape[0] < 1 or actions.shape[1:] != (self.action_horizon, self.width):
            raise ValueError("actions must be nonempty [B,action_horizon,width]")
        _tensor("body", body, actions.shape)
        _same_device(self.pacifica, actions=actions, body=body)
        key = str(layer)
        return self.body_action[key](actions, body) if key in self.body_action else actions
