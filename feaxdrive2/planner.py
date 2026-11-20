"""Portable dispatch around the source-derived paper model."""
from hashlib import sha256
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from .model.flow import FlowPlannerConfig
from .model.multimodal_flow import FeaXDriveAlpamayoDiffusionPlanner
from .model.camera_grounded_query import CameraGroundedQuery
from .model.camera_contract import CAMERA_KEYS
from .training.constants import PACIFICA
from .training.precision import strict_fp32

CONDITION_KEYS = ("multimodal_type_ids", "multimodal_position_ids", "multimodal_valid_mask", "coc_valid")
LAYOUTS = {"base": None, "rcf_no_spatial": None, "body_scene": "body_side_context",
           "centerline": "centerline", "body_corner": "footprint",
           "no_geometry_encoding": "body_side_no_geometry"}


def token_noise(tokens: Sequence[str], seed=0, *, device="cpu", dtype=torch.float32):
    """Same token/seed-to-noise mapping as NAVTEST; independent of batch order."""
    if not tokens or any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError("Supply at least one nonempty scenario token")
    rows = []
    for token in tokens:
        value = int.from_bytes(sha256(f"{seed}:{token}".encode()).digest()[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device=device).manual_seed(value)
        rows.append(torch.randn(8, 3, generator=generator, device=device, dtype=dtype))
    return torch.stack(rows)


class Planner(nn.Module):
    """Observation features -> five-step Flow trajectory, in metres/radians.

    ``forward`` is differentiable inference. Losses are explicit methods; eval()
    never changes their meaning. Frozen weights still allow input gradients.
    """

    def __init__(self, variant="body_scene"):
        super().__init__()
        if variant not in LAYOUTS:
            raise ValueError(f"Unknown planner variant: {variant}")
        self.variant = variant
        self.action_head = FeaXDriveAlpamayoDiffusionPlanner(
            FlowPlannerConfig(), alpamayo_conditioning_source="multimodal")
        layout = LAYOUTS[variant]
        if layout is not None:
            self.action_head.model.camera_grounded_query = CameraGroundedQuery(
                pacifica_geometry=PACIFICA, anchor_layout=layout)
        if variant == "rcf_no_spatial":
            self.action_head.rcf_no_body = self.action_head.model.rcf_no_body = True

    @property
    def has_spatial_retrieval(self):
        return LAYOUTS[self.variant] is not None

    def inputs(self, features: Mapping[str, torch.Tensor]):
        head = self.action_head
        dtype = next(head.parameters()).dtype
        history = features["history_trajectory"].to(dtype).flatten(1)
        status = features["status_feature"].to(dtype)
        if history.shape != (len(status), 12) or status.shape != (len(status), 8):
            raise ValueError("Expected history [B,4,3] and status [B,8]")
        data = {key: features[key] for key in CONDITION_KEYS}
        if self.has_spatial_retrieval:
            data.update({key: features[key] for key in (*CAMERA_KEYS, "vehicle_geometry")})
        data.update(his_traj=history, status_feature=status,
                    state=torch.cat((status, history), dim=1))
        return BatchFeature(data=data)

    def forward(self, features, noise, *, return_trace=False):
        if noise.shape != (len(features["status_feature"]), 8, 3) or not torch.isfinite(noise).all():
            raise ValueError("Initial noise must be finite [B,8,3]")
        with strict_fp32():
            return self.action_head.get_action(features["last_hidden_state"], self.inputs(features),
                init_actions=noise, return_trace=return_trace)

    def flow_matching_loss(self, features, trajectory, *, noise=None, time=None):
        if self.variant != "base":
            raise ValueError("Base planner learning requires variant='base'; use RC-F for spatial training")
        data = self.inputs(features)
        data["action"] = trajectory
        if noise is not None:
            data["flow_noise"] = noise
        if time is not None:
            data["flow_time"] = time
        return self.action_head(features["last_hidden_state"], data)

    def rcf_loss(self, features, trajectory, roles, reference, *, weights=None):
        """Roles: 0 high-risk, 1 HardSafe, 2 Normal. See docs/data.md for targets."""
        from .training import rcf
        if reference.variant != "base" or self.variant == "base":
            raise ValueError("RC-F uses an independent frozen Base Planner and an enhanced planner")
        if weights is None:
            weights = torch.full_like(roles, 1.0 / len(roles), dtype=torch.float32)
        reference_features = dict(features)
        # The Base reference has no geometry/camera input path.
        for key in (*CAMERA_KEYS, "vehicle_geometry"):
            reference_features.pop(key, None)
        with strict_fp32():
            fields = rcf.reference_fields(reference.action_head, reference_features,
                {"trajectory": trajectory}, roles, weights)
            data = self.inputs(features)
            data.update(action=trajectory, flow_noise=features["flow_noise"], flow_time=features["flow_time"])
            data.update(fields)
            return rcf.forward_loss(self.action_head, features["last_hidden_state"], data)


def build_planner(variant="body_scene", *, device="cpu"):
    """Construct the paper's 16-block model. No checkpoint is downloaded."""
    return Planner(variant).to(device=device, dtype=torch.float32)


def initialize_from_base(planner, base):
    """Copy every shared tensor; retain only newly initialized spatial tensors."""
    if base.variant != "base" or planner.variant == "base":
        raise ValueError("Expected enhanced planner and Base Planner")
    target = planner.action_head.state_dict()
    source = base.action_head.state_dict()
    missing = set(target) - set(source)
    if set(source) - set(target) or any(not k.startswith("model.camera_grounded_query.") for k in missing):
        raise ValueError("Base/enhanced architectures differ outside spatial retrieval")
    for key, value in source.items():
        if value.shape != target[key].shape:
            raise ValueError(f"Base tensor shape mismatch: {key}")
        target[key] = value
    planner.action_head.load_state_dict(target, strict=True)
