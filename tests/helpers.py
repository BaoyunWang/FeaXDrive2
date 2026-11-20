"""Small synthetic fixtures for interface/gradient checks, not benchmark data."""
import torch
from feaxdrive2.model.camera_grounded_query import CameraGroundedQuery
from feaxdrive2.training.constants import PACIFICA


def make_branch(batch=1):
    torch.manual_seed(7)
    branch = CameraGroundedQuery(width=16, visual_dim=6, pacifica_geometry=PACIFICA,
                                 anchor_layout="body_side_context").eval().requires_grad_(False)
    trajectory = torch.zeros(batch, 8, 3)
    trajectory[..., 0] = torch.arange(8)
    trajectory[..., 1] = torch.linspace(0., 1., 8)
    inputs = dict(geometry=torch.tensor(PACIFICA).expand(batch, 3).clone(),
        scene=torch.randn(batch, 32, 16), trajectory=trajectory, flow_time=torch.full((batch,), .4),
        camera_features=torch.randn(batch, 3, 6, 16, 24),
        camera_projection=torch.tensor([[2., 0., 0., 59.5], [0., -3., 0., 39.5],
                                         [0., 0., 0., 1.]]).expand(batch, 3, 3, 4).clone(),
        camera_image_size=torch.tensor([80, 120]).expand(batch, 3, 2).clone(),
        camera_valid=torch.ones(batch, 3, dtype=torch.bool),
        camera_intrinsics=torch.eye(3).expand(batch, 3, 3, 3).clone(),
        camera_distortion=torch.zeros(batch, 3, 5),
        camera_distortion_r2_max=torch.full((batch, 3), 1e6))
    field = torch.randn(batch, 3, 6, 1, 24).expand(-1, -1, -1, 16, -1)
    inputs["camera_features"] = (field + torch.arange(16)[None, None, None, :, None] * .001).clone()
    return branch, inputs


def features(batch=1):
    _, source = make_branch(batch)
    result = {k: v for k, v in source.items() if k.startswith("camera_")}
    result.update(vehicle_geometry=source["geometry"],
        camera_features=torch.randn(batch, 3, 1152, 16, 16),
        history_trajectory=torch.zeros(batch, 4, 3), status_feature=torch.zeros(batch, 8),
        last_hidden_state=torch.randn(batch, 8, 4096),
        multimodal_type_ids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).expand(batch, -1).clone(),
        multimodal_position_ids=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]).expand(batch, -1).clone(),
        multimodal_valid_mask=torch.ones(batch, 8, dtype=torch.bool), coc_valid=torch.ones(batch, dtype=torch.bool))
    return result
