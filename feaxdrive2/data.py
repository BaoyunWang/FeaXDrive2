"""Portable tensor records. Dataset licensing and sensor preprocessing are external."""
import json
from pathlib import Path
import torch


def validate_conditioning_source(rows, alignment_identity):
    if any(row.get("alignment_sha256") != alignment_identity for row in rows):
        raise ValueError("Feature manifest and alignment checkpoint differ; regenerate features with the selected VLM")


def read_manifest(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({r["token"] for r in rows}) != len(rows):
        raise ValueError("Manifest must contain unique, nonempty scenario records")
    for row in rows:
        source = Path(row["path"])
        row["path"] = str(source if source.is_absolute() else path.parent / source)
    return rows


def load_record(row, device):
    record = torch.load(row["path"], map_location="cpu", weights_only=True)
    if record.get("token") != row["token"]:
        raise ValueError("Manifest token differs from tensor record")
    features = record["features"]
    if not isinstance(features, dict) or any(not isinstance(v, torch.Tensor) for v in features.values()):
        raise ValueError("Features must be a tensor mapping, without pickled dataset objects")
    if features["status_feature"].shape != (1, 8):
        raise ValueError("Each portable record contains one scene with a batch dimension")
    features = {k: v.to(device=device, dtype=torch.float32 if v.is_floating_point() else v.dtype)
                for k, v in features.items()}
    trajectory = record.get("trajectory")
    if trajectory is not None:
        trajectory = trajectory.to(device=device, dtype=torch.float32)
        if trajectory.shape != (1, 8, 3) or not torch.isfinite(trajectory).all():
            raise ValueError("Training target must be finite [1,8,3]")
    return features, trajectory
