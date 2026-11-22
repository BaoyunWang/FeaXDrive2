"""Single-candidate inference from precomputed, checkpoint-matched features."""
import argparse
from pathlib import Path
import numpy as np
import torch
from feaxdrive2 import build_planner, token_noise
from feaxdrive2.checkpoints import load_planner, load_alignment, apply_alignment, checkpoint_identity
from feaxdrive2.data import read_manifest, load_record, validate_conditioning_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=("base", "rcf_no_spatial", "body_scene", "centerline", "body_corner", "no_geometry_encoding", "fixed_query"), default="body_scene")
    parser.add_argument("--alignment", help="Joint-alignment parameters; features must be generated with this same LoRA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = read_manifest(args.manifest)
    validate_conditioning_source(rows, checkpoint_identity(args.alignment) if args.alignment else None)
    planner = load_planner(build_planner(args.variant, device=args.device), args.checkpoint).eval()
    if args.alignment:
        apply_alignment(planner, load_alignment(args.alignment))
    predictions = []
    with torch.no_grad():
        for row in rows:
            features, _ = load_record(row, args.device)
            noise = token_noise([row["token"]], args.seed, device=args.device)
            predictions.append(planner(features, noise)["pred_traj"][0].cpu().numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, tokens=np.asarray([r["token"] for r in rows]),
                            trajectories=np.stack(predictions), interval_seconds=np.asarray(.5))
    print(f"Saved {len(rows)} single-candidate trajectories to {output}")


if __name__ == "__main__":
    main()
