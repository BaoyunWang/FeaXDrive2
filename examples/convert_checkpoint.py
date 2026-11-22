"""Export only the declared paper tensors from a trusted research checkpoint."""
import argparse
from pathlib import Path
import torch
from feaxdrive2 import build_planner
from feaxdrive2.checkpoints import load_file, validate_state, save_planner, ALIGNMENT_FORMAT
from feaxdrive2.training.constants import lora_configuration
from feaxdrive2.training.alignment_parameters import restore_dit


def convert_planner(saved, variant, prefix):
    planner = build_planner(variant)
    state = {k.removeprefix(prefix): v for k, v in saved["state_dict"].items() if k.startswith(prefix)}
    # Neither spatial layout nor query-update policy is encoded in tensor shapes.
    meta = saved.get("e2") or {}
    layout = getattr(planner.action_head.model, "camera_grounded_query", None)
    if layout is not None and meta.get("camera_anchor_layout", "footprint") != layout.anchor_layout:
        raise ValueError("Research checkpoint metadata does not match the requested spatial layout")
    if meta.get("spatial_query_update", "dynamic") != planner.spatial_query_update:
        raise ValueError("Research checkpoint query-update policy does not match the requested variant")
    validate_state(planner.action_head, state)
    planner.action_head.load_state_dict(state, strict=True)
    return planner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", required=True, choices=("planner", "alignment"))
    parser.add_argument("--variant", choices=("base", "rcf_no_spatial", "body_scene", "centerline", "body_corner", "no_geometry_encoding", "fixed_query"), default="body_scene")
    parser.add_argument("--prefix", default="agent.action_head.", help="Exact prefix in the original state_dict")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    saved = load_file(args.input)
    if args.kind == "alignment":
        if saved.get("configuration", {}).get("lora") != lora_configuration():
            raise ValueError("Research checkpoint LoRA configuration differs from the paper")
        planner = build_planner("body_scene")
        restore_dit(planner.action_head, saved["dit"])
        lora = saved["lora"]
        if len(lora) != 32 or any(not torch.isfinite(v).all() for v in lora.values()):
            raise ValueError("Expected 32 finite language LoRA tensors")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(format=ALIGNMENT_FORMAT, lora_config=lora_configuration(),
                        lora=lora, dit=saved["dit"]), args.output)
    else:
        planner = convert_planner(saved, args.variant, args.prefix)
        save_planner(planner, args.output)
    print(f"Exported {args.kind} tensors to {args.output}; datasets/optimizer/internal paths omitted")


if __name__ == "__main__":
    main()
