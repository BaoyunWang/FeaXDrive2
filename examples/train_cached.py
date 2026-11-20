"""Base/RC-F training from portable features, with scene-level accumulation.

This transparent single-device loop prioritizes faithful losses. Distributed
launch, on-disk cache generation, and validation scoring are external.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from feaxdrive2 import build_planner
from feaxdrive2.planner import initialize_from_base
from feaxdrive2.checkpoints import load_planner, save_planner
from feaxdrive2.data import read_manifest, load_record
from feaxdrive2.training.api import configure_rcf
from feaxdrive2.training.sampling import FourRoleSampler, ROLES as POOL_NAMES

ROLES = ("Collision", "NearCollision", "HardSafe", "Normal")
PROBABILITIES = (.25, .15, .25, .35)
ROLE_IDS = {"Collision": 0, "NearCollision": 0, "HardSafe": 1, "Normal": 2}


def schedule_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step + 1 - warmup) / max(1, total - warmup)
    return .1 + .9 * (1 + math.cos(math.pi * progress)) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", required=True, choices=("base", "rcf"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--variant", default="body_scene", choices=("body_scene", "rcf_no_spatial", "centerline", "body_corner", "no_geometry_encoding"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, help="Effective batch through accumulation; default 128 (base), 32 (RC-F)")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.epochs < 1 or args.updates < 1 or args.save_every < 1 or (args.batch_size is not None and args.batch_size < 1):
        raise ValueError("Epochs, updates, batch size, and save interval must be positive")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rows = read_manifest(args.manifest)
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Every training row must explicitly have split='train'")
    base_mode = args.objective == "base"
    size = args.batch_size or (128 if base_mode else 32)
    if base_mode:
        planner = build_planner("base", device=args.device).train()
        groups = [{"params": list(planner.parameters()), "lr": 1e-4}]
        total = args.epochs * math.ceil(len(rows) / size)
        warmup = max(1, total // 20)
        order = []
    else:
        if not args.base_checkpoint:
            raise ValueError("RC-F requires --base-checkpoint")
        if size != 32:
            raise ValueError("The reported RC-F quota schedule requires effective batch 32")
        reference = load_planner(build_planner("base", device=args.device), args.base_checkpoint).eval().requires_grad_(False)
        planner = build_planner(args.variant, device=args.device)
        initialize_from_base(planner, reference)
        groups = configure_rcf(planner)
        total, warmup = args.updates, min(200, args.updates)
        pools = {role: [i for i, row in enumerate(rows) if row.get("role") == role] for role in ROLES}
        if any(not pool for pool in pools.values()) or sum(map(len, pools.values())) != len(rows):
            raise ValueError("RC-F manifest needs all four roles, assigned by offline safety analysis")
        sampler = FourRoleSampler({key: pools[role] for key, role in zip(POOL_NAMES, ROLES)},
                                  seed=args.seed, steps=total)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=1e-4)
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    output.mkdir(parents=True)
    for step in range(total):
        if base_mode:
            if not order:
                order = rng.permutation(len(rows)).tolist()
            indices, order = order[:size], order[size:]
            selected = [rows[i] for i in indices]
        else:
            selected = [rows[index] for index in sampler.next_batch()]
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for row in selected:
            features, trajectory = load_record(row, args.device)
            if trajectory is None:
                raise ValueError("Training record is missing its trajectory target")
            noise = torch.randn(1, 8, 3, device=args.device)
            if base_mode:
                with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                    enabled=torch.device(args.device).type == "cuda"):
                    result = planner.flow_matching_loss(features, trajectory, noise=noise)
            else:
                features.update(flow_noise=noise, flow_time=torch.randint(5, (1,), device=args.device).float() / 5)
                role = torch.tensor([ROLE_IDS[row["role"]]], device=args.device)
                result = planner.rcf_loss(features, trajectory, role, reference)
            loss = result["loss"] / len(selected)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            loss_sum += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(planner.parameters(), 1., error_if_nonfinite=True)
        factor = schedule_factor(step, total, warmup)
        for group, lr in zip(optimizer.param_groups, learning_rates):
            group["lr"] = lr * factor
        optimizer.step()
        print(f"update={step+1}/{total} loss={loss_sum:.6f}", flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == total:
            save_planner(planner, output / f"planner-{step+1:06d}.pt")


if __name__ == "__main__":
    main()
