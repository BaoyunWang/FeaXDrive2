# Training and Adapted Feature Extraction

Run the commands below from the repository root.

The training order is base planner learning, offline sample construction,
RC-F, and joint VLM–Action Alignment. Dataset splitting, geometry-based scene
selection and official scoring remain dataset-side operations; no NAVTEST scene
may be used for training or checkpoint selection.

## Base Planner and RC-F

The portable single-device loop supports actual scene records and gradient
accumulation, including the four-role RC-F sampler:

```bash
python examples/train_cached.py --objective base --manifest /path/to/train.jsonl \
  --epochs 100 --batch-size 128 --output outputs/base
python examples/train_cached.py --objective rcf --manifest /path/to/rcf_train.jsonl \
  --base-checkpoint checkpoints/base.pt --updates 4000 --batch-size 32 \
  --output outputs/rcf
```

The example loop is an integration entry point, not the distributed cluster
runner used for the paper. It accumulates individual scenes rather than relying
on a fixed padded token length. Base learning uses BF16 on CUDA; RC-F uses FP32.
Checkpoints are saved without overwriting existing outputs. Official validation
and checkpoint selection are external to this loop.

RC-F retains the original four training roles and sampling probabilities:

| Role | Pool size | Sampling probability | Objective |
| --- | ---: | ---: | --- |
| Collision | 369 | 25% | High-risk trajectory supervision |
| NearCollision | 1,753 | 15% | High-risk trajectory supervision |
| HardSafe | 8,673 | 25% | Reference behavior preservation |
| Normal | 71,883 | 35% | Reference behavior preservation |

HardSafe and Normal share the preservation objective; their admission criteria
and sampling frequency differ. High-risk targets are safety-validated
demonstrations. Preservation interpolation targets are saved Base Planner
trajectories, while the free reference rollout uses the current shared noise.
See [data contract](data.md) for this distinction.

Role pools are shuffled and cycled independently. Each five-update cycle uses
quotas `(8,5,8,11)` for four updates and `(8,4,8,12)` for one, with global batch
32. This is coverage-deck sampling, not independent draws with replacement.

RC-F freezes the condition adapter and most of the planner. It trains the
spatial branch and self-attention in zero-based blocks 12 and 14, plus FFN/AdaLN
in blocks 12–15. Cross-attention stays frozen. Peak learning rates are `1e-4`
and `1e-5`, respectively; 4,000 updates use 200 warm-up updates and cosine decay
to one tenth. The optimizer is AdamW with `(0.9,0.95)`, weight decay `1e-4`, and
gradient clipping at 1.0.

### Fixed-Query Retrieval training ablation

Use `fixed_query` to train from the **same Base Planner** with body-corner
anchors, side-probe sampling, and the complete RC-F objective. The first Flow
evaluation uses zero metric query conditions. Its clean-trajectory estimate is
clipped, denormalized, and detached; the resulting retrieval locations are held
fixed for the remaining four evaluations. Flow states, time conditioning,
attention, and spatial context continue to update. Local auxiliary supervision
uses the corresponding query condition from this same rollout.

```bash
python examples/train_cached.py --objective rcf --variant fixed_query \
  --manifest /path/to/rcf_train.jsonl --base-checkpoint checkpoints/base.pt \
  --updates 4000 --batch-size 32 --save-every 1000 --seed 0 \
  --output outputs/fixed_query
```

This uses the same trainable parameters, four-role sampling, losses, optimizer,
and learning-rate schedule described above. The formal fixed-query run used
eight GPUs with four scenes per GPU (global batch 32), and selected the
1,000-update checkpoint from 1,000/2,000/3,000/4,000 using 4,231 validation
scenes. The reported dynamic comparator is the existing Body–Scene Grounded
Planner, selected at 1,000 updates from its historical 2,000-update run; both
used the 4,000-update learning-rate schedule. Use official validation to select the checkpoint; the portable loop does not
automate this step.

| NAVTEST configuration | At-fault collisions | Geometric contacts | PDMS |
| --- | ---: | ---: | ---: |
| Fixed-Query Retrieval | 131 | 143 | 88.426 |
| Body–Scene Grounded Planner | 123 | 136 | 88.550 |

These are the completed research runs on 12,146 scenes, not new results from
this source release. Fixed-query training is distinct from disabling query
updates only at evaluation on a dynamically trained checkpoint.

Convert the formal research checkpoint, or use the portable loop's selected
checkpoint directly, and keep `--variant fixed_query` for inference:

```bash
python examples/convert_checkpoint.py --kind planner --variant fixed_query \
  --prefix agent.action_head. --input /path/to/fixed_query.ckpt \
  --output checkpoints/fixed_query.pt
python examples/infer.py --variant fixed_query --checkpoint checkpoints/fixed_query.pt \
  --manifest /path/to/navtest.jsonl --seed 0 --output outputs/fixed_query_predictions.npz
```

Conversion requires `e2.spatial_query_update = "fixed_first_estimate"` and
`e2.camera_anchor_layout = "body_side_context"`. The public checkpoint retains
the policy through its `fixed_query` variant; a fixed-query checkpoint cannot
be loaded as `body_scene` despite having identical tensor shapes. No alignment
weights are applied for this ablation.

## Joint VLM–Action Alignment

```python
import torch
from feaxdrive2.training.api import build_alignment, alignment_optimizer
from feaxdrive2.checkpoints import save_alignment

model = build_alignment(planner, model_path=model_path, vlm_path=vlm_path)
optimizer = alignment_optimizer(model)
# features includes the native language boundary, original reference features,
# camera tensors and shared flow_noise/flow_time. See docs/data.md.
optimizer.zero_grad(set_to_none=True)
result = model(features, {"trajectory": targets}, roles, weights)
result["loss"].backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
optimizer.step()
save_alignment(model, "outputs/alignment.pt")
```

Alignment trains q/v LoRA in language layers 28–35 (rank 16, alpha 32, dropout
0) and the same 44 selected Action Flow-DiT tensors. Spatial retrieval, vision,
condition adaptation, and other weights remain frozen. The learning rates are
`3e-6` and `2e-6`; the effective batch is 16, with 5% warm-up and cosine decay to
one tenth over 2,200 updates. All preservation losses are multiplied by 0.2;
high-risk supervision is unchanged. The reference is an independent frozen
Body–Scene Grounded Planner. Fixed original reasoning tokens are replayed with
native prefill/decode during training; adapted autoregressive generation is used
at inference.

Alignment uses an independently admitted, Grounded-Planner-relative pool:
1,325 high-risk, 8,724 HardSafe and 72,383 Normal scenes. Its global-batch-16
quotas cycle through `(2,4,10)`, `(1,4,11)`, `(2,4,10)`, `(1,4,11)`, `(2,4,10)`.
Do not substitute the Base-relative four-role RC-F pool for alignment.

Only model/loss integration is provided for alignment, not a replacement for
the dataset-specific language-boundary cache builder or distributed trainer.

## Adapted feature extraction

Generate inference features with the selected alignment parameters:

```python
from feaxdrive2.checkpoints import load_alignment
from feaxdrive2.vlm.runtime import native_readout

alignment = load_alignment("checkpoints/alignment.pt")
readout = native_readout(model_path=model_path, vlm_path=vlm_path,
    processor_path=processor_path, alignment=alignment, device="cuda")
context = readout.extract_multimodal_context(sample, hidden_source="generation_capture")
```

Pack `context` with motion history and calibrated camera features using the
[data contract](data.md), and record the alignment identity in the inference
manifest. Pass the same alignment checkpoint to the
[inference entry point](../examples/infer.py) for the action-generator parameters.
Training uses fixed-token replay; inference uses adapted autoregressive generation.
