# Feature and training-data contract

All planner tensors must be on the same device. Planner inference and RC-F use
FP32; boolean masks and integer indices retain their types. Cached multimodal
and camera states are published as FP16, then promoted to FP32. Changing this
rounding boundary can change results.

## Planner inputs

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `last_hidden_state` | `[B,N,4096]` | Packed post-language visual/history/navigation/reasoning states |
| `multimodal_type_ids` | `[B,N]` | `0,1,2,3` for those four modalities; `-1` padding |
| `multimodal_position_ids` | `[B,N]` | Within-modality positions in `[0,255]` |
| `multimodal_valid_mask` | `[B,N]` bool | Valid token positions |
| `coc_valid` | `[B]` bool | Whether a closed reasoning sequence is available |
| `history_trajectory` | `[B,4,3]` | Planner motion history in its established rear-axle frame |
| `status_feature` | `[B,8]` | Navigation one-hot, planar velocity, planar acceleration, in that order |
| `vehicle_geometry` | `[B,3]` | Width, rear-axle-to-front, rear-axle-to-rear; metres |
| `camera_features` | `[B,3,1152,16,16]` | Current-frame pre-merger visual features |
| `camera_projection` | `[B,3,3,4]` | Calibrated projection matrix in processed-image coordinates |
| `camera_image_size` | `[B,3,2]` | Processed image **height, width** |
| `camera_valid` | `[B,3]` bool | Available cameras |
| `camera_intrinsics` | `[B,3,3,3]` | Intrinsics consistent with the processed image |
| `camera_distortion` | `[B,3,5]` | Brown–Conrady `k1,k2,p1,p2,k3` |
| `camera_distortion_r2_max` | `[B,3]` | Calibrated valid-radius bound |

The Base Planner and RC-F w/o Spatial Retrieval consume only the first seven
fields. The wrapper excludes camera/geometry inputs from these variants.
The paper uses Pacifica geometry `(2.297,4.049,1.127)`; use the same geometry
during inference and evaluation. A different footprint changes the experiment.

## VLM and current-frame feature construction

`vlm.planning_readout` retains the evaluated readout and packing operations.
It accepts a sample containing `token`, `image_paths` (three camera lists of four
historical/current frames), `camera_indices`, `num_frames_per_camera=4`,
`nav_text`, and Alpamayo-compatible `ego_history_xyz`/`ego_history_rot` tensors.
Alpamayo history tensors and the planner's four-point history have different
contracts; do not replace one with the other.

The three views are front-left, front, and front-right in the established cache
order. Camera embeddings, image paths, projection matrices and feature maps must
all use the same order. Sensor poses and lens/processor calibration must come
from the dataset; no synthetic default calibration is supplied.

```python
context = readout.extract_multimodal_context(sample, hidden_source="generation_capture")
features["last_hidden_state"] = context["multimodal_hidden_state"].unsqueeze(0).float().to(device)
for name in ("multimodal_type_ids", "multimodal_position_ids", "coc_valid"):
    features[name] = context[name].unsqueeze(0).to(device)
features["multimodal_valid_mask"] = features["multimodal_type_ids"].ge(0)

vision = readout.extract_pre_language_geometry(
    sample, geometry_source_hw=16, spatial_resampling="bilinear")
grid = vision["geometry_hidden_state"].reshape(3, 4, 16, 16, 1152)
features["camera_features"] = grid[:, -1].permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
```

The actual readout validates the underlying grouping; validate camera/frame IDs
when integrating a different dataset. Projection uses the local `z=0` plane,
positive optical depth and in-image checks. A valid projection does not assert
visibility or measure an obstacle's 3D clearance. Each side probe samples a
`3×3` neighborhood with offsets of one **feature cell**, not one image pixel.
The 1.5 m side offset specifies visual sampling, not a safety-distance threshold.

## Portable records

`examples/train_cached.py` and `examples/infer.py` read JSONL entries:

```json
{"token":"scenario_id","path":"records/scenario_id.pt","split":"train","role":"HardSafe"}
```

Paths may be relative to the manifest. Each `.pt` contains ordinary tensors and
primitive values, with a retained batch dimension of one:

```python
torch.save({"token": token, "features": features_cpu, "trajectory": target_cpu}, record_path)
```

Inference records may omit `trajectory` and `role`. Training rejects entries
not explicitly assigned to `train`. Do not add validation/test rows to training
manifests. The manifest's unique identities are also used to align predictions
with official scoring inputs.

For adapted-VLM inference, each JSONL row additionally records
`alignment_sha256=feaxdrive2.checkpoints.checkpoint_identity(alignment_path)`
at feature generation time. The inference entry point checks this against its
selected alignment file. Original-VLM manifests omit that field. This prevents
an adapted action generator from silently consuming original-VLM features.

For RC-F, targets depend on the role:

* Collision / NearCollision: safety-validated demonstration trajectory.
* HardSafe / Normal: saved Base Planner trajectory for the local interpolation
  bridge. The frozen Base Planner independently supplies current same-noise
  intermediate states and the final reference trajectory.

The loss encodes Collision and NearCollision as role `0`, HardSafe as `1`, and
Normal as `2`. For a batch averaged locally, use weights `1/B`; accumulation or
distributed training must preserve global scene averaging rather than normalize
each role separately.

## Native alignment boundary

Alignment uses fixed original reasoning IDs with native causal prefill/decode.
`vlm.planning_lora` provides `prepare_pixels`, `collate_payloads`, and
`planner_features`; `vlm.planning_readout` provides
`extract_llm_boundary_context`. Use native schema **3**, with
`hidden_source="generation_capture"` and `llm_prompt_length`. The older full
sequence replay is not the reported alignment protocol.

Each compact payload contains `token`, `split`, `image_paths`, `boundary`,
`reference_features`, `history_trajectory`, `status_feature`,
`high_command_one_hot`, and metadata identifying the native source pack.
Boundary fields include complete token IDs/masks, multimodal position IDs,
RoPE deltas, image grids, visual-token locations, native prompt length and special
token IDs. Original reference hidden states are stored in FP16. The code checks
these fields explicitly. Images/DeepStack are reconstructed through frozen
vision; compact payloads do not persist those activations.

```python
from feaxdrive2.vlm.planning_lora import prepare_pixels, collate_payloads

prepared = prepare_pixels(payload, processor_path)
features = collate_payloads([prepared], device="cuda")
features.update(camera_and_geometry_tensors)
features.update(flow_noise=noise, flow_time=flow_time)
result = alignment_model(features, {"trajectory": targets}, roles, weights)
```

Native replay currently uses per-device microbatch one; accumulate to effective
batch 16. The frozen reference is the original Body–Scene Grounded Planner.
The alignment loss replaces the preservation bridge endpoint with that
reference's same-noise final trajectory internally. Do not wrap the trainable
planner's forward in `no_grad`: gradients must flow from planning losses to
language LoRA and selected action parameters, while trajectory coordinates used
for spatial queries remain detached.
