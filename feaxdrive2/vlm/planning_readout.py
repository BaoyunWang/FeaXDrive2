# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Pure-Alpamayo Action-Expert planning readout.

This module deliberately has no NAVSIM/nuPlan imports so it can run in the
Python-3.12 ``alpamayo`` environment.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, LogitsProcessorList, StoppingCriteriaList

from alpamayo1_5 import helper
from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5, ExpertLogitsProcessor
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)

if __package__:
    from .last_kv_action_bridge import LastKVActionExpertBridge
    from .multimodal_context import (
        assemble_batched_generation_hidden,
        assemble_generation_hidden,
        pack_pre_language_visual_grid,
        pack_modalities,
        pool_fixed_groups,
        pool_visual_hidden,
        positions_between,
        rowwise_multinomial_generators,
    )
else:  # direct script execution: scripts/alpamayo is on sys.path
    from last_kv_action_bridge import LastKVActionExpertBridge
    from multimodal_context import (
        assemble_batched_generation_hidden,
        assemble_generation_hidden,
        pack_pre_language_visual_grid,
        pack_modalities,
        pool_fixed_groups,
        pool_visual_hidden,
        positions_between,
        rowwise_multinomial_generators,
    )


def stable_token_seed(token: str, base_seed: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    token_int = int.from_bytes(digest, byteorder="little", signed=False)
    return int((token_int ^ int(base_seed)) % (2**31 - 1))


def load_camera_tensor(image_paths: list[list[str]]) -> torch.Tensor:
    """Load camera-major image paths into [N_cam, N_frame, 3, H, W] uint8."""
    cameras = []
    for per_camera in image_paths:
        frames = []
        for path in per_camera:
            with Image.open(Path(path)) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
            frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
        cameras.append(torch.stack(frames, dim=0))
    return torch.stack(cameras, dim=0)


def require_local_metadata(path: str, label: str) -> Path:
    """Validate a local Hugging Face config/processor directory."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {resolved}")

    required_files = (
        "chat_template.json",
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
    )
    missing = [name for name in required_files if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{label} directory is incomplete: {resolved}; missing {', '.join(missing)}"
        )
    return resolved


class AlpamayoActionExpertReadout:
    """Extract pre-``action_out_proj`` hidden states under deterministic probes."""

    def __init__(
        self,
        model_path: str,
        vlm_path: str,
        processor_path: str,
        device: str = "cuda",
        num_probes: int = 4,
        probe_seed: int = 20260720,
        keep_tokens: int = 40,
        coc_seed: int = 20260720,
        max_generation_length: int = 256,
        top_p: float = 0.98,
        temperature: float = 0.6,
        expert_readout_layer: int | None = None,
        expert_execution_mode: str = "full",
        query_adapter_checkpoint: str | None = None,
        bridge_checkpoint: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.num_probes = int(num_probes)
        self.probe_seed = int(probe_seed)
        self.keep_tokens = int(keep_tokens)
        self.coc_seed = int(coc_seed)
        self.max_generation_length = int(max_generation_length)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.expert_readout_layer = (
            None if expert_readout_layer is None else int(expert_readout_layer)
        )
        self.expert_execution_mode = str(expert_execution_mode).lower()
        if self.expert_execution_mode not in {"full", "last_kv_bridge"}:
            raise ValueError(
                "expert_execution_mode must be 'full' or 'last_kv_bridge', got %r"
                % expert_execution_mode
            )
        if (
            self.expert_execution_mode == "last_kv_bridge"
            and self.expert_readout_layer is not None
        ):
            raise ValueError(
                "expert_readout_layer is incompatible with last_kv_bridge mode"
            )
        if query_adapter_checkpoint and bridge_checkpoint:
            raise ValueError(
                "Provide either query_adapter_checkpoint or bridge_checkpoint, not both"
            )

        local_vlm_path = require_local_metadata(vlm_path, "Alpamayo VLM metadata")
        local_processor_path = require_local_metadata(
            processor_path, "Alpamayo base processor metadata"
        )

        model_config_path = Path(model_path).expanduser().resolve() / "config.json"
        if not model_config_path.is_file():
            raise FileNotFoundError(f"Alpamayo model config does not exist: {model_config_path}")
        with model_config_path.open(encoding="utf-8") as config_file:
            model_config_dict = json.load(config_file)
        model_config_dict["vlm_name_or_path"] = str(local_vlm_path)
        model_config = Alpamayo1_5Config(**model_config_dict)

        self.model = Alpamayo1_5.from_pretrained(
            model_path,
            config=model_config,
            dtype=torch.bfloat16,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        expert_layers = getattr(self.model.expert, "layers", None)
        if expert_layers is None:
            raise AttributeError("Alpamayo Action Expert has no .layers module list")
        self.expert_num_layers = len(expert_layers)
        self.last_kv_bridge = None
        if self.expert_execution_mode == "last_kv_bridge":
            self.last_kv_bridge = LastKVActionExpertBridge(self.model)
            if bridge_checkpoint:
                self.last_kv_bridge.load_bridge(bridge_checkpoint)
            elif query_adapter_checkpoint:
                self.last_kv_bridge.load_query_adapter(query_adapter_checkpoint)
            self.last_kv_bridge.eval()
        elif self.expert_readout_layer is not None:
            if not 0 <= self.expert_readout_layer < self.expert_num_layers:
                raise ValueError(
                    "expert_readout_layer must be in [0, %d], got %d"
                    % (self.expert_num_layers - 1, self.expert_readout_layer)
                )
            # Intermediate-depth extraction only needs blocks through the target.
            # A forward hook below captures the raw block output before the model's
            # final norm; trimming avoids running the unused later expert blocks.
            self.model.expert.layers = torch.nn.ModuleList(
                list(expert_layers[: self.expert_readout_layer + 1])
            )
        self.processor = AutoProcessor.from_pretrained(
            str(local_processor_path),
            min_pixels=helper.MIN_PIXELS,
            max_pixels=helper.MAX_PIXELS,
            local_files_only=True,
        )
        self.processor.tokenizer = self.model.tokenizer

        n_tokens, action_dim = self.model.action_space.get_action_space_dims()
        if self.keep_tokens > n_tokens:
            raise ValueError("keep_tokens=%d > Alpamayo action tokens=%d" % (self.keep_tokens, n_tokens))
        self.n_diffusion_tokens = int(n_tokens)
        self.action_dim = int(action_dim)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.probe_seed)
        self.fixed_probes_cpu = torch.randn(
            self.num_probes,
            self.n_diffusion_tokens,
            self.action_dim,
            generator=gen,
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def prepare_pre_language_geometry(
        self,
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare the CPU-side Qwen3-VL inputs for GeoVision extraction.

        Keeping this work separate allows the cache extractor to overlap image
        decode and processor work with the preceding sample's GPU forward.  No
        model parameters or CUDA tensors are touched here.
        """
        image_frames = load_camera_tensor(sample["image_paths"])
        camera_indices = torch.tensor(
            sample["camera_indices"], dtype=torch.int64
        )
        num_frames = int(sample.get("num_frames_per_camera", 4))
        if image_frames.shape[:2] != (3, 4) or num_frames != 4:
            raise RuntimeError(
                "Formal GeoVision requires 3 cameras x 4 independent frames"
            )

        messages = helper.create_message(
            frames=image_frames.flatten(0, 1),
            camera_indices=camera_indices,
            num_frames_per_camera=num_frames,
            nav_text=sample.get("nav_text"),
            use_nav_prompt=True,
        )
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        return {
            "tokenized": dict(tokenized),
            "num_frames_per_camera": num_frames,
        }

    @torch.inference_mode()
    def extract_pre_language_geometry(
        self,
        sample: dict[str, Any],
        geometry_source_hw: int = 4,
        prepared: dict[str, Any] | None = None,
        *,
        spatial_resampling: str = "adaptive_avg",
    ) -> dict[str, Any]:
        """Extract only the final Vision-block output before Qwen3-VL merger.

        This path never runs the language Transformer or generates a CoC.  It
        is therefore suitable for extending an existing V1 post-L cache while
        preserving every V1 conditioning tensor bit-for-bit.
        """
        results = self.extract_pre_language_geometry_batch(
            [sample],
            geometry_source_hw=geometry_source_hw,
            prepared_batch=None if prepared is None else [prepared],
            spatial_resampling=spatial_resampling,
        )
        return results[0]

    @torch.inference_mode()
    def extract_pre_language_geometry_batch(
        self,
        samples: Sequence[dict[str, Any]],
        geometry_source_hw: int = 4,
        prepared_batch: Sequence[dict[str, Any]] | None = None,
        *,
        spatial_resampling: str = "adaptive_avg",
    ) -> list[dict[str, Any]]:
        """Extract independent GeoVision payloads in one Vision forward."""
        if not samples:
            raise ValueError("GeoVision batch must contain at least one sample")
        if spatial_resampling not in ("adaptive_avg", "bilinear"):
            raise ValueError("Unsupported spatial_resampling: %r" % spatial_resampling)
        if prepared_batch is None:
            prepared_batch = [
                self.prepare_pre_language_geometry(sample)
                for sample in samples
            ]
        if len(prepared_batch) != len(samples):
            raise ValueError("GeoVision samples/prepared batch length mismatch")

        pixel_values_batch = []
        image_grid_batch = []
        num_frames_batch = []
        pre_merger_counts = []
        for prepared in prepared_batch:
            tokenized = prepared["tokenized"]
            pixel_values = tokenized.get("pixel_values")
            image_grid_thw = tokenized.get("image_grid_thw")
            if pixel_values is None or image_grid_thw is None:
                raise RuntimeError(
                    "Alpamayo processor did not return image pixels and grid metadata"
                )
            if int(image_grid_thw.shape[0]) != 12:
                raise RuntimeError(
                    "Formal GeoVision requires 12 image grids, got %d"
                    % image_grid_thw.shape[0]
                )
            pixel_values_batch.append(pixel_values)
            image_grid_batch.append(image_grid_thw)
            num_frames_batch.append(
                int(prepared["num_frames_per_camera"])
            )
            pre_merger_counts.append(
                sum(
                    int(temporal) * int(height) * int(width)
                    for temporal, height, width in image_grid_thw.tolist()
                )
            )

        pixel_values = torch.cat(pixel_values_batch, dim=0).to(self.device)
        combined_image_grid = torch.cat(image_grid_batch, dim=0).to(
            self.device
        )

        visual_model = getattr(self.model.vlm.model, "visual", None)
        visual_merger = getattr(visual_model, "merger", None)
        if visual_model is None or visual_merger is None:
            raise AttributeError(
                "Qwen3-VL model has no visual.merger pre-language boundary"
            )
        pre_merger_outputs: list[torch.Tensor] = []

        def capture_pre_merger(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            if len(inputs) != 1:
                raise RuntimeError(
                    "Qwen3-VL visual.merger must receive one tensor"
                )
            if pre_merger_outputs:
                raise RuntimeError("Qwen3-VL visual.merger ran more than once")
            pre_merger_outputs.append(inputs[0].detach())

        hook_handle = visual_merger.register_forward_pre_hook(
            capture_pre_merger
        )
        try:
            visual_dtype = next(visual_model.parameters()).dtype
            with torch.autocast("cuda", dtype=torch.bfloat16):
                visual_model(
                    pixel_values.to(dtype=visual_dtype),
                    grid_thw=combined_image_grid,
                )
        finally:
            hook_handle.remove()
        if len(pre_merger_outputs) != 1:
            raise RuntimeError(
                "Expected one Qwen3-VL pre-merger output, captured %d"
                % len(pre_merger_outputs)
            )

        pre_merger_batch = pre_merger_outputs[0].split(
            pre_merger_counts, dim=0
        )
        if len(pre_merger_batch) != len(samples):
            raise RuntimeError("GeoVision pre-merger batch split failed")

        merge_size = getattr(
            self.processor.image_processor, "merge_size", None
        )
        if merge_size is None:
            merge_size = (
                self.model.vlm.config.vision_config.spatial_merge_size
            )
        merge_size = int(merge_size)
        results = []
        for sample, hidden, image_grid_thw, num_frames in zip(
            samples,
            pre_merger_batch,
            image_grid_batch,
            num_frames_batch,
        ):
            geometry_grid = pack_pre_language_visual_grid(
                hidden,
                image_grid_thw=image_grid_thw,
                spatial_merge_size=merge_size,
                camera_indices=sample["camera_indices"],
                num_frames_per_camera=num_frames,
                source_hw=int(geometry_source_hw),
                spatial_resampling=spatial_resampling,
            )
            results.append(
                {
                    "geometry_hidden_state": geometry_grid.hidden.to(
                        dtype=torch.float16
                    ).cpu().contiguous(),
                    "geometry_camera_ids": (
                        geometry_grid.camera_ids.cpu().contiguous()
                    ),
                    "geometry_frame_positions": (
                        geometry_grid.frame_positions.cpu().contiguous()
                    ),
                    "geometry_spatial_coordinates": (
                        geometry_grid.spatial_coordinates.cpu().contiguous()
                    ),
                    "geometry_group_ids": (
                        geometry_grid.group_ids.cpu().contiguous()
                    ),
                    "geometry_source_hw": int(geometry_source_hw),
                    "geometry_source_tokens": int(
                        geometry_grid.hidden.shape[0]
                    ),
                    "geometry_hidden_dim": int(
                        geometry_grid.hidden.shape[-1]
                    ),
                    "geometry_num_groups": int(
                        geometry_grid.group_ids.unique().numel()
                    ),
                    "geometry_spatial_merge_size": merge_size,
                    "geometry_spatial_resampling": spatial_resampling,
                    "geometry_pre_language_boundary": (
                        "qwen3vl_final_vision_block_pre_merger"
                    ),
                }
            )
        return results

    @torch.inference_mode()
    def _prepare_pre_action_context(
        self,
        sample: dict[str, Any],
        capture_generation_hidden: bool = False,
    ) -> dict[str, Any]:
        """Generate the deterministic pre-action sequence shared by readouts."""
        token = str(sample["token"])
        image_frames = load_camera_tensor(sample["image_paths"])
        camera_indices = torch.tensor(
            sample["camera_indices"], dtype=torch.int64
        )
        num_frames = int(sample.get("num_frames_per_camera", 4))

        messages = helper.create_message(
            frames=image_frames.flatten(0, 1),
            camera_indices=camera_indices,
            num_frames_per_camera=num_frames,
            nav_text=sample.get("nav_text"),
            use_nav_prompt=True,
        )
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        tokenized_data = helper.to_device(dict(tokenized), self.device)
        input_ids = tokenized_data.pop("input_ids")
        input_ids = self.model.fuse_traj_tokens(
            input_ids,
            {
                "ego_history_xyz": sample["ego_history_xyz"].to(self.device),
                "ego_history_rot": sample["ego_history_rot"].to(self.device),
            },
        )

        scene_seed = stable_token_seed(token, self.coc_seed)
        torch.manual_seed(scene_seed)
        torch.cuda.manual_seed_all(scene_seed)
        generation_config = copy.deepcopy(self.model.vlm.generation_config)
        generation_config.top_p = self.top_p
        generation_config.temperature = self.temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = 1
        generation_config.max_new_tokens = self.max_generation_length
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.pad_token_id = self.model.tokenizer.pad_token_id

        special_ids = {}
        for name in (
            "traj_future_start",
            "image_pad",
            "traj_history_start",
            "traj_history_end",
            "route_start",
            "route_end",
            "cot_start",
            "cot_end",
        ):
            special_token = to_special_token(name)
            token_id = int(
                self.model.tokenizer.convert_tokens_to_ids(special_token)
            )
            if token_id < 0 or token_id == self.model.tokenizer.unk_token_id:
                raise RuntimeError(f"Tokenizer does not define {special_token}")
            special_ids[name] = token_id

        future_start_id = special_ids["traj_future_start"]
        stopping_criteria = StoppingCriteriaList(
            [StopAfterEOS(eos_token_id=future_start_id)]
        )
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.model.config.traj_token_start_idx,
                    traj_vocab_size=self.model.config.traj_vocab_size,
                )
            ]
        )
        captured_hidden_chunks: list[torch.Tensor] = []
        capture_handle = None
        if capture_generation_hidden:

            def _capture_final_hidden(_module, _inputs, output) -> None:
                hidden = (
                    output.last_hidden_state
                    if hasattr(output, "last_hidden_state")
                    else output[0]
                )
                if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                    raise RuntimeError(
                        "VLM generation hook expected last hidden [B,L,D]"
                    )
                if hidden.shape[0] != 1:
                    raise RuntimeError(
                        "Generation-capture E3 requires one generated sequence"
                    )
                captured_hidden_chunks.append(hidden.detach())

            capture_handle = self.model.vlm.model.register_forward_hook(
                _capture_final_hidden
            )
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vlm_outputs = self.model.vlm.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                    logits_processor=logits_processor,
                    use_cache=True,
                    **tokenized_data,
                )
        finally:
            if capture_handle is not None:
                capture_handle.remove()

        sequence = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=future_start_id,
            pad_token_id=self.model.tokenizer.pad_token_id,
        )[0]
        del vlm_outputs
        future_positions = torch.nonzero(
            sequence.eq(future_start_id), as_tuple=False
        ).flatten()
        if future_positions.numel():
            context_end = int(future_positions[0].item())
        else:
            non_padding = torch.nonzero(
                sequence.ne(self.model.tokenizer.pad_token_id), as_tuple=False
            ).flatten()
            if non_padding.numel() == 0:
                raise RuntimeError(f"Generated sequence for {token} is empty")
            context_end = int(non_padding[-1].item()) + 1
        context_sequence = sequence[:context_end]
        context_attention_mask = torch.ones(
            (1, context_sequence.numel()),
            dtype=tokenized_data["attention_mask"].dtype,
            device=context_sequence.device,
        )
        image_grid_thw = tokenized_data.get("image_grid_thw")
        if image_grid_thw is None:
            raise RuntimeError("Alpamayo processor did not return image_grid_thw")
        generation_hidden = None
        if capture_generation_hidden:
            generation_hidden = assemble_generation_hidden(
                captured_hidden_chunks,
                context_length=int(context_sequence.numel()),
            )

        return {
            "token": token,
            "sequence": sequence,
            "context_sequence": context_sequence,
            "context_attention_mask": context_attention_mask,
            "tokenized_data": tokenized_data,
            "image_grid_thw": image_grid_thw,
            "special_ids": special_ids,
            "scene_seed": scene_seed,
            "route_required": sample.get("nav_text") is not None,
            "generation_hidden_state": generation_hidden,
            "generation_capture_chunks": len(captured_hidden_chunks),
        }

    @torch.inference_mode()
    def _prepare_pre_action_context_batch(
        self,
        samples: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate several OnePass contexts with independent per-scene RNGs."""
        samples = list(samples)
        if len(samples) < 2:
            raise ValueError("Batched OnePass generation requires at least two samples")

        scene_seeds = [stable_token_seed(str(sample["token"]), self.coc_seed)
                       for sample in samples]
        row_generators = []
        for scene_seed in scene_seeds:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(scene_seed)
            row_generators.append(generator)

        with rowwise_multinomial_generators(row_generators):
            messages_batch = []
            expected_images = []
            for sample in samples:
                token = str(sample["token"])
                image_frames = load_camera_tensor(sample["image_paths"])
                camera_indices = torch.tensor(
                    sample["camera_indices"], dtype=torch.int64
                )
                num_frames = int(sample.get("num_frames_per_camera", 4))
                image_count = sum(len(paths) for paths in sample["image_paths"])
                if image_count != 12 or image_frames.shape[:2] != (3, 4):
                    raise RuntimeError(
                        "Batched E3 requires 3 cameras x 4 frames per sample"
                    )
                messages_batch.append(
                    helper.create_message(
                        frames=image_frames.flatten(0, 1),
                        camera_indices=camera_indices,
                        num_frames_per_camera=num_frames,
                        nav_text=sample.get("nav_text"),
                        use_nav_prompt=True,
                    )
                )
                expected_images.append(image_count)

            tokenizer = self.processor.tokenizer
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            try:
                tokenized = self.processor.apply_chat_template(
                    messages_batch,
                    tokenize=True,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    padding=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            finally:
                tokenizer.padding_side = original_padding_side

            tokenized_data = helper.to_device(dict(tokenized), self.device)
            input_ids = tokenized_data.pop("input_ids")
            prompt_attention_mask = tokenized_data["attention_mask"].clone()
            input_ids = self.model.fuse_traj_tokens(
                input_ids,
                {
                    "ego_history_xyz": torch.cat(
                        [sample["ego_history_xyz"] for sample in samples], dim=0
                    ).to(self.device),
                    "ego_history_rot": torch.cat(
                        [sample["ego_history_rot"] for sample in samples], dim=0
                    ).to(self.device),
                },
            )

            generation_config = copy.deepcopy(self.model.vlm.generation_config)
            generation_config.top_p = self.top_p
            generation_config.temperature = self.temperature
            generation_config.do_sample = True
            generation_config.num_return_sequences = 1
            generation_config.max_new_tokens = self.max_generation_length
            generation_config.output_logits = False
            generation_config.return_dict_in_generate = True
            generation_config.pad_token_id = self.model.tokenizer.pad_token_id

            special_ids = {}
            for name in (
                "traj_future_start",
                "image_pad",
                "traj_history_start",
                "traj_history_end",
                "route_start",
                "route_end",
                "cot_start",
                "cot_end",
            ):
                special_token = to_special_token(name)
                token_id = int(
                    self.model.tokenizer.convert_tokens_to_ids(special_token)
                )
                if token_id < 0 or token_id == self.model.tokenizer.unk_token_id:
                    raise RuntimeError(f"Tokenizer does not define {special_token}")
                special_ids[name] = token_id

            batch_size = len(samples)
            future_start_id = special_ids["traj_future_start"]

            class _StopEachRowAfterEOS(StopAfterEOS):
                def __call__(self, input_ids, scores, **kwargs):
                    count = input_ids.shape[0]
                    if self.eos_found is None:
                        self.eos_found = torch.zeros(count, dtype=torch.bool, device=input_ids.device)
                    if tuple(self.eos_found.shape) != (count,):
                        raise RuntimeError("Per-row EOS criterion batch size changed")
                    # Match single-scene StopAfterEOS: consume one further token
                    # after first EOS so that its hidden/cache has been evaluated.
                    finished = self.eos_found.clone()
                    self.eos_found |= input_ids[:, -1].eq(self.eos_token_id)
                    return finished

            captured_hidden_chunks: list[torch.Tensor] = []

            def _capture_final_hidden(_module, _inputs, output) -> None:
                hidden = (
                    output.last_hidden_state
                    if hasattr(output, "last_hidden_state")
                    else output[0]
                )
                if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                    raise RuntimeError(
                        "VLM generation hook expected last hidden [B,L,D]"
                    )
                if hidden.shape[0] != batch_size:
                    raise RuntimeError(
                        "Generation hook batch mismatch: expected %d, got %d"
                        % (batch_size, hidden.shape[0])
                    )
                captured_hidden_chunks.append(hidden.detach())

            capture_handle = self.model.vlm.model.register_forward_hook(
                _capture_final_hidden
            )
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vlm_outputs = self.model.vlm.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        stopping_criteria=StoppingCriteriaList(
                            [_StopEachRowAfterEOS(eos_token_id=future_start_id)]
                        ),
                        logits_processor=LogitsProcessorList(
                            [
                                ExpertLogitsProcessor(
                                    traj_token_offset=(
                                        self.model.config.traj_token_start_idx
                                    ),
                                    traj_vocab_size=self.model.config.traj_vocab_size,
                                )
                            ]
                        ),
                        use_cache=True,
                        **tokenized_data,
                    )
            finally:
                capture_handle.remove()

            padded_sequences = replace_padding_after_eos(
                token_ids=vlm_outputs.sequences,
                eos_token_id=future_start_id,
                pad_token_id=self.model.tokenizer.pad_token_id,
            )
            del vlm_outputs
            prompt_width = int(input_ids.shape[1])
            context_sequences = []
            full_sequences = []
            for row in range(batch_size):
                prompt_sequence = padded_sequences[row, :prompt_width][
                    prompt_attention_mask[row].bool()
                ]
                sequence = torch.cat(
                    [prompt_sequence, padded_sequences[row, prompt_width:]], dim=0
                )
                future_positions = torch.nonzero(
                    sequence.eq(future_start_id), as_tuple=False
                ).flatten()
                if future_positions.numel():
                    context_end = int(future_positions[0].item())
                else:
                    non_padding = torch.nonzero(
                        sequence.ne(self.model.tokenizer.pad_token_id),
                        as_tuple=False,
                    ).flatten()
                    if non_padding.numel() == 0:
                        raise RuntimeError(
                            "Generated sequence for %s is empty"
                            % samples[row]["token"]
                        )
                    context_end = int(non_padding[-1].item()) + 1
                full_sequences.append(sequence)
                context_sequences.append(sequence[:context_end])

            generation_hidden = assemble_batched_generation_hidden(
                captured_hidden_chunks,
                prompt_attention_mask,
                [int(sequence.numel()) for sequence in context_sequences],
            )
            image_grid_thw = tokenized_data.get("image_grid_thw")
            if image_grid_thw is None:
                raise RuntimeError("Alpamayo processor did not return image_grid_thw")
            if int(image_grid_thw.shape[0]) != sum(expected_images):
                raise RuntimeError(
                    "Batched image-grid count mismatch: expected %d, got %d"
                    % (sum(expected_images), image_grid_thw.shape[0])
                )

            prepared_batch = []
            image_cursor = 0
            for row, sample in enumerate(samples):
                image_count = expected_images[row]
                sample_image_grid = image_grid_thw[
                    image_cursor : image_cursor + image_count
                ]
                image_cursor += image_count
                context_sequence = context_sequences[row]
                prepared_batch.append(
                    {
                        "token": str(sample["token"]),
                        "sequence": full_sequences[row],
                        "context_sequence": context_sequence,
                        "context_attention_mask": torch.ones(
                            (1, context_sequence.numel()),
                            dtype=prompt_attention_mask.dtype,
                            device=context_sequence.device,
                        ),
                        "tokenized_data": {},
                        "image_grid_thw": sample_image_grid,
                        "special_ids": special_ids,
                        "scene_seed": scene_seeds[row],
                        "route_required": sample.get("nav_text") is not None,
                        "generation_hidden_state": generation_hidden[row],
                        "generation_capture_chunks": len(captured_hidden_chunks),
                    }
                )
            return prepared_batch

    @torch.inference_mode()
    def _replay_multimodal_hidden(
        self, prepared: dict[str, Any]
    ) -> torch.Tensor:
        context_inputs = dict(prepared["tokenized_data"])
        context_inputs["attention_mask"] = prepared[
            "context_attention_mask"
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.vlm.model(
                input_ids=prepared["context_sequence"].unsqueeze(0),
                use_cache=False,
                **context_inputs,
            ).last_hidden_state[0]

    @staticmethod
    def _select_available_hidden(
        vlm_hidden: torch.Tensor,
        positions: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        if positions.numel() and int(positions.max().item()) >= vlm_hidden.shape[0]:
            raise RuntimeError(
                "Generation capture did not evaluate a selected %s token: "
                "last_position=%d captured=%d"
                % (label, int(positions.max().item()), vlm_hidden.shape[0])
            )
        return vlm_hidden.index_select(0, positions)

    def _pack_multimodal_context(
        self,
        sample: dict[str, Any],
        prepared: dict[str, Any],
        vlm_hidden: torch.Tensor,
        *,
        max_coc_tokens: int,
        visual_pool_size: int,
        history_tokens: int,
        hidden_source: str,
    ) -> dict[str, Any]:
        if vlm_hidden.ndim != 2:
            raise ValueError("VLM context hidden must have shape [L,D]")
        sequence = prepared["sequence"]
        context_sequence = prepared["context_sequence"]
        special_ids = prepared["special_ids"]
        if vlm_hidden.shape[0] > context_sequence.numel():
            raise RuntimeError(
                "VLM hidden is longer than the pre-action context: %d > %d"
                % (vlm_hidden.shape[0], context_sequence.numel())
            )
        available_sequence = context_sequence[: vlm_hidden.shape[0]]

        image_grid_thw = prepared["image_grid_thw"]
        expected_images = sum(len(paths) for paths in sample["image_paths"])
        if expected_images != 12:
            raise RuntimeError(
                "E3 expects 3 cameras x 4 frames (12 images), got %d"
                % expected_images
            )
        if int(image_grid_thw.shape[0]) != expected_images:
            raise RuntimeError(
                "Expected %d image grids from input paths, got %d"
                % (expected_images, image_grid_thw.shape[0])
            )
        merge_size = getattr(self.processor.image_processor, "merge_size", None)
        if merge_size is None:
            merge_size = self.model.vlm.config.vision_config.spatial_merge_size
        vision_hidden = pool_visual_hidden(
            vlm_hidden,
            available_sequence,
            image_token_id=special_ids["image_pad"],
            image_grid_thw=image_grid_thw,
            spatial_merge_size=int(merge_size),
            output_hw=int(visual_pool_size),
        )

        history_positions = positions_between(
            context_sequence,
            special_ids["traj_history_start"],
            special_ids["traj_history_end"],
            label="trajectory-history",
        )
        if int(history_positions.numel()) != 48:
            raise RuntimeError(
                "E3 expects 48 trajectory-history tokens, got %d"
                % history_positions.numel()
            )
        history_hidden = self._select_available_hidden(
            vlm_hidden, history_positions, label="trajectory-history"
        )
        history_hidden = pool_fixed_groups(history_hidden, int(history_tokens))

        route_positions = positions_between(
            context_sequence,
            special_ids["route_start"],
            special_ids["route_end"],
            label="route",
            required=prepared["route_required"],
        )
        if int(route_positions.numel()) > 256:
            raise RuntimeError(
                "E3 route token count exceeds the 256-position adapter limit"
            )
        route_hidden = self._select_available_hidden(
            vlm_hidden, route_positions, label="route"
        )

        coc_positions = positions_between(
            context_sequence,
            special_ids["cot_start"],
            special_ids["cot_end"],
            label="CoC",
            required=False,
            use_last_start=True,
        )
        coc_valid = bool(coc_positions.numel())
        if coc_valid:
            coc_positions = coc_positions[: int(max_coc_tokens)]
        coc_hidden = self._select_available_hidden(
            vlm_hidden, coc_positions, label="CoC"
        )

        packed_hidden, type_ids, position_ids = pack_modalities(
            [vision_hidden, history_hidden, route_hidden, coc_hidden]
        )
        text_extra = extract_text_tokens(
            self.model.tokenizer, sequence.unsqueeze(0)
        )
        cot = ""
        if coc_valid and isinstance(text_extra, dict) and "cot" in text_extra:
            cot_values = text_extra["cot"]
            cot = (
                str(cot_values[0])
                if isinstance(cot_values, (list, tuple, np.ndarray)) and len(cot_values)
                else str(cot_values)
            )
        is_capture = hidden_source == "generation_capture"
        payload = {
            "multimodal_hidden_state": packed_hidden.to(
                dtype=torch.float16
            ).cpu().contiguous(),
            "multimodal_type_ids": type_ids.cpu().contiguous(),
            "multimodal_position_ids": position_ids.cpu().contiguous(),
            "coc_valid": torch.tensor(coc_valid, dtype=torch.bool),
            "alpamayo_cot": cot,
            "alpamayo_cot_valid": coc_valid,
            "alpamayo_coc_seed": prepared["scene_seed"],
            "alpamayo_multimodal_hidden_dim": int(packed_hidden.shape[-1]),
            "alpamayo_multimodal_source_tokens": int(packed_hidden.shape[0]),
            "alpamayo_multimodal_counts": torch.bincount(
                type_ids, minlength=4
            ).cpu(),
            "alpamayo_readout": (
                "vlm_multimodal_context_generation_capture_v1"
                if is_capture
                else "vlm_multimodal_context_pre_action_v1"
            ),
        }
        if is_capture:
            payload.update(
                {
                    "alpamayo_hidden_source": hidden_source,
                    "alpamayo_vlm_reforward": False,
                    "alpamayo_generation_capture_chunks": int(
                        prepared["generation_capture_chunks"]
                    ),
                    "alpamayo_generation_capture_tokens": int(
                        vlm_hidden.shape[0]
                    ),
                    "alpamayo_generation_context_tokens": int(
                        context_sequence.numel()
                    ),
                }
            )
        return payload

    @torch.inference_mode()
    def extract_multimodal_context(
        self,
        sample: dict[str, Any],
        max_coc_tokens: int = 256,
        visual_pool_size: int = 2,
        history_tokens: int = 16,
        hidden_source: str = "replay",
    ) -> dict[str, Any]:
        """Extract E3 post-L context without loading the Action Expert.

        ``generation_capture`` retains final-layer hidden from the original
        causal generation and removes the complete teacher-forced VLM replay.
        The existing ``replay`` path remains the default for compatibility.
        """
        if not 1 <= int(max_coc_tokens) <= 256:
            raise ValueError("max_coc_tokens must be in [1, 256]")
        hidden_source = str(hidden_source).lower()
        if hidden_source not in {"replay", "generation_capture"}:
            raise ValueError(
                "hidden_source must be 'replay' or 'generation_capture'"
            )
        prepared = self._prepare_pre_action_context(
            sample,
            capture_generation_hidden=hidden_source == "generation_capture",
        )
        if hidden_source == "generation_capture":
            vlm_hidden = prepared["generation_hidden_state"]
            if vlm_hidden is None:
                raise RuntimeError("Generation capture returned no VLM hidden")
        else:
            vlm_hidden = self._replay_multimodal_hidden(prepared)
        return self._pack_multimodal_context(
            sample,
            prepared,
            vlm_hidden,
            max_coc_tokens=int(max_coc_tokens),
            visual_pool_size=int(visual_pool_size),
            history_tokens=int(history_tokens),
            hidden_source=hidden_source,
        )

    @torch.inference_mode()
    def extract_multimodal_context_batch(
        self,
        samples: Sequence[dict[str, Any]],
        max_coc_tokens: int = 256,
        visual_pool_size: int = 2,
        history_tokens: int = 16,
    ) -> list[dict[str, Any]]:
        """Extract generation-capture E3 contexts for a true VLM batch."""
        samples = list(samples)
        if not samples:
            raise ValueError("OnePass batch must contain at least one sample")
        if len(samples) == 1:
            return [
                self.extract_multimodal_context(
                    samples[0],
                    max_coc_tokens=max_coc_tokens,
                    visual_pool_size=visual_pool_size,
                    history_tokens=history_tokens,
                    hidden_source="generation_capture",
                )
            ]
        if not 1 <= int(max_coc_tokens) <= 256:
            raise ValueError("max_coc_tokens must be in [1, 256]")
        prepared_batch = self._prepare_pre_action_context_batch(samples)
        return [
            self._pack_multimodal_context(
                sample,
                prepared,
                prepared["generation_hidden_state"],
                max_coc_tokens=int(max_coc_tokens),
                visual_pool_size=int(visual_pool_size),
                history_tokens=int(history_tokens),
                hidden_source="generation_capture",
            )
            for sample, prepared in zip(samples, prepared_batch)
        ]

    @torch.inference_mode()
    def compare_multimodal_generation_capture(
        self,
        sample: dict[str, Any],
        max_coc_tokens: int = 256,
        visual_pool_size: int = 2,
        history_tokens: int = 16,
    ) -> dict[str, dict[str, Any]]:
        """Build replay and OnePass payloads from one deterministic generation."""
        prepared = self._prepare_pre_action_context(
            sample, capture_generation_hidden=True
        )
        captured_hidden = prepared["generation_hidden_state"]
        if captured_hidden is None:
            raise RuntimeError("Generation capture returned no VLM hidden")
        replay_hidden = self._replay_multimodal_hidden(prepared)
        common = {
            "max_coc_tokens": int(max_coc_tokens),
            "visual_pool_size": int(visual_pool_size),
            "history_tokens": int(history_tokens),
        }
        return {
            "generation_capture": self._pack_multimodal_context(
                sample,
                prepared,
                captured_hidden,
                hidden_source="generation_capture",
                **common,
            ),
            "replay": self._pack_multimodal_context(
                sample,
                prepared,
                replay_hidden,
                hidden_source="replay",
                **common,
            ),
        }

    @torch.inference_mode()
    def extract_llm_boundary_context(
        self,
        sample: dict[str, Any],
        max_coc_tokens: int = 256,
        visual_pool_size: int = 2,
        history_tokens: int = 16,
    ) -> dict[str, Any]:
        """Cache the frozen-Vision boundary for online language-layer LoRA.

        The cache contains no post-language hidden state.  Token embeddings are
        recomputed from the frozen language embedding table at training time;
        only the expensive frozen Vision outputs (including DeepStack) are
        persisted.  This leaves every language decoder layer differentiable.
        """
        if not 1 <= int(max_coc_tokens) <= 256:
            raise ValueError("max_coc_tokens must be in [1, 256]")
        if int(visual_pool_size) != 2 or int(history_tokens) != 16:
            raise ValueError(
                "Safe-Context locks visual_pool_size=2 and history_tokens=16"
            )

        prepared = self._prepare_pre_action_context(sample)
        context_sequence = prepared["context_sequence"]
        attention_mask = prepared["context_attention_mask"]
        tokenized_data = prepared["tokenized_data"]
        image_grid_thw = prepared["image_grid_thw"]
        special_ids = prepared["special_ids"]

        expected_images = sum(len(paths) for paths in sample["image_paths"])
        if expected_images != 12 or int(image_grid_thw.shape[0]) != 12:
            raise RuntimeError(
                "Safe-Context LLM boundary requires 3 cameras x 4 frames"
            )
        pixel_values = tokenized_data.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("Alpamayo processor did not return pixel_values")

        qwen_model = self.model.vlm.model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            image_parts, deepstack_parts = qwen_model.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = torch.cat(list(image_parts), dim=0)
            token_embeds = qwen_model.get_input_embeddings()(
                context_sequence.unsqueeze(0)
            )
        image_mask, _ = qwen_model.get_placeholder_mask(
            context_sequence.unsqueeze(0),
            inputs_embeds=token_embeds,
            image_features=image_embeds,
        )
        visual_pos_mask = image_mask[..., 0]
        visual_tokens = int(visual_pos_mask.sum().item())
        if image_embeds.shape[0] != visual_tokens:
            raise RuntimeError(
                "Vision boundary token mismatch: embeds=%d mask=%d"
                % (image_embeds.shape[0], visual_tokens)
            )
        if not deepstack_parts:
            raise RuntimeError("Qwen3-VL returned no DeepStack Vision features")
        for layer, deepstack in enumerate(deepstack_parts):
            if deepstack.shape != image_embeds.shape:
                raise RuntimeError(
                    "DeepStack layer %d shape %s != image embeds %s"
                    % (layer, tuple(deepstack.shape), tuple(image_embeds.shape))
                )
        position_ids, rope_deltas = qwen_model.get_rope_index(
            context_sequence.unsqueeze(0),
            image_grid_thw,
            None,
            attention_mask=attention_mask,
        )

        coc_positions = positions_between(
            context_sequence,
            special_ids["cot_start"],
            special_ids["cot_end"],
            label="CoC",
            required=False,
            use_last_start=True,
        )
        coc_valid = bool(coc_positions.numel())
        text_extra = extract_text_tokens(
            self.model.tokenizer, prepared["sequence"].unsqueeze(0)
        )
        cot = ""
        if coc_valid and isinstance(text_extra, dict) and "cot" in text_extra:
            cot_values = text_extra["cot"]
            cot = (
                str(cot_values[0])
                if isinstance(cot_values, (list, tuple, np.ndarray))
                and len(cot_values)
                else str(cot_values)
            )

        merge_size = getattr(self.processor.image_processor, "merge_size", None)
        if merge_size is None:
            merge_size = self.model.vlm.config.vision_config.spatial_merge_size
        scalar_ids = {
            f"llm_{name}_token_id": torch.tensor(value, dtype=torch.int32)
            for name, value in special_ids.items()
        }
        return {
            "llm_context_input_ids": context_sequence.to(
                dtype=torch.int32
            ).cpu().contiguous(),
            "llm_attention_mask": attention_mask[0].to(
                dtype=torch.bool
            ).cpu().contiguous(),
            "llm_position_ids": position_ids[:, 0].to(
                dtype=torch.int32
            ).cpu().contiguous(),
            "llm_rope_deltas": rope_deltas[0].to(
                dtype=torch.int32
            ).cpu().contiguous(),
            "llm_visual_pos_mask": visual_pos_mask[0].cpu().contiguous(),
            "llm_image_embeds": image_embeds.to(
                dtype=torch.float16
            ).cpu().contiguous(),
            "llm_deepstack_visual_embeds": torch.stack(
                list(deepstack_parts), dim=0
            ).to(dtype=torch.float16).cpu().contiguous(),
            "llm_image_grid_thw": image_grid_thw.to(
                dtype=torch.int32
            ).cpu().contiguous(),
            "llm_spatial_merge_size": torch.tensor(
                int(merge_size), dtype=torch.int32
            ),
            "llm_visual_pool_size": torch.tensor(
                int(visual_pool_size), dtype=torch.int32
            ),
            "llm_history_tokens": torch.tensor(
                int(history_tokens), dtype=torch.int32
            ),
            "llm_max_coc_tokens": torch.tensor(
                int(max_coc_tokens), dtype=torch.int32
            ),
            "llm_route_required": torch.tensor(
                bool(prepared["route_required"]), dtype=torch.bool
            ),
            "coc_valid": torch.tensor(coc_valid, dtype=torch.bool),
            "alpamayo_cot": cot,
            "alpamayo_cot_valid": coc_valid,
            "alpamayo_coc_seed": prepared["scene_seed"],
            "alpamayo_llm_hidden_dim": int(image_embeds.shape[-1]),
            "alpamayo_llm_context_tokens": int(context_sequence.numel()),
            "alpamayo_llm_visual_tokens": visual_tokens,
            "alpamayo_llm_deepstack_layers": len(deepstack_parts),
            "alpamayo_readout": "qwen3vl_frozen_vision_llm_boundary_v1",
            **scalar_ids,
        }

    @torch.inference_mode()
    def extract_semantic(
        self,
        sample: dict[str, Any],
        max_semantic_tokens: int = 64,
    ) -> dict[str, Any]:
        """Extract contextual VLM hidden states for generated CoC text tokens.

        This path stops before the Action Expert. The returned tokens are the
        final Alpamayo VLM states strictly between ``cot_start`` and
        ``cot_end`` and therefore remain grounded in the multimodal prompt.
        """
        token = str(sample["token"])
        image_frames = load_camera_tensor(sample["image_paths"])
        camera_indices = torch.tensor(sample["camera_indices"], dtype=torch.int64)
        num_frames = int(sample.get("num_frames_per_camera", 4))

        messages = helper.create_message(
            frames=image_frames.flatten(0, 1),
            camera_indices=camera_indices,
            num_frames_per_camera=num_frames,
            nav_text=sample.get("nav_text"),
            use_nav_prompt=True,
        )
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        tokenized_data = helper.to_device(dict(tokenized), self.device)

        input_ids = tokenized_data.pop("input_ids")
        input_ids = self.model.fuse_traj_tokens(
            input_ids,
            {
                "ego_history_xyz": sample["ego_history_xyz"].to(self.device),
                "ego_history_rot": sample["ego_history_rot"].to(self.device),
            },
        )

        scene_seed = stable_token_seed(token, self.coc_seed)
        torch.manual_seed(scene_seed)
        torch.cuda.manual_seed_all(scene_seed)

        generation_config = copy.deepcopy(self.model.vlm.generation_config)
        generation_config.top_p = self.top_p
        generation_config.temperature = self.temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = 1
        generation_config.max_new_tokens = self.max_generation_length
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.pad_token_id = self.model.tokenizer.pad_token_id

        trajectory_start_id = self.model.tokenizer.convert_tokens_to_ids(
            to_special_token("traj_future_start")
        )
        stopping_criteria = StoppingCriteriaList(
            [StopAfterEOS(eos_token_id=trajectory_start_id)]
        )
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.model.config.traj_token_start_idx,
                    traj_vocab_size=self.model.config.traj_vocab_size,
                )
            ]
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.model.vlm.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=True,
                **tokenized_data,
            )

        sequence = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=trajectory_start_id,
            pad_token_id=self.model.tokenizer.pad_token_id,
        )[0]
        # Release the generation KV cache before the full contextual readout.
        del vlm_outputs
        cot_start_id = self.model.tokenizer.convert_tokens_to_ids(
            to_special_token("cot_start")
        )
        cot_end_id = self.model.tokenizer.convert_tokens_to_ids(
            to_special_token("cot_end")
        )
        start_positions = torch.nonzero(sequence == cot_start_id, as_tuple=False).flatten()
        if start_positions.numel() == 0:
            raise RuntimeError(f"Generated sequence for {token} has no cot_start token")
        cot_start = int(start_positions[-1].item())
        end_positions = torch.nonzero(
            (sequence == cot_end_id)
            & (torch.arange(sequence.numel(), device=sequence.device) > cot_start),
            as_tuple=False,
        ).flatten()
        if end_positions.numel() == 0:
            # Alpamayo occasionally exhausts max_new_tokens without closing
            # CoC. Treat that as an explicit missing condition instead of
            # injecting an arbitrary truncated/degenerate generation.
            hidden_dim = int(self.model.vlm.config.text_config.hidden_size)
            return {
                "semantic_hidden_state": torch.zeros(
                    1, hidden_dim, dtype=torch.float16
                ),
                "semantic_token_ids": torch.empty(0, dtype=torch.long),
                "alpamayo_cot": "",
                "alpamayo_cot_valid": False,
                "alpamayo_coc_seed": scene_seed,
                "alpamayo_semantic_tokens": 0,
                "alpamayo_semantic_hidden_dim": hidden_dim,
                "alpamayo_readout": "vlm_coc_contextual_hidden_v1_null",
            }
        cot_end = int(end_positions[0].item())
        semantic_end = min(cot_end, cot_start + 1 + int(max_semantic_tokens))
        if semantic_end <= cot_start + 1:
            raise RuntimeError(f"Generated sequence for {token} has an empty CoC")

        # Re-run only through cot_end to obtain exact contextual token states
        # from the VLM backbone. This deliberately does not call the Action Expert.
        semantic_sequence = sequence[: cot_end + 1].unsqueeze(0)
        semantic_inputs = dict(tokenized_data)
        semantic_inputs["attention_mask"] = torch.ones_like(
            semantic_sequence, dtype=tokenized_data["attention_mask"].dtype
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_hidden = self.model.vlm.model(
                input_ids=semantic_sequence,
                use_cache=False,
                **semantic_inputs,
            ).last_hidden_state

        semantic_hidden = vlm_hidden[
            0, cot_start + 1 : semantic_end
        ].to(dtype=torch.float16).cpu().contiguous()
        semantic_token_ids = sequence[cot_start + 1 : semantic_end].cpu().contiguous()

        text_extra = extract_text_tokens(self.model.tokenizer, sequence.unsqueeze(0))
        cot = text_extra.get("cot", [""])[0]
        return {
            "semantic_hidden_state": semantic_hidden,
            "semantic_token_ids": semantic_token_ids,
            "alpamayo_cot": str(cot),
            "alpamayo_cot_valid": True,
            "alpamayo_coc_seed": scene_seed,
            "alpamayo_semantic_tokens": int(semantic_hidden.shape[0]),
            "alpamayo_semantic_hidden_dim": int(semantic_hidden.shape[-1]),
            "alpamayo_readout": "vlm_coc_contextual_hidden_v1",
        }

    @torch.inference_mode()
    def extract(
        self,
        sample: dict[str, Any],
        include_action_input_hidden: bool = False,
    ) -> dict[str, Any]:
        token = str(sample["token"])
        image_frames = load_camera_tensor(sample["image_paths"])
        camera_indices = torch.tensor(sample["camera_indices"], dtype=torch.int64)
        num_frames = int(sample.get("num_frames_per_camera", 4))

        messages = helper.create_message(
            frames=image_frames.flatten(0, 1),
            camera_indices=camera_indices,
            num_frames_per_camera=num_frames,
            nav_text=sample.get("nav_text"),
            use_nav_prompt=True,
        )
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        tokenized_data = helper.to_device(dict(tokenized), self.device)

        ego_history_xyz = sample["ego_history_xyz"].to(self.device)
        ego_history_rot = sample["ego_history_rot"].to(self.device)
        input_ids = tokenized_data.pop("input_ids")
        input_ids = self.model.fuse_traj_tokens(
            input_ids,
            {
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
            },
        )

        # Per-token deterministic CoC sampling, independent of extraction order/resume.
        scene_seed = stable_token_seed(token, self.coc_seed)
        torch.manual_seed(scene_seed)
        torch.cuda.manual_seed_all(scene_seed)

        generation_config = copy.deepcopy(self.model.vlm.generation_config)
        generation_config.top_p = self.top_p
        generation_config.temperature = self.temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = 1
        generation_config.max_new_tokens = self.max_generation_length
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.pad_token_id = self.model.tokenizer.pad_token_id

        eos_token_id = self.model.tokenizer.convert_tokens_to_ids(
            to_special_token("traj_future_start")
        )
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.model.config.traj_token_start_idx,
                    traj_vocab_size=self.model.config.traj_vocab_size,
                )
            ]
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.model.vlm.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=True,
                **tokenized_data,
            )

        vlm_outputs.rope_deltas = self.model.vlm.model.rope_deltas
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.model.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        # One CoC, K deterministic action probes.  The full expert evaluates
        # probes together.  E2 evaluates them sequentially so the 36-layer VLM
        # cache is never repeated four times merely to consume KV35.
        if self.num_probes > 1 and self.last_kv_bridge is None:
            prompt_cache.batch_repeat_interleave(self.num_probes)
        sequences = vlm_outputs.sequences.repeat_interleave(self.num_probes, dim=0)
        rope_deltas = vlm_outputs.rope_deltas.repeat_interleave(self.num_probes, dim=0)
        offset = self.model._find_eos_offset(
            sequences=sequences,
            eos_token_id=eos_token_id,
            device=self.device,
        )
        prefix_mask = tokenized_data.get("attention_mask")
        if prefix_mask is not None:
            prefix_mask = prefix_mask.repeat_interleave(self.num_probes, dim=0)

        position_ids, attention_mask = self.model._build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=rope_deltas,
            kv_cache_seq_len=prefill_seq_len,
            n_diffusion_tokens=self.n_diffusion_tokens,
            b_star=self.num_probes,
            device=self.device,
            prefix_mask=prefix_mask,
        )

        probes = self.fixed_probes_cpu.to(self.device)
        t = torch.zeros((self.num_probes, 1, 1), device=self.device, dtype=probes.dtype)
        forward_kwargs = {}
        if self.model.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        captured_layer: dict[str, torch.Tensor] = {}
        hook_handle = None
        if self.expert_readout_layer is not None:
            target_block = self.model.expert.layers[self.expert_readout_layer]

            def _capture_block_output(_module, _inputs, output) -> None:
                hidden_output = output[0] if isinstance(output, tuple) else output
                captured_layer["hidden"] = hidden_output

            hook_handle = target_block.register_forward_hook(_capture_block_output)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.last_kv_bridge is not None:
                bridge_hidden = []
                bridge_raw_queries = []
                for probe_index in range(self.num_probes):
                    probe_hidden, probe_raw_queries, _ = self.last_kv_bridge(
                        probes[probe_index : probe_index + 1],
                        t[probe_index : probe_index + 1],
                        position_ids=position_ids[
                            :, probe_index : probe_index + 1
                        ],
                        prompt_cache=prompt_cache,
                        attention_mask=attention_mask[
                            probe_index : probe_index + 1
                        ],
                        is_causal=not self.model.config.expert_non_causal_attention,
                        return_queries=True,
                    )
                    bridge_hidden.append(probe_hidden)
                    bridge_raw_queries.append(probe_raw_queries)
                hidden = torch.cat(bridge_hidden, dim=0)
                future_token_embeds = torch.cat(bridge_raw_queries, dim=0)
                expert_out = None
            else:
                future_token_embeds = self.model.action_in_proj(probes, t)
                if future_token_embeds.dim() == 2:
                    future_token_embeds = future_token_embeds.view(
                        self.num_probes, self.n_diffusion_tokens, -1
                    )
                try:
                    expert_out = self.model.expert(
                        inputs_embeds=future_token_embeds,
                        position_ids=position_ids,
                        past_key_values=prompt_cache,
                        attention_mask=attention_mask,
                        use_cache=True,
                        **forward_kwargs,
                    )
                finally:
                    if hook_handle is not None:
                        hook_handle.remove()

        if self.last_kv_bridge is not None:
            readout_name = "last_vlm_kv_ae35_bridge_pre_action_out_proj_t0_v1"
        elif self.expert_readout_layer is None:
            assert expert_out is not None
            hidden = expert_out.last_hidden_state
            readout_name = "action_expert_pre_action_out_proj_t0_fixed_gaussian_v1"
        else:
            if "hidden" not in captured_layer:
                raise RuntimeError(
                    "Action Expert layer %d forward hook did not capture output"
                    % self.expert_readout_layer
                )
            hidden = captured_layer["hidden"]
            readout_name = (
                "action_expert_layer%02d_pre_final_norm_t0_fixed_gaussian_v1"
                % self.expert_readout_layer
            )

        hidden = hidden[:, -self.n_diffusion_tokens :]
        hidden = hidden[:, : self.keep_tokens].to(dtype=torch.float16).cpu().contiguous()

        text_extra = extract_text_tokens(self.model.tokenizer, vlm_outputs.sequences)
        cot = None
        if isinstance(text_extra, dict) and "cot" in text_extra:
            value = text_extra["cot"]
            if isinstance(value, (list, tuple, np.ndarray)) and len(value):
                cot = str(value[0])
            else:
                cot = str(value)

        result = {
            "last_hidden_state": hidden,  # [K,40,2048], pre-action_out_proj
            "alpamayo_cot": cot,
            "alpamayo_num_probes": self.num_probes,
            "alpamayo_probe_seed": self.probe_seed,
            "alpamayo_coc_seed": scene_seed,
            "alpamayo_source_tokens": self.keep_tokens,
            "alpamayo_hidden_dim": int(hidden.shape[-1]),
            "alpamayo_readout": readout_name,
            "alpamayo_expert_readout_layer": self.expert_readout_layer,
            "alpamayo_expert_num_layers": self.expert_num_layers,
            "alpamayo_expert_execution_mode": self.expert_execution_mode,
        }
        if self.last_kv_bridge is not None:
            result.update(
                {
                    "alpamayo_vlm_kv_layer": self.last_kv_bridge.layer_index,
                    "alpamayo_action_expert_layer": self.last_kv_bridge.layer_index,
                    "alpamayo_query_adapter_gate": self.last_kv_bridge.query_adapter.gate.detach()
                    .float()
                    .cpu()
                    .item(),
                }
            )
        if include_action_input_hidden:
            result["action_input_hidden_state"] = future_token_embeds[
                :, : self.keep_tokens
            ].to(dtype=torch.float16).cpu().contiguous()
        return result

    @torch.inference_mode()
    def extract_direct_trajectory(
        self,
        sample: dict[str, Any],
        action_dt_s: float = 0.1,
        target_dt_s: float = 0.5,
        target_horizon: int = 8,
    ) -> dict[str, Any]:
        """Run Alpamayo's official Action-Expert sampler and return NAVSIM poses."""
        if (
            self.expert_readout_layer is not None
            or self.expert_execution_mode != "full"
        ):
            raise RuntimeError("Direct trajectory sampling requires the full Action Expert")

        token = str(sample["token"])
        image_frames = load_camera_tensor(sample["image_paths"])
        camera_indices = torch.tensor(sample["camera_indices"], dtype=torch.int64)
        num_frames = int(sample.get("num_frames_per_camera", 4))
        messages = helper.create_message(
            frames=image_frames.flatten(0, 1),
            camera_indices=camera_indices,
            num_frames_per_camera=num_frames,
            nav_text=sample.get("nav_text"),
            use_nav_prompt=True,
        )
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": helper.to_device(dict(tokenized), self.device),
            "ego_history_xyz": sample["ego_history_xyz"].to(self.device),
            "ego_history_rot": sample["ego_history_rot"].to(self.device),
        }

        scene_seed = stable_token_seed(token, self.coc_seed)
        torch.manual_seed(scene_seed)
        torch.cuda.manual_seed_all(scene_seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=self.top_p,
                temperature=self.temperature,
                num_traj_samples=1,
                num_traj_sets=1,
                max_generation_length=self.max_generation_length,
                return_extra=True,
            )

        # Official shape: [B, num_sets, num_samples, 64, ...]. The first
        # future point is at 0.1 s, so 0.5 s steps are indices 4,9,...,39.
        xyz = pred_xyz[0, 0, 0]
        rot = pred_rot[0, 0, 0]
        indices = [
            int(round((step + 1) * target_dt_s / action_dt_s)) - 1
            for step in range(int(target_horizon))
        ]
        if indices[-1] >= xyz.shape[0]:
            raise ValueError(
                "Direct trajectory needs source index %d but only has %d points"
                % (indices[-1], xyz.shape[0])
            )
        xyz = xyz[indices]
        rot = rot[indices]
        yaw = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
        trajectory = torch.stack((xyz[..., 0], xyz[..., 1], yaw), dim=-1)

        cot = ""
        if isinstance(extra, dict) and "cot" in extra:
            cot_values = extra["cot"]
            try:
                cot = str(cot_values[0, 0, 0])
            except (IndexError, TypeError):
                cot = str(cot_values)
        return {
            "trajectory": trajectory.to(dtype=torch.float32).cpu().contiguous(),
            "alpamayo_cot": cot,
            "alpamayo_coc_seed": scene_seed,
            "alpamayo_direct_source_indices": indices,
            "alpamayo_action_dt_s": float(action_dt_s),
            "navsim_target_dt_s": float(target_dt_s),
            "alpamayo_num_inference_steps": int(self.model.diffusion.num_inference_steps),
            "alpamayo_sampling_temperature": 1.0,
            "alpamayo_readout": "official_action_expert_flow_trajectory_v1",
        }
