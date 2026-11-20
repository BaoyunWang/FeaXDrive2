# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Frozen-Vision, all-language-layer QKVO-LoRA replay for Safe-Context."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, Sequence

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence

from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from .multimodal_context import (
    pack_modalities,
    pool_fixed_groups,
    pool_visual_hidden,
    positions_between,
)


QKVO_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _single_batch_scalar(features: Dict[str, torch.Tensor], key: str) -> int:
    values = features[key].detach().reshape(-1).long()
    if values.numel() == 0 or not torch.all(values.eq(values[0])):
        raise ValueError(f"Batch must use one locked value for {key}")
    return int(values[0].item())


def configure_all_layer_qkvo_lora(
    language_model: torch.nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    expected_layers: int,
) -> torch.nn.Module:
    """Freeze a Qwen text model and add LoRA only to every layer's QKVO."""
    layers = getattr(language_model, "layers", None)
    if layers is None or len(layers) != int(expected_layers):
        actual = None if layers is None else len(layers)
        raise RuntimeError(
            f"Expected {expected_layers} Qwen language layers, got {actual}"
        )
    for parameter in language_model.parameters():
        parameter.requires_grad = False

    matched = [
        name
        for name, module in language_model.named_modules()
        if name.rsplit(".", 1)[-1] in QKVO_TARGETS
        and isinstance(module, torch.nn.Linear)
    ]
    expected_targets = int(expected_layers) * len(QKVO_TARGETS)
    if len(matched) != expected_targets:
        raise RuntimeError(
            f"Expected {expected_targets} language QKVO projections, got "
            f"{len(matched)}"
        )

    peft_model = get_peft_model(
        language_model,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            target_modules=list(QKVO_TARGETS),
            bias="none",
        ),
    )
    trainable = [
        name for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA parameters may remain trainable in Qwen")
    return peft_model


class AlpamayoLanguageLoRABackbone(torch.nn.Module):
    """Replay cached frozen-Vision inputs through the trainable LoRA language stack."""

    def __init__(
        self,
        *,
        model_path: str,
        vlm_path: str,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        expected_language_layers: int = 36,
        gradient_checkpointing: bool = True,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        model_root = Path(model_path).expanduser().resolve()
        config_path = model_root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing Alpamayo config: {config_path}")
        local_vlm = Path(vlm_path).expanduser().resolve()
        if not (local_vlm / "config.json").is_file():
            raise FileNotFoundError(f"Missing local VLM metadata: {local_vlm}")

        with config_path.open(encoding="utf-8") as stream:
            config_dict = json.load(stream)
        config_dict["vlm_name_or_path"] = str(local_vlm)
        model_config = Alpamayo1_5Config(**config_dict)
        full_model = Alpamayo1_5.from_pretrained(
            str(model_root),
            config=model_config,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        language_model = full_model.vlm.model.language_model
        self.hidden_size = int(language_model.config.hidden_size)
        self.num_hidden_layers = len(language_model.layers)
        self.language_model = configure_all_layer_qkvo_lora(
            language_model,
            rank=int(lora_rank),
            alpha=int(lora_alpha),
            dropout=float(lora_dropout),
            expected_layers=int(expected_language_layers),
        )
        self.language_model.config.use_cache = False
        if gradient_checkpointing:
            self.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # The Vision encoder, Action Expert, output head, and tokenizer remain
        # frozen/offline and are not retained on the training GPU.
        del full_model
        gc.collect()
        self.language_model.to(torch.device(device))

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

    def _pack_post_language(
        self,
        hidden: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_token_id = _single_batch_scalar(
            features, "llm_image_pad_token_id"
        )
        history_start = _single_batch_scalar(
            features, "llm_traj_history_start_token_id"
        )
        history_end = _single_batch_scalar(
            features, "llm_traj_history_end_token_id"
        )
        route_start = _single_batch_scalar(
            features, "llm_route_start_token_id"
        )
        route_end = _single_batch_scalar(
            features, "llm_route_end_token_id"
        )
        cot_start = _single_batch_scalar(features, "llm_cot_start_token_id")
        cot_end = _single_batch_scalar(features, "llm_cot_end_token_id")
        merge_size = _single_batch_scalar(
            features, "llm_spatial_merge_size"
        )
        visual_pool_size = _single_batch_scalar(
            features, "llm_visual_pool_size"
        )
        history_tokens = _single_batch_scalar(features, "llm_history_tokens")
        max_coc_tokens = _single_batch_scalar(
            features, "llm_max_coc_tokens"
        )

        packed_hidden = []
        packed_type_ids = []
        packed_position_ids = []
        coc_valid_values = []
        for batch_index in range(hidden.shape[0]):
            valid_length = int(
                features["llm_attention_mask"][batch_index].sum().item()
            )
            token_ids = features["llm_context_input_ids"][
                batch_index, :valid_length
            ].long()
            sample_hidden = hidden[batch_index, :valid_length]
            grid_count = int(
                features["llm_image_grid_valid_mask"][batch_index]
                .sum()
                .item()
            )
            image_grid_thw = features["llm_image_grid_thw"][
                batch_index, :grid_count
            ].long()
            vision_hidden = pool_visual_hidden(
                sample_hidden,
                token_ids,
                image_token_id=image_token_id,
                image_grid_thw=image_grid_thw,
                spatial_merge_size=merge_size,
                output_hw=visual_pool_size,
            )
            history_positions = positions_between(
                token_ids,
                history_start,
                history_end,
                label="trajectory-history",
            )
            if int(history_positions.numel()) != 48:
                raise RuntimeError(
                    "Safe-Context expects 48 trajectory-history tokens"
                )
            history_hidden = pool_fixed_groups(
                sample_hidden.index_select(0, history_positions),
                history_tokens,
            )
            route_positions = positions_between(
                token_ids,
                route_start,
                route_end,
                label="route",
                required=bool(
                    features["llm_route_required"][batch_index].item()
                ),
            )
            route_hidden = sample_hidden.index_select(0, route_positions)
            coc_positions = positions_between(
                token_ids,
                cot_start,
                cot_end,
                label="CoC",
                required=False,
                use_last_start=True,
            )
            coc_valid = bool(coc_positions.numel())
            if coc_valid:
                coc_positions = coc_positions[:max_coc_tokens]
            coc_hidden = sample_hidden.index_select(0, coc_positions)
            packed, type_ids, position_ids = pack_modalities(
                [vision_hidden, history_hidden, route_hidden, coc_hidden]
            )
            packed_hidden.append(packed)
            packed_type_ids.append(type_ids)
            packed_position_ids.append(position_ids)
            coc_valid_values.append(coc_valid)

        padded_hidden = pad_sequence(
            packed_hidden, batch_first=True, padding_value=0.0
        )
        type_ids = pad_sequence(
            packed_type_ids, batch_first=True, padding_value=-1
        )
        position_ids = pad_sequence(
            packed_position_ids, batch_first=True, padding_value=0
        )
        valid_mask = type_ids.ge(0)
        coc_valid = torch.tensor(
            coc_valid_values, dtype=torch.bool, device=hidden.device
        )
        return padded_hidden, type_ids, position_ids, valid_mask, coc_valid

    def _language_inputs(self, features):
        input_ids = features["llm_context_input_ids"].long()
        attention_mask = features["llm_attention_mask"].bool()
        visual_pos_mask = features["llm_visual_pos_mask"].bool()
        image_valid_mask = features["llm_image_valid_mask"].bool()
        image_embeds = features["llm_image_embeds"]
        deepstack = features["llm_deepstack_visual_embeds"]

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        compact_images = image_embeds[image_valid_mask].to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        if int(visual_pos_mask.sum().item()) != compact_images.shape[0]:
            raise RuntimeError("Cached image embeddings do not match visual mask")
        expanded_visual_mask = visual_pos_mask.unsqueeze(-1).expand_as(
            inputs_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            expanded_visual_mask,
            compact_images.reshape(-1),
        )

        deepstack_layers: Sequence[torch.Tensor] = tuple(
            deepstack[:, layer][image_valid_mask].to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            for layer in range(deepstack.shape[1])
        )
        return inputs_embeds, attention_mask, visual_pos_mask, deepstack_layers

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs_embeds, attention_mask, visual_pos_mask, deepstack_layers = self._language_inputs(features)
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=features["llm_position_ids"].long().permute(1, 0, 2),
            use_cache=False,
            visual_pos_masks=visual_pos_mask,
            deepstack_visual_embeds=list(deepstack_layers),
        )
        return self._pack_post_language(outputs.last_hidden_state, features)
