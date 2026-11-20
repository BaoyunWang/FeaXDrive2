# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""FeaXDrive Flow planner conditioned by cached Alpamayo 1.5 representations."""

from typing import Optional, Sequence

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from .flow import (
    FlowPlanner,
    FlowPlannerConfig,
)
from .conditioning import (
    AlpamayoMultimodalAdapter,
    AlpamayoPlanningAdapter,
    AlpamayoSemanticAdapter,
)


class FeaXDriveAlpamayoDiffusionPlanner(FlowPlanner):
    """Same FeaXDrive planner core; only the foundation-feature adapter changes."""

    def __init__(
        self,
        config: FlowPlannerConfig,
        alpamayo_hidden_dim: int = 2048,
        alpamayo_semantic_hidden_dim: int = 4096,
        alpamayo_multimodal_hidden_dim: int = 4096,
        alpamayo_multimodal_token_budget: Sequence[int] = (16, 4, 4, 8),
        alpamayo_multimodal_use_gated_global_fusion: bool = True,
        alpamayo_source_tokens: int = 40,
        alpamayo_adapter_dropout: float = 0.0,
        alpamayo_conditioning_source: str = "action_expert",
    ):
        self.alpamayo_conditioning_source = str(alpamayo_conditioning_source).lower()
        planning_sources = {"action_expert", "last_kv_action_expert"}
        if self.alpamayo_conditioning_source not in planning_sources | {
            "semantic",
            "multimodal",
        }:
            raise ValueError(
                "alpamayo_conditioning_source must be 'action_expert', "
                "'last_kv_action_expert', or "
                "'semantic', or 'multimodal', got "
                f"{alpamayo_conditioning_source!r}"
            )
        if self.alpamayo_conditioning_source == "multimodal":
            # E3 supplies a balanced modality-global anchor below instead of
            # the base class's raw mean over every condition token.
            config.use_condition_mean_fusion = False
        super().__init__(config)
        if self.alpamayo_conditioning_source in planning_sources:
            self.feature_encoder = AlpamayoPlanningAdapter(
                input_dim=int(alpamayo_hidden_dim),
                output_dim=int(config.input_embedding_dim),
                action_horizon=int(config.action_horizon),
                source_tokens=int(alpamayo_source_tokens),
                dropout=float(alpamayo_adapter_dropout),
            )
        elif self.alpamayo_conditioning_source == "semantic":
            self.feature_encoder = AlpamayoSemanticAdapter(
                input_dim=int(alpamayo_semantic_hidden_dim),
                output_dim=int(config.input_embedding_dim),
                dropout=float(alpamayo_adapter_dropout),
            )
        else:
            self.feature_encoder = AlpamayoMultimodalAdapter(
                input_dim=int(alpamayo_multimodal_hidden_dim),
                output_dim=int(config.input_embedding_dim),
                token_budget=alpamayo_multimodal_token_budget,
                dropout=float(alpamayo_adapter_dropout),
            )
            self.multimodal_use_gated_global_fusion = bool(
                alpamayo_multimodal_use_gated_global_fusion
            )
            if self.multimodal_use_gated_global_fusion:
                width = int(config.input_embedding_dim)
                self.multimodal_global_projector = nn.Linear(4 * width, width)
                self.multimodal_global_gate = nn.Parameter(torch.zeros(()))

    def _encode_vl_conditioning(self, vl_features, action_input=None):
        if self.alpamayo_conditioning_source == "multimodal":
            raise RuntimeError(
                "Multimodal conditioning requires type/position metadata; "
                "call _prepare_vl_conditioning instead"
            )
        del action_input
        return self.feature_encoder(vl_features)

    def _prepare_vl_conditioning(
        self,
        vl_features: torch.Tensor,
        action_input: Optional[BatchFeature] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.alpamayo_conditioning_source == "multimodal":
            if action_input is None:
                raise ValueError("E3 multimodal conditioning requires action_input")
            required = ("multimodal_type_ids", "multimodal_position_ids")
            missing = [key for key in required if key not in action_input]
            if missing:
                raise KeyError(
                    "E3 multimodal action_input missing keys %s" % missing
                )
            valid_mask = action_input.get("multimodal_valid_mask")
            if valid_mask is None:
                valid_mask = action_input.multimodal_type_ids.ge(0)
            encoded = self.feature_encoder(
                vl_features,
                action_input.multimodal_type_ids,
                action_input.multimodal_position_ids,
                valid_mask,
            )
            return encoded, None

        if self.alpamayo_conditioning_source != "semantic":
            return self.feature_encoder(vl_features), None

        # Semantic sequences are padded with exact zeros by the training
        # collate function. Mask those positions in every DiT cross-attention.
        valid_tokens = vl_features.ne(0).any(dim=-1)
        missing_condition = ~valid_tokens.any(dim=1)
        # A zero sentinel denotes Alpamayo generations without a valid closed
        # CoC. Its projected bias acts as a learned null condition; truncated
        # generated text is deliberately not injected.
        if bool(missing_condition.any()):
            valid_tokens = valid_tokens.clone()
            valid_tokens[missing_condition, 0] = True
        encoded = self.feature_encoder(vl_features)
        encoded = encoded * valid_tokens.unsqueeze(-1).to(encoded.dtype)
        return encoded, valid_tokens

    def _fuse_trajectory_inputs(
        self,
        history_features: torch.Tensor,
        condition_features: torch.Tensor,
        action_features: torch.Tensor,
        condition_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.alpamayo_conditioning_source != "multimodal":
            return super()._fuse_trajectory_inputs(
                history_features,
                condition_features,
                action_features,
                condition_attention_mask=condition_attention_mask,
            )

        fused = self.fusion_projector(
            torch.cat([history_features, action_features], dim=2)
        )
        if not self.multimodal_use_gated_global_fusion:
            return fused
        expected_tokens = self.feature_encoder.output_tokens
        if condition_features.shape[1] != expected_tokens:
            raise ValueError(
                "Expected %d E3 condition tokens, got %d"
                % (expected_tokens, condition_features.shape[1])
            )
        if condition_attention_mask is not None and (
            condition_attention_mask.shape != condition_features.shape[:2]
        ):
            raise ValueError(
                "condition_attention_mask must match condition token axes"
            )
        modality_means = []
        for group_slice in self.feature_encoder.group_slices:
            group = condition_features[:, group_slice]
            plain_mean = group.mean(dim=1)
            if condition_attention_mask is None:
                modality_means.append(plain_mean)
                continue
            group_mask = condition_attention_mask[:, group_slice].to(
                device=group.device, dtype=torch.bool
            )
            weights = group_mask.unsqueeze(-1).to(dtype=group.dtype)
            valid_count = weights.sum(dim=1)
            masked_mean = (group * weights).sum(dim=1) / valid_count.clamp_min(1.0)
            masked_mean = torch.where(
                valid_count.gt(0), masked_mean, torch.zeros_like(masked_mean)
            )
            # Retain the original reduction for fully valid samples so the
            # established Safe-Context computation stays unchanged.
            all_valid = group_mask.all(dim=1, keepdim=True)
            modality_means.append(
                torch.where(all_valid, plain_mean, masked_mean)
            )
        global_condition = self.multimodal_global_projector(
            torch.cat(modality_means, dim=-1)
        )
        return fused + torch.tanh(self.multimodal_global_gate) * global_condition.unsqueeze(1)
