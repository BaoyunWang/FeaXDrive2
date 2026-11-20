# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Single-layer Alpamayo Action-Expert bridge for the E2 experiment.

This module runs in the dedicated Python-3.12 ``alpamayo`` environment.  It
keeps only the final pretrained Action-Expert decoder block, whose original
layer index matches the final VLM KV-cache slot, and exposes its hidden state
before Alpamayo's ``action_out_proj``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Small dependency-free RMSNorm used by the trainable query adapter."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(hidden_dim)))
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden.dtype
        normalized = hidden.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(input_dtype) * self.weight.to(input_dtype)


class ResidualActionQueryAdapter(nn.Module):
    """Adapt Action-Expert input queries without changing the initial function.

    The final official expert block was trained on the output of the preceding
    35 blocks, not directly on ``action_in_proj`` embeddings.  This residual
    adapter learns that depth alignment.  Its scalar gate starts at zero, so an
    untrained E2 bridge is exactly ``AE35(action_in_proj(x, t), KV35)``.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 2,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        inner_dim = hidden_dim * int(expansion)
        self.norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, hidden_dim),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = self.mlp(self.norm(hidden))
        return hidden + self.gate.to(hidden.dtype) * residual


class LastKVActionExpertBridge(nn.Module):
    """Run one official Action-Expert block against its matching final VLM KV.

    Construction intentionally prunes ``alpamayo_model.expert.layers`` to the
    selected final block *after* pretrained weights have been loaded.  The
    block retains its original internal ``layer_idx`` (normally 35), so
    Transformers reads only KV slot 35 even though the generated cache object
    still contains every VLM layer.

    The Alpamayo VLM and ``action_in_proj`` remain frozen.  The query adapter,
    AE35 block, and expert final norm remain trainable so a later bridge-fit
    stage can depth-align the query while preserving the pretrained expert.
    """

    def __init__(
        self,
        alpamayo_model: nn.Module,
        layer_index: int = -1,
        query_adapter_expansion: int = 2,
        freeze_action_in_proj: bool = True,
        freeze_expert: bool = False,
    ) -> None:
        super().__init__()
        expert = getattr(alpamayo_model, "expert", None)
        expert_layers = getattr(expert, "layers", None)
        if expert is None or expert_layers is None or len(expert_layers) == 0:
            raise AttributeError("Alpamayo model has no non-empty expert.layers")

        self.original_num_layers = int(len(expert_layers))
        normalized_index = int(layer_index)
        if normalized_index < 0:
            normalized_index += self.original_num_layers
        if not 0 <= normalized_index < self.original_num_layers:
            raise ValueError(
                "layer_index must be in [%d, %d], got %d"
                % (0, self.original_num_layers - 1, layer_index)
            )
        if normalized_index != self.original_num_layers - 1:
            raise ValueError(
                "E2 requires the final Action-Expert block; got layer %d of %d"
                % (normalized_index, self.original_num_layers)
            )

        selected_layer = expert_layers[normalized_index]
        internal_index = self._infer_internal_layer_index(selected_layer)
        if internal_index is not None and internal_index != normalized_index:
            raise ValueError(
                "Selected expert block reports layer_idx=%d, expected %d"
                % (internal_index, normalized_index)
            )

        # The pretrained model has already been loaded, so removing the unused
        # expert blocks is safe and materially reduces extraction memory.
        expert.layers = nn.ModuleList([selected_layer])
        self.expert = expert
        self.action_in_proj = alpamayo_model.action_in_proj
        self.layer_index = normalized_index

        hidden_dim = int(getattr(expert.config, "hidden_size"))
        self.query_adapter = ResidualActionQueryAdapter(
            hidden_dim=hidden_dim,
            expansion=query_adapter_expansion,
        )

        reference_parameter = next(selected_layer.parameters())
        self.query_adapter.to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        if freeze_action_in_proj:
            self.action_in_proj.requires_grad_(False)
        if freeze_expert:
            self.expert.requires_grad_(False)

    @staticmethod
    def _infer_internal_layer_index(layer: nn.Module) -> int | None:
        attention = getattr(layer, "self_attn", None)
        value = getattr(attention, "layer_idx", None)
        return None if value is None else int(value)

    @staticmethod
    def _cache_depth(prompt_cache: Any) -> int | None:
        for attribute in ("layers", "key_cache"):
            value = getattr(prompt_cache, attribute, None)
            if value is not None:
                try:
                    return len(value)
                except TypeError:
                    pass
        if isinstance(prompt_cache, (tuple, list)):
            return len(prompt_cache)
        return None

    def _validate_cache(self, prompt_cache: Any) -> None:
        cache_depth = self._cache_depth(prompt_cache)
        if cache_depth is not None and cache_depth <= self.layer_index:
            raise ValueError(
                "E2 needs VLM KV slot %d, but cache depth is %d"
                % (self.layer_index, cache_depth)
            )

    @staticmethod
    def _cache_seq_length(prompt_cache: Any) -> int | None:
        get_seq_length = getattr(prompt_cache, "get_seq_length", None)
        if get_seq_length is None:
            return None
        try:
            return int(get_seq_length())
        except TypeError:
            return int(get_seq_length(0))

    @staticmethod
    def _restore_cache(prompt_cache: Any, sequence_length: int | None) -> None:
        crop = getattr(prompt_cache, "crop", None)
        if crop is not None and sequence_length is not None:
            crop(sequence_length)

    def load_query_adapter(self, checkpoint_path: str) -> None:
        """Load an E2 query-adapter checkpoint without touching official weights."""
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(checkpoint, dict) and "query_adapter" in checkpoint:
            checkpoint = checkpoint["query_adapter"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError("E2 query-adapter checkpoint must contain a state dict")

        prefix = "query_adapter."
        if any(str(key).startswith(prefix) for key in checkpoint):
            checkpoint = {
                str(key)[len(prefix) :]: value
                for key, value in checkpoint.items()
                if str(key).startswith(prefix)
            }
        self.query_adapter.load_state_dict(checkpoint, strict=True)

    def load_bridge(self, checkpoint_path: str) -> None:
        """Load a complete fitted bridge checkpoint, including AE35 and final norm."""
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(checkpoint, dict) and "bridge" in checkpoint:
            checkpoint = checkpoint["bridge"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError("E2 bridge checkpoint must contain a state dict")

        for prefix in ("last_kv_bridge.", "bridge."):
            if any(str(key).startswith(prefix) for key in checkpoint):
                checkpoint = {
                    str(key)[len(prefix) :]: value
                    for key, value in checkpoint.items()
                    if str(key).startswith(prefix)
                }
                break
        self.load_state_dict(checkpoint, strict=True)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        prompt_cache: Any,
        attention_mask: torch.Tensor,
        is_causal: bool = False,
        return_queries: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final-block hidden states aligned with the action-token axis."""
        if noisy_actions.ndim < 3:
            raise ValueError(
                "Expected noisy actions with [B,T,...], got %s"
                % (tuple(noisy_actions.shape),)
            )
        self._validate_cache(prompt_cache)
        action_tokens = int(noisy_actions.shape[1])
        raw_queries = self.action_in_proj(noisy_actions, timesteps)
        if raw_queries.dim() == 2:
            raw_queries = raw_queries.view(noisy_actions.shape[0], action_tokens, -1)
        adapted_queries = self.query_adapter(raw_queries)

        prefix_length = self._cache_seq_length(prompt_cache)
        try:
            expert_output = self.expert(
                inputs_embeds=adapted_queries,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                is_causal=bool(is_causal),
            )
        finally:
            # Match Alpamayo's official denoising loop: action-token KV must not
            # leak into a later probe or Flow step.
            self._restore_cache(prompt_cache, prefix_length)

        hidden = expert_output.last_hidden_state[:, -action_tokens:]
        if return_queries:
            return hidden, raw_queries, adapted_queries
        return hidden
