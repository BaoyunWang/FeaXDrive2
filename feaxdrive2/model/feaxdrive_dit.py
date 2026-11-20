# FeaXDrive2 release: extracted or reorganized from the research implementation.
# ------------------------------------------------------------------------
# Modified from LightningDiT(https://github.com/hustvl/LightningDiT)
# Copyright (c) Xiaomi Corporation. All rights reserved.
# ------------------------------------------------------------------------
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np

from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)

from .blocks.rmsnorm import RMSNorm
from .blocks.attention import Attention
from .blocks.rope import RotaryEmbedding
from .blocks.encoder import SwiGLUFFN


class TimestepEncoder(nn.Module):
    """Encodes scalar timesteps into a high-dimensional vector."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = self.timestep_embedder.linear_1.weight.dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb


class FinalLayer(nn.Module):
    """
    The final output layer of the DiT model.

    Encapsulates the final adaptive layer normalization and the projection to
    the output dimension.
    """
    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, output_dim)
        self.modulation_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def modulate(self, x, shift, scale):
        if shift is None:
            return x * (1 + scale.unsqueeze(1))
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor,
                geometry_modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        params = self.modulation_proj(conditioning)
        if geometry_modulation is not None:
            params = params + geometry_modulation
        shift, scale = params.chunk(2, dim=1)
        x = self.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        if norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(dim)

        self.attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
        )

        if norm_type == "layer_norm":
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm2 = RMSNorm(dim)

        self.ffn = SwiGLUFFN(
            dim,
            bias=True,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def modulate(self, x, shift, scale):
        if shift is None:
            return x * (1 + scale.unsqueeze(1))
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        rotary_embedder: Optional[nn.Module] = None,
        geometry_modulation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mod_params = self.adaLN_modulation(conditioning)
        if geometry_modulation is not None:
            mod_params = mod_params + geometry_modulation
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = \
            mod_params.chunk(6, dim=1)

        normed_states = self.norm1(hidden_states)
        modulated_states = self.modulate(normed_states, shift_attn, scale_attn)

        attn_output = self.attn(
            modulated_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            rotary_embedder=rotary_embedder
        )
        hidden_states = hidden_states + gate_attn.unsqueeze(1) * attn_output

        normed_states = self.norm2(hidden_states)
        modulated_states = self.modulate(normed_states, shift_ffn, scale_ffn)
        ffn_output = self.ffn(modulated_states)

        hidden_states = hidden_states + gate_ffn.unsqueeze(1) * ffn_output

        return hidden_states


class SparseSemanticCrossAttention(nn.Module):
    """One-way semantic injection from frozen/source tokens into trajectory tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        attention_bias: bool = False,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = RMSNorm(dim, eps=norm_eps)
        self.attn = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=dim,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        gate_enabled: bool,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        rotary_embedder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        residual = self.attn(
            self.norm(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            rotary_embedder=rotary_embedder,
        )
        # Both ablations execute the identical module graph. Multiplying the
        # parameter by zero keeps M1 structurally identical to M3 while making
        # the semantic residual exactly inactive.
        gate = torch.tanh(self.gate)
        if not gate_enabled:
            gate = gate * 0.0
        return hidden_states + gate * residual


class LightningDiT(nn.Module):

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 48,
        output_dim: int = 512,
        num_layers: int = 16,
        dropout: float = 0.0,
        attention_bias: bool = True,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        interleave_attention: bool = True,
        semantic_fusion_mode: str = "current",
        sparse_cross_attention_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.output_dim = output_dim
        self.interleave_attention = interleave_attention
        self.semantic_fusion_mode = str(semantic_fusion_mode).lower()
        if self.semantic_fusion_mode not in {
            "current",
            "condition_only",
            "sparse_gated",
            "sparse_interleaved",
        }:
            raise ValueError(
                "semantic_fusion_mode must be one of 'current', "
                "'condition_only', 'sparse_gated', or "
                "'sparse_interleaved', got "
                f"{semantic_fusion_mode!r}"
            )

        sparse_layers = [5, 10] if sparse_cross_attention_layers is None else list(
            sparse_cross_attention_layers
        )
        if len(set(sparse_layers)) != len(sparse_layers):
            raise ValueError("sparse_cross_attention_layers must not contain duplicates")
        if any(layer < 0 or layer >= num_layers for layer in sparse_layers):
            raise ValueError(
                "sparse_cross_attention_layers must contain zero-based DiT layer "
                f"indices in [0, {num_layers - 1}], got {sparse_layers}"
            )
        self.sparse_cross_attention_layers = tuple(sorted(sparse_layers))

        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim
        )

        self.rotary_embedder = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=8,
        )

        use_current_fusion = self.semantic_fusion_mode == "current"
        if use_current_fusion:
            self.main_cross_attention_layers = tuple(
                idx
                for idx in range(num_layers)
                if idx % 2 != 0 or not interleave_attention
            )
        elif self.semantic_fusion_mode == "sparse_interleaved":
            self.main_cross_attention_layers = self.sparse_cross_attention_layers
        else:
            self.main_cross_attention_layers = ()

        self.transformer_blocks = nn.ModuleList([
            LightningDiTBlock(
                dim=self.inner_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                attention_bias=attention_bias,
                norm_type=norm_type,
                norm_eps=norm_eps,
                cross_attention_dim=(
                    self.inner_dim
                    if idx in self.main_cross_attention_layers
                    else None
                )
            ) for idx in range(num_layers)
        ])

        self.sparse_cross_attention = nn.ModuleDict()
        if self.semantic_fusion_mode in {"condition_only", "sparse_gated"}:
            self.sparse_cross_attention = nn.ModuleDict({
                str(idx): SparseSemanticCrossAttention(
                    dim=self.inner_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                )
                for idx in self.sparse_cross_attention_layers
            })

        self.final_layer = FinalLayer(self.inner_dim, output_dim)

        self._initialize_weights()

    def semantic_gate_values(self) -> dict[str, torch.Tensor]:
        """Return effective sparse semantic gates keyed by zero-based layer index."""
        gate_enabled = self.semantic_fusion_mode == "sparse_gated"
        return {
            layer: torch.tanh(block.gate) if gate_enabled else block.gate * 0.0
            for layer, block in self.sparse_cross_attention.items()
        }

    def _initialize_weights(self) -> None:
        """
        Initializes weights for stable training.
        - Initializes positional embeddings with sine/cosine values.
        - Zeroes out the weights of modulation and final output layers.
        """
        def zero_out_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        for block in self.transformer_blocks:
            block.adaLN_modulation.apply(zero_out_init)
        self.final_layer.modulation_proj.apply(zero_out_init)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        conditioning_features: torch.Tensor,
        timesteps: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        vehicle_geometry: Optional[torch.Tensor] = None,
        camera_context: Optional[dict] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the UnifiedDiT model.

        Args:
            hidden_states (torch.Tensor): Input sequence. Shape: (B, N, D).
            encoder_hidden_states (torch.Tensor): Context sequence for cross-attention.
            conditioning_features (torch.Tensor): Additional conditioning features.
            timesteps (torch.LongTensor): Diffusion timesteps.
            return_hidden_states (bool): If True, returns the output and a list of
                all intermediate hidden states.

        Returns:
            The final output tensor, or a tuple of the output and all hidden states.
        """
        seq_len = hidden_states.shape[1]
        # hidden_states = hidden_states + self.pos_embed[:, :seq_len, :]

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        conditioning_features = conditioning_features.contiguous()

        time_embedding = self.timestep_encoder(timesteps)
        conditioning = time_embedding + conditioning_features
        geometry = None
        body = None
        has_geometry = hasattr(self, "geometry_conditioner")
        has_query = hasattr(self, "footprint_query")
        has_camera = hasattr(self, "camera_grounded_query")
        if sum((has_geometry, has_query, has_camera)) > 1:
            raise ValueError("Embodiment branches are mutually exclusive")
        if has_geometry or has_query or has_camera:
            if vehicle_geometry is None or vehicle_geometry.shape != (hidden_states.shape[0], 3):
                raise ValueError("E2 requires explicit vehicle_geometry [B,3]")
        if has_geometry:
            geometry = self.geometry_conditioner(vehicle_geometry)
        if has_query:
            # Keep this outside frozen/no_grad condition encoding: the new scene
            # retrieval has trainable parameters. Reuse only within this forward.
            body = self.footprint_query(vehicle_geometry, encoder_hidden_states,
                                        encoder_attention_mask)
        if has_camera:
            if camera_context is None:
                raise ValueError("Camera Query requires explicit spatial features and causal history")
            body = self.camera_grounded_query(vehicle_geometry, encoder_hidden_states, **camera_context)

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            use_cross_attention = idx in self.main_cross_attention_layers
            current_encoder_states = encoder_hidden_states if use_cross_attention else None

            hidden_states = block(
                hidden_states,
                conditioning=conditioning,
                encoder_hidden_states=current_encoder_states,
                encoder_attention_mask=(
                    encoder_attention_mask if use_cross_attention else None
                ),
                rotary_embedder=self.rotary_embedder,
                geometry_modulation=geometry[0][idx] if geometry is not None else None,
            )
            sparse_key = str(idx)
            if sparse_key in self.sparse_cross_attention:
                sparse_block = self.sparse_cross_attention[sparse_key]
                hidden_states = sparse_block(
                    hidden_states,
                    encoder_hidden_states,
                    gate_enabled=self.semantic_fusion_mode == "sparse_gated",
                    encoder_attention_mask=encoder_attention_mask,
                    rotary_embedder=self.rotary_embedder,
                )
            if body is not None:
                branch = self.camera_grounded_query if has_camera else self.footprint_query
                hidden_states = branch.inject(idx, hidden_states, body)
            all_hidden_states.append(hidden_states)

        output = self.final_layer(hidden_states, conditioning,
                                  geometry[1] if geometry is not None else None)

        return (output, all_hidden_states) if return_hidden_states else output
