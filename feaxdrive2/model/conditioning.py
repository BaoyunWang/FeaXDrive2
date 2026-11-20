# FeaXDrive2 release: extracted or reorganized from the research implementation.
"""Learnable adapters from Alpamayo hidden states to FeaXDrive tokens."""

from typing import Sequence

import torch
from torch import nn

from .blocks.attention import Attention


class AlpamayoPlanningAdapter(nn.Module):
    """Convert cached Action-Expert hidden states into 8 FeaXDrive planning tokens.

    Input:
      [B, K, T, C] where K is deterministic action probes, T is usually 40
      (the first 4.0 s of Alpamayo's 10 Hz action slots), and C=2048.

    The adapter intentionally avoids global average pooling:
      1. learn attention over K probes independently at every future slot;
      2. learn attention inside each 0.5 s temporal window (5 x 0.1 s slots);
      3. project each aligned future token into the FeaXDrive DiT width.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 384,
        action_horizon: int = 8,
        source_tokens: int = 40,
        dropout: float = 0.0,
    ):
        super().__init__()
        if source_tokens % action_horizon != 0:
            raise ValueError(
                "source_tokens (%d) must be divisible by action_horizon (%d)"
                % (source_tokens, action_horizon)
            )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.action_horizon = int(action_horizon)
        self.source_tokens = int(source_tokens)
        self.tokens_per_action = self.source_tokens // self.action_horizon

        self.probe_norm = nn.LayerNorm(self.input_dim)
        self.probe_score = nn.Linear(self.input_dim, 1, bias=False)

        self.temporal_norm = nn.LayerNorm(self.input_dim)
        self.temporal_score = nn.Linear(self.input_dim, 1, bias=False)

        self.proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim),
        )

        # Start close to uniform attention but keep the aggregation learnable.
        nn.init.zeros_(self.probe_score.weight)
        nn.init.zeros_(self.temporal_score.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            # Backward-compatible K=1 cache: [B,T,C] -> [B,1,T,C]
            hidden = hidden.unsqueeze(1)
        if hidden.ndim != 4:
            raise ValueError("Expected [B,K,T,C] Alpamayo hidden, got %s" % (tuple(hidden.shape),))

        b, _, t, c = hidden.shape
        if c != self.input_dim:
            raise ValueError("Expected hidden dim %d, got %d" % (self.input_dim, c))
        if t < self.source_tokens:
            raise ValueError("Need at least %d source tokens, got %d" % (self.source_tokens, t))
        hidden = hidden[:, :, : self.source_tokens, :]

        # Probe attention: [B,K,T,C] -> [B,T,C]
        probe_logits = self.probe_score(self.probe_norm(hidden)).squeeze(-1)
        probe_weights = torch.softmax(probe_logits.float(), dim=1).to(hidden.dtype)
        hidden = (hidden * probe_weights.unsqueeze(-1)).sum(dim=1)

        # Temporal alignment: 40 x 0.1 s -> 8 x 0.5 s.
        hidden = hidden.reshape(
            b, self.action_horizon, self.tokens_per_action, self.input_dim
        )
        temporal_logits = self.temporal_score(self.temporal_norm(hidden)).squeeze(-1)
        temporal_weights = torch.softmax(temporal_logits.float(), dim=2).to(hidden.dtype)
        hidden = (hidden * temporal_weights.unsqueeze(-1)).sum(dim=2)

        return self.proj(hidden)


class AlpamayoSemanticAdapter(nn.Module):
    """Project contextual Alpamayo CoC tokens without temporal/action pooling."""

    def __init__(
        self,
        input_dim: int = 4096,
        output_dim: int = 384,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(
                "Expected [B,T,C] Alpamayo semantic hidden, got %s"
                % (tuple(hidden.shape),)
            )
        if hidden.shape[-1] != self.input_dim:
            raise ValueError(
                "Expected semantic hidden dim %d, got %d"
                % (self.input_dim, hidden.shape[-1])
            )
        return self.proj(hidden)


class MaskedQueryResampler(nn.Module):
    """Compress one padded source modality into a fixed learned-query budget."""

    def __init__(
        self,
        dim: int,
        num_queries: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")

        self.dim = int(dim)
        self.num_queries = int(num_queries)
        self.queries = nn.Parameter(torch.empty(self.num_queries, self.dim))
        self.null_context = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.query_norm = nn.LayerNorm(self.dim)
        self.context_norm = nn.LayerNorm(self.dim)
        self.attn = Attention(
            query_dim=self.dim,
            cross_attention_dim=self.dim,
            heads=int(num_heads),
            dim_head=self.dim // int(num_heads),
            dropout=float(dropout),
            bias=True,
        )
        self.ffn_norm = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * self.dim, self.dim),
        )
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(
        self,
        context: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError(
                "Expected context [B,N,D], got %s" % (tuple(context.shape),)
            )
        if valid_mask.shape != context.shape[:2]:
            raise ValueError(
                "valid_mask shape %s does not match context %s"
                % (tuple(valid_mask.shape), tuple(context.shape[:2]))
            )

        valid_mask = valid_mask.to(device=context.device, dtype=torch.bool)
        batch_size = context.shape[0]
        missing = ~valid_mask.any(dim=1, keepdim=True)
        null_context = self.null_context.to(dtype=context.dtype).expand(
            batch_size, -1, -1
        )
        context = torch.cat([context, null_context], dim=1)
        valid_mask = torch.cat([valid_mask, missing], dim=1)

        queries = self.queries.to(dtype=context.dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        queries = queries + self.attn(
            self.query_norm(queries),
            encoder_hidden_states=self.context_norm(context),
            attention_mask=valid_mask,
        )
        return queries + self.ffn(self.ffn_norm(queries))


class AlpamayoMultimodalAdapter(nn.Module):
    """Resample full Alpamayo VLM context into 32 balanced condition tokens.

    The packed source sequence carries four deterministic modality ids:
      0 vision, 1 ego-history, 2 route/navigation, 3 generated CoC.
    Each modality has its own normalization and learned query bank while the
    expensive 4096-to-planner-width projection is shared.
    """

    NUM_MODALITIES = 4

    def __init__(
        self,
        input_dim: int = 4096,
        output_dim: int = 384,
        token_budget: Sequence[int] = (16, 4, 4, 8),
        max_position_embeddings: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        budgets = tuple(int(value) for value in token_budget)
        if len(budgets) != self.NUM_MODALITIES or any(value <= 0 for value in budgets):
            raise ValueError(
                "token_budget must contain four positive values, got %s"
                % (budgets,)
            )
        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.token_budget = budgets
        self.max_position_embeddings = int(max_position_embeddings)
        self.modality_norms = nn.ModuleList(
            [nn.LayerNorm(self.input_dim) for _ in range(self.NUM_MODALITIES)]
        )
        self.source_proj = nn.Linear(self.input_dim, self.output_dim)
        self.modality_embedding = nn.Embedding(
            self.NUM_MODALITIES, self.output_dim
        )
        self.position_embedding = nn.Embedding(
            self.max_position_embeddings, self.output_dim
        )
        self.resamplers = nn.ModuleList(
            [
                MaskedQueryResampler(
                    dim=self.output_dim,
                    num_queries=budget,
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                )
                for budget in self.token_budget
            ]
        )

    @property
    def output_tokens(self) -> int:
        return sum(self.token_budget)

    @property
    def group_slices(self) -> tuple[slice, ...]:
        result = []
        start = 0
        for budget in self.token_budget:
            result.append(slice(start, start + budget))
            start += budget
        return tuple(result)

    def forward(
        self,
        hidden: torch.Tensor,
        modality_ids: torch.Tensor,
        position_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(
                "Expected hidden [B,N,C], got %s" % (tuple(hidden.shape),)
            )
        if hidden.shape[-1] != self.input_dim:
            raise ValueError(
                "Expected hidden dim %d, got %d"
                % (self.input_dim, hidden.shape[-1])
            )
        expected_shape = hidden.shape[:2]
        for name, value in (
            ("modality_ids", modality_ids),
            ("position_ids", position_ids),
            ("valid_mask", valid_mask),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    "%s shape %s does not match hidden %s"
                    % (name, tuple(value.shape), tuple(expected_shape))
                )

        modality_ids = modality_ids.to(device=hidden.device, dtype=torch.long)
        position_ids = position_ids.to(device=hidden.device, dtype=torch.long)
        valid_mask = valid_mask.to(device=hidden.device, dtype=torch.bool)
        valid_mask = valid_mask & modality_ids.ge(0)

        if bool((modality_ids[valid_mask] >= self.NUM_MODALITIES).any()):
            raise ValueError("Valid modality ids must be in [0, 3]")
        valid_positions = position_ids[valid_mask]
        if bool((valid_positions < 0).any()) or bool(
            (valid_positions >= self.max_position_embeddings).any()
        ):
            raise ValueError(
                "Valid position ids must be in [0, %d]"
                % (self.max_position_embeddings - 1)
            )

        normalized = torch.zeros_like(hidden)
        for modality, norm in enumerate(self.modality_norms):
            modality_mask = valid_mask & modality_ids.eq(modality)
            # Execute every expert on every batch so DDP with
            # find_unused_parameters=False remains valid even when an entire
            # batch has no closed CoC. The mask keeps non-matching tokens zero.
            normalized = normalized + norm(hidden) * modality_mask.unsqueeze(-1).to(
                hidden.dtype
            )

        encoded = self.source_proj(normalized)
        safe_modality_ids = modality_ids.clamp(min=0, max=self.NUM_MODALITIES - 1)
        safe_position_ids = position_ids.clamp(
            min=0, max=self.max_position_embeddings - 1
        )
        encoded = (
            encoded
            + self.modality_embedding(safe_modality_ids)
            + self.position_embedding(safe_position_ids)
        )
        encoded = encoded * valid_mask.unsqueeze(-1).to(encoded.dtype)

        outputs = []
        for modality, resampler in enumerate(self.resamplers):
            modality_mask = valid_mask & modality_ids.eq(modality)
            outputs.append(resampler(encoded, modality_mask))
        return torch.cat(outputs, dim=1)
