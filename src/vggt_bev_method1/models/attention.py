from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiheadAttention(nn.Module):
    """Exact or global linearized multi-head attention.

    Linear mode preserves all-query/all-token connectivity without materializing
    the quadratic attention matrix, making one query per 512x512 BEV cell viable.
    """

    def __init__(self, hidden_dim: int, heads: int, *, mode: str = "linear") -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if mode not in ("linear", "exact"):
            raise ValueError("attention mode must be 'linear' or 'exact'")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.mode = mode
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def _split(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(context))
        v = self._split(self.v_proj(context))
        if self.mode == "exact":
            output = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = self.head_dim**-0.25
            q_feature = F.elu(q * scale) + 1.0
            k_feature = F.elu(k * scale) + 1.0
            key_value = torch.einsum("bhmd,bhme->bhde", k_feature, v)
            key_sum = k_feature.sum(dim=2)
            denominator = torch.einsum(
                "bhqd,bhd->bhq", q_feature, key_sum
            ).clamp_min(1e-6)
            output = torch.einsum(
                "bhqd,bhde,bhq->bhqe",
                q_feature,
                key_value,
                denominator.reciprocal(),
            )
        output = output.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], self.hidden_dim
        )
        return self.out_proj(output)


class DeformableCrossAttention(nn.Module):
    """PDF Method I deformable cross-attention over spatial token pyramids.

    Each BEV query predicts a small set of image sampling locations. Sampling
    is performed at every VGGT feature level and every input frame, then
    aggregated without constructing a Q-by-image-token attention matrix.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        *,
        levels: int,
        samples: int,
        query_chunk_size: int,
        maximum_offset: float = 0.75,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if levels <= 0 or samples <= 0 or query_chunk_size <= 0:
            raise ValueError("levels, samples, and query_chunk_size must be positive")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.levels = levels
        self.samples = samples
        self.query_chunk_size = query_chunk_size
        self.maximum_offset = maximum_offset
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projections = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(levels)]
        )
        self.value_projections = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(levels)]
        )
        self.query_offset_projection = nn.Linear(
            hidden_dim,
            levels * samples * 2,
        )
        self.frame_offset_projections = nn.ModuleList(
            [nn.Linear(hidden_dim, samples * 2) for _ in range(levels)]
        )
        self.attention_bias_projection = nn.Linear(
            hidden_dim,
            heads * levels * samples,
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        pyramid: list[torch.Tensor],
        reference_grid: torch.Tensor,
    ) -> torch.Tensor:
        if len(pyramid) != self.levels:
            raise ValueError(
                f"expected {self.levels} pyramid levels, got {len(pyramid)}"
            )
        batch, query_count, _ = query.shape
        if reference_grid.shape != (query_count, 2):
            raise ValueError("reference_grid must have shape [Q, 2]")

        projected_levels = []
        frame_counts = set()
        for level_index, feature in enumerate(pyramid):
            if feature.ndim != 5 or feature.shape[0] != batch:
                raise ValueError("pyramid levels must have shape [B, N, C, H, W]")
            level_batch, frames, channels, height, width = feature.shape
            if channels != self.hidden_dim:
                raise ValueError("pyramid channel width does not match hidden_dim")
            frame_counts.add(frames)
            flattened = feature.reshape(
                level_batch * frames,
                channels,
                height,
                width,
            )
            key = self.key_projections[level_index](flattened)
            value = self.value_projections[level_index](flattened)
            frame_summary = feature.mean(dim=(-2, -1))
            frame_offsets = self.frame_offset_projections[level_index](
                frame_summary
            ).view(batch, frames, self.samples, 2)
            frame_offsets = torch.tanh(frame_offsets) * (self.maximum_offset * 0.5)
            projected_levels.append(
                (
                    key.view(level_batch, frames, channels, height, width),
                    value.view(level_batch, frames, channels, height, width),
                    frame_offsets,
                )
            )
        if len(frame_counts) != 1:
            raise ValueError("all pyramid levels must contain the same frame count")

        outputs = []
        reference_grid = reference_grid.to(device=query.device, dtype=query.dtype)
        for start in range(0, query_count, self.query_chunk_size):
            end = min(start + self.query_chunk_size, query_count)
            query_chunk = query[:, start:end]
            chunk_size = end - start
            query_heads = self.query_projection(query_chunk).view(
                batch,
                chunk_size,
                self.heads,
                self.head_dim,
            ).permute(0, 2, 1, 3)
            query_offsets = self.query_offset_projection(query_chunk).view(
                batch,
                chunk_size,
                self.levels,
                self.samples,
                2,
            )
            query_offsets = torch.tanh(query_offsets) * self.maximum_offset
            attention_bias = self.attention_bias_projection(query_chunk).view(
                batch,
                chunk_size,
                self.heads,
                self.levels,
                self.samples,
            ).permute(0, 2, 1, 3, 4)
            running_max = query.new_full(
                (batch, self.heads, chunk_size),
                -torch.inf,
            )
            denominator = query.new_zeros(batch, self.heads, chunk_size)
            numerator = query.new_zeros(
                batch,
                self.heads,
                chunk_size,
                self.head_dim,
            )
            for level_index, (
                key_feature,
                value_feature,
                frame_offsets,
            ) in enumerate(projected_levels):
                _, frames, _, height, width = key_feature.shape
                grid = (
                    reference_grid[start:end][None, :, None, None, :]
                    + query_offsets[:, :, level_index, None]
                    + frame_offsets[:, None]
                ).clamp(-1.0, 1.0)
                grid = (
                    grid.permute(0, 2, 1, 3, 4)
                    .reshape(batch * frames, chunk_size, self.samples, 2)
                )
                sampled_key = F.grid_sample(
                    key_feature.reshape(
                        batch * frames,
                        self.hidden_dim,
                        height,
                        width,
                    ),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampled_value = F.grid_sample(
                    value_feature.reshape(
                        batch * frames,
                        self.hidden_dim,
                        height,
                        width,
                    ),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                sampled_key = sampled_key.view(
                    batch,
                    frames,
                    self.heads,
                    self.head_dim,
                    chunk_size,
                    self.samples,
                ).permute(0, 2, 4, 1, 5, 3)
                sampled_value = sampled_value.view(
                    batch,
                    frames,
                    self.heads,
                    self.head_dim,
                    chunk_size,
                    self.samples,
                ).permute(0, 2, 4, 1, 5, 3)
                scores = (
                    query_heads[:, :, :, None, None] * sampled_key
                ).sum(dim=-1) * (self.head_dim**-0.5)
                scores = scores + attention_bias[
                    :, :, :, level_index, :
                ][:, :, :, None, :]
                level_max = scores.amax(dim=(3, 4))
                new_max = torch.maximum(running_max, level_max)
                previous_scale = torch.exp(running_max - new_max)
                weights = torch.exp(
                    scores - new_max[:, :, :, None, None]
                )
                numerator = (
                    numerator * previous_scale[..., None]
                    + (weights[..., None] * sampled_value).sum(dim=(3, 4))
                )
                denominator = (
                    denominator * previous_scale + weights.sum(dim=(3, 4))
                )
                running_max = new_max
            normalized = numerator / denominator.clamp_min(1e-6)[..., None]
            output = normalized.transpose(1, 2).reshape(
                batch,
                chunk_size,
                self.hidden_dim,
            )
            outputs.append(self.output_projection(output))
        return torch.cat(outputs, dim=1)


class DirectDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        *,
        self_attention_mode: str,
        cross_attention_mode: str,
        feature_levels: int,
        deformable_samples: int,
        cross_query_chunk_size: int,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        if cross_attention_mode not in ("linear", "exact", "deformable"):
            raise ValueError(
                "cross_attention_mode must be linear, exact, or deformable"
            )
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiheadAttention(
            hidden_dim, heads, mode=self_attention_mode
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention_mode = cross_attention_mode
        self.cross_attention = (
            DeformableCrossAttention(
                hidden_dim,
                heads,
                levels=feature_levels,
                samples=deformable_samples,
                query_chunk_size=cross_query_chunk_size,
            )
            if cross_attention_mode == "deformable"
            else MultiheadAttention(
                hidden_dim,
                heads,
                mode=cross_attention_mode,
            )
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | list[torch.Tensor],
        reference_grid: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_norm(query)
        query = query + self.self_attention(normalized, normalized)
        normalized_query = self.cross_norm(query)
        if self.cross_attention_mode == "deformable":
            if not isinstance(context, list):
                raise TypeError("deformable cross-attention requires a feature pyramid")
            normalized_context = [
                self.context_norm(level.movedim(2, -1)).movedim(-1, 2)
                for level in context
            ]
            cross = self.cross_attention(
                normalized_query,
                normalized_context,
                reference_grid,
            )
        else:
            if not isinstance(context, torch.Tensor):
                raise TypeError("global cross-attention requires a flat context tensor")
            cross = self.cross_attention(
                normalized_query,
                self.context_norm(context),
            )
        query = query + cross
        return query + self.ffn(self.ffn_norm(query))
