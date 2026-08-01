from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from vggt_bev_method1.config import Supervision

from .attention import DirectDecoderBlock

DecoderOutputMode = Literal[
    "legacy",
    "observed_categorical",
    "complete_evidential",
]


def normalized_position_features(size: int) -> torch.Tensor:
    """Unit-width BEV cell centers plus scale-independent harmonics."""

    cell_size = 1.0 / size
    axis = torch.linspace(
        -0.5 + cell_size / 2,
        0.5 - cell_size / 2,
        size,
    )
    z, x = torch.meshgrid(axis.flip(0), axis, indexing="ij")
    normalized_x = x * 2.0
    normalized_z = z * 2.0
    features = [x, z, normalized_x, normalized_z]
    for frequency in (1.0, 2.0, 4.0):
        features.extend(
            [
                torch.sin(math.pi * frequency * normalized_x),
                torch.cos(math.pi * frequency * normalized_x),
                torch.sin(math.pi * frequency * normalized_z),
                torch.cos(math.pi * frequency * normalized_z),
            ]
        )
    return torch.stack(features, dim=-1).view(size * size, -1)


class DirectBEVDecoder(nn.Module):
    """PDF Method I decoder with one learned query per final BEV cell."""

    def __init__(
        self,
        *,
        supervision: Supervision,
        output_size: int,
        hidden_dim: int,
        geometry_cue_dim: int,
        heads: int,
        layers: int,
        self_attention_mode: str,
        cross_attention_mode: str,
        feature_levels: int,
        deformable_samples: int,
        cross_query_chunk_size: int,
        gradient_checkpointing: bool,
        query_parameter_chunk_size: int = 32768,
        output_mode: DecoderOutputMode = "legacy",
    ) -> None:
        super().__init__()
        if output_size <= 0 or query_parameter_chunk_size <= 0:
            raise ValueError("output size and query chunk size must be positive")
        if output_mode not in (
            "legacy",
            "observed_categorical",
            "complete_evidential",
        ):
            raise ValueError(f"unsupported decoder output mode: {output_mode}")
        self.supervision = supervision
        self.output_mode = output_mode
        self.output_size = output_size
        self.gradient_checkpointing = gradient_checkpointing
        self.cross_attention_mode = cross_attention_mode
        query_count = output_size * output_size
        self.cell_embedding_chunks = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        1,
                        min(query_parameter_chunk_size, query_count - start),
                        hidden_dim,
                    )
                )
                for start in range(0, query_count, query_parameter_chunk_size)
            ]
        )
        for embedding in self.cell_embedding_chunks:
            nn.init.normal_(embedding, std=0.02)
        positions = normalized_position_features(output_size)
        self.register_buffer("normalized_positions", positions, persistent=False)
        self.position_projection = nn.Linear(positions.shape[-1], hidden_dim)
        self.scale_projection = nn.Sequential(
            nn.Linear(geometry_cue_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DirectDecoderBlock(
                    hidden_dim,
                    heads,
                    self_attention_mode=self_attention_mode,
                    cross_attention_mode=cross_attention_mode,
                    feature_levels=feature_levels,
                    deformable_samples=deformable_samples,
                    cross_query_chunk_size=cross_query_chunk_size,
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        if output_mode == "observed_categorical":
            self.class_head = nn.Linear(hidden_dim, 3)
        elif output_mode == "complete_evidential":
            self.evidence_head = nn.Linear(hidden_dim, 2)
        elif supervision == "joint":
            self.occupancy_heads = nn.ModuleDict(
                {
                    task: nn.Linear(hidden_dim, 1)
                    for task in ("observed", "complete")
                }
            )
            self.observation_heads = nn.ModuleDict(
                {
                    task: nn.Linear(hidden_dim, 1)
                    for task in ("observed", "complete")
                }
            )
        else:
            self.occupancy_head = nn.Linear(hidden_dim, 1)
            # Unknown means camera-unobserved for the observed task and
            # undefined target-map support for the complete task.
            self.observation_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        context: torch.Tensor | list[torch.Tensor],
        geometry_cue: torch.Tensor,
        *,
        extent_normalized_scale: torch.Tensor,
        geometry_gate: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        batch = (
            context.shape[0]
            if isinstance(context, torch.Tensor)
            else context[0].shape[0]
        )
        context_dtype = (
            context.dtype
            if isinstance(context, torch.Tensor)
            else context[0].dtype
        )
        if isinstance(context, list) and self.cross_attention_mode != "deformable":
            context = torch.cat(
                [
                    level.permute(0, 1, 3, 4, 2).flatten(1, 3)
                    for level in context
                ],
                dim=1,
            )
        if geometry_cue.ndim != 2 or geometry_cue.shape[0] != batch:
            raise ValueError("geometry_cue must have shape [B, C]")
        gated_geometry = geometry_cue * float(geometry_gate)
        if extent_normalized_scale.shape != (batch,):
            raise ValueError("extent_normalized_scale must have shape [B]")
        normalized_extent = extent_normalized_scale.float()
        if not torch.isfinite(normalized_extent).all() or not (
            normalized_extent > 0
        ).all():
            raise ValueError("external P1A extent must be finite and positive")
        normalized_cell_size = normalized_extent / self.output_size
        position_features = self.normalized_positions.to(context_dtype)[None].expand(
            batch,
            -1,
            -1,
        ).clone()
        position_features[..., :2] *= normalized_extent.to(context_dtype)[
            :, None, None
        ]
        position = self.position_projection(position_features)
        scale_input = torch.cat(
            (
                torch.log1p(normalized_extent)[:, None],
                torch.log1p(normalized_cell_size)[:, None],
                gated_geometry,
            ),
            dim=-1,
        )
        scale = self.scale_projection(scale_input)[:, None]
        cell_embedding = torch.cat(tuple(self.cell_embedding_chunks), dim=1)
        query = cell_embedding.expand(batch, -1, -1) + position + scale
        reference_grid = torch.stack(
            (
                self.normalized_positions[:, 2],
                -self.normalized_positions[:, 3],
            ),
            dim=-1,
        )
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                if isinstance(context, list):
                    def run_block(
                        current_query: torch.Tensor,
                        *pyramid: torch.Tensor,
                        current_block: DirectDecoderBlock = block,
                    ) -> torch.Tensor:
                        return current_block(
                            current_query,
                            list(pyramid),
                            reference_grid,
                        )

                    query = checkpoint(
                        run_block,
                        query,
                        *context,
                        use_reentrant=False,
                    )
                    continue
                query = checkpoint(
                    block,
                    query,
                    context,
                    reference_grid,
                    use_reentrant=False,
                )
            else:
                query = block(query, context, reference_grid)
        query = self.final_norm(query)
        coordinate_contract = {
            "extent_normalized_scale": normalized_extent,
            "cell_size_normalized_scale": normalized_cell_size,
            "bounds_normalized_scale": torch.stack(
                (
                    -0.5 * normalized_extent,
                    0.5 * normalized_extent,
                    -0.5 * normalized_extent,
                    0.5 * normalized_extent,
                ),
                dim=-1,
            ),
        }
        if self.output_mode == "observed_categorical":
            class_logits = self.class_head(query).view(
                batch,
                self.output_size,
                self.output_size,
                3,
            )
            return {
                "class_logits": class_logits.permute(0, 3, 1, 2).contiguous(),
                **coordinate_contract,
            }
        if self.output_mode == "complete_evidential":
            raw_evidence = self.evidence_head(query).view(
                batch,
                self.output_size,
                self.output_size,
                2,
            )
            # A Beta distribution replaces scalar sigmoid confidence. Channel
            # zero is occupied evidence and channel one is free evidence.
            evidence = F.softplus(raw_evidence.float())
            alpha_occupied = evidence[..., 0] + 1.0
            beta_free = evidence[..., 1] + 1.0
            strength = alpha_occupied + beta_free
            return {
                "raw_evidence": raw_evidence.permute(0, 3, 1, 2).contiguous(),
                "alpha_occupied": alpha_occupied,
                "beta_free": beta_free,
                "evidence_strength": strength,
                "occupancy_probability": alpha_occupied / strength,
                "occupancy_distribution_variance": (
                    alpha_occupied
                    * beta_free
                    / (strength.square() * (strength + 1.0))
                ),
                "epistemic_uncertainty": 2.0 / strength,
                "evidence_confidence": (
                    1.0 - 2.0 / strength
                ).clamp(0.0, 1.0),
                **coordinate_contract,
            }
        if self.supervision == "joint":
            return {
                **{
                    task: {
                        "occupancy_logit": self.occupancy_heads[task](query)[
                            ..., 0
                        ].view(batch, self.output_size, self.output_size),
                        "observed_logit": self.observation_heads[task](query)[
                            ..., 0
                        ].view(batch, self.output_size, self.output_size),
                    }
                    for task in ("observed", "complete")
                },
                **coordinate_contract,
            }
        return {
            "occupancy_logit": self.occupancy_head(query)[..., 0].view(
                batch, self.output_size, self.output_size
            ),
            "observed_logit": self.observation_head(query)[..., 0].view(
                batch, self.output_size, self.output_size
            ),
            **coordinate_contract,
        }
