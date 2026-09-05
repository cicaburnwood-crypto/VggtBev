from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..m05_plus_contract import (
    BEV_REFERENCE_FRAME,
    DATASET_INPUT_ORDER,
    GEOMETRY_CONDITIONING,
    HISTORY_ORDER,
    PATCH_FUSION,
    PATCH_TOKEN_STREAMS,
    PIPELINE_ID,
    PREFIX_CONDITIONING,
    PREFIX_CONTEXT_TRAINING,
    TEMPORAL_EXECUTION,
    VGGT_INPUT_ORDER,
)
from .attention import DeformableCrossAttention, MultiheadAttention
from .m04 import ParallelQueryBranch, SpatialResidualBlock
from .m05 import DenseNativeQuery, M05Head, StructuredFrameReliabilityHead
from .method1 import MetricScaleTokenHead, MultiScaleTokenProjector


class DPTLiteRefineBlock(nn.Module):
    """Small spatial residual block used by the patch-token top-down path."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm_1 = nn.GroupNorm(1, channels)
        self.conv_1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm_2 = nn.GroupNorm(1, channels)
        self.conv_2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.conv_1(F.gelu(self.norm_1(feature)))
        residual = self.conv_2(F.gelu(self.norm_2(residual)))
        return feature + residual


class DPTLiteLocalGlobalPyramid(nn.Module):
    """Fuse cached VGGT depths while preserving local/global patch streams.

    VGGT-Omega concatenates a 1024D within-frame state and a 1024D
    inter-frame state at every cached layer.  Each half is normalized,
    projected and fused top-down independently.  They meet only after the
    multi-layer DPT-lite refinement at each output pyramid level.
    """

    def __init__(
        self,
        layers: tuple[int, ...],
        input_dim: int,
        hidden_dim: int,
        stream_dim: int,
        spatial_scales: tuple[float, ...],
    ) -> None:
        super().__init__()
        if len(layers) != len(spatial_scales) or not layers:
            raise ValueError("every cached token layer needs one spatial scale")
        if input_dim % 2:
            raise ValueError("local/global VGGT token width must be even")
        if hidden_dim <= 0 or stream_dim <= 0:
            raise ValueError("DPT-lite channel dimensions must be positive")
        if any(scale <= 0 for scale in spatial_scales):
            raise ValueError("DPT-lite spatial scales must be positive")
        if tuple(layers) != tuple(sorted(layers)):
            raise ValueError("DPT-lite cached layers must run shallow to deep")
        if any(
            shallow_scale < deep_scale
            for shallow_scale, deep_scale in zip(
                spatial_scales[:-1], spatial_scales[1:], strict=True
            )
        ):
            raise ValueError(
                "DPT-lite spatial scales must run high to low resolution"
            )
        self.layers = layers
        self.spatial_scales = spatial_scales
        self.input_dim = int(input_dim)
        self.input_stream_dim = input_dim // 2
        self.stream_dim = int(stream_dim)
        self.hidden_dim = int(hidden_dim)
        self.local_projections = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.LayerNorm(self.input_stream_dim),
                    nn.Linear(self.input_stream_dim, stream_dim),
                )
                for layer in layers
            }
        )
        self.global_projections = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.LayerNorm(self.input_stream_dim),
                    nn.Linear(self.input_stream_dim, stream_dim),
                )
                for layer in layers
            }
        )
        self.local_level_embedding = nn.Parameter(
            torch.empty(len(layers), 1, 1, 1, stream_dim)
        )
        self.global_level_embedding = nn.Parameter(
            torch.empty(len(layers), 1, 1, 1, stream_dim)
        )
        self.local_refinement = nn.ModuleList(
            [DPTLiteRefineBlock(stream_dim) for _ in layers]
        )
        self.global_refinement = nn.ModuleList(
            [DPTLiteRefineBlock(stream_dim) for _ in layers]
        )
        self.stream_fusion = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(stream_dim * 2, hidden_dim, 1),
                    nn.GELU(),
                    DPTLiteRefineBlock(hidden_dim),
                )
                for _ in layers
            ]
        )
        nn.init.normal_(self.local_level_embedding, std=0.02)
        nn.init.normal_(self.global_level_embedding, std=0.02)

    @staticmethod
    def _resize_frames(
        feature: torch.Tensor, size: tuple[int, int]
    ) -> torch.Tensor:
        batch, frames, channels, _, _ = feature.shape
        return F.interpolate(
            feature.flatten(0, 1),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).view(batch, frames, channels, *size)

    @staticmethod
    def _refine_frames(
        feature: torch.Tensor, block: nn.Module
    ) -> torch.Tensor:
        batch, frames, _, height, width = feature.shape
        refined = block(feature.flatten(0, 1))
        return refined.view(batch, frames, refined.shape[1], height, width)

    def _project_streams(
        self,
        tokens: dict[int, torch.Tensor],
        patch_grid: tuple[int, int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        patch_height, patch_width = patch_grid
        local_levels: list[torch.Tensor] = []
        global_levels: list[torch.Tensor] = []
        for level_index, layer in enumerate(self.layers):
            if layer not in tokens:
                raise KeyError(f"missing cached VGGT layer {layer}")
            value = tokens[layer]
            if value.shape[-1] != self.input_dim:
                raise ValueError(
                    f"cached VGGT layer {layer} width must be {self.input_dim}"
                )
            batch, frames, patches, _ = value.shape
            if patches != patch_height * patch_width:
                raise ValueError(
                    f"layer {layer} has {patches} tokens but grid is "
                    f"{patch_height}x{patch_width}"
                )
            local = self.local_projections[str(layer)](
                value[..., : self.input_stream_dim]
            )
            global_ = self.global_projections[str(layer)](
                value[..., self.input_stream_dim :]
            )
            local = local + self.local_level_embedding[level_index]
            global_ = global_ + self.global_level_embedding[level_index]
            size = (
                max(1, round(patch_height * self.spatial_scales[level_index])),
                max(1, round(patch_width * self.spatial_scales[level_index])),
            )
            local = local.view(
                batch, frames, patch_height, patch_width, self.stream_dim
            ).permute(0, 1, 4, 2, 3)
            global_ = global_.view(
                batch, frames, patch_height, patch_width, self.stream_dim
            ).permute(0, 1, 4, 2, 3)
            local_levels.append(self._resize_frames(local, size))
            global_levels.append(self._resize_frames(global_, size))
        return local_levels, global_levels

    def forward(
        self,
        tokens: dict[int, torch.Tensor],
        patch_grid: tuple[int, int],
    ) -> list[torch.Tensor]:
        local_levels, global_levels = self._project_streams(tokens, patch_grid)
        outputs: list[torch.Tensor | None] = [None] * len(self.layers)
        local_state: torch.Tensor | None = None
        global_state: torch.Tensor | None = None
        for level_index in range(len(self.layers) - 1, -1, -1):
            size = local_levels[level_index].shape[-2:]
            local_state = (
                local_levels[level_index]
                if local_state is None
                else local_levels[level_index]
                + self._resize_frames(local_state, size)
            )
            global_state = (
                global_levels[level_index]
                if global_state is None
                else global_levels[level_index]
                + self._resize_frames(global_state, size)
            )
            local_state = self._refine_frames(
                local_state, self.local_refinement[level_index]
            )
            global_state = self._refine_frames(
                global_state, self.global_refinement[level_index]
            )
            fused = torch.cat((local_state, global_state), dim=2)
            outputs[level_index] = self._refine_frames(
                fused, self.stream_fusion[level_index]
            )
        if any(level is None for level in outputs):
            raise RuntimeError("DPT-lite did not produce every pyramid level")
        return [level for level in outputs if level is not None]


class PrefixContextBlock(nn.Module):
    """Content-only self-attention over the ordered special-token sequence."""

    def __init__(self, hidden_dim: int, heads: int, expansion: int = 4) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiheadAttention(hidden_dim, heads, mode="exact")
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_reliability: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(
            normalized, normalized, token_reliability
        )
        return tokens + self.ffn(self.ffn_norm(tokens))


class RoleSeparatedPrefixTrunk(nn.Module):
    """Contextualize Camera/Register tokens without pooling or geometry heads.

    Camera and Register inputs use different projections and remain distinct
    token slots throughout.  The module predicts no pose, depth, scale or
    calibration quantity; gradients arrive only through the BEV objective.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        maximum_history: int,
        maximum_prefix_tokens: int,
    ) -> None:
        super().__init__()
        if layers <= 0 or maximum_history <= 0 or maximum_prefix_tokens < 2:
            raise ValueError("prefix context dimensions must be positive")
        self.maximum_history = int(maximum_history)
        self.maximum_prefix_tokens = int(maximum_prefix_tokens)
        self.hidden_dim = int(hidden_dim)
        self.camera_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim)
        )
        self.register_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim)
        )
        self.frame_age_embedding = nn.Parameter(
            torch.empty(maximum_history, hidden_dim)
        )
        self.camera_type_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.register_type_embedding = nn.Parameter(
            torch.empty(maximum_prefix_tokens - 1, hidden_dim)
        )
        self.latest_reference_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.blocks = nn.ModuleList(
            [PrefixContextBlock(hidden_dim, heads) for _ in range(layers)]
        )
        self.camera_output_norm = nn.LayerNorm(hidden_dim)
        self.register_output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.frame_age_embedding, std=0.02)
        nn.init.normal_(self.camera_type_embedding, std=0.02)
        nn.init.normal_(self.register_type_embedding, std=0.02)
        nn.init.normal_(self.latest_reference_embedding, std=0.02)

    def forward(
        self,
        prefix_tokens: torch.Tensor,
        *,
        frame_reliability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prefix_tokens.ndim != 4:
            raise ValueError("prefix tokens must have shape [B,N,P,C]")
        batch, frames, prefix_count, _ = prefix_tokens.shape
        if not 1 <= frames <= self.maximum_history:
            raise ValueError("frame count exceeds prefix context contract")
        if prefix_count != self.maximum_prefix_tokens:
            raise ValueError("prefix context requires its exact token count")
        camera = self.camera_projection(prefix_tokens[:, :, 0])
        registers = self.register_projection(prefix_tokens[:, :, 1:])
        frame_age = self.frame_age_embedding[:frames][None, :, None, :]
        camera = (
            camera[:, :, None, :]
            + frame_age
            + self.camera_type_embedding[None, None, None, :]
        )
        registers = (
            registers
            + frame_age
            + self.register_type_embedding[: prefix_count - 1][None, None]
        )
        tokens = torch.cat((camera, registers), dim=2)
        tokens = tokens.clone()
        tokens[:, 0] = tokens[:, 0] + self.latest_reference_embedding
        tokens = tokens.reshape(batch, frames * prefix_count, self.hidden_dim)
        token_reliability = None
        if frame_reliability is not None:
            if frame_reliability.shape != (batch, frames):
                raise ValueError("frame_reliability must have shape [B,N]")
            token_reliability = frame_reliability.repeat_interleave(
                prefix_count, dim=1
            )
        for block in self.blocks:
            tokens = block(tokens, token_reliability)
        tokens = tokens.view(batch, frames, prefix_count, self.hidden_dim)
        camera_context = self.camera_output_norm(tokens[:, :, 0])
        register_context = self.register_output_norm(tokens[:, :, 1:])
        return camera_context, register_context


class PerCellPrefixReader(nn.Module):
    """Let each BEV cell read distinct Camera/Register token content."""

    def __init__(
        self,
        *,
        prefix_dim: int,
        hidden_dim: int,
        heads: int,
        query_chunk_size: int,
    ) -> None:
        super().__init__()
        if query_chunk_size <= 0:
            raise ValueError("prefix query chunk size must be positive")
        self.query_chunk_size = int(query_chunk_size)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.camera_projection = nn.Sequential(
            nn.LayerNorm(prefix_dim), nn.Linear(prefix_dim, hidden_dim)
        )
        self.register_projection = nn.Sequential(
            nn.LayerNorm(prefix_dim), nn.Linear(prefix_dim, hidden_dim)
        )
        self.attention = MultiheadAttention(
            hidden_dim, heads, mode="exact", projection_fusion="kv"
        )

    def forward(
        self,
        query: torch.Tensor,
        camera_token: torch.Tensor,
        register_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if camera_token.ndim != 2 or register_tokens.ndim != 3:
            raise ValueError("per-cell prefix inputs have invalid ranks")
        if camera_token.shape[0] != query.shape[0]:
            raise ValueError("camera/query batches must match")
        if register_tokens.shape[0] != query.shape[0]:
            raise ValueError("register/query batches must match")
        memory = torch.cat(
            (
                self.camera_projection(camera_token)[:, None],
                self.register_projection(register_tokens),
            ),
            dim=1,
        )
        normalized = self.query_norm(query)
        chunks = []
        for start in range(0, query.shape[1], self.query_chunk_size):
            chunks.append(
                self.attention(
                    normalized[:, start : start + self.query_chunk_size],
                    memory,
                )
            )
        return torch.cat(chunks, dim=1)


class ProjectedDenseMetricQuery(nn.Module):
    """Full native cell identity with a wider shared decoder state."""

    def __init__(
        self,
        *,
        size: int,
        extent_m: float,
        content_dim: int,
        hidden_dim: int,
        fourier_bands: int,
    ) -> None:
        super().__init__()
        self.content = DenseNativeQuery(
            size=size,
            extent_m=extent_m,
            hidden_dim=content_dim,
            fourier_bands=fourier_bands,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(content_dim),
            nn.Linear(content_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def size(self) -> int:
        return self.content.size

    @property
    def extent_m(self) -> float:
        return self.content.extent_m

    @property
    def reference_grid(self) -> torch.Tensor:
        return self.content.reference_grid

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.projection(self.content(batch_size))


class TemporalFrameProposal(nn.Module):
    """Create one BEV residual proposal from one historical frame."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        feature_levels: int,
        layers: int,
        deformable_samples: int,
        cross_query_chunk_size: int,
    ) -> None:
        super().__init__()
        self.state_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(layers)]
        )
        self.context_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(layers)]
        )
        self.attentions = nn.ModuleList(
            [
                DeformableCrossAttention(
                    hidden_dim,
                    heads,
                    levels=feature_levels,
                    samples=deformable_samples,
                    query_chunk_size=cross_query_chunk_size,
                )
                for _ in range(layers)
            ]
        )
        self.frame_conditions = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(layers)]
        )
        self.delta_projections = nn.ModuleList(
            [
                nn.Sequential(
                    # This path is structurally zero preserving: a zero
                    # attention update can never synthesize a residual from
                    # LayerNorm affine terms or Linear biases.
                    nn.LayerNorm(hidden_dim, elementwise_affine=False),
                    nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
                )
                for _ in range(layers)
            ]
        )
        # Fresh M05+ starts as an exact latest-anchor model. Historical
        # proposals become active through learning instead of injecting a
        # random residual on the first optimizer step.
        for projection in self.delta_projections:
            nn.init.zeros_(projection[-1].weight)

    def forward(
        self,
        anchor: torch.Tensor,
        frame_pyramid: list[torch.Tensor],
        reference_grid: torch.Tensor,
        frame_context: torch.Tensor,
    ) -> torch.Tensor:
        proposal = anchor
        for state_norm, context_norm, attention, condition, projection in zip(
            self.state_norms,
            self.context_norms,
            self.attentions,
            self.frame_conditions,
            self.delta_projections,
            strict=True,
        ):
            context = [
                context_norm(level.movedim(2, -1)).movedim(-1, 2)
                for level in frame_pyramid
            ]
            conditioned = condition(frame_context)
            if conditioned.ndim == 2:
                conditioned = conditioned[:, None, :].expand_as(proposal)
            if conditioned.shape != proposal.shape:
                raise ValueError(
                    "frame context must resolve to one feature per BEV query"
                )
            query = proposal + conditioned
            update = attention(state_norm(query), context, reference_grid)
            proposal = proposal + projection(update)
        # Both tensors are in the same decoder state space. In particular,
        # zero historical evidence produces an exact zero residual.
        return proposal - anchor


class PerQueryTemporalAttention(nn.Module):
    """Choose historical evidence independently at every BEV query.

    A learned null candidate lets a query preserve the latest anchor when no
    historical frame is useful. The softmax is only across homogeneous frame
    proposals; camera/register token roles remain separated upstream.
    """

    def __init__(
        self,
        hidden_dim: int,
        maximum_history: int,
        *,
        null_initial_probability: float,
    ) -> None:
        super().__init__()
        if not 0.0 < null_initial_probability < 1.0:
            raise ValueError("null_initial_probability must be in (0,1)")
        self.state_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.delta_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.frame_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1)
        self.null_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.frame_age_bias = nn.Parameter(torch.zeros(maximum_history - 1))
        initial_log_odds = math.log(
            null_initial_probability / (1.0 - null_initial_probability)
        )
        self.register_buffer(
            "null_initial_log_odds",
            torch.tensor(initial_log_odds, dtype=torch.float32),
        )
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        # The learned scorer starts as a residual around the explicit prior.
        # Its initialization therefore cannot vary with anchor content.
        nn.init.zeros_(self.null_score[-1].weight)
        nn.init.zeros_(self.null_score[-1].bias)

    def forward(
        self,
        anchor: torch.Tensor,
        proposals: list[torch.Tensor],
        frame_contexts: list[torch.Tensor],
        frame_reliabilities: list[torch.Tensor],
        frame_ages: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not proposals:
            empty = anchor.new_zeros((anchor.shape[0], 0))
            return anchor, empty, anchor.new_ones((anchor.shape[0],))
        scores = []
        state_feature = self.state_projection(anchor)
        for proposal, context, reliability, age in zip(
            proposals,
            frame_contexts,
            frame_reliabilities,
            frame_ages,
            strict=True,
        ):
            projected_context = self.frame_projection(context)
            if projected_context.ndim == 2:
                projected_context = projected_context[:, None, :].expand_as(
                    state_feature
                )
            if projected_context.shape != state_feature.shape:
                raise ValueError(
                    "temporal frame context must be per-frame or per-query"
                )
            feature = torch.tanh(
                state_feature
                + self.delta_projection(proposal)
                + projected_context
            )
            value = self.score(feature).squeeze(-1)
            value = value + reliability.float().clamp_min(1e-4).log()[:, None]
            value = value + self.frame_age_bias[age - 1]
            scores.append(value)
        history_scores = torch.stack(scores, dim=1).float()
        history_count = history_scores.shape[1]
        null_score = self.null_score(anchor).squeeze(-1).unsqueeze(1).float()
        null_score = (
            null_score
            + math.log(history_count)
            + self.null_initial_log_odds
        )
        weights = torch.softmax(
            torch.cat((null_score, history_scores), dim=1), dim=1
        ).to(dtype=anchor.dtype)
        history_weights = weights[:, 1:]
        stacked = torch.stack(proposals, dim=1)
        update = (history_weights.unsqueeze(-1) * stacked).sum(dim=1)
        merged = anchor + update
        return (
            merged,
            history_weights.float().mean(dim=2),
            weights[:, 0].float().mean(dim=1),
        )


class M05PlusHead(nn.Module):
    """Higher-capacity fixed-metric M05 with per-query frame selection."""

    pipeline_id = PIPELINE_ID

    def __init__(
        self,
        *,
        cached_layers: tuple[int, ...],
        spatial_scales: tuple[float, ...],
        vggt_token_dim: int,
        hidden_dim: int,
        query_content_dim: int,
        heads: int,
        latest_decoder_layers: int,
        temporal_proposal_layers: int,
        scale_hidden_dim: int,
        scale_decoder_layers: int,
        self_attention_mode: str,
        deformable_samples: int,
        cross_query_chunk_size: int,
        merged_bev_size: int,
        merged_extent_m: float,
        query_fourier_bands: int,
        shared_refinement_layers: int,
        routing_refinement_layers: int,
        evidence_refinement_layers: int,
        predict_scale_uncertainty: bool,
        patch_stream_dim: int,
        prefix_context_hidden_dim: int,
        prefix_context_heads: int,
        prefix_context_layers: int,
        maximum_history: int,
        maximum_prefix_tokens: int,
        frame_reliability_hidden_dim: int,
        frame_reliability_minimum: float,
        frame_reliability_maximum: float,
        temporal_null_initial_probability: float = 0.90,
    ) -> None:
        super().__init__()
        self.maximum_history = int(maximum_history)
        self.maximum_prefix_tokens = int(maximum_prefix_tokens)
        self.vggt_token_dim = int(vggt_token_dim)
        self.temporal_null_initial_probability = float(
            temporal_null_initial_probability
        )
        self.spatial_token_projector = DPTLiteLocalGlobalPyramid(
            cached_layers,
            vggt_token_dim,
            hidden_dim,
            patch_stream_dim,
            spatial_scales,
        )
        self.scale_token_projector = MultiScaleTokenProjector(
            cached_layers, vggt_token_dim, scale_hidden_dim, spatial_scales
        )
        self.prefix_context_trunk = RoleSeparatedPrefixTrunk(
            input_dim=vggt_token_dim,
            hidden_dim=prefix_context_hidden_dim,
            heads=prefix_context_heads,
            layers=prefix_context_layers,
            maximum_history=maximum_history,
            maximum_prefix_tokens=maximum_prefix_tokens,
        )
        self.prefix_reader = PerCellPrefixReader(
            prefix_dim=prefix_context_hidden_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            query_chunk_size=cross_query_chunk_size,
        )
        self.frame_reliability = StructuredFrameReliabilityHead(
            vggt_token_dim,
            frame_reliability_hidden_dim,
            maximum_prefix_tokens=maximum_prefix_tokens,
            minimum=frame_reliability_minimum,
            maximum=frame_reliability_maximum,
        )
        self.scale_frame_reliability = StructuredFrameReliabilityHead(
            vggt_token_dim,
            frame_reliability_hidden_dim,
            maximum_prefix_tokens=maximum_prefix_tokens,
            minimum=frame_reliability_minimum,
            maximum=frame_reliability_maximum,
        )
        self.query = ProjectedDenseMetricQuery(
            size=merged_bev_size,
            extent_m=merged_extent_m,
            content_dim=query_content_dim,
            hidden_dim=hidden_dim,
            fourier_bands=query_fourier_bands,
        )
        self.latest_branch = ParallelQueryBranch(
            hidden_dim=hidden_dim,
            heads=heads,
            layers=latest_decoder_layers,
            feature_levels=len(cached_layers),
            self_attention_mode=self_attention_mode,
            cross_attention_mode="deformable",
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
        )
        self.history_proposal = TemporalFrameProposal(
            hidden_dim=hidden_dim,
            heads=heads,
            feature_levels=len(cached_layers),
            layers=temporal_proposal_layers,
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
        )
        self.temporal_attention = PerQueryTemporalAttention(
            hidden_dim,
            maximum_history,
            null_initial_probability=temporal_null_initial_probability,
        )
        self.shared_refinement = nn.Sequential(
            *[
                SpatialResidualBlock(hidden_dim)
                for _ in range(shared_refinement_layers)
            ]
        )
        self.routing_refinement = nn.Sequential(
            *[
                SpatialResidualBlock(hidden_dim)
                for _ in range(routing_refinement_layers)
            ]
        )
        self.evidence_refinement = nn.Sequential(
            *[
                SpatialResidualBlock(hidden_dim)
                for _ in range(evidence_refinement_layers)
            ]
        )
        self.routing_head = nn.Conv2d(hidden_dim, 2, 1)
        self.evidence_head = nn.Conv2d(hidden_dim, 2, 1)
        self.scale_decoder = MetricScaleTokenHead(
            scale_hidden_dim,
            heads,
            layers=scale_decoder_layers,
            predict_uncertainty=predict_scale_uncertainty,
        )

    @property
    def merged_bev_size(self) -> int:
        return self.query.size

    @property
    def merged_extent_m(self) -> float:
        return self.query.extent_m

    def _decode_state(
        self, state: torch.Tensor, *, assemble_runtime_outputs: bool
    ) -> dict:
        spatial = state.transpose(1, 2).reshape(
            state.shape[0], -1, self.merged_bev_size, self.merged_bev_size
        )
        shared = self.shared_refinement(spatial)
        routing = self.routing_refinement(shared)
        evidence = self.evidence_refinement(shared)
        return M05Head._assemble_bev(
            self.evidence_head(evidence),
            self.routing_head(routing),
            assemble_runtime_outputs=assemble_runtime_outputs,
        )

    def forward(
        self,
        extraction: dict,
        *,
        include_merged: bool = True,
        include_scale: bool = False,
        include_latest_auxiliary: bool = False,
        assemble_runtime_outputs: bool = True,
    ) -> dict:
        prefix = extraction["camera_register_tokens"]
        if prefix.ndim != 4:
            raise ValueError("M05+ prefix tokens must have shape [B,N,P,C]")
        batch, frames, prefix_count, channels = prefix.shape
        if not 1 <= frames <= self.maximum_history:
            raise ValueError("M05+ RGB history length is outside its contract")
        if prefix_count != self.maximum_prefix_tokens:
            raise ValueError(
                "M05+ requires exactly one camera token plus the configured "
                f"register tokens ({self.maximum_prefix_tokens} total)"
            )
        if channels != self.vggt_token_dim:
            raise ValueError(
                f"M05+ prefix width must be {self.vggt_token_dim}, got {channels}"
            )
        patch_height, patch_width = extraction["patch_grid"]
        tokens = extraction.get("tokens")
        if not isinstance(tokens, dict):
            raise ValueError("M05+ extraction tokens must be a layer dictionary")
        for layer in self.spatial_token_projector.layers:
            value = tokens.get(layer)
            if value is None:
                raise KeyError(f"missing cached VGGT layer {layer}")
            expected = (batch, frames, patch_height * patch_width, channels)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"cached VGGT layer {layer} must have shape {expected}, "
                    f"got {tuple(value.shape)}"
                )
        reliability = self.frame_reliability(prefix)
        output: dict = {
            "frame_reliability": reliability,
            "effective_frame_reliability": reliability,
            "frame_keep_mask": torch.ones_like(reliability, dtype=torch.bool),
        }
        if include_merged:
            pyramid = self.spatial_token_projector(
                extraction["tokens"], extraction["patch_grid"]
            )
            camera_context, register_context = self.prefix_context_trunk(
                prefix, frame_reliability=reliability
            )
            query = self.query(batch)
            latest_prefix = self.prefix_reader(
                query,
                camera_context[:, 0],
                register_context[:, 0],
            )
            anchor = self.latest_branch(
                query + latest_prefix,
                [level[:, :1] for level in pyramid],
                self.query.reference_grid,
            )
            if include_latest_auxiliary:
                output["latest_auxiliary_bev"] = self._decode_state(
                    anchor, assemble_runtime_outputs=assemble_runtime_outputs
                )
            proposals: list[torch.Tensor] = []
            contexts: list[torch.Tensor] = []
            reliabilities: list[torch.Tensor] = []
            ages: list[int] = []
            # Extraction order is latest -> oldest. Index zero owns VGGT's
            # special first-frame token and anchors the latest-centric BEV.
            frame_order = tuple(range(1, frames))
            for frame_index in frame_order:
                frame_pyramid = [
                    level[:, frame_index : frame_index + 1]
                    for level in pyramid
                ]
                frame_context = self.prefix_reader(
                    anchor,
                    camera_context[:, frame_index],
                    register_context[:, frame_index],
                )
                proposals.append(
                    self.history_proposal(
                        anchor,
                        frame_pyramid,
                        self.query.reference_grid,
                        frame_context,
                    )
                )
                contexts.append(frame_context)
                reliabilities.append(reliability[:, frame_index])
                ages.append(frame_index)
            merged, attention_mean, null_mean = self.temporal_attention(
                anchor, proposals, contexts, reliabilities, ages
            )
            output["history_frame_indices_newest_to_oldest"] = frame_order
            output["history_original_indices_newest_to_oldest"] = tuple(
                range(frames - 2, -1, -1)
            )
            output["temporal_frame_attention_mean"] = attention_mean
            output["temporal_null_attention_mean"] = null_mean
            # Compatibility diagnostic name used by the common trainer.
            output["history_update_gate_mean"] = attention_mean
            output["merged_bev"] = self._decode_state(
                merged, assemble_runtime_outputs=assemble_runtime_outputs
            )
        if include_scale:
            scale_reliability = self.scale_frame_reliability(prefix)
            output["scale_frame_reliability"] = scale_reliability
            scale_pyramid = self.scale_token_projector(
                extraction["tokens"], extraction["patch_grid"]
            )
            output["scale"] = self.scale_decoder(
                scale_pyramid, frame_reliability=scale_reliability
            )
        return output


class M05PlusSystem(nn.Module):
    def __init__(self, adapter: nn.Module, **head_arguments) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = M05PlusHead(**head_arguments)

    def unwrapped_head(self) -> M05PlusHead:
        head = getattr(self.head, "module", self.head)
        if not isinstance(head, M05PlusHead):
            raise TypeError("unexpected M05+ head type")
        return head

    def extract(self, images: torch.Tensor) -> dict:
        # Dataset windows stay chronological for GT construction. Reverse only
        # at the frozen-VGGT boundary so the original latest frame receives
        # VGGT's unique first-frame camera/register-token role.
        latest_first_images = images.flip(1)
        if getattr(self.adapter, "supports_native_token_dtype", False):
            return self.adapter.aggregate(
                latest_first_images,
                preserve_token_dtype=images.device.type == "cuda",
            )
        return self.adapter.aggregate(latest_first_images)

    def decode_scale_teacher(self, extraction: dict) -> dict:
        teacher = self.adapter.decode_scale_teacher(extraction)
        # GT depth tensors remain chronological. Restore the teacher tensors
        # to that order for training-only metric-scale label fitting.
        for key in ("estimated_depth_vggt", "estimated_depth_confidence"):
            value = teacher.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                teacher[key] = value.flip(1)
        return teacher

    def forward_head(self, extraction: dict, **arguments) -> dict:
        prediction = self.head(extraction, **arguments)
        head = self.unwrapped_head()
        return {
            **prediction,
            "pipeline_id": head.pipeline_id,
            "runtime_inputs": ("rgb_window",),
            "dataset_input_order": DATASET_INPUT_ORDER,
            "vggt_input_order": VGGT_INPUT_ORDER,
            "vggt_anchor_original_frame": BEV_REFERENCE_FRAME,
            "runtime_passes": 1,
            "single_bev_present": False,
            "latest_auxiliary_runtime_output": False,
            "relative_pose_head_present": False,
            "explicit_geometry_module_present": False,
            "geometry_auxiliary_loss_present": False,
            "prefix_context_training": PREFIX_CONTEXT_TRAINING,
            "extrinsic_input_present": False,
            "camera_height_input_present": False,
            "external_fusion_present": False,
            "runtime_postprocessing_present": False,
            "merged_source": "latest anchor plus per-query temporal attention",
            "coordinate_mode": "fixed_metric",
            "merged_extent_m": head.merged_extent_m,
            "merged_bounds_m": (
                -head.merged_extent_m / 2.0,
                head.merged_extent_m / 2.0,
                -head.merged_extent_m / 2.0,
                head.merged_extent_m / 2.0,
            ),
            "merged_cell_size_m": head.merged_extent_m / head.merged_bev_size,
            "merged_output_size": head.merged_bev_size,
            "scale_unit": "meter_per_vggt_runtime_unit",
            "scale_is_merged_input": False,
            "scale_default_runtime_output": False,
            "scale_output_present": "scale" in prediction,
            "geometry_conditioning": GEOMETRY_CONDITIONING,
            "patch_fusion": PATCH_FUSION,
            "patch_token_streams": PATCH_TOKEN_STREAMS,
            "patch_local_input_dim": head.spatial_token_projector.input_stream_dim,
            "patch_global_input_dim": head.spatial_token_projector.input_stream_dim,
            "patch_stream_dim": head.spatial_token_projector.stream_dim,
            "prefix_conditioning": PREFIX_CONDITIONING,
            "prefix_token_pooling_present": False,
            "prefix_context_hidden_dim": head.prefix_context_trunk.hidden_dim,
            "temporal_execution": TEMPORAL_EXECUTION,
            "history_order": HISTORY_ORDER,
            "strict_zero_history_residual": True,
            "temporal_null_history_count_invariant": True,
            "temporal_null_initial_probability": (
                head.temporal_null_initial_probability
            ),
            "expected_prefix_tokens": head.maximum_prefix_tokens,
            "maximum_history": head.maximum_history,
            "orientation": "latest ego centered; forward is image-up",
        }

    def forward(
        self, images: torch.Tensor, *, include_scale: bool = False
    ) -> dict:
        extraction = self.extract(images)
        for private_key in ("_aggregated", "_patch_start", "_images"):
            extraction.pop(private_key, None)
        if images.device.type != "cuda":
            return self.forward_head(extraction, include_scale=include_scale)
        runtime_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        extraction["tokens"] = {
            layer: value.to(dtype=runtime_dtype)
            for layer, value in extraction["tokens"].items()
        }
        extraction["camera_register_tokens"] = extraction[
            "camera_register_tokens"
        ].to(dtype=runtime_dtype)
        with torch.autocast("cuda", dtype=runtime_dtype):
            return self.forward_head(extraction, include_scale=include_scale)
