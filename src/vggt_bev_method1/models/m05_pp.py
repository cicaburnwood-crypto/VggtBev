from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..m05_pp_contract import (
    BEV_REFERENCE_FRAME,
    DATASET_INPUT_ORDER,
    GEOMETRY_CONDITIONING,
    HISTORY_ORDER,
    PATCH_FUSION,
    PATCH_TOKEN_STREAMS,
    PIPELINE_ID,
    PREFIX_CONDITIONING,
    PREFIX_CONTEXT_TRAINING,
    RESOLUTION_HIERARCHY,
    TEMPORAL_EXECUTION,
    VGGT_INPUT_ORDER,
)
from .m04 import ParallelQueryBranch, SpatialResidualBlock
from .m05 import M05Head, StructuredFrameReliabilityHead
from .m05_plus import (
    DPTLiteLocalGlobalPyramid,
    M05PlusHead,
    M05PlusSystem,
    PerCellPrefixReader,
    PerQueryTemporalAttention,
    ProjectedDenseMetricQuery,
    RoleSeparatedPrefixTrunk,
    TemporalFrameProposal,
)
from .method1 import MetricScaleTokenHead, MultiScaleTokenProjector


class LightweightSpatialResidualBlock(nn.Module):
    """Depthwise-separable residual refinement for the final 512 grid."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.depthwise = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            3,
            padding=1,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(F.gelu(self.norm(value)))
        return value + self.pointwise(F.gelu(update))


class LatestHighResolutionCorrection(nn.Module):
    """One shallow 512-grid correction that reads latest-frame patches only."""

    def __init__(
        self,
        *,
        coarse_size: int,
        fine_size: int,
        extent_m: float,
        content_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        feature_levels: int,
        self_attention_mode: str,
        deformable_samples: int,
        cross_query_chunk_size: int,
        query_fourier_bands: int,
    ) -> None:
        super().__init__()
        if fine_size <= coarse_size:
            raise ValueError("fine correction grid must exceed the coarse grid")
        if layers != 1:
            raise ValueError("M05++ keeps exactly one high-resolution correction")
        self.coarse_size = int(coarse_size)
        self.fine_size = int(fine_size)
        self.fine_query = ProjectedDenseMetricQuery(
            size=fine_size,
            extent_m=extent_m,
            content_dim=content_dim,
            hidden_dim=hidden_dim,
            fourier_bands=query_fourier_bands,
        )
        self.latest_patch_decoder = ParallelQueryBranch(
            hidden_dim=hidden_dim,
            heads=heads,
            layers=layers,
            feature_levels=feature_levels,
            self_attention_mode=self_attention_mode,
            cross_attention_mode="deformable",
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
        )
        self.delta_norm = nn.LayerNorm(hidden_dim)
        self.delta_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # The new fine path starts as an exact upsample of the trained coarse
        # semantics, then learns only evidence-supported high-frequency deltas.
        nn.init.zeros_(self.delta_projection.weight)

    def forward(
        self,
        coarse_state: torch.Tensor,
        latest_pyramid: list[torch.Tensor],
    ) -> torch.Tensor:
        batch, queries, channels = coarse_state.shape
        if queries != self.coarse_size * self.coarse_size:
            raise ValueError("coarse state does not match the correction grid")
        spatial = coarse_state.transpose(1, 2).reshape(
            batch, channels, self.coarse_size, self.coarse_size
        )
        upsampled = F.interpolate(
            spatial,
            size=(self.fine_size, self.fine_size),
            mode="bilinear",
            align_corners=False,
        ).flatten(2).transpose(1, 2)
        fine_query = upsampled + self.fine_query(batch)
        latest_update = self.latest_patch_decoder(
            fine_query,
            latest_pyramid,
            self.fine_query.reference_grid,
        )
        return upsampled + self.delta_projection(self.delta_norm(latest_update))


class M05PPHead(M05PlusHead):
    """M05++: coarse temporal reasoning with a latest-guided 512 correction."""

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
        fine_correction_layers: int,
        scale_hidden_dim: int,
        scale_decoder_layers: int,
        self_attention_mode: str,
        deformable_samples: int,
        cross_query_chunk_size: int,
        coarse_bev_size: int,
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
        history_proposal_batch_size: int = 1,
    ) -> None:
        nn.Module.__init__(self)
        if coarse_bev_size >= merged_bev_size:
            raise ValueError("M05++ coarse grid must be smaller than final output")
        if history_proposal_batch_size <= 0:
            raise ValueError("history proposal batch size must be positive")
        self.maximum_history = int(maximum_history)
        self.maximum_prefix_tokens = int(maximum_prefix_tokens)
        self.vggt_token_dim = int(vggt_token_dim)
        self.history_proposal_batch_size = int(history_proposal_batch_size)
        self.channels_last_spatial = False
        self.temporal_null_initial_probability = float(
            temporal_null_initial_probability
        )
        self.coarse_bev_size = int(coarse_bev_size)
        self.final_bev_size = int(merged_bev_size)
        self._merged_extent_m = float(merged_extent_m)

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
        reliability_arguments = dict(
            maximum_prefix_tokens=maximum_prefix_tokens,
            minimum=frame_reliability_minimum,
            maximum=frame_reliability_maximum,
        )
        self.frame_reliability = StructuredFrameReliabilityHead(
            vggt_token_dim,
            frame_reliability_hidden_dim,
            **reliability_arguments,
        )
        self.scale_frame_reliability = StructuredFrameReliabilityHead(
            vggt_token_dim,
            frame_reliability_hidden_dim,
            **reliability_arguments,
        )
        self.coarse_query = ProjectedDenseMetricQuery(
            size=coarse_bev_size,
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
        self.coarse_refinement = nn.Sequential(
            *[
                SpatialResidualBlock(hidden_dim)
                for _ in range(shared_refinement_layers)
            ]
        )
        self.high_resolution_correction = LatestHighResolutionCorrection(
            coarse_size=coarse_bev_size,
            fine_size=merged_bev_size,
            extent_m=merged_extent_m,
            content_dim=query_content_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            layers=fine_correction_layers,
            feature_levels=len(cached_layers),
            self_attention_mode=self_attention_mode,
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
            query_fourier_bands=query_fourier_bands,
        )
        self.routing_refinement = nn.Sequential(
            *[
                LightweightSpatialResidualBlock(hidden_dim)
                for _ in range(routing_refinement_layers)
            ]
        )
        self.evidence_refinement = nn.Sequential(
            *[
                LightweightSpatialResidualBlock(hidden_dim)
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
        return self.final_bev_size

    @property
    def merged_extent_m(self) -> float:
        return self._merged_extent_m

    def _refine_coarse(self, state: torch.Tensor) -> torch.Tensor:
        spatial = state.transpose(1, 2).reshape(
            state.shape[0], -1, self.coarse_bev_size, self.coarse_bev_size
        )
        if self.channels_last_spatial:
            spatial = spatial.contiguous(memory_format=torch.channels_last)
        return self.coarse_refinement(spatial).flatten(2).transpose(1, 2)

    def _decode_fine_state(
        self,
        state: torch.Tensor,
        *,
        assemble_runtime_outputs: bool,
    ) -> dict:
        spatial = state.transpose(1, 2).reshape(
            state.shape[0], -1, self.final_bev_size, self.final_bev_size
        )
        if self.channels_last_spatial:
            spatial = spatial.contiguous(memory_format=torch.channels_last)
        routing = self.routing_refinement(spatial)
        evidence = self.evidence_refinement(spatial)
        return M05Head._assemble_bev(
            self.evidence_head(evidence),
            self.routing_head(routing),
            assemble_runtime_outputs=assemble_runtime_outputs,
        )

    def _validate_extraction(self, extraction: dict) -> tuple[torch.Tensor, int, int]:
        prefix = extraction["camera_register_tokens"]
        if prefix.ndim != 4:
            raise ValueError("M05++ prefix tokens must have shape [B,N,P,C]")
        batch, frames, prefix_count, channels = prefix.shape
        if not 1 <= frames <= self.maximum_history:
            raise ValueError("M05++ RGB history length is outside its contract")
        if prefix_count != self.maximum_prefix_tokens:
            raise ValueError("M05++ requires one Camera plus sixteen Register tokens")
        if channels != self.vggt_token_dim:
            raise ValueError("M05++ prefix width does not match VGGT")
        patch_height, patch_width = extraction["patch_grid"]
        tokens = extraction.get("tokens")
        if not isinstance(tokens, dict):
            raise ValueError("M05++ extraction tokens must be a layer dictionary")
        for layer in self.spatial_token_projector.layers:
            value = tokens.get(layer)
            expected = (batch, frames, patch_height * patch_width, channels)
            if value is None or tuple(value.shape) != expected:
                raise ValueError(
                    f"cached VGGT layer {layer} must have shape {expected}"
                )
        return prefix, batch, frames

    def forward(
        self,
        extraction: dict,
        *,
        include_merged: bool = True,
        include_scale: bool = False,
        include_latest_auxiliary: bool = False,
        assemble_runtime_outputs: bool = True,
    ) -> dict:
        prefix, batch, frames = self._validate_extraction(extraction)
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
                prefix,
                frame_reliability=reliability,
            )
            query = self.coarse_query(batch)
            latest_prefix = self.prefix_reader(
                query,
                camera_context[:, 0],
                register_context[:, 0],
            )
            anchor = self.latest_branch(
                query + latest_prefix,
                [level[:, :1] for level in pyramid],
                self.coarse_query.reference_grid,
            )
            frame_order = tuple(range(1, frames))
            proposal_chunks: list[torch.Tensor] = []
            context_chunks: list[torch.Tensor] = []
            for history_start in range(
                1,
                frames,
                self.history_proposal_batch_size,
            ):
                history_end = min(
                    history_start + self.history_proposal_batch_size,
                    frames,
                )
                history_count = history_end - history_start
                batched_anchor = anchor[:, None].expand(
                    -1,
                    history_count,
                    -1,
                    -1,
                ).reshape(batch * history_count, anchor.shape[1], anchor.shape[2])
                batched_context = self.prefix_reader(
                    batched_anchor,
                    camera_context[:, history_start:history_end].reshape(
                        batch * history_count,
                        -1,
                    ),
                    register_context[:, history_start:history_end].reshape(
                        batch * history_count,
                        register_context.shape[2],
                        register_context.shape[3],
                    ),
                )
                batched_pyramid = [
                    level[:, history_start:history_end].reshape(
                        batch * history_count,
                        1,
                        level.shape[2],
                        level.shape[3],
                        level.shape[4],
                    )
                    for level in pyramid
                ]
                proposal_chunks.append(
                    self.history_proposal(
                        batched_anchor,
                        batched_pyramid,
                        self.coarse_query.reference_grid,
                        batched_context,
                    ).view(batch, history_count, anchor.shape[1], anchor.shape[2])
                )
                context_chunks.append(
                    batched_context.view(
                        batch,
                        history_count,
                        anchor.shape[1],
                        anchor.shape[2],
                    )
                )
            if proposal_chunks:
                proposal_tensor = torch.cat(proposal_chunks, dim=1)
                context_tensor = torch.cat(context_chunks, dim=1)
            else:
                proposal_tensor = anchor.new_empty(
                    batch,
                    0,
                    anchor.shape[1],
                    anchor.shape[2],
                )
                context_tensor = anchor.new_empty(
                    batch,
                    0,
                    anchor.shape[1],
                    anchor.shape[2],
                )
            merged, attention_mean, null_mean = self.temporal_attention(
                anchor,
                proposal_tensor,
                context_tensor,
                reliability[:, 1:],
                frame_order,
            )
            if frames == 1:
                merged = merged + self._unused_temporal_zero(anchor)
            output["history_frame_indices_newest_to_oldest"] = frame_order
            output["history_original_indices_newest_to_oldest"] = tuple(
                range(frames - 2, -1, -1)
            )
            output["temporal_frame_attention_mean"] = attention_mean
            output["temporal_null_attention_mean"] = null_mean
            output["history_update_gate_mean"] = attention_mean

            if include_latest_auxiliary:
                coarse = self._refine_coarse(torch.cat((anchor, merged), dim=0))
                latest_pyramid = [
                    torch.cat((level[:, :1], level[:, :1]), dim=0)
                    for level in pyramid
                ]
                fine = self.high_resolution_correction(coarse, latest_pyramid)
                decoded = self._decode_fine_state(
                    fine,
                    assemble_runtime_outputs=assemble_runtime_outputs,
                )
                output["latest_auxiliary_bev"], output["merged_bev"] = (
                    self._split_decoded_batch(decoded, batch)
                )
            else:
                coarse = self._refine_coarse(merged)
                fine = self.high_resolution_correction(
                    coarse,
                    [level[:, :1] for level in pyramid],
                )
                output["merged_bev"] = self._decode_fine_state(
                    fine,
                    assemble_runtime_outputs=assemble_runtime_outputs,
                )
        if include_scale:
            scale_reliability = self.scale_frame_reliability(prefix)
            output["scale_frame_reliability"] = scale_reliability
            scale_pyramid = self.scale_token_projector(
                extraction["tokens"], extraction["patch_grid"]
            )
            output["scale"] = self.scale_decoder(
                scale_pyramid,
                frame_reliability=scale_reliability,
            )
        return output


class M05PPSystem(M05PlusSystem):
    def __init__(self, adapter: nn.Module, **head_arguments) -> None:
        nn.Module.__init__(self)
        self.adapter = adapter
        self.head = M05PPHead(**head_arguments)

    def unwrapped_head(self) -> M05PPHead:
        head = getattr(self.head, "module", self.head)
        if not isinstance(head, M05PPHead):
            raise TypeError("unexpected M05++ head type")
        return head

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
            "merged_source": "coarse temporal state plus latest-patch fine correction",
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
            "coarse_bev_size": head.coarse_bev_size,
            "resolution_hierarchy": RESOLUTION_HIERARCHY,
            "latest_high_resolution_skip_present": True,
            "heavy_spatial_refinement_size": head.coarse_bev_size,
            "final_refinement_mode": "depthwise_separable",
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

