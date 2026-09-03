from __future__ import annotations

import math

import torch
from torch import nn

from .attention import DeformableCrossAttention
from .m04 import ParallelQueryBranch, SpatialResidualBlock, _native_query_coordinates
from .method1 import MetricScaleTokenHead, MultiScaleTokenProjector
from .p1b_probability import (
    compose_semantic,
    decode_binary_prediction,
    fuse_pixel_routing,
)
from .p1d import FrameReliabilityHead
from .wtbd_merge_scale import ImplicitGeometryContextTrunk


class DenseNativeQuery(nn.Module):
    """One learned content vector per native BEV cell plus metric-free position.

    M04's row/column factorization was economical, but it also coupled distant
    edge cells through the same factors. M05 restores the full native query
    capacity used by the strong Single baseline. No latent raster is resized.
    """

    def __init__(
        self,
        *,
        size: int,
        extent_vggt: float,
        hidden_dim: int,
        fourier_bands: int = 4,
    ) -> None:
        super().__init__()
        if fourier_bands <= 0:
            raise ValueError("M05 Fourier band count must be positive")
        self.size = int(size)
        self.extent_vggt = float(extent_vggt)
        normalized, reference = _native_query_coordinates(size, extent_vggt)
        self.register_buffer("normalized_coordinates", normalized, persistent=True)
        self.register_buffer("reference_grid", reference, persistent=True)
        self.register_buffer(
            "fourier_frequencies",
            2.0 ** torch.arange(fourier_bands, dtype=torch.float32),
            persistent=True,
        )
        self.query_content = nn.Parameter(
            torch.empty(size * size, hidden_dim)
        )
        coordinate_dim = 2 + fourier_bands * 2 * 2
        self.coordinate_projection = nn.Sequential(
            nn.Linear(coordinate_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.query_content, std=0.02)

    def _coordinate_features(self) -> torch.Tensor:
        coordinates = self.normalized_coordinates
        phase = (
            coordinates[:, :, None]
            * self.fourier_frequencies[None, None, :]
            * math.pi
        )
        return torch.cat(
            (
                coordinates,
                phase.sin().flatten(1),
                phase.cos().flatten(1),
            ),
            dim=1,
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("M05 query batch size must be positive")
        position = self.coordinate_projection(
            self._coordinate_features().to(dtype=self.query_content.dtype)
        )
        return (self.query_content + position).unsqueeze(0).expand(
            batch_size, -1, -1
        )


class ReverseHistoryUpdate(nn.Module):
    """Shared newest-to-oldest residual update for one historical frame.

    The latest-frame state is never replaced by a historical proposal. Each
    frame contributes a learned residual whose gate sees the current state,
    the proposed evidence, implicit geometry, and learned frame reliability.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        feature_levels: int,
        deformable_samples: int,
        cross_query_chunk_size: int,
        gate_initial_bias: float,
    ) -> None:
        super().__init__()
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.frame_condition = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_attention = DeformableCrossAttention(
            hidden_dim,
            heads,
            levels=feature_levels,
            samples=deformable_samples,
            query_chunk_size=cross_query_chunk_size,
        )
        self.delta_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate_state = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_delta = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_frame = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_bias = nn.Parameter(
            torch.full((hidden_dim,), float(gate_initial_bias))
        )

    def forward(
        self,
        state: torch.Tensor,
        frame_pyramid: list[torch.Tensor],
        reference_grid: torch.Tensor,
        frame_context: torch.Tensor,
        frame_reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frame_context.shape != (state.shape[0], state.shape[2]):
            raise ValueError("M05 frame context must have shape [B,C]")
        if frame_reliability.shape != (state.shape[0],):
            raise ValueError("M05 frame reliability must have shape [B]")
        conditioned = state + self.frame_condition(frame_context)[:, None, :]
        normalized_context = [
            self.context_norm(level.movedim(2, -1)).movedim(-1, 2)
            for level in frame_pyramid
        ]
        cross = self.cross_attention(
            self.state_norm(conditioned),
            normalized_context,
            reference_grid,
        )
        delta = self.delta_projection(cross)
        # Adding log reliability to a gate logit is a smooth odds correction:
        # r<1 suppresses and r>1 promotes a historical update without allowing
        # an unbounded multiplicative residual.
        reliability_log_odds = frame_reliability.float().clamp_min(1e-4).log()
        gate = torch.sigmoid(
            self.gate_state(self.state_norm(state))
            + self.gate_delta(delta)
            + self.gate_frame(frame_context)[:, None, :]
            + self.gate_bias
            + reliability_log_odds[:, None, None].to(dtype=state.dtype)
        )
        return state + gate * delta, gate.float().mean(dim=(1, 2))


class M05Head(nn.Module):
    """Latest-anchored reverse-history Merged BEV and metric-scale head."""

    pipeline_id = "M05-LATEST-ANCHORED-REVERSE-GATED-MERGED-SCALE-NLL"

    def __init__(
        self,
        *,
        cached_layers: tuple[int, ...] = (4, 11, 17, 23),
        spatial_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
        vggt_token_dim: int = 2048,
        hidden_dim: int = 64,
        heads: int = 8,
        latest_decoder_layers: int = 2,
        history_update_layers: int = 1,
        scale_decoder_layers: int = 2,
        self_attention_mode: str = "linear",
        deformable_samples: int = 4,
        # 65,536 was the throughput knee on RTX PRO 6000 Blackwell: it removes
        # most Python/checkpoint launch overhead while retaining bounded
        # deformable-attention activations. This is execution-only and does not
        # change parameters, predictions, losses, or checkpoint compatibility.
        cross_query_chunk_size: int = 65536,
        merged_bev_size: int = 512,
        merged_extent_vggt: float = 6.5,
        query_fourier_bands: int = 4,
        refinement_layers: int = 1,
        predict_scale_uncertainty: bool = True,
        implicit_geometry_hidden_dim: int = 256,
        implicit_geometry_heads: int = 8,
        implicit_geometry_layers: int = 4,
        maximum_history: int = 10,
        maximum_prefix_tokens: int = 17,
        frame_reliability_hidden_dim: int = 256,
        frame_reliability_minimum: float = 0.25,
        frame_reliability_maximum: float = 1.75,
        history_gate_initial_bias: float = -1.0,
    ) -> None:
        super().__init__()
        if latest_decoder_layers <= 0 or history_update_layers <= 0:
            raise ValueError("M05 decoder depths must be positive")
        if refinement_layers <= 0:
            raise ValueError("M05 requires at least one spatial refinement block")
        self.maximum_history = int(maximum_history)
        projector_args = (
            cached_layers,
            vggt_token_dim,
            hidden_dim,
            spatial_scales,
        )
        self.spatial_token_projector = MultiScaleTokenProjector(*projector_args)
        self.scale_token_projector = MultiScaleTokenProjector(*projector_args)
        self.implicit_geometry_trunk = ImplicitGeometryContextTrunk(
            input_dim=vggt_token_dim,
            hidden_dim=implicit_geometry_hidden_dim,
            output_dim=hidden_dim,
            heads=implicit_geometry_heads,
            layers=implicit_geometry_layers,
            maximum_history=maximum_history,
            maximum_prefix_tokens=maximum_prefix_tokens,
        )
        self.frame_reliability = FrameReliabilityHead(
            vggt_token_dim,
            frame_reliability_hidden_dim,
            minimum=frame_reliability_minimum,
            maximum=frame_reliability_maximum,
        )
        self.scale_frame_reliability = FrameReliabilityHead(
            vggt_token_dim,
            frame_reliability_hidden_dim,
            minimum=frame_reliability_minimum,
            maximum=frame_reliability_maximum,
        )
        self.query = DenseNativeQuery(
            size=merged_bev_size,
            extent_vggt=merged_extent_vggt,
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
        self.history_updates = nn.ModuleList(
            [
                ReverseHistoryUpdate(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    feature_levels=len(cached_layers),
                    deformable_samples=deformable_samples,
                    cross_query_chunk_size=cross_query_chunk_size,
                    gate_initial_bias=history_gate_initial_bias,
                )
                for _ in range(history_update_layers)
            ]
        )
        self.refinement = nn.Sequential(
            *[SpatialResidualBlock(hidden_dim) for _ in range(refinement_layers)]
        )
        self.evidence_head = nn.Conv2d(hidden_dim, 2, 1)
        self.routing_head = nn.Conv2d(hidden_dim, 2, 1)
        self.scale_decoder = MetricScaleTokenHead(
            hidden_dim,
            heads,
            layers=scale_decoder_layers,
            predict_uncertainty=predict_scale_uncertainty,
        )

    @property
    def merged_bev_size(self) -> int:
        return self.query.size

    @property
    def merged_extent_vggt(self) -> float:
        return self.query.extent_vggt

    @staticmethod
    def _add_frame_embedding(
        pyramid: list[torch.Tensor],
        frame_embedding: torch.Tensor,
    ) -> list[torch.Tensor]:
        return [
            level + frame_embedding[:, :, :, None, None].to(dtype=level.dtype)
            for level in pyramid
        ]

    @staticmethod
    def _assemble_bev(
        raw_evidence: torch.Tensor,
        routing_raw: torch.Tensor,
        *,
        assemble_runtime_outputs: bool,
    ) -> dict:
        guessed = decode_binary_prediction(
            raw_evidence,
            "evidential",
            include_diagnostics=assemble_runtime_outputs,
        )
        routing_raw = routing_raw.float()
        observed_logit = routing_raw[:, 0]
        support_logit = routing_raw[:, 1]
        support_probability = torch.sigmoid(support_logit)
        output = {
            "guessed": guessed,
            "observed_gate_logit": observed_logit,
            "fov_support_logit": support_logit,
            "fov_support_probability": support_probability,
        }
        if not assemble_runtime_outputs:
            return output
        observed_probability = torch.sigmoid(observed_logit)
        guessed_region = 1.0 - observed_probability
        occupancy = guessed["occupancy_probability"].float()
        routing_probability = torch.stack(
            (
                observed_probability,
                guessed_region * (1.0 - occupancy),
                guessed_region * occupancy,
            ),
            dim=1,
        )
        fused = fuse_pixel_routing(
            routing_probability,
            guessed,
            support_probability,
            "evidential",
        )
        output.update(
            {
                "routing_probability": routing_probability,
                "routing_class": routing_probability.argmax(dim=1),
                "observed_gate_probability": observed_probability,
                "guessed_region_probability": guessed_region,
                "fused": fused,
                "occupancy_probability": fused["occupancy_probability"],
                "navigation_confidence": fused["navigation_confidence"],
                "fov_complete_semantic": compose_semantic(
                    fused, support_probability
                ),
            }
        )
        return output

    def _decode_state(
        self,
        state: torch.Tensor,
        *,
        assemble_runtime_outputs: bool,
    ) -> dict:
        spatial = state.transpose(1, 2).reshape(
            state.shape[0],
            -1,
            self.merged_bev_size,
            self.merged_bev_size,
        )
        spatial = self.refinement(spatial)
        return self._assemble_bev(
            self.evidence_head(spatial),
            self.routing_head(spatial),
            assemble_runtime_outputs=assemble_runtime_outputs,
        )

    def forward(
        self,
        extraction: dict,
        *,
        include_merged: bool = True,
        include_scale: bool = True,
        include_latest_auxiliary: bool = False,
        assemble_runtime_outputs: bool = True,
    ) -> dict:
        if "camera_register_tokens" not in extraction:
            raise KeyError("M05 requires frozen VGGT camera/register tokens")
        prefix = extraction["camera_register_tokens"]
        if prefix.ndim != 4 or not 1 <= prefix.shape[1] <= self.maximum_history:
            raise ValueError("M05 RGB history length is outside its contract")
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
            batch, frames = prefix.shape[:2]
            latest_geometry = self.implicit_geometry_trunk(
                prefix[:, -1:], frame_reliability=reliability[:, -1:]
            )
            geometry = self.implicit_geometry_trunk(
                prefix, frame_reliability=reliability
            ) if frames > 1 else latest_geometry
            latest_pyramid = self._add_frame_embedding(
                [level[:, -1:] for level in pyramid],
                latest_geometry,
            )
            state = self.latest_branch(
                self.query(batch),
                latest_pyramid,
                self.query.reference_grid,
            )
            if include_latest_auxiliary:
                output["latest_auxiliary_bev"] = self._decode_state(
                    state,
                    assemble_runtime_outputs=assemble_runtime_outputs,
                )
            frame_order = tuple(range(frames - 2, -1, -1))
            gate_means: list[torch.Tensor] = []
            for frame_index in frame_order:
                frame_pyramid = self._add_frame_embedding(
                    [
                        level[:, frame_index : frame_index + 1]
                        for level in pyramid
                    ],
                    geometry[:, frame_index : frame_index + 1],
                )
                for update in self.history_updates:
                    state, gate_mean = update(
                        state,
                        frame_pyramid,
                        self.query.reference_grid,
                        geometry[:, frame_index],
                        reliability[:, frame_index],
                    )
                    gate_means.append(gate_mean)
            output["history_frame_indices_newest_to_oldest"] = frame_order
            output["history_update_gate_mean"] = (
                torch.stack(gate_means, dim=1)
                if gate_means
                else reliability.new_zeros((batch, 0))
            )
            output["merged_bev"] = self._decode_state(
                state,
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


class M05System(nn.Module):
    """Frozen VGGT aggregation plus one end-to-end, non-runtime-recurrent head."""

    def __init__(self, adapter: nn.Module, **head_arguments) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = M05Head(**head_arguments)

    def unwrapped_head(self) -> M05Head:
        head = getattr(self.head, "module", self.head)
        if not isinstance(head, M05Head):
            raise TypeError("unexpected M05 head type")
        return head

    def extract(self, images: torch.Tensor) -> dict:
        if getattr(self.adapter, "supports_native_token_dtype", False):
            return self.adapter.aggregate(
                images,
                preserve_token_dtype=images.device.type == "cuda",
            )
        return self.adapter.aggregate(images)

    def decode_scale_teacher(self, extraction: dict) -> dict:
        return self.adapter.decode_scale_teacher(extraction)

    def forward_head(self, extraction: dict, **arguments) -> dict:
        prediction = self.head(extraction, **arguments)
        head = self.unwrapped_head()
        return {
            **prediction,
            "pipeline_id": head.pipeline_id,
            "runtime_inputs": ("rgb_window",),
            "runtime_passes": 1,
            "single_bev_present": False,
            "latest_auxiliary_runtime_output": False,
            "relative_pose_head_present": False,
            "extrinsic_input_present": False,
            "camera_height_input_present": False,
            "external_fusion_present": False,
            "runtime_postprocessing_present": False,
            "merged_source": "latest anchor plus reverse gated historical updates",
            "coordinate_mode": "vggt_native_units",
            "merged_extent_vggt": head.merged_extent_vggt,
            "merged_output_size": head.merged_bev_size,
            "scale_unit": "meter_per_vggt_runtime_unit",
            "scale_is_merged_input": False,
            "geometry_conditioning": "latest_anchor_reverse_gated_history",
            "maximum_history": head.maximum_history,
            "orientation": "latest ego centered; forward is image-up",
        }

    def forward(self, images: torch.Tensor) -> dict:
        extraction = self.extract(images)
        for private_key in ("_aggregated", "_patch_start", "_images"):
            extraction.pop(private_key, None)
        if images.device.type != "cuda":
            return self.forward_head(extraction)
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
            return self.forward_head(extraction)
