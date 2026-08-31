from __future__ import annotations

import math

import torch
from torch import nn

from .attention import DirectDecoderBlock
from .method1 import MetricScaleTokenHead, MultiScaleTokenProjector
from .p1b_probability import (
    compose_semantic,
    decode_binary_prediction,
    fuse_pixel_routing,
)
from .p1d import FrameReliabilityHead
from .wtbd_merge_scale import ImplicitGeometryContextTrunk


def _native_query_coordinates(
    size: int,
    extent_vggt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if size <= 0 or extent_vggt <= 0.0:
        raise ValueError("M04 query size and extent must be positive")
    cell = extent_vggt / size
    pixel = torch.arange(size, dtype=torch.float32)
    x = -extent_vggt / 2.0 + (pixel + 0.5) * cell
    z = extent_vggt / 2.0 - (pixel + 0.5) * cell
    z_grid, x_grid = torch.meshgrid(z, x, indexing="ij")
    normalized = torch.stack(
        (x_grid / (extent_vggt / 2.0), z_grid / (extent_vggt / 2.0)),
        dim=-1,
    ).reshape(-1, 2)
    reference = torch.stack((normalized[:, 0], -normalized[:, 1]), dim=-1)
    return normalized, reference


class FactorizedNativeQuery(nn.Module):
    """Continuous native-grid query without one Parameter per BEV pixel.

    Row and column factors retain exact native-pixel identity, while fixed
    Fourier coordinates give the decoder a smooth spatial prior.  The query is
    evaluated directly at the output resolution; no latent raster is resized.
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
            raise ValueError("M04 Fourier band count must be positive")
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
        self.row_embedding = nn.Parameter(torch.empty(size, hidden_dim))
        self.column_embedding = nn.Parameter(torch.empty(size, hidden_dim))
        coordinate_dim = 2 + fourier_bands * 2 * 2
        self.coordinate_projection = nn.Sequential(
            nn.Linear(coordinate_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.column_embedding, std=0.02)

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
            raise ValueError("M04 query batch size must be positive")
        factor = (
            self.row_embedding[:, None, :] + self.column_embedding[None, :, :]
        ).reshape(self.size * self.size, -1)
        coordinate = self.coordinate_projection(
            self._coordinate_features().to(dtype=factor.dtype)
        )
        return (factor + coordinate).unsqueeze(0).expand(batch_size, -1, -1)


class ParallelQueryBranch(nn.Module):
    """One independent latest or history query branch."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        layers: int,
        feature_levels: int,
        self_attention_mode: str,
        cross_attention_mode: str,
        deformable_samples: int,
        cross_query_chunk_size: int,
    ) -> None:
        super().__init__()
        self.cross_attention_mode = str(cross_attention_mode)
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
        self.output_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _flatten_context(pyramid: list[torch.Tensor]) -> torch.Tensor:
        batch = pyramid[0].shape[0]
        return torch.cat(
            [
                level.permute(0, 1, 3, 4, 2).reshape(
                    batch, -1, level.shape[2]
                )
                for level in pyramid
            ],
            dim=1,
        )

    @staticmethod
    def _flatten_weights(
        pyramid: list[torch.Tensor],
        frame_reliability: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames = frame_reliability.shape
        if any(level.shape[:2] != (batch, frames) for level in pyramid):
            raise ValueError("M04 pyramid and reliability frame counts disagree")
        return torch.cat(
            [
                frame_reliability.repeat_interleave(
                    level.shape[-2] * level.shape[-1], dim=1
                )
                for level in pyramid
            ],
            dim=1,
        )

    def forward(
        self,
        query: torch.Tensor,
        pyramid: list[torch.Tensor],
        reference_grid: torch.Tensor,
        *,
        frame_reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not pyramid:
            raise ValueError("M04 query branch requires a token pyramid")
        context_weight = None
        if self.cross_attention_mode == "deformable":
            context: torch.Tensor | list[torch.Tensor] = pyramid
            if frame_reliability is not None:
                raise ValueError("deformable latest branch does not use frame weights")
        else:
            context = self._flatten_context(pyramid)
            if frame_reliability is not None:
                context_weight = self._flatten_weights(
                    pyramid, frame_reliability
                )
        for block in self.blocks:
            query = block(
                query,
                context,
                reference_grid,
                context_weight=context_weight,
            )
        return self.output_norm(query)


class SpatialResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.network = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(self.norm(value))


class M04Head(nn.Module):
    """Parallel latest-anchor/history Merged BEV and metric-scale head."""

    pipeline_id = "M04-PARALLEL-ANCHOR-HISTORY-MERGED-SCALE-NLL"

    def __init__(
        self,
        *,
        cached_layers: tuple[int, ...] = (4, 11, 17, 23),
        spatial_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
        vggt_token_dim: int = 2048,
        hidden_dim: int = 64,
        heads: int = 8,
        latest_decoder_layers: int = 2,
        history_decoder_layers: int = 2,
        scale_decoder_layers: int = 2,
        self_attention_mode: str = "linear",
        deformable_samples: int = 4,
        cross_query_chunk_size: int = 4096,
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
    ) -> None:
        super().__init__()
        if latest_decoder_layers <= 0 or history_decoder_layers <= 0:
            raise ValueError("M04 decoder depths must be positive")
        if refinement_layers <= 0:
            raise ValueError("M04 requires at least one spatial refinement block")
        projector_args = (
            cached_layers,
            vggt_token_dim,
            hidden_dim,
            spatial_scales,
        )
        self.maximum_history = int(maximum_history)
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
        # Keep the scalar-scale objective from updating the BEV temporal
        # routing policy (and vice versa). Both heads see only frozen VGGT
        # prefix tokens, but they share no trainable reliability parameters.
        self.scale_frame_reliability = FrameReliabilityHead(
            vggt_token_dim,
            frame_reliability_hidden_dim,
            minimum=frame_reliability_minimum,
            maximum=frame_reliability_maximum,
        )
        self.query = FactorizedNativeQuery(
            size=merged_bev_size,
            extent_vggt=merged_extent_vggt,
            hidden_dim=hidden_dim,
            fourier_bands=query_fourier_bands,
        )
        branch_args = dict(
            hidden_dim=hidden_dim,
            heads=heads,
            feature_levels=len(cached_layers),
            self_attention_mode=self_attention_mode,
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
        )
        self.latest_branch = ParallelQueryBranch(
            layers=latest_decoder_layers,
            cross_attention_mode="deformable",
            **branch_args,
        )
        self.history_branch = ParallelQueryBranch(
            layers=history_decoder_layers,
            cross_attention_mode="linear",
            **branch_args,
        )
        self.latest_reference_projection = nn.Sequential(
            nn.LayerNorm(vggt_token_dim),
            nn.Linear(vggt_token_dim, hidden_dim),
        )
        self.history_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
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
            level
            + frame_embedding[:, :, :, None, None].to(dtype=level.dtype)
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

    def forward(
        self,
        extraction: dict,
        *,
        include_merged: bool = True,
        include_scale: bool = True,
        assemble_runtime_outputs: bool = True,
    ) -> dict:
        if "camera_register_tokens" not in extraction:
            raise KeyError("M04 requires frozen VGGT camera/register tokens")
        prefix = extraction["camera_register_tokens"]
        if prefix.ndim != 4 or not 1 <= prefix.shape[1] <= self.maximum_history:
            raise ValueError("M04 RGB history length is outside its contract")
        reliability = self.frame_reliability(prefix)
        frame_keep_mask = torch.ones_like(reliability, dtype=torch.bool)
        output: dict = {
            "frame_reliability": reliability,
            "effective_frame_reliability": reliability,
            "frame_keep_mask": frame_keep_mask,
        }
        if include_merged:
            spatial_pyramid = self.spatial_token_projector(
                extraction["tokens"], extraction["patch_grid"]
            )
            batch, frames = prefix.shape[:2]
            base_query = self.query(batch)
            latest = self.latest_branch(
                base_query,
                [level[:, -1:] for level in spatial_pyramid],
                self.query.reference_grid,
            )
            if frames == 1:
                history = torch.zeros_like(latest)
            else:
                frame_embedding = self.implicit_geometry_trunk(
                    prefix, frame_reliability=reliability
                )
                history_pyramid = self._add_frame_embedding(
                    [level[:, :-1] for level in spatial_pyramid],
                    frame_embedding[:, :-1],
                )
                latest_reference = self.latest_reference_projection(
                    prefix[:, -1].mean(dim=1)
                )
                history = self.history_branch(
                    base_query + latest_reference[:, None, :],
                    history_pyramid,
                    self.query.reference_grid,
                    frame_reliability=reliability[:, :-1],
                )
            gate = self.fusion_gate(torch.cat((latest, history), dim=-1))
            fused = latest + gate * self.history_projection(history)
            spatial = fused.transpose(1, 2).reshape(
                batch,
                -1,
                self.merged_bev_size,
                self.merged_bev_size,
            )
            spatial = self.refinement(spatial)
            output["merged_bev"] = self._assemble_bev(
                self.evidence_head(spatial),
                self.routing_head(spatial),
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


class M04System(nn.Module):
    """Frozen VGGT aggregation plus one non-recurrent M04 head."""

    def __init__(self, adapter: nn.Module, **head_arguments) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = M04Head(**head_arguments)

    def unwrapped_head(self) -> M04Head:
        head = getattr(self.head, "module", self.head)
        if not isinstance(head, M04Head):
            raise TypeError("unexpected M04 head type")
        return head

    def extract(self, images: torch.Tensor) -> dict:
        # The frozen CUDA trunk already emits BF16 tokens under autocast and
        # the train/runtime head consumes BF16.  Preserve that native dtype so
        # M04 does not materialize multi-gigabyte FP32 copies only to cast them
        # straight back to BF16 before decoding.  The default adapter contract
        # remains FP32 for every other consumer.
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
            "relative_pose_head_present": False,
            "extrinsic_input_present": False,
            "camera_height_input_present": False,
            "external_fusion_present": False,
            "runtime_postprocessing_present": False,
            "merged_source": "frozen VGGT tokens with parallel latest/history branches",
            "coordinate_mode": "vggt_native_units",
            "merged_extent_vggt": head.merged_extent_vggt,
            "merged_output_size": head.merged_bev_size,
            "scale_unit": "meter_per_vggt_runtime_unit",
            "scale_is_merged_input": False,
            "geometry_conditioning": "parallel_latest_anchor_and_implicit_history",
            "maximum_history": head.maximum_history,
            "orientation": "latest ego centered; forward is image-up",
        }

    def forward(self, images: torch.Tensor) -> dict:
        extraction = self.extract(images)
        # Runtime never invokes the frozen depth teacher. Release its large
        # aggregator/image references before native-grid BEV decoding.
        for private_key in ("_aggregated", "_patch_start", "_images"):
            extraction.pop(private_key, None)
        if images.device.type != "cuda":
            return self.forward_head(extraction)

        # Match the training/evaluation execution contract. The frozen VGGT
        # adapter exposes cached tokens as FP32 for general consumers, but M04
        # is trained under CUDA AMP. Replacing those references before the
        # native 512-grid head also releases the unnecessary FP32 copies.
        runtime_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
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
