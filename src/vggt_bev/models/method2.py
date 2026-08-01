from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from vggt_bev.config import BatchBEVGridSpec, BEVGridSpec, LabelValues
from vggt_bev.geometry.direct_projection import (
    camera_origins_3d_in_latest_camera,
    camera_points_to_latest_camera,
    project_vggt_geometry_bevs,
)
from vggt_bev.geometry.frames import (
    camera_origins_in_reference_bev,
    opencv_points_to_reference_bev,
)
from vggt_bev.geometry.ground import (
    GroundAlignment,
    align_geometry_to_ground,
    align_geometry_to_ground_raw,
    stabilize_sequence_intrinsics,
)
from vggt_bev.geometry.lift import backproject_pixels, sample_image_at_pixels
from vggt_bev.geometry.splat import bilinear_splat, raycast_free_evidence

from .decoder import BEVDecoder
from .vggt_adapter import FrozenVGGTAdapter

DEFAULT_LABEL_VALUES = LabelValues()
MetricScaleMode = Literal[
    "learned_global",
    "camera_height",
    "vggt_raw",
    "vggt_normalized",
]


class Method2BEVHead(nn.Module):
    """Lift, confidence-weight, bilinearly splat, raycast, then refine in 2D."""

    def __init__(
        self,
        *,
        feature_dim: int = 2048,
        hidden_dim: int = 96,
        output_size: int = 512,
        obstacle_height_band_m: tuple[float, float] = (0.0, 1.4),
        filter_by_height: bool = True,
        ray_steps: int = 32,
    ) -> None:
        super().__init__()
        if obstacle_height_band_m[0] >= obstacle_height_band_m[1]:
            raise ValueError("invalid obstacle height band")
        self.obstacle_height_band_m = obstacle_height_band_m
        self.filter_by_height = filter_by_height
        self.ray_steps = ray_steps
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim + 3),
            nn.Linear(feature_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = BEVDecoder(hidden_dim + 2, hidden_dim, output_size)

    def forward(
        self,
        patch_features: torch.Tensor,
        points_bev: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        camera_origins: torch.Tensor,
        grid: BEVGridSpec,
    ) -> dict[str, torch.Tensor]:
        height = points_bev[..., 2]
        radial_distance = torch.linalg.vector_norm(points_bev[..., :2], dim=-1)
        log_confidence = torch.log(confidence.clamp_min(1e-6))
        geometric = torch.stack((height, radial_distance, log_confidence), dim=-1)
        encoded = self.point_encoder(torch.cat((patch_features, geometric), dim=-1))

        minimum_height, maximum_height = self.obstacle_height_band_m
        surface_valid = valid & torch.isfinite(points_bev).all(dim=-1)
        if self.filter_by_height:
            surface_valid = (
                surface_valid
                & (height >= minimum_height)
                & (height <= maximum_height)
            )
        projected, surface_evidence = bilinear_splat(
            points_bev[..., :2], encoded, confidence, surface_valid, grid
        )
        free_evidence = raycast_free_evidence(
            camera_origins,
            points_bev[..., :2],
            confidence,
            valid & torch.isfinite(points_bev).all(dim=-1),
            grid,
            steps=self.ray_steps,
        )
        decoder_input = torch.cat((projected, torch.log1p(surface_evidence), free_evidence), dim=1)
        output = self.decoder(decoder_input)
        output.update(
            {
                "surface_evidence": surface_evidence,
                "free_evidence": free_evidence,
            }
        )
        return output


class LearnedVGGTNormalizer(nn.Module):
    """Predict a per-sequence zoom in dimensionless VGGT-normalized space."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        *,
        initial_single_span: float = 2.0,
        single_output_size: int = 512,
    ) -> None:
        super().__init__()
        if initial_single_span <= 0:
            raise ValueError("initial_single_span must be positive")
        if single_output_size <= 1:
            raise ValueError("single_output_size must be greater than one")
        self.initial_units_per_pixel = (
            float(initial_single_span) / single_output_size
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.constant_(
            self.predictor[-1].bias,
            math.log(math.expm1(1.0)),
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if patch_features.shape[:-1] != confidence.shape:
            raise ValueError("normalizer features and confidence must align")
        if valid.shape != confidence.shape:
            raise ValueError("normalizer valid mask must align with confidence")
        weights = (
            confidence
            * valid.to(confidence.dtype)
        ).clamp_min(0.0)
        denominator = weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        pooled = (
            patch_features * weights[..., None]
        ).sum(dim=(1, 2)) / denominator[:, 0]
        zoom = torch.nn.functional.softplus(
            self.predictor(pooled).squeeze(-1)
        )
        return self.initial_units_per_pixel * (zoom + 1e-6)


class Method2System(nn.Module):
    """Frozen VGGT extractor plus one or two trainable Method II BEV heads.

    When ``merged_head`` is present, VGGT processes the cumulative history once. The
    single head receives geometry from only the latest valid frame, while the merged
    head receives geometry from every valid frame.
    """

    def __init__(
        self,
        adapter: FrozenVGGTAdapter,
        head: Method2BEVHead,
        *,
        merged_head: Method2BEVHead | None = None,
        bev_feature_size: int = 128,
        merged_bev_feature_size: int | None = None,
        initial_depth_scale: float = 1.0,
        learn_depth_scale: bool = True,
        metric_scale_mode: MetricScaleMode = "learned_global",
        normalizer_hidden_dim: int = 96,
        initial_normalized_single_span: float = 2.0,
        stabilize_intrinsics: bool = False,
        minimum_confidence: float = 0.05,
        minimum_depth_m: float = 0.05,
        maximum_depth_m: float = 20.0,
        ground_minimum_points: int = 48,
        ground_candidate_quantile: float = 0.55,
        ground_maximum_candidate_quantile: float = 0.995,
        ground_irls_iterations: int = 5,
        ground_huber_delta: float = 2.5,
        ground_maximum_tilt_degrees: float = 35.0,
    ) -> None:
        super().__init__()
        if initial_depth_scale <= 0:
            raise ValueError("initial_depth_scale must be positive")
        if metric_scale_mode not in (
            "learned_global",
            "camera_height",
            "vggt_raw",
            "vggt_normalized",
        ):
            raise ValueError(
                "metric_scale_mode must be 'learned_global', 'camera_height', "
                "'vggt_raw', or 'vggt_normalized'"
            )
        if (
            metric_scale_mode
            in ("camera_height", "vggt_raw", "vggt_normalized")
            and learn_depth_scale
        ):
            raise ValueError(
                f"{metric_scale_mode} cannot also learn a global depth scale"
            )
        if not 0.0 <= minimum_confidence < 1.0:
            raise ValueError("minimum_confidence must be in [0, 1)")
        if not 0.0 < minimum_depth_m < maximum_depth_m:
            raise ValueError("depth limits must satisfy 0 < minimum < maximum")
        self.adapter = adapter
        self.head = head
        self.merged_head = merged_head
        self.bev_feature_size = bev_feature_size
        self.merged_bev_feature_size = (
            int(merged_bev_feature_size)
            if merged_bev_feature_size is not None
            else bev_feature_size
        )
        if self.bev_feature_size <= 1 or self.merged_bev_feature_size <= 1:
            raise ValueError("BEV feature sizes must be greater than one")
        self.metric_scale_mode = metric_scale_mode
        self.stabilize_intrinsics = stabilize_intrinsics
        self.minimum_confidence = minimum_confidence
        self.minimum_depth_m = minimum_depth_m
        self.maximum_depth_m = maximum_depth_m
        self.ground_options = {
            "minimum_points": ground_minimum_points,
            "candidate_quantile": ground_candidate_quantile,
            "maximum_candidate_quantile": (
                ground_maximum_candidate_quantile
            ),
            "irls_iterations": ground_irls_iterations,
            "huber_delta": ground_huber_delta,
            "maximum_tilt_degrees": ground_maximum_tilt_degrees,
        }
        initial_log_scale = torch.tensor(math.log(initial_depth_scale), dtype=torch.float32)
        self.log_depth_scale = nn.Parameter(initial_log_scale, requires_grad=learn_depth_scale)
        self.normalizer = LearnedVGGTNormalizer(
            feature_dim=self.head.point_encoder[0].normalized_shape[0] - 3,
            hidden_dim=normalizer_hidden_dim,
            initial_single_span=initial_normalized_single_span,
            single_output_size=self.head.decoder.output_size,
        )
        self.normalizer.requires_grad_(metric_scale_mode == "vggt_normalized")

    @property
    def depth_scale(self) -> torch.Tensor:
        return self.log_depth_scale.exp()

    @staticmethod
    def _grid(extents: torch.Tensor, size: int) -> BEVGridSpec:
        if not torch.allclose(extents, extents[:1]):
            raise ValueError("all samples in a batch must have the same target extent")
        return BEVGridSpec(
            extent_m=float(extents[0].detach().cpu()),
            height=size,
            width=size,
        )

    @staticmethod
    def _latest_frame_mask(frame_valid: torch.Tensor) -> torch.Tensor:
        if not frame_valid.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid frame")
        latest = frame_valid.sum(dim=1, dtype=torch.long) - 1
        mask = torch.zeros_like(frame_valid)
        mask.scatter_(1, latest[:, None], True)
        return mask

    @staticmethod
    def _reference_scale(
        points_bev: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return a per-sequence characteristic radius in raw VGGT coordinates."""

        radial = torch.linalg.vector_norm(points_bev[..., :2], dim=-1)
        usable = (
            valid
            & torch.isfinite(radial)
            & torch.isfinite(confidence)
            & (radial > 1e-6)
        )
        weights = confidence * usable.to(confidence.dtype)
        numerator = (radial * weights).sum(dim=(1, 2))
        denominator = weights.sum(dim=(1, 2))
        reference = numerator / denominator.clamp_min(1e-6)
        return torch.where(
            (denominator > 0) & torch.isfinite(reference) & (reference > 1e-6),
            reference,
            torch.ones_like(reference),
        )

    def _normalized_grids(
        self,
        extraction: dict[str, torch.Tensor],
        points_bev: torch.Tensor,
        camera_origins: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        BatchBEVGridSpec,
        BatchBEVGridSpec | None,
        dict[str, torch.Tensor],
    ]:
        reference_scale = self._reference_scale(
            points_bev,
            confidence,
            valid,
        )
        point_scale = reference_scale[:, None, None, None]
        origin_scale = reference_scale[:, None, None]
        normalized_points = points_bev / point_scale
        normalized_origins = camera_origins / origin_scale
        units_per_output_pixel = self.normalizer(
            extraction["patch_features"],
            confidence,
            valid,
        )
        single_span = (
            units_per_output_pixel * self.head.decoder.output_size
        )
        single_grid = BatchBEVGridSpec(
            single_span,
            self.bev_feature_size,
            self.bev_feature_size,
        )
        merged_grid: BatchBEVGridSpec | None = None
        merged_span = single_span.new_zeros(single_span.shape)
        if self.merged_head is not None:
            merged_span = (
                units_per_output_pixel
                * self.merged_head.decoder.output_size
            )
            merged_grid = BatchBEVGridSpec(
                merged_span,
                self.merged_bev_feature_size,
                self.merged_bev_feature_size,
            )
        report = {
            "reference_scale_vggt": reference_scale,
            "normalized_units_per_output_pixel": units_per_output_pixel,
            "single_span_normalized": single_span,
            "merged_span_normalized": merged_span,
        }
        return (
            normalized_points,
            normalized_origins,
            single_grid,
            merged_grid,
            report,
        )

    def forward(self, batch: dict) -> dict:
        return_geometry = bool(batch.get("return_geometry", False))
        use_predicted_geometry = bool(batch.get("use_predicted_geometry", False))
        extraction = batch.get("vggt_extraction")
        if extraction is None:
            extraction = self.adapter(
                batch["images"],
                include_geometry=return_geometry or use_predicted_geometry,
            )
        centers = extraction["patch_centers"]
        raw_depth = extraction["depth"]
        patch_image_valid = sample_image_at_pixels(
            batch["image_valid"][:, :, None].to(raw_depth.dtype),
            centers,
            mode="nearest",
        )[..., 0] > 0.5
        raw_valid = (
            batch["frame_valid"][:, :, None]
            & patch_image_valid
            & torch.isfinite(raw_depth)
            & (raw_depth > 0)
        )
        # VGGT confidence is 1 + exp(logit), so this exactly recovers sigmoid(logit).
        confidence = (
            (extraction["confidence"] - 1.0) / extraction["confidence"]
        ).clamp(0.0, 1.0)
        ground_alignment: GroundAlignment | None = None
        stable_intrinsics: torch.Tensor | None = None
        if use_predicted_geometry:
            if (
                self.metric_scale_mode
                not in ("vggt_raw", "vggt_normalized")
                and "camera_height_m" not in batch
            ):
                raise ValueError(
                    "VGGT-predicted geometry requires configured camera_height_m"
                )
            stable_intrinsics = extraction["estimated_intrinsics"]
            if self.stabilize_intrinsics:
                stable_intrinsics = stabilize_sequence_intrinsics(
                    stable_intrinsics,
                    batch["frame_valid"],
                )
            points_camera = backproject_pixels(
                raw_depth,
                centers,
                stable_intrinsics,
            )
            points_reference = camera_points_to_latest_camera(
                points_camera,
                extraction["estimated_camera_from_world"],
                batch["frame_valid"],
            )
            camera_origins_reference = camera_origins_3d_in_latest_camera(
                extraction["estimated_camera_from_world"],
                batch["frame_valid"],
            )
            if self.metric_scale_mode in (
                "camera_height",
                "vggt_raw",
                "vggt_normalized",
            ):
                if self.metric_scale_mode == "camera_height":
                    ground_alignment = align_geometry_to_ground(
                        points_reference,
                        camera_origins_reference,
                        confidence,
                        # Ground fitting already confidence-weights points. Keep
                        # low-confidence finite geometry available so sparse views
                        # can still establish metric scale; the stricter threshold
                        # remains in force for BEV feature projection below.
                        raw_valid,
                        batch["camera_height_m"],
                        **self.ground_options,
                    )
                else:
                    ground_alignment = align_geometry_to_ground_raw(
                        points_reference,
                        camera_origins_reference,
                        confidence,
                        raw_valid,
                        **self.ground_options,
                    )
                applied_scale = ground_alignment.metric_scale
                metric_depth = raw_depth * applied_scale[:, None, None]
                points_bev = ground_alignment.points
                camera_origins = ground_alignment.camera_origins[..., :2]
            else:
                applied_scale = self.depth_scale
                metric_depth = raw_depth * applied_scale
                scaled_points_reference = points_reference * applied_scale
                points_bev = torch.stack(
                    (
                        scaled_points_reference[..., 0],
                        scaled_points_reference[..., 2],
                        batch["camera_height_m"][:, None, None]
                        - scaled_points_reference[..., 1],
                    ),
                    dim=-1,
                )
                camera_origins = (
                    camera_origins_reference[..., (0, 2)] * applied_scale
                )
        else:
            applied_scale = self.depth_scale
            metric_depth = raw_depth * applied_scale
            points_camera = backproject_pixels(
                metric_depth,
                centers,
                batch["intrinsics"],
            )
            points_bev = opencv_points_to_reference_bev(
                points_camera,
                batch["camera_to_world"],
                batch["reference_world_from_bev"],
                batch["floor_y"],
            )
            camera_origins = camera_origins_in_reference_bev(
                batch["camera_to_world"],
                batch["reference_world_from_bev"],
            )
        valid = (
            raw_valid
            & torch.isfinite(metric_depth)
            & (metric_depth >= self.minimum_depth_m)
            & (metric_depth <= self.maximum_depth_m)
            & (confidence >= self.minimum_confidence)
        )
        normalization: dict[str, torch.Tensor] | None = None
        if self.metric_scale_mode == "vggt_normalized":
            (
                points_bev,
                camera_origins,
                single_grid,
                merged_grid,
                normalization,
            ) = self._normalized_grids(
                extraction,
                points_bev,
                camera_origins,
                confidence,
                valid,
            )
        else:
            single_extent_key = (
                "target_extent_m"
                if self.merged_head is None
                else "single_target_extent_m"
            )
            single_grid = self._grid(
                batch[single_extent_key],
                self.bev_feature_size,
            )
            merged_grid = (
                self._grid(
                    batch["merged_target_extent_m"],
                    self.merged_bev_feature_size,
                )
                if self.merged_head is not None
                else None
            )

        if self.merged_head is None:
            output = self.head(
                extraction["patch_features"],
                points_bev,
                confidence,
                valid,
                camera_origins,
                single_grid,
            )
            output["depth_scale"] = applied_scale
            if normalization is not None:
                output["normalization"] = normalization
            if ground_alignment is not None:
                output["ground"] = self._ground_report(ground_alignment)
            return output

        single_valid = (
            valid
            & self._latest_frame_mask(batch["frame_valid"])[:, :, None]
        )
        single = self.head(
            extraction["patch_features"],
            points_bev,
            confidence,
            single_valid,
            camera_origins,
            single_grid,
        )
        if bool(batch.get("single_only", False)):
            output = {
                "single": single,
                "depth_scale": applied_scale,
            }
            if normalization is not None:
                output["normalization"] = normalization
            if ground_alignment is not None:
                output["ground"] = self._ground_report(ground_alignment)
            return output
        assert merged_grid is not None
        merged = self.merged_head(
            extraction["patch_features"],
            points_bev,
            confidence,
            valid,
            camera_origins,
            merged_grid,
        )
        output = {
            "single": single,
            "merged": merged,
            "depth_scale": applied_scale,
        }
        if normalization is not None:
            output["normalization"] = normalization
        if ground_alignment is not None:
            output["ground"] = self._ground_report(ground_alignment)
        if return_geometry:
            if use_predicted_geometry:
                camera_height_m = batch.get("camera_height_m")
                if camera_height_m is None:
                    camera_height_m = torch.ones(
                        batch["images"].shape[0],
                        dtype=raw_depth.dtype,
                        device=raw_depth.device,
                    )
            else:
                latest = self._latest_frame_mask(batch["frame_valid"])
                latest_index = latest.to(torch.long).argmax(dim=1)
                batch_index = torch.arange(
                    batch["images"].shape[0],
                    device=batch["images"].device,
                )
                camera_height_m = (
                    batch["camera_to_world"][
                        batch_index,
                        latest_index,
                        1,
                        3,
                    ]
                    - batch["floor_y"]
                )
            if normalization is not None:
                geometry_single_extent: float | torch.Tensor = (
                    normalization["single_span_normalized"]
                )
                geometry_merged_extent: float | torch.Tensor = (
                    normalization["merged_span_normalized"]
                )
                coordinate_divisor = normalization[
                    "reference_scale_vggt"
                ]
            else:
                geometry_single_extent = float(
                    batch["single_target_extent_m"][0].detach().cpu()
                )
                geometry_merged_extent = float(
                    batch["merged_target_extent_m"][0].detach().cpu()
                )
                coordinate_divisor = None
            output["geometry"] = project_vggt_geometry_bevs(
                dense_depth=extraction["dense_depth"],
                dense_confidence=extraction["dense_confidence"],
                estimated_intrinsics=extraction["estimated_intrinsics"],
                estimated_camera_from_world=extraction[
                    "estimated_camera_from_world"
                ],
                image_valid=batch["image_valid"],
                frame_valid=batch["frame_valid"],
                camera_height_m=camera_height_m,
                single_extent_m=geometry_single_extent,
                merged_extent_m=geometry_merged_extent,
                output_size=self.head.decoder.output_size,
                merged_output_size=(
                    self.merged_head.decoder.output_size
                ),
                depth_scale=applied_scale,
                stabilized_intrinsics=stable_intrinsics,
                ground_alignment=ground_alignment,
                coordinate_divisor=coordinate_divisor,
            )
        return output

    @staticmethod
    def _ground_report(alignment: GroundAlignment) -> dict[str, torch.Tensor]:
        return {
            "normal": alignment.normal,
            "origin": alignment.origin,
            "right": alignment.right,
            "forward": alignment.forward,
            "predicted_camera_height": alignment.predicted_camera_height,
            "metric_scale": alignment.metric_scale,
            "inlier_fraction": alignment.inlier_fraction,
            "fallback_used": alignment.fallback_used,
        }


def render_observed_bev(
    occupancy_logit: torch.Tensor,
    observed_logit: torch.Tensor,
    *,
    threshold: float = 0.5,
    labels: LabelValues = DEFAULT_LABEL_VALUES,
) -> torch.Tensor:
    """Render model logits to the simulator's occupied/unknown/free uint8 convention."""

    occupied = occupancy_logit.sigmoid() >= threshold
    observed = observed_logit.sigmoid() >= threshold
    output = torch.full_like(occupancy_logit, labels.unknown, dtype=torch.uint8)
    output[observed & ~occupied] = labels.free
    output[observed & occupied] = labels.occupied
    return output
