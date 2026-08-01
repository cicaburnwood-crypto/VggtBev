from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .vggt_adapter import world_camera_centers


@dataclass(frozen=True)
class GeometryBuilderConfig:
    """Runtime P1A fixed-grid metric-geometry settings."""

    sample_stride: int = 4
    confidence_threshold: float = 0.25
    minimum_points: int = 64
    single_extent_normalized_scale: float = 6.5
    merged_extent_normalized_scale: float = 10.0
    minimum_ground_quality: float = 0.05
    ransac_hypotheses: int = 48
    huber_iterations: int = 4
    fov_chunk_size: int = 65536

    def __post_init__(self) -> None:
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")
        if not 0.0 <= self.confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be in [0, 1)")
        if self.minimum_points < 3:
            raise ValueError("minimum_points must be at least three")
        if self.single_extent_normalized_scale != 6.5:
            raise ValueError("P1A single normalized-scale extent must be 6.5")
        if self.merged_extent_normalized_scale != 10.0:
            raise ValueError("P1A merged normalized-scale extent must be 10.0")
        if not 0.0 <= self.minimum_ground_quality <= 1.0:
            raise ValueError("minimum_ground_quality must be in [0, 1]")
        if self.ransac_hypotheses <= 0 or self.huber_iterations <= 0:
            raise ValueError("RANSAC hypotheses and Huber iterations must be positive")
        if self.fov_chunk_size <= 0:
            raise ValueError("fov_chunk_size must be positive")


def _confidence_probability(confidence: torch.Tensor) -> torch.Tensor:
    return ((confidence - 1.0) / confidence.clamp_min(1.0)).clamp(0.0, 1.0)


def _latest_camera_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_from_world: torch.Tensor,
    *,
    sample_stride: int,
) -> torch.Tensor:
    """Unproject sampled depth and align every frame to the latest camera."""

    batch, frames, height, width = depth.shape
    rows = torch.arange(
        0,
        height,
        sample_stride,
        device=depth.device,
        dtype=depth.dtype,
    )
    columns = torch.arange(
        0,
        width,
        sample_stride,
        device=depth.device,
        dtype=depth.dtype,
    )
    v, u = torch.meshgrid(rows, columns, indexing="ij")
    sampled_depth = depth[..., ::sample_stride, ::sample_stride]
    fx = intrinsics[..., 0, 0, None, None].clamp_min(1e-6)
    fy = intrinsics[..., 1, 1, None, None].clamp_min(1e-6)
    cx = intrinsics[..., 0, 2, None, None]
    cy = intrinsics[..., 1, 2, None, None]
    camera_points = torch.stack(
        (
            sampled_depth * (u.view(1, 1, *u.shape) - cx) / fx,
            sampled_depth * (v.view(1, 1, *v.shape) - cy) / fy,
            sampled_depth,
        ),
        dim=-1,
    )

    rotation = camera_from_world[..., :3, :3]
    translation = camera_from_world[..., :3, 3]
    world_points = (
        rotation.transpose(-1, -2)[..., None, None, :, :]
        @ (camera_points - translation[..., None, None, :])[..., None]
    )[..., 0]
    latest_rotation = rotation[:, -1:]
    latest_translation = translation[:, -1:]
    latest_points = (
        latest_rotation[..., None, None, :, :] @ world_points[..., None]
    )[..., 0] + latest_translation[..., None, None, :]
    return latest_points.view(batch, frames, -1, 3)


def _fit_ground_frame(
    points: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
    *,
    minimum_points: int,
    ransac_hypotheses: int,
    huber_iterations: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Estimate a confidence-weighted deterministic RANSAC/Huber ground."""

    normals = []
    origins = []
    rights = []
    forwards = []
    qualities = []
    confidence_qualities = []
    residual_qualities = []
    inlier_qualities = []
    temporal_qualities = []
    camera_up = points.new_tensor([0.0, -1.0, 0.0])
    camera_forward = points.new_tensor([0.0, 0.0, 1.0])
    for batch_index in range(points.shape[0]):
        sample_points = points[batch_index].reshape(-1, 3)
        sample_valid = valid[batch_index].reshape(-1)
        sample_confidence = confidence[batch_index].reshape(-1)
        frame_ids = (
            torch.arange(
                points.shape[1],
                device=points.device,
            )[:, None]
            .expand(points.shape[1], points.shape[2])
            .reshape(-1)
        )
        selected = sample_points[sample_valid]
        selected_confidence = sample_confidence[sample_valid]
        selected_frames = frame_ids[sample_valid]
        if selected.shape[0] >= minimum_points:
            lower_cut = torch.quantile(selected[:, 1], 0.55)
            lower = selected[:, 1] >= lower_cut
            candidates = selected[lower]
            candidate_confidence = selected_confidence[lower].clamp_min(1e-3)
            candidate_frames = selected_frames[lower]
            if candidates.shape[0] < minimum_points:
                candidates = selected
                candidate_confidence = selected_confidence.clamp_min(1e-3)
                candidate_frames = selected_frames

            count = candidates.shape[0]
            hypotheses = min(ransac_hypotheses, count)
            first = torch.linspace(
                0,
                count - 1,
                hypotheses,
                device=points.device,
            ).round().long()
            second = (first + max(1, count // 3)) % count
            third = (first + max(2, 2 * count // 3 + 1)) % count
            first_point = candidates[first]
            hypothesis_normal = torch.cross(
                candidates[second] - first_point,
                candidates[third] - first_point,
                dim=-1,
            )
            hypothesis_norm = hypothesis_normal.norm(dim=-1)
            hypothesis_valid = hypothesis_norm > 1e-6
            hypothesis_normal = hypothesis_normal / hypothesis_norm.clamp_min(
                1e-6
            )[:, None]
            orientation = torch.sign(hypothesis_normal @ camera_up)
            orientation = torch.where(
                orientation == 0,
                torch.ones_like(orientation),
                orientation,
            )
            hypothesis_normal = hypothesis_normal * orientation[:, None]
            hypothesis_offset = -(
                hypothesis_normal * first_point
            ).sum(dim=-1)
            distance_scale = candidates.norm(dim=-1).median().clamp_min(1e-6)
            huber_delta = (0.015 * distance_scale).clamp_min(1e-4)
            hypothesis_residual = (
                hypothesis_normal[:, None, :] * candidates[None]
            ).sum(dim=-1) + hypothesis_offset[:, None]
            hypothesis_score = (
                candidate_confidence[None]
                * (
                    1.0
                    - hypothesis_residual.abs() / huber_delta
                ).clamp(0.0, 1.0)
            ).sum(dim=-1)
            hypothesis_score = torch.where(
                hypothesis_valid,
                hypothesis_score,
                torch.full_like(hypothesis_score, -1.0),
            )
            normal = hypothesis_normal[hypothesis_score.argmax()]

            robust_weights = candidate_confidence
            center = candidates.median(dim=0).values
            for _ in range(huber_iterations):
                weight_sum = robust_weights.sum().clamp_min(1e-6)
                center = (
                    candidates * robust_weights[:, None]
                ).sum(dim=0) / weight_sum
                centered = candidates - center
                covariance = (
                    centered.transpose(0, 1)
                    @ (centered * robust_weights[:, None])
                ) / weight_sum
                _, eigenvectors = torch.linalg.eigh(covariance.float())
                refined = eigenvectors[:, 0].to(points.dtype)
                if torch.dot(refined, normal) < 0:
                    refined = -refined
                normal = F.normalize(refined, dim=0)
                residual = ((candidates - center) @ normal).abs()
                robust_weights = candidate_confidence * torch.clamp(
                    huber_delta / residual.clamp_min(1e-6),
                    max=1.0,
                )
            if torch.dot(normal, camera_up) < 0:
                normal = -normal
            normal = F.normalize(normal, dim=0)
            offset = -torch.dot(normal, center)
            origin = -offset * normal
            residual = ((candidates - origin) @ normal).abs().median()
            residual_quality = torch.exp(
                -residual / huber_delta.clamp_min(1e-6)
            )
            point_quality = candidates.new_tensor(
                min(candidates.shape[0] / float(minimum_points * 4), 1.0)
            )
            confidence_quality = candidate_confidence.mean().clamp(0.0, 1.0)
            candidate_residual = ((candidates - origin) @ normal).abs()
            inlier = candidate_residual <= 2.0 * huber_delta
            inlier_quality = (
                candidate_confidence[inlier].sum()
                / candidate_confidence.sum().clamp_min(1e-6)
            ).clamp(0.0, 1.0)
            frame_inlier_ratios = []
            for frame_index in torch.unique(candidate_frames):
                frame_mask = candidate_frames == frame_index
                frame_inlier_ratios.append(
                    candidate_confidence[frame_mask & inlier].sum()
                    / candidate_confidence[frame_mask].sum().clamp_min(1e-6)
                )
            temporal_quality = (
                (
                    1.0
                    - torch.stack(frame_inlier_ratios).std(
                        unbiased=False
                    )
                ).clamp(0.0, 1.0)
                if frame_inlier_ratios
                else candidates.new_tensor(0.0)
            )
            quality = (
                point_quality
                * confidence_quality
                * residual_quality
                * inlier_quality
                * temporal_quality
            )
        else:
            normal = camera_up
            origin = points.new_zeros(3)
            quality = points.new_tensor(0.0)
            confidence_quality = points.new_tensor(0.0)
            residual_quality = points.new_tensor(0.0)
            inlier_quality = points.new_tensor(0.0)
            temporal_quality = points.new_tensor(0.0)
        forward = camera_forward - torch.dot(camera_forward, normal) * normal
        if torch.linalg.vector_norm(forward) < 1e-4:
            forward = camera_forward
        forward = F.normalize(forward, dim=0)
        right = F.normalize(torch.cross(forward, normal, dim=0), dim=0)
        normals.append(normal)
        origins.append(origin)
        rights.append(right)
        forwards.append(forward)
        qualities.append(quality)
        confidence_qualities.append(confidence_quality)
        residual_qualities.append(residual_quality)
        inlier_qualities.append(inlier_quality)
        temporal_qualities.append(temporal_quality)
    return (
        torch.stack(normals),
        torch.stack(origins),
        torch.stack(rights),
        torch.stack(forwards),
        torch.stack(qualities).clamp(0.0, 1.0),
        torch.stack(confidence_qualities).clamp(0.0, 1.0),
        torch.stack(residual_qualities).clamp(0.0, 1.0),
        torch.stack(inlier_qualities).clamp(0.0, 1.0),
        torch.stack(temporal_qualities).clamp(0.0, 1.0),
    )


def _runtime_fov_support(
    *,
    intrinsics: torch.Tensor,
    camera_from_world: torch.Tensor,
    ground_origin: torch.Tensor,
    ground_normal: torch.Tensor,
    ground_right: torch.Tensor,
    ground_forward: torch.Tensor,
    metric_per_native: torch.Tensor,
    geometry_valid: torch.Tensor,
    output_size: int,
    output_extent_m: float,
    source_extent_m: float,
    image_width: int,
    frame_indices: tuple[int, ...] | None,
    chunk_size: int,
) -> torch.Tensor:
    """Project a fixed ground grid into same-run VGGT cameras."""

    batch, frames = camera_from_world.shape[:2]
    selected_frames = (
        tuple(range(frames)) if frame_indices is None else frame_indices
    )
    cell = output_extent_m / output_size
    axis = torch.linspace(
        -output_extent_m / 2.0 + cell / 2.0,
        output_extent_m / 2.0 - cell / 2.0,
        output_size,
        device=intrinsics.device,
        dtype=intrinsics.dtype,
    )
    z, x = torch.meshgrid(axis.flip(0), axis, indexing="ij")
    x = x.reshape(-1)
    z = z.reshape(-1)
    support = torch.zeros(
        batch,
        output_size * output_size,
        device=intrinsics.device,
        dtype=torch.bool,
    )
    camera_forward = intrinsics.new_tensor([0.0, 0.0, 1.0])
    rotation = camera_from_world[..., :3, :3]
    translation = camera_from_world[..., :3, 3]
    latest_rotation = rotation[:, -1]
    latest_translation = translation[:, -1]
    half_source = source_extent_m / 2.0
    for batch_index in range(batch):
        if not bool(geometry_valid[batch_index]):
            continue
        normal = ground_normal[batch_index]
        latest_from_world = latest_rotation[batch_index]
        latest_translation_value = latest_translation[batch_index]
        frame_geometry = []
        for frame_index in selected_frames:
            source_rotation = rotation[batch_index, frame_index]
            source_translation = translation[batch_index, frame_index]
            source_center_world = -(
                source_rotation.transpose(0, 1) @ source_translation
            )
            source_center_latest = (
                latest_from_world @ source_center_world
                + latest_translation_value
            )
            signed_distance = torch.dot(
                source_center_latest - ground_origin[batch_index],
                normal,
            )
            source_foot = source_center_latest - signed_distance * normal
            source_forward_world = (
                source_rotation.transpose(0, 1) @ camera_forward
            )
            source_forward_latest = latest_from_world @ source_forward_world
            source_forward_ground = (
                source_forward_latest
                - torch.dot(source_forward_latest, normal) * normal
            )
            if source_forward_ground.norm() < 1e-5:
                source_forward_ground = ground_forward[batch_index]
            source_forward_ground = F.normalize(
                source_forward_ground,
                dim=0,
            )
            source_right_ground = F.normalize(
                torch.cross(source_forward_ground, normal, dim=0),
                dim=0,
            )
            source_from_latest_rotation = (
                source_rotation @ latest_from_world.transpose(0, 1)
            )
            source_from_latest_translation = (
                source_translation
                - source_from_latest_rotation @ latest_translation_value
            )
            frame_geometry.append(
                (
                    frame_index,
                    source_foot,
                    source_right_ground,
                    source_forward_ground,
                    source_from_latest_rotation,
                    source_from_latest_translation,
                )
            )

        native_per_metric = metric_per_native[batch_index].reciprocal()
        for start in range(0, x.numel(), chunk_size):
            stop = min(start + chunk_size, x.numel())
            points_latest = (
                ground_origin[batch_index][None]
                + x[start:stop, None]
                * native_per_metric
                * ground_right[batch_index][None]
                + z[start:stop, None]
                * native_per_metric
                * ground_forward[batch_index][None]
            )
            chunk_support = torch.zeros(
                stop - start,
                device=intrinsics.device,
                dtype=torch.bool,
            )
            for (
                frame_index,
                source_foot,
                source_right_ground,
                source_forward_ground,
                source_from_latest_rotation,
                source_from_latest_translation,
            ) in frame_geometry:
                source_points = (
                    points_latest @ source_from_latest_rotation.transpose(0, 1)
                    + source_from_latest_translation
                )
                source_z = source_points[:, 2]
                intrinsic = intrinsics[batch_index, frame_index]
                projected_u = (
                    intrinsic[0, 0]
                    * source_points[:, 0]
                    / source_z.clamp_min(1e-6)
                    + intrinsic[0, 2]
                )
                ground_delta = points_latest - source_foot[None]
                local_x = ground_delta @ source_right_ground
                local_z = ground_delta @ source_forward_ground
                chunk_support |= (
                    (source_z > 1e-6)
                    & (projected_u >= 0)
                    & (projected_u < image_width)
                    & (local_x.abs() <= half_source)
                    & (local_z >= 0)
                    & (local_z <= half_source)
                )
            support[batch_index, start:stop] = chunk_support
    return support.view(batch, output_size, output_size)


def build_p1a_geometry(
    *,
    depth: torch.Tensor,
    confidence: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_from_world: torch.Tensor,
    camera_height_m: torch.Tensor,
    config: GeometryBuilderConfig,
) -> dict[str, torch.Tensor]:
    """Recover metric scale and construct fixed 6.5/10 P1A grid conditions."""

    dense_depth = depth[..., 0] if depth.ndim == 5 else depth
    if camera_height_m.ndim != 1 or camera_height_m.shape[0] != depth.shape[0]:
        raise ValueError("camera_height_m must have shape [B]")
    points = _latest_camera_points(
        dense_depth,
        intrinsics,
        camera_from_world,
        sample_stride=config.sample_stride,
    )
    confidence_probability = _confidence_probability(confidence)[
        ..., :: config.sample_stride, :: config.sample_stride
    ].reshape(points.shape[:-1])
    sampled_depth = dense_depth[
        ..., :: config.sample_stride, :: config.sample_stride
    ].reshape(points.shape[:-1])
    valid = (
        torch.isfinite(points).all(dim=-1)
        & torch.isfinite(sampled_depth)
        & (sampled_depth > 0)
        & (confidence_probability >= config.confidence_threshold)
    )
    (
        normal,
        origin_native,
        right,
        forward,
        ground_quality,
        confidence_quality,
        residual_quality,
        inlier_quality,
        temporal_quality,
    ) = _fit_ground_frame(
        points,
        valid,
        confidence_probability,
        minimum_points=config.minimum_points,
        ransac_hypotheses=config.ransac_hypotheses,
        huber_iterations=config.huber_iterations,
    )
    ground_distance_native = torch.linalg.vector_norm(
        origin_native,
        dim=-1,
    )
    height_valid = torch.isfinite(camera_height_m) & (camera_height_m > 0)
    distance_valid = torch.isfinite(ground_distance_native) & (
        ground_distance_native > 1e-6
    )
    scale_valid = (
        height_valid
        & distance_valid
        & (ground_quality >= config.minimum_ground_quality)
    )
    metric_per_native = camera_height_m / ground_distance_native.clamp_min(1e-6)
    safe_metric_per_native = torch.where(
        scale_valid,
        metric_per_native,
        torch.ones_like(metric_per_native),
    )
    valid_fraction = valid.float().mean(dim=(1, 2))
    geometry_quality = (
        ground_quality * scale_valid.to(ground_quality.dtype)
    ).clamp(0.0, 1.0)

    single_extent = torch.full_like(
        safe_metric_per_native,
        config.single_extent_normalized_scale,
    )
    merged_extent = torch.full_like(
        safe_metric_per_native,
        config.merged_extent_normalized_scale,
    )
    single_fov_support = _runtime_fov_support(
        intrinsics=intrinsics,
        camera_from_world=camera_from_world,
        ground_origin=origin_native,
        ground_normal=normal,
        ground_right=right,
        ground_forward=forward,
        metric_per_native=safe_metric_per_native,
        geometry_valid=scale_valid,
        output_size=512,
        output_extent_m=config.single_extent_normalized_scale,
        source_extent_m=config.single_extent_normalized_scale,
        image_width=dense_depth.shape[-1],
        frame_indices=(dense_depth.shape[1] - 1,),
        chunk_size=config.fov_chunk_size,
    )
    merged_fov_support = _runtime_fov_support(
        intrinsics=intrinsics,
        camera_from_world=camera_from_world,
        ground_origin=origin_native,
        ground_normal=normal,
        ground_right=right,
        ground_forward=forward,
        metric_per_native=safe_metric_per_native,
        geometry_valid=scale_valid,
        output_size=800,
        output_extent_m=config.merged_extent_normalized_scale,
        source_extent_m=config.single_extent_normalized_scale,
        image_width=dense_depth.shape[-1],
        frame_indices=None,
        chunk_size=config.fov_chunk_size,
    )
    return {
        # The output grid uses fixed normalized-scale coordinates. With the
        # explicit camera-height anchor, one grid unit is one metre.
        "single_extent_normalized_scale": single_extent,
        "merged_extent_normalized_scale": merged_extent,
        "single_extent_m": single_extent,
        "merged_extent_m": merged_extent,
        "normalized_scale_m_per_unit": torch.ones_like(single_extent),
        "metric_per_vggt_native_unit": safe_metric_per_native,
        "vggt_native_per_metric_unit": safe_metric_per_native.reciprocal(),
        "ground_distance_vggt_native": ground_distance_native,
        "ground_normal_latest_camera": normal,
        "ground_origin_vggt_native": origin_native,
        "ground_origin_m": origin_native * safe_metric_per_native[:, None],
        "ground_right_latest_camera": right,
        "ground_forward_latest_camera": forward,
        "ground_quality": ground_quality,
        "ground_confidence_quality": confidence_quality,
        "ground_residual_quality": residual_quality,
        "ground_inlier_quality": inlier_quality,
        "ground_temporal_quality": temporal_quality,
        "scale_quality": geometry_quality,
        "geometry_quality": geometry_quality,
        "geometry_valid": scale_valid,
        "valid_geometry_fraction": valid_fraction,
        "camera_height_m": camera_height_m,
        "camera_centers_vggt_native": world_camera_centers(camera_from_world),
        "single_fov_support": single_fov_support,
        "merged_fov_support": merged_fov_support,
        "single_fov_support_fraction": single_fov_support.float().mean(
            dim=(1, 2)
        ),
        "merged_fov_support_fraction": merged_fov_support.float().mean(
            dim=(1, 2)
        ),
    }


def p1a_geometry_cue(
    base_cue: torch.Tensor,
    geometry: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Append explicit metric-anchor/ground conditions to VGGT summaries."""

    extra = torch.cat(
        (
            geometry["metric_per_vggt_native_unit"].clamp_min(1e-8).log()[
                :, None
            ],
            geometry["ground_distance_vggt_native"].clamp_min(1e-8).log()[
                :, None
            ],
            geometry["ground_normal_latest_camera"],
            geometry["ground_origin_m"],
            geometry["valid_geometry_fraction"][:, None],
            geometry["geometry_quality"][:, None],
        ),
        dim=-1,
    )
    return torch.cat((base_cue, extra), dim=-1)


def fit_metric_per_native_scale(
    predicted_centers: torch.Tensor,
    ground_truth_centers_m: torch.Tensor,
    *,
    minimum_baseline_m: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training-only odometry oracle using one isotropic scalar."""

    if predicted_centers.shape != ground_truth_centers_m.shape:
        raise ValueError("predicted and GT camera trajectories must have equal shape")
    predicted_steps = torch.linalg.vector_norm(
        predicted_centers[:, 1:] - predicted_centers[:, :-1],
        dim=-1,
    )
    metric_steps = torch.linalg.vector_norm(
        ground_truth_centers_m[:, 1:] - ground_truth_centers_m[:, :-1],
        dim=-1,
    )
    scales = []
    qualities = []
    for batch_index in range(predicted_centers.shape[0]):
        valid = (
            torch.isfinite(predicted_steps[batch_index])
            & torch.isfinite(metric_steps[batch_index])
            & (predicted_steps[batch_index] > 1e-6)
            & (metric_steps[batch_index] >= minimum_baseline_m)
        )
        ratios = metric_steps[batch_index][valid] / predicted_steps[
            batch_index
        ][valid]
        if ratios.numel() == 0:
            scales.append(predicted_centers.new_tensor(float("nan")))
            qualities.append(predicted_centers.new_tensor(0.0))
            continue
        median = ratios.median()
        deviation = (
            (ratios - median).abs() / median.clamp_min(1e-6)
        ).median()
        scales.append(median)
        qualities.append((1.0 - deviation).clamp(0.0, 1.0))
    return torch.stack(scales), torch.stack(qualities)


def validate_p1a_training_geometry(
    batch: dict,
    extraction: dict,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Compare runtime height scale to GT odometry without changing labels."""

    odometry_scale, odometry_quality = fit_metric_per_native_scale(
        extraction["p1a_geometry"]["camera_centers_vggt_native"],
        batch["alignment_camera_centers_m"],
    )
    runtime_scale = extraction["p1a_geometry"][
        "metric_per_vggt_native_unit"
    ]
    comparison_valid = (
        torch.isfinite(odometry_scale)
        & (odometry_quality > 0)
        & extraction["p1a_geometry"]["geometry_valid"]
    )
    safe_oracle = torch.where(
        comparison_valid,
        odometry_scale,
        runtime_scale,
    )
    scale_log_error = (
        runtime_scale.clamp_min(1e-8).log()
        - safe_oracle.clamp_min(1e-8).log()
    ).abs()
    return batch, {
        "runtime_metric_per_native": runtime_scale,
        "odometry_metric_per_native": odometry_scale,
        "scale_log_error": scale_log_error,
        "scale_validation_quality": odometry_quality,
        "scale_validation_valid": comparison_valid,
        "single_output_extent_m": extraction["p1a_geometry"][
            "single_extent_m"
        ],
        "merged_output_extent_m": extraction["p1a_geometry"][
            "merged_extent_m"
        ],
        "geometry_quality": extraction["p1a_geometry"]["geometry_quality"],
        "geometry_valid": extraction["p1a_geometry"]["geometry_valid"],
    }
