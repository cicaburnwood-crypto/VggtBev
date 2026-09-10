from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CameraHeightScaleCalibration:
    """Per-window camera-height bridge between VGGT, BEV, and metres."""

    camera_height_vggt: torch.Tensor
    camera_metric_scale_m_per_vggt: torch.Tensor
    model_scale_bev_per_vggt: torch.Tensor
    bev_metric_scale_m_per_bev: torch.Tensor
    ground_normal_latest_camera: torch.Tensor
    ground_inlier_fraction: torch.Tensor
    ground_ransac_confidence: torch.Tensor
    ground_plane_relative_residual: torch.Tensor
    ground_fallback_used: torch.Tensor


@dataclass(frozen=True)
class _GroundEstimate:
    height: torch.Tensor
    normal: torch.Tensor
    inlier_fraction: torch.Tensor
    confidence: torch.Tensor
    relative_residual: torch.Tensor


def _stabilize_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [B,N,3,3]")
    stable = intrinsics.median(dim=1).values
    if (
        not torch.isfinite(stable).all()
        or (stable[:, 0, 0] <= 0).any()
        or (stable[:, 1, 1] <= 0).any()
    ):
        raise ValueError("VGGT returned invalid camera intrinsics")
    stable[:, 0, 1] = 0.0
    stable[:, 1, 0] = 0.0
    stable[:, 2, 0] = 0.0
    stable[:, 2, 1] = 0.0
    stable[:, 2, 2] = 1.0
    return stable[:, None].expand_as(intrinsics).clone()


def _dense_points_in_camera(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    pixel_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if depth.ndim != 4:
        raise ValueError("depth must have shape [B,N,H,W]")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")
    batch, frames, height, width = depth.shape
    if intrinsics.shape != (batch, frames, 3, 3):
        raise ValueError("intrinsics must have shape [B,N,3,3]")
    rows = torch.arange(
        0, height, pixel_stride, device=depth.device, dtype=depth.dtype
    )
    columns = torch.arange(
        0, width, pixel_stride, device=depth.device, dtype=depth.dtype
    )
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    sampled = depth[:, :, ::pixel_stride, ::pixel_stride].reshape(
        batch, frames, -1
    )
    u = column_grid.reshape(1, 1, -1)
    v = row_grid.reshape(1, 1, -1)
    fx = intrinsics[..., 0, 0, None]
    fy = intrinsics[..., 1, 1, None]
    cx = intrinsics[..., 0, 2, None]
    cy = intrinsics[..., 1, 2, None]
    points = torch.stack(
        (
            (u - cx) / fx * sampled,
            (v - cy) / fy * sampled,
            sampled,
        ),
        dim=-1,
    )
    return points, v.expand(batch, frames, -1), u.expand(batch, frames, -1)


def _weighted_quantile(
    values: torch.Tensor, weights: torch.Tensor, quantile: float
) -> torch.Tensor:
    if values.ndim != 1 or weights.shape != values.shape or values.numel() == 0:
        raise ValueError("weighted quantile expects non-empty aligned vectors")
    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order].clamp_min(1e-8)
    cumulative = torch.cumsum(sorted_weights, dim=0)
    index = torch.searchsorted(cumulative, cumulative[-1] * float(quantile))
    return sorted_values[index.clamp_max(values.numel() - 1)]


def _axis_coverage(values: torch.Tensor, extent: int, bins: int = 10) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    occupied = torch.unique(
        (values * bins / max(extent, 1)).floor().long().clamp(0, bins - 1)
    )
    return values.new_tensor(float(occupied.numel()) / float(bins))


def _ransac_frame_ground_plane(
    points: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    rows: torch.Tensor,
    columns: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    minimum_points: int,
    iterations: int,
    maximum_points: int,
    relative_inlier_threshold: float,
    minimum_vertical_alignment: float,
) -> _GroundEstimate | None:
    """Fit the lower-image ground plane with deterministic confidence RANSAC."""

    mask = (
        valid
        & torch.isfinite(points).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence > 0.0)
        & (points[:, 1] > 0.0)
        & (rows >= float(image_height) * 0.42)
    )
    usable = points[mask]
    weights = confidence[mask].clamp(1e-4, 1.0)
    usable_rows = rows[mask]
    usable_columns = columns[mask]
    if usable.shape[0] < minimum_points:
        return None

    # Horizontal furniture is normally closer to the camera than the floor.
    # Bias sampling to the deeper half while scoring against all lower-image
    # points; this improves floor discovery without hard-coding a height.
    y_threshold = _weighted_quantile(usable[:, 1], weights, 0.50)
    candidates = torch.nonzero(usable[:, 1] >= y_threshold).flatten()
    if candidates.numel() < minimum_points:
        candidates = torch.arange(usable.shape[0], device=usable.device)
    if usable.shape[0] > maximum_points:
        stride = max(1, usable.shape[0] // maximum_points)
        keep = torch.arange(0, usable.shape[0], stride, device=usable.device)[
            :maximum_points
        ]
        usable = usable[keep]
        weights = weights[keep]
        usable_rows = usable_rows[keep]
        usable_columns = usable_columns[keep]
        y_threshold = _weighted_quantile(usable[:, 1], weights, 0.50)
        candidates = torch.nonzero(usable[:, 1] >= y_threshold).flatten()
    if candidates.numel() < 3:
        return None

    total_weight = weights.sum().clamp_min(1e-6)
    best_score: torch.Tensor | None = None
    best_inliers: torch.Tensor | None = None
    count = int(candidates.numel())
    # Prime strides provide deterministic, well-distributed triples without
    # touching global RNG state or making the result request-order dependent.
    for iteration in range(int(iterations)):
        indices = candidates[
            torch.tensor(
                [
                    (iteration * 17 + 3) % count,
                    (iteration * 43 + 11) % count,
                    (iteration * 97 + 29) % count,
                ],
                device=usable.device,
            )
        ]
        first, second, third = usable[indices]
        normal = torch.linalg.cross(second - first, third - first)
        normal_norm = torch.linalg.vector_norm(normal)
        if not torch.isfinite(normal_norm) or float(normal_norm) <= 1e-7:
            continue
        normal = normal / normal_norm
        vertical_alignment = normal[1].abs()
        if float(vertical_alignment) < minimum_vertical_alignment:
            continue
        offset = -torch.dot(normal, first)
        height = offset.abs()
        if not torch.isfinite(height) or float(height) <= 1e-6:
            continue
        relative_residuals = (usable @ normal + offset).abs() / height
        inliers = relative_residuals <= relative_inlier_threshold
        if int(inliers.sum()) < minimum_points:
            continue
        inlier_weight = weights[inliers].sum()
        inlier_fraction = inlier_weight / total_weight
        row_coverage = _axis_coverage(usable_rows[inliers], image_height)
        column_coverage = _axis_coverage(usable_columns[inliers], image_width)
        if float(row_coverage) < 0.20 or float(column_coverage) < 0.30:
            continue
        median_residual = _weighted_quantile(
            relative_residuals[inliers], weights[inliers], 0.5
        )
        residual_quality = torch.exp(
            -median_residual / max(relative_inlier_threshold * 0.5, 1e-6)
        )
        depth_preference = (height / usable[:, 1].quantile(0.95)).clamp(0.0, 1.0)
        score = (
            inlier_fraction
            * torch.sqrt(row_coverage * column_coverage)
            * vertical_alignment
            * residual_quality
            * torch.sqrt(depth_preference.clamp_min(1e-4))
        )
        if best_score is None or float(score) > float(best_score):
            best_score = score
            best_inliers = inliers

    if best_inliers is None:
        return None

    # Weighted least-squares refinement on the winning consensus set.
    inlier_points = usable[best_inliers]
    inlier_weights = weights[best_inliers]
    normalized_weights = inlier_weights / inlier_weights.sum().clamp_min(1e-6)
    centroid = (inlier_points * normalized_weights[:, None]).sum(dim=0)
    centered = inlier_points - centroid
    covariance = (centered * normalized_weights[:, None]).T @ centered
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    if normal[1] > 0.0:
        normal = -normal
    offset = -torch.dot(normal, centroid)
    height = offset.abs()
    if not torch.isfinite(height) or float(height) <= 1e-6:
        return None
    residuals = (usable @ normal + offset).abs() / height
    inliers = residuals <= relative_inlier_threshold
    if int(inliers.sum()) < minimum_points:
        return None
    inlier_fraction = weights[inliers].sum() / total_weight
    row_coverage = _axis_coverage(usable_rows[inliers], image_height)
    column_coverage = _axis_coverage(usable_columns[inliers], image_width)
    relative_residual = _weighted_quantile(residuals[inliers], weights[inliers], 0.5)
    alignment = normal[1].abs().clamp(0.0, 1.0)
    support_quality = (inlier_fraction / 0.25).clamp(0.0, 1.0)
    residual_quality = torch.exp(
        -relative_residual / max(relative_inlier_threshold * 0.5, 1e-6)
    )
    ransac_confidence = (
        support_quality
        * torch.sqrt(row_coverage * column_coverage)
        * alignment
        * residual_quality
    ).clamp(0.0, 1.0)
    if not torch.isfinite(ransac_confidence):
        return None
    return _GroundEstimate(
        height=height,
        normal=normal,
        inlier_fraction=inlier_fraction,
        confidence=ransac_confidence,
        relative_residual=relative_residual,
    )


def _fuse_frame_estimates(
    estimates: list[_GroundEstimate], frame_count: int
) -> _GroundEstimate | None:
    if not estimates:
        return None
    heights = torch.stack([item.height for item in estimates])
    confidences = torch.stack([item.confidence for item in estimates]).clamp_min(1e-6)
    log_heights = heights.log()
    center = _weighted_quantile(log_heights, confidences, 0.5)
    residual = (log_heights - center).abs()
    mad = _weighted_quantile(residual, confidences, 0.5)
    threshold = torch.maximum(mad * 3.0, mad.new_tensor(0.08))
    consistent = residual <= threshold
    if int(consistent.sum()) < (2 if frame_count >= 3 else 1):
        return None
    weights = confidences[consistent]
    weights = weights / weights.sum().clamp_min(1e-6)
    fused_height = (heights[consistent] * weights).sum()
    fused_normal = torch.stack(
        [item.normal for item, keep in zip(estimates, consistent.tolist()) if keep]
    )
    fused_normal = (fused_normal * weights[:, None]).sum(dim=0)
    fused_normal = fused_normal / torch.linalg.vector_norm(fused_normal).clamp_min(1e-6)
    inlier = torch.stack([item.inlier_fraction for item in estimates])[consistent]
    plane_residual = torch.stack(
        [item.relative_residual for item in estimates]
    )[consistent]
    return _GroundEstimate(
        height=fused_height,
        normal=fused_normal,
        inlier_fraction=(inlier * weights).sum(),
        confidence=(confidences[consistent] * weights).sum()
        * consistent.float().mean(),
        relative_residual=(plane_residual * weights).sum(),
    )


@torch.no_grad()
def calibrate_bev_scale_from_camera_height(
    *,
    depth_vggt: torch.Tensor,
    confidence_vggt: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_from_world_vggt: torch.Tensor,
    physical_camera_height_m: torch.Tensor,
    model_scale_token_bev_per_vggt: torch.Tensor,
    pixel_stride: int = 4,
    minimum_points: int = 48,
    ransac_iterations: int = 128,
    ransac_maximum_points: int = 3072,
    ransac_relative_inlier_threshold: float = 0.025,
    minimum_vertical_alignment: float = 0.90,
) -> CameraHeightScaleCalibration:
    """Estimate camera height by RANSAC and apply the historical scale chain.

    The returned BEV scale is ``(physical_height / VGGT_height) / ScaleToken``.
    Temporal smoothing is intentionally owned by the runtime so its history is
    exactly the previous BoQ-selected frames rather than arbitrary RGB frames.
    """

    depth = depth_vggt[..., 0] if depth_vggt.ndim == 5 else depth_vggt
    confidence = confidence_vggt[..., 0] if confidence_vggt.ndim == 5 else confidence_vggt
    if depth.ndim != 4 or confidence.shape != depth.shape:
        raise ValueError("VGGT depth and confidence must have shape [B,N,H,W]")
    batch, frames = depth.shape[:2]
    if camera_from_world_vggt.shape != (batch, frames, 3, 4):
        raise ValueError("camera_from_world_vggt must have shape [B,N,3,4]")
    if physical_camera_height_m.shape != (batch,):
        raise ValueError("physical_camera_height_m must have shape [B]")
    if model_scale_token_bev_per_vggt.shape != (batch,):
        raise ValueError("model scale token must have shape [B]")
    if (
        not torch.isfinite(physical_camera_height_m).all()
        or (physical_camera_height_m <= 0).any()
        or not torch.isfinite(model_scale_token_bev_per_vggt).all()
        or (model_scale_token_bev_per_vggt <= 0).any()
    ):
        raise ValueError("camera height and model scale token must be positive")

    depth = depth.float()
    confidence = confidence.float()
    intrinsics = _stabilize_intrinsics(intrinsics.float())
    points, sampled_rows, sampled_columns = _dense_points_in_camera(
        depth, intrinsics, pixel_stride=pixel_stride
    )
    sampled_depth = depth[:, :, ::pixel_stride, ::pixel_stride].reshape(points.shape[:-1])
    sampled_confidence = confidence[:, :, ::pixel_stride, ::pixel_stride].reshape(
        points.shape[:-1]
    )
    # VGGT commonly emits evidence (>1), but synthetic tests and some model
    # builds emit a direct [0,1] probability. Support both contracts.
    if float(sampled_confidence.detach().amax()) <= 1.0001:
        confidence_probability = sampled_confidence.clamp(0.0, 1.0)
    else:
        confidence_probability = (
            (sampled_confidence - 1.0) / sampled_confidence.clamp_min(1.0)
        ).clamp(0.0, 1.0)
    valid = torch.isfinite(sampled_depth) & (sampled_depth > 0.0)

    heights: list[torch.Tensor] = []
    normals: list[torch.Tensor] = []
    inlier_fractions: list[torch.Tensor] = []
    ransac_confidences: list[torch.Tensor] = []
    plane_residuals: list[torch.Tensor] = []
    fallbacks: list[bool] = []
    for batch_index in range(batch):
        estimates: list[_GroundEstimate] = []
        for frame_index in range(frames):
            estimate = _ransac_frame_ground_plane(
                points[batch_index, frame_index],
                confidence_probability[batch_index, frame_index],
                valid[batch_index, frame_index],
                sampled_rows[batch_index, frame_index],
                sampled_columns[batch_index, frame_index],
                image_height=depth.shape[-2],
                image_width=depth.shape[-1],
                minimum_points=minimum_points,
                iterations=ransac_iterations,
                maximum_points=ransac_maximum_points,
                relative_inlier_threshold=ransac_relative_inlier_threshold,
                minimum_vertical_alignment=minimum_vertical_alignment,
            )
            if estimate is not None:
                estimates.append(estimate)
        fused = _fuse_frame_estimates(estimates, frames)
        fallback = fused is None
        if fallback:
            height = (
                physical_camera_height_m[batch_index]
                / model_scale_token_bev_per_vggt[batch_index]
            )
            normal = depth.new_tensor((0.0, -1.0, 0.0))
            inlier = depth.new_zeros(())
            ransac_confidence = depth.new_zeros(())
            plane_residual = depth.new_full((), float("inf"))
        else:
            height = fused.height
            normal = fused.normal
            inlier = fused.inlier_fraction
            ransac_confidence = fused.confidence
            plane_residual = fused.relative_residual
        heights.append(height)
        normals.append(normal)
        inlier_fractions.append(inlier)
        ransac_confidences.append(ransac_confidence)
        plane_residuals.append(plane_residual)
        fallbacks.append(fallback)

    camera_height_vggt = torch.stack(heights)
    camera_metric_scale = physical_camera_height_m / camera_height_vggt
    bev_metric_scale = camera_metric_scale / model_scale_token_bev_per_vggt
    if (
        not torch.isfinite(camera_metric_scale).all()
        or (camera_metric_scale <= 0).any()
        or not torch.isfinite(bev_metric_scale).all()
        or (bev_metric_scale <= 0).any()
    ):
        raise ValueError("camera-height scale calibration is invalid")
    return CameraHeightScaleCalibration(
        camera_height_vggt=camera_height_vggt,
        camera_metric_scale_m_per_vggt=camera_metric_scale,
        model_scale_bev_per_vggt=model_scale_token_bev_per_vggt,
        bev_metric_scale_m_per_bev=bev_metric_scale,
        ground_normal_latest_camera=torch.stack(normals),
        ground_inlier_fraction=torch.stack(inlier_fractions),
        ground_ransac_confidence=torch.stack(ransac_confidences),
        ground_plane_relative_residual=torch.stack(plane_residuals),
        ground_fallback_used=torch.tensor(
            fallbacks, dtype=torch.bool, device=depth.device
        ),
    )
