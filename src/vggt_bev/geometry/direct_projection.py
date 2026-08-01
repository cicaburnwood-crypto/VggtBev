from __future__ import annotations

import torch

from vggt_bev.config import BatchBEVGridSpec, BEVGridSpec, LabelValues
from vggt_bev.geometry.ground import GroundAlignment, project_to_ground_frame

DEFAULT_LABEL_VALUES = LabelValues()


def _latest_valid_indices(frame_valid: torch.Tensor) -> torch.Tensor:
    if frame_valid.ndim != 2 or not frame_valid.any(dim=1).all():
        raise ValueError("every batch item must contain at least one valid frame")
    return frame_valid.sum(dim=1, dtype=torch.long) - 1


def scale_camera_translations(
    camera_from_world: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply the common VGGT reconstruction scale to camera translations."""

    scale_value = scale
    if scale.ndim == 1:
        scale_value = scale[:, None, None, None]
    return torch.cat(
        (
            camera_from_world[..., :3],
            camera_from_world[..., 3:] * scale_value,
        ),
        dim=-1,
    )


def camera_points_to_latest_camera(
    camera_points: torch.Tensor,
    camera_from_world: torch.Tensor,
    frame_valid: torch.Tensor,
) -> torch.Tensor:
    """Transform OpenCV camera points into the latest valid camera frame."""

    if camera_points.ndim != 4 or camera_points.shape[-1] != 3:
        raise ValueError("camera_points must have shape [B, N, P, 3]")
    batch, frames = camera_points.shape[:2]
    if camera_from_world.shape != (batch, frames, 3, 4):
        raise ValueError("camera_from_world must have shape [B, N, 3, 4]")
    latest_indices = _latest_valid_indices(frame_valid)
    rotation = camera_from_world[..., :3]
    translation = camera_from_world[..., 3]
    points_world = torch.einsum(
        "bnji,bnpj->bnpi",
        rotation,
        camera_points - translation[:, :, None, :],
    )
    batch_index = torch.arange(batch, device=camera_points.device)
    reference_rotation = rotation[batch_index, latest_indices]
    reference_translation = translation[batch_index, latest_indices]
    return (
        torch.einsum("bij,bnpj->bnpi", reference_rotation, points_world)
        + reference_translation[:, None, None, :]
    )


def camera_origins_in_latest_camera(
    camera_from_world: torch.Tensor,
    frame_valid: torch.Tensor,
) -> torch.Tensor:
    """Return each predicted camera center as (right, forward) in the latest frame."""

    return camera_origins_3d_in_latest_camera(
        camera_from_world,
        frame_valid,
    )[..., (0, 2)]


def camera_origins_3d_in_latest_camera(
    camera_from_world: torch.Tensor,
    frame_valid: torch.Tensor,
) -> torch.Tensor:
    """Return every camera center in the latest OpenCV camera frame."""

    batch, frames = frame_valid.shape
    if camera_from_world.shape != (batch, frames, 3, 4):
        raise ValueError("camera_from_world must have shape [B, N, 3, 4]")
    rotation = camera_from_world[..., :3]
    translation = camera_from_world[..., 3]
    origins_world = -torch.einsum("bnji,bnj->bni", rotation, translation)
    latest_indices = _latest_valid_indices(frame_valid)
    batch_index = torch.arange(batch, device=camera_from_world.device)
    reference_rotation = rotation[batch_index, latest_indices]
    reference_translation = translation[batch_index, latest_indices]
    origins_reference = (
        torch.einsum("bij,bnj->bni", reference_rotation, origins_world)
        + reference_translation[:, None, :]
    )
    return origins_reference


def _points_in_latest_camera(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_from_world: torch.Tensor,
    frame_valid: torch.Tensor,
    *,
    pixel_stride: int,
) -> torch.Tensor:
    """Unproject VGGT depth and express every point in the latest camera frame."""

    if depth.ndim != 4:
        raise ValueError("depth must have shape [B, N, H, W]")
    batch, frames, height, width = depth.shape
    if intrinsics.shape != (batch, frames, 3, 3):
        raise ValueError("intrinsics must have shape [B, N, 3, 3]")
    if camera_from_world.shape != (batch, frames, 3, 4):
        raise ValueError("camera_from_world must have shape [B, N, 3, 4]")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")

    rows = torch.arange(
        0,
        height,
        pixel_stride,
        device=depth.device,
        dtype=depth.dtype,
    )
    columns = torch.arange(
        0,
        width,
        pixel_stride,
        device=depth.device,
        dtype=depth.dtype,
    )
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    columns_flat = column_grid.reshape(1, 1, -1)
    rows_flat = row_grid.reshape(1, 1, -1)
    sampled_depth = depth[:, :, ::pixel_stride, ::pixel_stride].reshape(
        batch,
        frames,
        -1,
    )

    fx = intrinsics[..., 0, 0, None]
    fy = intrinsics[..., 1, 1, None]
    cx = intrinsics[..., 0, 2, None]
    cy = intrinsics[..., 1, 2, None]
    camera_points = torch.stack(
        (
            (columns_flat - cx) / fx * sampled_depth,
            (rows_flat - cy) / fy * sampled_depth,
            sampled_depth,
        ),
        dim=-1,
    )

    return camera_points_to_latest_camera(
        camera_points,
        camera_from_world,
        frame_valid,
    )


def _rasterize(
    points_bev: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    frame_mask: torch.Tensor,
    grid: BEVGridSpec | BatchBEVGridSpec,
    *,
    obstacle_height_band_m: tuple[float, float],
    floor_height_band_m: tuple[float, float],
    minimum_confidence: float,
    labels: LabelValues,
) -> torch.Tensor:
    batch, frames, points, _ = points_bev.shape
    if confidence.shape != (batch, frames, points):
        raise ValueError("confidence must align with projected points")
    if valid.shape != confidence.shape:
        raise ValueError("valid must align with confidence")
    if frame_mask.shape != (batch, frames):
        raise ValueError("frame_mask must have shape [B, N]")

    right_forward = points_bev[..., :2]
    height_above_floor = points_bev[..., 2]
    pixels = grid.spatial_to_pixel(right_forward)
    columns = pixels[..., 0].round().to(torch.long)
    rows = pixels[..., 1].round().to(torch.long)
    point_valid = (
        valid
        & frame_mask[:, :, None]
        & torch.isfinite(points_bev).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= minimum_confidence)
        & (columns >= 0)
        & (columns < grid.width)
        & (rows >= 0)
        & (rows < grid.height)
    )

    minimum_obstacle, maximum_obstacle = obstacle_height_band_m
    minimum_floor, maximum_floor = floor_height_band_m
    obstacle = (
        point_valid
        & (height_above_floor >= minimum_obstacle)
        & (height_above_floor <= maximum_obstacle)
    )
    floor = (
        point_valid
        & (height_above_floor >= minimum_floor)
        & (height_above_floor <= maximum_floor)
    )

    batch_offset = (
        torch.arange(batch, device=points_bev.device)[:, None, None]
        * grid.height
        * grid.width
    )
    safe_columns = columns.clamp(0, grid.width - 1)
    safe_rows = rows.clamp(0, grid.height - 1)
    linear = (batch_offset + safe_rows * grid.width + safe_columns).reshape(-1)
    total_cells = batch * grid.height * grid.width
    obstacle_evidence = confidence.new_zeros(total_cells)
    floor_evidence = confidence.new_zeros(total_cells)
    obstacle_evidence.scatter_add_(
        0,
        linear,
        torch.where(obstacle, confidence, 0.0).reshape(-1),
    )
    floor_evidence.scatter_add_(
        0,
        linear,
        torch.where(floor, confidence, 0.0).reshape(-1),
    )
    obstacle_evidence = obstacle_evidence.reshape(batch, grid.height, grid.width)
    floor_evidence = floor_evidence.reshape(batch, grid.height, grid.width)

    output = torch.full(
        (batch, grid.height, grid.width),
        labels.unknown,
        dtype=torch.uint8,
        device=points_bev.device,
    )
    output[floor_evidence > 0] = labels.free
    output[obstacle_evidence > 0] = labels.occupied
    return output


def project_vggt_geometry_bevs(
    *,
    dense_depth: torch.Tensor,
    dense_confidence: torch.Tensor,
    estimated_intrinsics: torch.Tensor,
    estimated_camera_from_world: torch.Tensor,
    image_valid: torch.Tensor,
    frame_valid: torch.Tensor,
    camera_height_m: torch.Tensor,
    single_extent_m: float | torch.Tensor,
    merged_extent_m: float | torch.Tensor,
    output_size: int,
    depth_scale: torch.Tensor,
    merged_output_size: int | None = None,
    pixel_stride: int = 1,
    obstacle_height_band_m: tuple[float, float] = (0.12, 1.4),
    floor_height_band_m: tuple[float, float] = (-0.12, 0.12),
    minimum_confidence: float = 0.05,
    labels: LabelValues = DEFAULT_LABEL_VALUES,
    stabilized_intrinsics: torch.Tensor | None = None,
    ground_alignment: GroundAlignment | None = None,
    coordinate_divisor: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Create head-free BEVs from VGGT-estimated depth, intrinsics, and poses.

    VGGT reconstruction is scale-ambiguous. A strict Method II checkpoint supplies
    a ground alignment and per-window camera-height scale; legacy checkpoints use
    their learned global scale. No trained BEV head or simulator pose participates.
    """

    if dense_depth.ndim != 5 or dense_depth.shape[-1] != 1:
        raise ValueError("dense_depth must have shape [B, N, H, W, 1]")
    raw_depth = dense_depth[..., 0]
    intrinsics = (
        stabilized_intrinsics
        if stabilized_intrinsics is not None
        else estimated_intrinsics
    )
    if ground_alignment is None:
        depth_multiplier = (
            depth_scale[:, None, None, None]
            if depth_scale.ndim == 1
            else depth_scale
        )
        depth = raw_depth * depth_multiplier
        camera_from_world = scale_camera_translations(
            estimated_camera_from_world,
            depth_scale,
        )
    else:
        depth = raw_depth
        camera_from_world = estimated_camera_from_world
    if dense_confidence.shape != depth.shape:
        raise ValueError("dense_confidence must match dense_depth spatial dimensions")
    if image_valid.shape != depth.shape:
        raise ValueError("image_valid must match dense_depth spatial dimensions")
    if frame_valid.shape != depth.shape[:2]:
        raise ValueError("frame_valid must have shape [B, N]")
    if not 0.0 <= minimum_confidence < 1.0:
        raise ValueError("minimum_confidence must be in [0, 1)")

    latest = _latest_valid_indices(frame_valid)
    points_reference = _points_in_latest_camera(
        depth,
        intrinsics,
        camera_from_world,
        frame_valid,
        pixel_stride=pixel_stride,
    )
    if ground_alignment is None:
        points_bev = torch.stack(
            (
                points_reference[..., 0],
                points_reference[..., 2],
                camera_height_m[:, None, None] - points_reference[..., 1],
            ),
            dim=-1,
        )
    else:
        points_bev = project_to_ground_frame(
            points_reference,
            normal=ground_alignment.normal,
            origin=ground_alignment.origin,
            right=ground_alignment.right,
            forward=ground_alignment.forward,
            metric_scale=ground_alignment.metric_scale,
        )
    if coordinate_divisor is not None:
        if coordinate_divisor.shape != (points_bev.shape[0],):
            raise ValueError("coordinate_divisor must have shape [B]")
        points_bev = points_bev / coordinate_divisor[:, None, None, None]
    confidence = (
        (dense_confidence - 1.0) / dense_confidence.clamp_min(1.0)
    )[:, :, ::pixel_stride, ::pixel_stride].reshape(points_reference.shape[:-1])
    valid = (
        image_valid[:, :, ::pixel_stride, ::pixel_stride].reshape(
            points_reference.shape[:-1]
        )
        & torch.isfinite(depth[:, :, ::pixel_stride, ::pixel_stride]).reshape(
            points_reference.shape[:-1]
        )
        & (depth[:, :, ::pixel_stride, ::pixel_stride] > 0).reshape(
            points_reference.shape[:-1]
        )
    )
    single_frames = torch.zeros_like(frame_valid)
    single_frames.scatter_(1, latest[:, None], True)

    common = {
        "points_bev": points_bev,
        "confidence": confidence,
        "valid": valid,
        "obstacle_height_band_m": obstacle_height_band_m,
        "floor_height_band_m": floor_height_band_m,
        "minimum_confidence": minimum_confidence,
        "labels": labels,
    }
    def grid_spec(
        extent: float | torch.Tensor,
        size: int,
    ) -> BEVGridSpec | BatchBEVGridSpec:
        if isinstance(extent, torch.Tensor):
            return BatchBEVGridSpec(extent, size, size)
        return BEVGridSpec(extent, size, size)

    single = _rasterize(
        frame_mask=single_frames,
        grid=grid_spec(single_extent_m, output_size),
        **common,
    )
    resolved_merged_output_size = merged_output_size or output_size
    merged = _rasterize(
        frame_mask=frame_valid,
        grid=grid_spec(merged_extent_m, resolved_merged_output_size),
        **common,
    )
    return {
        "single_labels": single,
        "merged_labels": merged,
        "estimated_intrinsics": intrinsics,
        "estimated_camera_from_world": estimated_camera_from_world,
    }
