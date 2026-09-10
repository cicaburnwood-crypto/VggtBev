from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from vggt_bev_method1.config import LabelValues


def local_fov_polygon(
    horizontal_fov_degrees: float,
    *,
    source_extent_m: float = 6.5,
) -> np.ndarray:
    """Return the unobstructed horizontal-FOV footprint in local BEV x/z.

    The camera is at the BEV origin and looks along positive z. The footprint
    is clipped by the square source-complete coverage. No depth or visibility
    test is performed, so geometry behind an obstacle remains inside the mask.
    """

    if not 0.0 < horizontal_fov_degrees < 180.0:
        raise ValueError("horizontal FOV must be inside (0, 180) degrees")
    if source_extent_m <= 0:
        raise ValueError("source BEV extent must be positive")
    half_extent = source_extent_m / 2.0
    half_angle = math.radians(horizontal_fov_degrees / 2.0)
    tangent = math.tan(half_angle)
    if tangent <= 1.0:
        x_at_far = half_extent * tangent
        vertices = (
            (0.0, 0.0),
            (-x_at_far, half_extent),
            (x_at_far, half_extent),
        )
    else:
        z_at_side = half_extent / tangent
        vertices = (
            (0.0, 0.0),
            (-half_extent, z_at_side),
            (-half_extent, half_extent),
            (half_extent, half_extent),
            (half_extent, z_at_side),
        )
    return np.asarray(vertices, dtype=np.float64)


def _metric_to_pixel(
    polygon_xz: np.ndarray,
    *,
    output_size: int,
    output_extent_m: float,
) -> list[tuple[float, float]]:
    cell = output_extent_m / output_size
    column = (polygon_xz[:, 0] + output_extent_m / 2.0) / cell - 0.5
    row = (output_extent_m / 2.0 - polygon_xz[:, 1]) / cell - 0.5
    return list(zip(column.tolist(), row.tolist(), strict=True))


def fov_union_mask(
    world_from_bev_planar: np.ndarray,
    *,
    target_frame: int,
    horizontal_fov_degrees: float,
    output_size: int,
    output_extent_m: float,
    source_extent_m: float = 6.5,
) -> torch.Tensor:
    """Rasterize historical camera-FOV union in the latest ego BEV frame."""

    transforms = np.asarray(world_from_bev_planar, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 3):
        raise ValueError("world_from_bev_planar must have shape [frames,3,3]")
    if not 0 <= target_frame < transforms.shape[0]:
        raise ValueError("target frame lies outside the pose sequence")
    if output_size <= 0 or output_extent_m <= 0:
        raise ValueError("output BEV size and extent must be positive")

    local_polygon = local_fov_polygon(
        horizontal_fov_degrees,
        source_extent_m=source_extent_m,
    )
    homogeneous = np.concatenate(
        (local_polygon, np.ones((local_polygon.shape[0], 1))),
        axis=1,
    ).T
    target_from_world = np.linalg.inv(transforms[target_frame])
    image = Image.new("L", (output_size, output_size), 0)
    draw = ImageDraw.Draw(image)
    for source_from_local in transforms[: target_frame + 1]:
        target_polygon = target_from_world @ source_from_local @ homogeneous
        target_xz = (target_polygon[:2] / target_polygon[2:3]).T
        draw.polygon(
            _metric_to_pixel(
                target_xz,
                output_size=output_size,
                output_extent_m=output_extent_m,
            ),
            fill=1,
        )
    return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).bool()


def cap_complete_and_visible_to_fov(
    complete: torch.Tensor,
    visible: torch.Tensor,
    fov_support: torch.Tensor,
    *,
    labels: LabelValues,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create FOV-complete content and its visible/occluded partition."""

    if complete.shape != visible.shape or complete.shape != fov_support.shape:
        raise ValueError("complete, visible and FOV support shapes must match")
    complete_valid = complete != labels.unknown
    effective_support = fov_support.bool() & complete_valid
    fov_complete = torch.full_like(complete, labels.unknown)
    fov_complete[effective_support] = complete[effective_support]

    fov_visible = torch.full_like(visible, labels.unknown)
    directly_visible = effective_support & (visible != labels.unknown)
    fov_visible[directly_visible] = visible[directly_visible]
    if bool(
        (
            directly_visible
            & (fov_visible != fov_complete)
        ).any()
    ):
        raise ValueError("visible BEV labels disagree with complete collision truth")
    return fov_complete, fov_visible, effective_support


def load_world_from_bev_planar(
    records: Sequence[dict],
    *,
    expected_frames: int,
) -> np.ndarray:
    matrices = []
    for expected_frame, record in enumerate(records):
        if int(record.get("frame_id", -1)) != expected_frame:
            raise ValueError("camera extrinsics are not in contiguous frame order")
        extrinsic = record.get("extrinsic", record)
        matrix = np.asarray(
            extrinsic.get("world_from_bev_planar"),
            dtype=np.float64,
        )
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("world_from_bev_planar is missing or invalid")
        if abs(float(np.linalg.det(matrix))) < 1e-8:
            raise ValueError("world_from_bev_planar is singular")
        matrices.append(matrix)
    if len(matrices) != expected_frames:
        raise ValueError(
            "camera extrinsic count does not match session frame_count"
        )
    return np.stack(matrices)
