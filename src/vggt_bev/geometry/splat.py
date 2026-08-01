from __future__ import annotations

import torch

from vggt_bev.config import BatchBEVGridSpec, BEVGridSpec

GridSpec = BEVGridSpec | BatchBEVGridSpec


def bilinear_splat(
    points_right_forward: torch.Tensor,
    features: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    grid: GridSpec,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Confidence-weighted differentiable splat into an ego BEV raster.

    Args use shapes BxNxPx2, BxNxPxC, BxNxP, and BxNxP. The returned tensors are
    BxCxHxW normalized features and Bx1xHxW accumulated evidence.
    """

    if points_right_forward.shape[:-1] != features.shape[:-1]:
        raise ValueError("points and features must share B, N, P dimensions")
    if points_right_forward.shape[-1] != 2:
        raise ValueError("points must end in (right, forward)")
    if weights.shape != points_right_forward.shape[:-1] or valid.shape != weights.shape:
        raise ValueError("weights and valid must have shape [B, N, P]")

    batch = features.shape[0]
    channels = features.shape[-1]
    points = points_right_forward.reshape(batch, -1, 2)
    values = features.reshape(batch, -1, channels)
    point_weights = weights.reshape(batch, -1).to(features.dtype)
    point_valid = valid.reshape(batch, -1)
    pixels = grid.spatial_to_pixel(points)

    x0 = torch.floor(pixels[..., 0])
    y0 = torch.floor(pixels[..., 1])
    dx = pixels[..., 0] - x0
    dy = pixels[..., 1] - y0
    x0 = x0.to(torch.long)
    y0 = y0.to(torch.long)

    total_cells = batch * grid.height * grid.width
    numerator = features.new_zeros((total_cells, channels))
    denominator = features.new_zeros((total_cells, 1))
    batch_offset = (
        torch.arange(batch, device=features.device)[:, None] * grid.height * grid.width
    )

    neighbors = (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x0 + 1, y0, dx * (1.0 - dy)),
        (x0, y0 + 1, (1.0 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    )
    finite = torch.isfinite(pixels).all(dim=-1) & torch.isfinite(point_weights)
    for column, row, interpolation_weight in neighbors:
        inside = (
            point_valid
            & finite
            & (column >= 0)
            & (column < grid.width)
            & (row >= 0)
            & (row < grid.height)
        )
        safe_column = column.clamp(0, grid.width - 1)
        safe_row = row.clamp(0, grid.height - 1)
        linear = batch_offset + safe_row * grid.width + safe_column
        combined = point_weights * interpolation_weight * inside.to(features.dtype)
        flat_index = linear.reshape(-1)
        flat_weight = combined.reshape(-1, 1)
        numerator.scatter_add_(
            0,
            flat_index[:, None].expand(-1, channels),
            (values * combined[..., None]).reshape(-1, channels),
        )
        denominator.scatter_add_(0, flat_index[:, None], flat_weight)

    normalized = numerator / denominator.clamp_min(eps)
    normalized = normalized.view(batch, grid.height, grid.width, channels).permute(0, 3, 1, 2)
    evidence = denominator.view(batch, grid.height, grid.width, 1).permute(0, 3, 1, 2)
    return normalized, evidence


def raycast_free_evidence(
    camera_origins: torch.Tensor,
    endpoints: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    grid: GridSpec,
    *,
    steps: int = 32,
) -> torch.Tensor:
    """Splat explicit free-space evidence along rays, excluding their surface endpoints."""

    if steps < 2:
        raise ValueError("steps must be at least two")
    if camera_origins.shape != endpoints.shape[:2] + (2,):
        raise ValueError("camera_origins must have shape [B, N, 2]")
    base_fractions = torch.linspace(
        0.0,
        1.0,
        steps,
        device=endpoints.device,
        dtype=endpoints.dtype,
    )
    origin_points = camera_origins[:, :, None, :]
    direction = endpoints - origin_points
    ray_length = torch.linalg.vector_norm(direction, dim=-1)
    half_extent = grid.half_extent
    cell_size = grid.cell_size
    if isinstance(half_extent, torch.Tensor):
        half_extent = half_extent[:, None, None]
        cell_size = cell_size[:, None, None]
    epsilon = torch.finfo(endpoints.dtype).eps
    positive_infinity = torch.full_like(direction[..., 0], float("inf"))
    exit_fractions: list[torch.Tensor] = []
    for axis in range(2):
        component = direction[..., axis]
        origin_component = origin_points[..., axis]
        exit_fractions.append(
            torch.where(
                component > epsilon,
                (half_extent - origin_component) / component,
                torch.where(
                    component < -epsilon,
                    (-half_extent - origin_component) / component,
                    positive_infinity,
                ),
            )
        )
    boundary_fraction = torch.minimum(
        exit_fractions[0],
        exit_fractions[1],
    ).clamp(0.0, 1.0)
    maximum_fraction = (
        boundary_fraction
        - cell_size / ray_length.clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    fractions = maximum_fraction[..., None] * base_fractions
    origins = camera_origins[:, :, None, None, :]
    rays = origins + fractions[..., None] * (
        endpoints[:, :, :, None, :] - origins
    )
    batch, frames, points = endpoints.shape[:3]
    ray_points = rays.reshape(batch, frames, points * steps, 2)
    ray_valid = valid[..., None].expand(-1, -1, -1, steps).reshape(batch, frames, -1)
    ray_weights = (
        weights[..., None].expand(-1, -1, -1, steps).reshape(batch, frames, -1) / steps
    )
    ones = endpoints.new_ones((batch, frames, points * steps, 1))
    _, evidence = bilinear_splat(ray_points, ones, ray_weights, ray_valid, grid)
    return torch.log1p(evidence)
