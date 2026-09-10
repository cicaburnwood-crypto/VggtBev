from __future__ import annotations

import torch


def metric_points_to_vggt_units(
    points_m: torch.Tensor,
    lambda_m_per_vggt: torch.Tensor,
) -> torch.Tensor:
    """Convert metric coordinates/translations into same-window VGGT units."""

    if points_m.shape[-1] != 3:
        raise ValueError("points_m must end in XYZ coordinates")
    if bool((lambda_m_per_vggt <= 0).any()):
        raise ValueError("metric scale must be positive")
    scale = lambda_m_per_vggt
    while scale.ndim < points_m.ndim:
        scale = scale.unsqueeze(-1)
    return points_m / scale


def project_vggt_points(
    points_world_vggt: torch.Tensor,
    camera_from_world_vggt: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project VGGT-unit world points with OpenCV camera-from-world E and K."""

    if points_world_vggt.shape[-1] != 3:
        raise ValueError("points must end in XYZ")
    if camera_from_world_vggt.shape[-2:] != (3, 4):
        raise ValueError("camera_from_world must end in 3x4")
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must end in 3x3")
    rotation = camera_from_world_vggt[..., :3, :3]
    translation = camera_from_world_vggt[..., :3, 3]
    camera = (
        rotation @ points_world_vggt.unsqueeze(-1)
    ).squeeze(-1) + translation
    pixel_h = (intrinsics @ camera.unsqueeze(-1)).squeeze(-1)
    positive_depth = camera[..., 2] > 0
    pixel = pixel_h[..., :2] / pixel_h[..., 2:].clamp_min(1e-6)
    return pixel, positive_depth
