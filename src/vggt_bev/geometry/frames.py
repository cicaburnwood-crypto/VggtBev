from __future__ import annotations

import torch


def habitat_from_opencv(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Map OpenCV camera axes (+x right, +y down, +z forward) to Habitat sensor axes."""

    return torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device, dtype=dtype))


def opencv_points_to_reference_bev(
    points_camera: torch.Tensor,
    camera_to_world: torch.Tensor,
    reference_world_from_bev: torch.Tensor,
    floor_y: torch.Tensor,
) -> torch.Tensor:
    """Transform BxNxPx3 OpenCV points into (right, forward, height) in reference BEV.

    The simulator extrinsic uses Habitat sensor-local coordinates. The explicit axis
    conversion here is therefore required; omitting it mirrors both forward and vertical.
    """

    if points_camera.ndim != 4 or points_camera.shape[-1] != 3:
        raise ValueError("points_camera must have shape [B, N, P, 3]")
    batch, frames = points_camera.shape[:2]
    if camera_to_world.shape != (batch, frames, 4, 4):
        raise ValueError("camera_to_world must have shape [B, N, 4, 4]")
    if reference_world_from_bev.shape != (batch, 3, 3):
        raise ValueError("reference_world_from_bev must have shape [B, 3, 3]")

    ones = torch.ones_like(points_camera[..., :1])
    points_cv_h = torch.cat((points_camera, ones), dim=-1)
    axis_transform = habitat_from_opencv(device=points_camera.device, dtype=points_camera.dtype)
    points_habitat = torch.einsum("ij,bnpj->bnpi", axis_transform, points_cv_h)
    points_world = torch.einsum("bnij,bnpj->bnpi", camera_to_world, points_habitat)

    world_xz_h = torch.stack(
        (points_world[..., 0], points_world[..., 2], torch.ones_like(points_world[..., 0])),
        dim=-1,
    )
    reference_from_world = torch.linalg.inv(reference_world_from_bev)
    points_bev_h = torch.einsum("bij,bnpj->bnpi", reference_from_world, world_xz_h)
    height = points_world[..., 1] - floor_y[:, None, None]
    return torch.stack((points_bev_h[..., 0], points_bev_h[..., 1], height), dim=-1)


def camera_origins_in_reference_bev(
    camera_to_world: torch.Tensor, reference_world_from_bev: torch.Tensor
) -> torch.Tensor:
    """Return BxNx2 camera origins as (right, forward) in the target ego frame."""

    world_xz_h = torch.stack(
        (
            camera_to_world[..., 0, 3],
            camera_to_world[..., 2, 3],
            torch.ones_like(camera_to_world[..., 0, 3]),
        ),
        dim=-1,
    )
    reference_from_world = torch.linalg.inv(reference_world_from_bev)
    origins = torch.einsum("bij,bnj->bni", reference_from_world, world_xz_h)
    return origins[..., :2]

