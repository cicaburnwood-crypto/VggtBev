from __future__ import annotations

import torch
import torch.nn.functional as F


def patch_centers(
    height: int,
    width: int,
    patch_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return row-major patch-token centers in image pixel-center coordinates."""

    if height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be divisible by patch_size")
    rows = torch.arange(height // patch_size, device=device, dtype=dtype)
    columns = torch.arange(width // patch_size, device=device, dtype=dtype)
    v, u = torch.meshgrid(rows, columns, indexing="ij")
    u = (u + 0.5) * patch_size - 0.5
    v = (v + 0.5) * patch_size - 0.5
    return torch.stack((u, v), dim=-1).reshape(-1, 2)


def sample_image_at_pixels(
    image: torch.Tensor, pixels_uv: torch.Tensor, *, mode: str = "bilinear"
) -> torch.Tensor:
    """Sample BxNxCxHxW at P pixel centers, returning BxNxPxC."""

    if image.ndim != 5:
        raise ValueError("image must have shape [B, N, C, H, W]")
    batch, frames, channels, height, width = image.shape
    if pixels_uv.ndim == 2:
        pixels_uv = pixels_uv[None, None].expand(batch, frames, -1, -1)
    if pixels_uv.shape[:2] != (batch, frames) or pixels_uv.shape[-1] != 2:
        raise ValueError("pixels_uv must have shape [P, 2] or [B, N, P, 2]")

    u = pixels_uv[..., 0]
    v = pixels_uv[..., 1]
    normalized_u = 2.0 * u / max(width - 1, 1) - 1.0
    normalized_v = 2.0 * v / max(height - 1, 1) - 1.0
    grid = torch.stack((normalized_u, normalized_v), dim=-1).reshape(batch * frames, -1, 1, 2)
    flat_image = image.reshape(batch * frames, channels, height, width)
    sampled = F.grid_sample(
        flat_image,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[..., 0].transpose(1, 2).reshape(batch, frames, -1, channels)


def backproject_pixels(
    depth: torch.Tensor, pixels_uv: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """Backproject BxNxP metric depths into OpenCV camera coordinates."""

    if depth.ndim != 3:
        raise ValueError("depth must have shape [B, N, P]")
    batch, frames, points = depth.shape
    if pixels_uv.ndim == 2:
        pixels_uv = pixels_uv[None, None].expand(batch, frames, -1, -1)
    if pixels_uv.shape != (batch, frames, points, 2):
        raise ValueError("pixels_uv is incompatible with depth")
    if intrinsics.shape != (batch, frames, 3, 3):
        raise ValueError("intrinsics must have shape [B, N, 3, 3]")

    pixel_h = torch.cat((pixels_uv, torch.ones_like(pixels_uv[..., :1])), dim=-1)
    rays = torch.einsum("bnij,bnpj->bnpi", torch.linalg.inv(intrinsics), pixel_h)
    return rays * depth[..., None]

