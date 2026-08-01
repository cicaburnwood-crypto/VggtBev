from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from vggt_bev_method1.config import LabelValues


@dataclass(frozen=True)
class P2BRegionMasks:
    valid: torch.Tensor
    observed: torch.Tensor
    guessed: torch.Tensor
    observed_free: torch.Tensor
    observed_surface: torch.Tensor
    occupied: torch.Tensor
    gate_target: torch.Tensor


@dataclass(frozen=True)
class PackedRayBank:
    indices: torch.Tensor
    valid: torch.Tensor
    angles: torch.Tensor


def p2b_region_masks(
    complete: torch.Tensor,
    visible: torch.Tensor,
    support: torch.Tensor,
    *,
    labels: LabelValues = LabelValues(),
) -> P2BRegionMasks:
    """Build mutually exclusive GT routing masks without predicted gating."""

    if not (complete.shape == visible.shape == support.shape):
        raise ValueError("complete, visible and support targets must align")
    valid = complete != labels.unknown
    if not torch.equal(valid, support.bool()):
        raise ValueError("P2B support must equal the valid complete target")
    observed = (visible != labels.unknown) & valid
    guessed = valid & ~observed
    if bool((observed & (visible != complete)).any()):
        raise ValueError("visible target disagrees with complete target")
    occupied = complete == labels.occupied
    observed_free = observed & ~occupied
    observed_surface = observed & occupied
    return P2BRegionMasks(
        valid=valid,
        observed=observed,
        guessed=guessed,
        observed_free=observed_free,
        observed_surface=observed_surface,
        occupied=occupied,
        gate_target=observed.to(torch.float32),
    )


def _supercover_line(
    start_row: int,
    start_column: int,
    end_row: int,
    end_column: int,
) -> list[tuple[int, int]]:
    delta_column = end_column - start_column
    delta_row = end_row - start_row
    column_step = 1 if delta_column >= 0 else -1
    row_step = 1 if delta_row >= 0 else -1
    nx = abs(delta_column)
    ny = abs(delta_row)
    column = start_column
    row = start_row
    points = [(row, column)]
    ix = 0
    iy = 0
    while ix < nx or iy < ny:
        decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if decision == 0:
            column += column_step
            row += row_step
            ix += 1
            iy += 1
        elif decision < 0:
            column += column_step
            ix += 1
        else:
            row += row_step
            iy += 1
        points.append((row, column))
    return points


def _support_endpoints(support: torch.Tensor) -> list[tuple[int, int, float]]:
    """Choose the farthest support boundary cell in each angular bin."""

    if support.ndim != 2:
        raise ValueError("one support raster is required")
    height, width = support.shape
    origin_row = height // 2
    origin_column = width // 2
    target = support.detach().to(device="cpu", dtype=torch.bool)
    outside = 1.0 - F.pad(target[None, None].float(), (1, 1, 1, 1))
    neighbour_outside = F.max_pool2d(outside, kernel_size=3, stride=1)[0, 0]
    boundary = target & (neighbour_outside > 0)
    coordinates = boundary.nonzero(as_tuple=False)
    # Four bins per raster width over 360 degrees yields about one ray per
    # far-edge cell for a 90-degree FOV (roughly 512 rays at 512x512).
    angular_bins = 4 * max(height, width)
    selected: dict[int, tuple[int, int, float, float]] = {}
    for row_tensor, column_tensor in coordinates:
        row = int(row_tensor)
        column = int(column_tensor)
        right = column - origin_column
        forward = origin_row - row
        if right == 0 and forward == 0:
            continue
        angle = math.atan2(right, forward)
        bin_index = int(round((angle + math.pi) / (2.0 * math.pi) * angular_bins))
        distance = float(right * right + forward * forward)
        previous = selected.get(bin_index)
        if previous is None or distance > previous[3]:
            selected[bin_index] = (row, column, angle, distance)
    return [
        (row, column, angle)
        for row, column, angle, _ in sorted(
            selected.values(), key=lambda value: value[2]
        )
    ]


def build_packed_ray_bank(
    support: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> PackedRayBank:
    """Build deterministic target rays for one support raster."""

    if support.ndim != 2:
        raise ValueError("one support raster is required")
    height, width = support.shape
    origin_row = height // 2
    origin_column = width // 2
    support_cpu = support.detach().to(device="cpu", dtype=torch.bool)
    rays: list[list[int]] = []
    angles: list[float] = []
    for end_row, end_column, angle in _support_endpoints(support_cpu):
        points = _supercover_line(
            origin_row, origin_column, end_row, end_column
        )
        connected: list[int] = []
        started = False
        for row, column in points:
            inside = bool(support_cpu[row, column])
            if inside:
                started = True
                connected.append(row * width + column)
            elif started:
                break
        if connected:
            rays.append(connected)
            angles.append(angle)
    maximum = max((len(ray) for ray in rays), default=1)
    indices = torch.zeros((len(rays), maximum), dtype=torch.long)
    valid = torch.zeros((len(rays), maximum), dtype=torch.bool)
    for ray_index, ray in enumerate(rays):
        indices[ray_index, : len(ray)] = torch.tensor(ray, dtype=torch.long)
        valid[ray_index, : len(ray)] = True
    target_device = support.device if device is None else device
    return PackedRayBank(
        indices=indices.to(target_device),
        valid=valid.to(target_device),
        angles=torch.tensor(angles, dtype=torch.float32, device=target_device),
    )
