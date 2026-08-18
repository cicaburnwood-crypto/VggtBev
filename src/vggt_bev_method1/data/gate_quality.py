from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GateLeakRepairConfig:
    """Conservative, raster-only repair policy for observed-free GT leaks.

    A pixel is never removed merely because RGB depth and the BEV disagree.
    Broad-lobe repair requires all of the following:

    * a very short break between two observed occupied-surface pixels;
    * closing that break disconnects a free lobe from the camera origin;
    * the disconnected lobe becomes occluded from the origin by the repaired
      surface for most of its area.

    A second branch removes only long, narrow appendages with a small contact
    neck to the broad observed region.  This catches needle-like raster rays
    without shaving ordinary broad FOV boundaries.  Ambiguous pixels are
    reported separately so a caller can ignore only their Gate supervision.
    """

    occupied_value: int = 0
    unknown_value: int = 112
    free_value: int = 255
    maximum_surface_gap_pixels: int = 1
    minimum_lobe_pixels: int = 24
    minimum_occluded_fraction: float = 0.90
    angular_bins: int = 4096
    line_of_sight_margin_pixels: float = 1.25
    origin_seed_radius_pixels: float = 7.0
    thin_branch_opening_radius_pixels: int = 3
    minimum_thin_branch_pixels: int = 16
    minimum_thin_branch_elongation: float = 1.5
    maximum_thin_branch_contact_fraction: float = 0.50


@dataclass(frozen=True)
class GateLeakRepairResult:
    repair_remove: np.ndarray
    thin_branch_remove: np.ndarray
    ambiguous_ignore: np.ndarray
    inserted_surface: np.ndarray
    raw_origin_reachable: np.ndarray
    repaired_origin_reachable: np.ndarray
    closed_surface_occluded: np.ndarray
    candidate_component_count: int
    accepted_component_count: int
    accepted_thin_branch_count: int


DEFAULT_GATE_LEAK_REPAIR_CONFIG = GateLeakRepairConfig()


_LINE_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _line_coordinates(
    height: int,
    width: int,
    direction: tuple[int, int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return [
            (np.full(width, row, dtype=np.int32), np.arange(width, dtype=np.int32))
            for row in range(height)
        ]
    if (dr, dc) == (1, 0):
        return [
            (
                np.arange(height, dtype=np.int32),
                np.full(height, column, dtype=np.int32),
            )
            for column in range(width)
        ]

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    if (dr, dc) == (1, 1):
        starts = [(0, column) for column in range(width)]
        starts.extend((row, 0) for row in range(1, height))
    else:
        starts = [(0, column) for column in range(width)]
        starts.extend((row, width - 1) for row in range(1, height))
    for start_row, start_column in starts:
        length = min(
            height - start_row,
            width - start_column if dc > 0 else start_column + 1,
        )
        step = np.arange(length, dtype=np.int32)
        lines.append(
            (
                start_row + step,
                start_column + dc * step,
            )
        )
    return lines


def _short_surface_breaks(
    surface: np.ndarray,
    free: np.ndarray,
    maximum_gap_pixels: int,
) -> np.ndarray:
    """Return only free runs bracketed by surface on one of four axes."""

    if maximum_gap_pixels < 1:
        return np.zeros_like(surface, dtype=bool)
    if maximum_gap_pixels == 1:
        inserted = np.zeros_like(surface, dtype=bool)
        inserted[:, 1:-1] |= (
            surface[:, :-2] & free[:, 1:-1] & surface[:, 2:]
        )
        inserted[1:-1, :] |= (
            surface[:-2, :] & free[1:-1, :] & surface[2:, :]
        )
        inserted[1:-1, 1:-1] |= (
            surface[:-2, :-2] & free[1:-1, 1:-1] & surface[2:, 2:]
        )
        inserted[1:-1, 1:-1] |= (
            surface[:-2, 2:] & free[1:-1, 1:-1] & surface[2:, :-2]
        )
        return inserted
    height, width = surface.shape
    inserted = np.zeros_like(surface, dtype=bool)
    for direction in _LINE_DIRECTIONS:
        for rows, columns in _line_coordinates(height, width, direction):
            values = surface[rows, columns]
            occupied_positions = np.flatnonzero(values)
            if occupied_positions.size < 2:
                continue
            for left, right in zip(
                occupied_positions[:-1],
                occupied_positions[1:],
                strict=True,
            ):
                gap_size = int(right - left - 1)
                if gap_size < 1 or gap_size > maximum_gap_pixels:
                    continue
                gap = slice(left + 1, right)
                gap_rows = rows[gap]
                gap_columns = columns[gap]
                if bool(free[gap_rows, gap_columns].all()):
                    inserted[gap_rows, gap_columns] = True
    return inserted


def _origin_seed(free: np.ndarray, radius: float) -> np.ndarray:
    height, width = free.shape
    row_center = (height - 1) / 2.0
    column_center = (width - 1) / 2.0
    rows, columns = np.ogrid[:height, :width]
    near_origin = (
        (rows - row_center) ** 2 + (columns - column_center) ** 2 <= radius**2
    )
    seed = free & near_origin
    if bool(seed.any()):
        return seed

    free_rows, free_columns = np.nonzero(free)
    if not free_rows.size:
        return np.zeros_like(free, dtype=bool)
    distance_squared = (free_rows - row_center) ** 2 + (
        free_columns - column_center
    ) ** 2
    nearest = int(np.argmin(distance_squared))
    seed[free_rows[nearest], free_columns[nearest]] = True
    return seed


def _origin_reachable(free: np.ndarray, seed: np.ndarray) -> np.ndarray:
    if not bool(seed.any()):
        return np.zeros_like(free, dtype=bool)
    return ndimage.binary_propagation(
        seed & free,
        structure=np.ones((3, 3), dtype=bool),
        mask=free,
    )


def _disk(radius: int) -> np.ndarray:
    rows, columns = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return rows * rows + columns * columns <= radius * radius


def _thin_observed_branches(
    free: np.ndarray,
    *,
    opening_radius: int,
    minimum_pixels: int,
    minimum_elongation: float,
    maximum_contact_fraction: float,
) -> tuple[np.ndarray, int]:
    """Prune long, needle-like free appendages without shaving broad borders."""

    empty = np.zeros_like(free, dtype=bool)
    if opening_radius < 1 or not bool(free.any()):
        return empty, 0
    opened = ndimage.binary_opening(free, structure=_disk(opening_radius))
    top_hat = free & ~opened
    component_labels, component_count = ndimage.label(
        top_hat,
        structure=np.ones((3, 3), dtype=bool),
    )
    contact_band = ndimage.binary_dilation(
        opened,
        structure=np.ones((3, 3), dtype=bool),
    )
    accepted = np.zeros_like(free, dtype=bool)
    accepted_count = 0
    for component_id in range(1, component_count + 1):
        component = component_labels == component_id
        area = int(component.sum())
        if area < minimum_pixels:
            continue
        rows, columns = np.nonzero(component)
        covariance = np.cov(np.stack((rows, columns)), bias=True)
        eigenvalues = np.linalg.eigvalsh(covariance)
        elongation = float(
            np.sqrt((eigenvalues[-1] + 1e-6) / (eigenvalues[0] + 1e-6))
        )
        if elongation < minimum_elongation:
            continue
        contact = int((component & contact_band).sum())
        if contact / area > maximum_contact_fraction:
            continue
        accepted |= component
        accepted_count += 1
    return accepted, accepted_count


def _first_surface_radius(
    surface: np.ndarray,
    angular_bins: int,
) -> np.ndarray:
    """Rasterize each surface cell's angular footprint into a polar z-buffer."""

    height, width = surface.shape
    center_row = height / 2.0
    center_column = width / 2.0
    rows, columns = np.nonzero(surface)
    first = np.full(angular_bins, np.inf, dtype=np.float32)
    if not rows.size:
        return first

    dy = center_row - (rows.astype(np.float32) + 0.5)
    dx = columns.astype(np.float32) + 0.5 - center_column
    radius = np.hypot(dx, dy)
    angle = np.mod(np.arctan2(dx, dy), 2.0 * np.pi)
    # Half of a cell diagonal is a conservative angular footprint.  Expanding
    # the surface angularly reduces false "visible through one raster corner"
    # cases without expanding it radially into valid floor.
    half_angle = np.minimum(np.pi, np.arctan2(np.sqrt(0.5), np.maximum(radius, 0.5)))
    bin_scale = angular_bins / (2.0 * np.pi)
    for cell_angle, cell_half_angle, cell_radius in zip(
        angle,
        half_angle,
        radius,
        strict=True,
    ):
        center_bin = int(np.floor(cell_angle * bin_scale)) % angular_bins
        half_bins = max(1, int(np.ceil(cell_half_angle * bin_scale)))
        candidate_radius = max(0.0, float(cell_radius) - np.sqrt(0.5))
        offsets = np.arange(-half_bins, half_bins + 1, dtype=np.int32)
        bins = (center_bin + offsets) % angular_bins
        np.minimum.at(first, bins, candidate_radius)
    return first


def _surface_occluded(
    free: np.ndarray,
    surface: np.ndarray,
    *,
    angular_bins: int,
    margin_pixels: float,
) -> np.ndarray:
    """Conservatively require the full angular width of a cell to be blocked."""

    first = _first_surface_radius(surface, angular_bins)
    height, width = free.shape
    center_row = height / 2.0
    center_column = width / 2.0
    rows, columns = np.nonzero(free)
    occluded = np.zeros_like(free, dtype=bool)
    if not rows.size:
        return occluded

    dy = center_row - (rows.astype(np.float32) + 0.5)
    dx = columns.astype(np.float32) + 0.5 - center_column
    radius = np.hypot(dx, dy)
    angle = np.mod(np.arctan2(dx, dy), 2.0 * np.pi)
    half_angle = np.minimum(np.pi, np.arctan2(np.sqrt(0.5), np.maximum(radius, 0.5)))
    bin_scale = angular_bins / (2.0 * np.pi)
    center_bins = np.floor(angle * bin_scale).astype(np.int64) % angular_bins
    half_bins = np.maximum(1, np.ceil(half_angle * bin_scale).astype(np.int64))
    left_bins = (center_bins - half_bins) % angular_bins
    right_bins = (center_bins + half_bins) % angular_bins
    nearest_on_samples = np.maximum.reduce(
        (first[left_bins], first[center_bins], first[right_bins])
    )
    blocked = nearest_on_samples + margin_pixels < radius
    occluded[rows[blocked], columns[blocked]] = True
    return occluded


def repair_observed_gate_leaks(
    masked_bev: np.ndarray,
    *,
    config: GateLeakRepairConfig = DEFAULT_GATE_LEAK_REPAIR_CONFIG,
) -> GateLeakRepairResult:
    """Find only high-confidence observed-free lobes behind tiny surface gaps."""

    labels = np.asarray(masked_bev)
    if labels.ndim != 2:
        raise ValueError("masked_bev must be a two-dimensional label raster")
    allowed = {
        config.occupied_value,
        config.unknown_value,
        config.free_value,
    }
    if not set(np.unique(labels).tolist()) <= allowed:
        raise ValueError("masked_bev contains labels outside the configured contract")

    free = labels == config.free_value
    surface = labels == config.occupied_value
    thin_branch_remove, accepted_thin_branch_count = _thin_observed_branches(
        free,
        opening_radius=config.thin_branch_opening_radius_pixels,
        minimum_pixels=config.minimum_thin_branch_pixels,
        minimum_elongation=config.minimum_thin_branch_elongation,
        maximum_contact_fraction=config.maximum_thin_branch_contact_fraction,
    )
    inserted_surface = _short_surface_breaks(
        surface,
        free,
        config.maximum_surface_gap_pixels,
    )
    empty = np.zeros_like(free, dtype=bool)
    if not bool(inserted_surface.any()) or not bool(free.any()):
        return GateLeakRepairResult(
            repair_remove=thin_branch_remove,
            thin_branch_remove=thin_branch_remove,
            ambiguous_ignore=empty.copy(),
            inserted_surface=inserted_surface,
            raw_origin_reachable=empty.copy(),
            repaired_origin_reachable=empty.copy(),
            closed_surface_occluded=empty.copy(),
            candidate_component_count=0,
            accepted_component_count=0,
            accepted_thin_branch_count=accepted_thin_branch_count,
        )

    seed = _origin_seed(free, config.origin_seed_radius_pixels)
    raw_reachable = _origin_reachable(free, seed)
    repaired_free = free & ~inserted_surface
    repaired_reachable = _origin_reachable(repaired_free, seed)
    disconnected = raw_reachable & ~repaired_reachable
    closed_surface = surface | inserted_surface
    closed_occluded = _surface_occluded(
        free,
        closed_surface,
        angular_bins=config.angular_bins,
        margin_pixels=config.line_of_sight_margin_pixels,
    )

    component_labels, component_count = ndimage.label(
        disconnected,
        structure=np.ones((3, 3), dtype=bool),
    )
    repair_remove = np.zeros_like(free, dtype=bool)
    ambiguous_ignore = np.zeros_like(free, dtype=bool)
    accepted = 0
    for component_id in range(1, component_count + 1):
        component = component_labels == component_id
        component_pixels = int(component.sum())
        if component_pixels < config.minimum_lobe_pixels:
            # A sealed pixel or tiny raster chip is not a lobe.  It is left
            # exactly as collected; neither the target nor Gate loss changes.
            continue
        non_gap = component & ~inserted_surface
        non_gap_pixels = int(non_gap.sum())
        occluded_fraction = float((non_gap & closed_occluded).sum()) / max(
            non_gap_pixels,
            1,
        )
        if occluded_fraction >= config.minimum_occluded_fraction:
            repair_remove |= component
            accepted += 1
        else:
            ambiguous_ignore |= component

    return GateLeakRepairResult(
        repair_remove=repair_remove | thin_branch_remove,
        thin_branch_remove=thin_branch_remove,
        ambiguous_ignore=ambiguous_ignore,
        inserted_surface=inserted_surface,
        raw_origin_reachable=raw_reachable,
        repaired_origin_reachable=repaired_reachable,
        closed_surface_occluded=closed_occluded,
        candidate_component_count=int(component_count),
        accepted_component_count=accepted,
        accepted_thin_branch_count=accepted_thin_branch_count,
    )
