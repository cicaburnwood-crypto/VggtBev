from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ConservativeVisibilityConfig:
    """Numerical policy for conservative, cell-footprint visibility.

    ``angular_oversample`` controls only angular discretization accuracy.  It
    does not define a branch width or a morphology threshold.  The number of
    bins scales with raster radius, so the same code works at other BEV sizes.
    """

    occupied_value: int = 0
    unknown_value: int = 112
    free_value: int = 255
    angular_oversample: int = 4
    raster_boundary_uncertainty_layers: int = 2


@dataclass(frozen=True)
class ConservativeVisibilityResult:
    repair_remove: np.ndarray
    disconnected_remove: np.ndarray
    partial_cell_remove: np.ndarray
    boundary_partial_keep: np.ndarray
    post_visibility_disconnected_remove: np.ndarray
    raw_origin_reachable: np.ndarray
    repaired_origin_reachable: np.ndarray
    angular_bins: int


DEFAULT_CONSERVATIVE_VISIBILITY_CONFIG = ConservativeVisibilityConfig()


def _origin_component(mask: np.ndarray) -> np.ndarray:
    """Return the 8-connected component seeded at the exact BEV origin."""

    height, width = mask.shape
    origin = (height // 2, width // 2)
    if not bool(mask[origin]):
        return np.zeros_like(mask, dtype=bool)
    seed = np.zeros_like(mask, dtype=bool)
    seed[origin] = True
    return ndimage.binary_propagation(
        seed,
        structure=np.ones((3, 3), dtype=bool),
        mask=mask,
    )


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, value - 1).bit_length()


def _angular_bin_count(shape: tuple[int, int], oversample: int) -> int:
    if oversample < 1:
        raise ValueError("angular_oversample must be positive")
    height, width = shape
    maximum_radius = float(np.hypot(height / 2.0, width / 2.0))
    # At maximum range, one cell subtends approximately 1/r radians.  This
    # samples that smallest cell footprint ``oversample`` times.
    requested = int(np.ceil(2.0 * np.pi * maximum_radius * oversample))
    return _next_power_of_two(max(requested, 1024))


def _cell_intervals(
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unwrapped angular intervals and radial entry distances.

    Angles use the BEV convention: zero points up/forward and positive angles
    point right.  Each raster cell is treated as its complete closed square,
    rather than as a point sample at its centre.
    """

    height, width = shape
    origin_row = height // 2
    origin_column = width // 2
    dy = origin_row - rows.astype(np.float64)
    dx = columns.astype(np.float64) - origin_column
    center_angle = np.arctan2(dx, dy)

    corner_angles = []
    for row_offset in (-0.5, 0.5):
        for column_offset in (-0.5, 0.5):
            corner_dy = dy - row_offset
            corner_dx = dx + column_offset
            angle = np.arctan2(corner_dx, corner_dy)
            relative = np.angle(np.exp(1j * (angle - center_angle)))
            corner_angles.append(relative)
    relative_corners = np.stack(corner_angles, axis=0)
    interval_start = center_angle + relative_corners.min(axis=0)
    interval_end = center_angle + relative_corners.max(axis=0)

    nearest_dx = np.maximum(np.abs(dx) - 0.5, 0.0)
    nearest_dy = np.maximum(np.abs(dy) - 0.5, 0.0)
    entry_radius = np.hypot(nearest_dx, nearest_dy)
    return interval_start, interval_end, entry_radius, center_angle


def _first_obstacle_radius(
    obstacle: np.ndarray,
    *,
    angular_bins: int,
) -> np.ndarray:
    """Rasterize complete occupied-cell angular footprints into a z-buffer."""

    rows, columns = np.nonzero(obstacle)
    first = np.full(angular_bins, np.inf, dtype=np.float32)
    if not rows.size:
        return first
    starts, ends, entry_radii, _ = _cell_intervals(
        rows,
        columns,
        shape=obstacle.shape,
    )
    scale = angular_bins / (2.0 * np.pi)
    start_bins = np.floor(starts * scale).astype(np.int64)
    end_bins = np.ceil(ends * scale).astype(np.int64)
    for start, end, radius in zip(
        start_bins,
        end_bins,
        entry_radii,
        strict=True,
    ):
        bins = np.arange(start, end + 1, dtype=np.int64) % angular_bins
        np.minimum.at(first, bins, np.float32(radius))
    return first


def _circular_range_minimum(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Query inclusive circular range minima in O(log N) preprocessing."""

    count = int(values.size)
    lengths = ends - starts + 1
    if bool((lengths <= 0).any()) or bool((lengths > count).any()):
        raise ValueError("angular interval length lies outside the bin domain")
    doubled = np.concatenate((values, values))
    table = [doubled]
    stride = 1
    while stride * 2 <= doubled.size:
        previous = table[-1]
        stride *= 2
        table.append(
            np.minimum(previous[: doubled.size - stride + 1], previous[stride // 2 :])
        )

    left = np.mod(starts, count)
    right = left + lengths - 1
    levels = np.floor(np.log2(lengths)).astype(np.int64)
    span = np.left_shift(1, levels)
    output = np.empty(starts.shape, dtype=values.dtype)
    for level in np.unique(levels):
        selected = levels == level
        level_values = table[int(level)]
        output[selected] = np.minimum(
            level_values[left[selected]],
            level_values[right[selected] - span[selected] + 1],
        )
    return output


def conservative_observed_gate_repair(
    masked_bev: np.ndarray,
    complete_bev: np.ndarray,
    *,
    config: ConservativeVisibilityConfig = DEFAULT_CONSERVATIVE_VISIBILITY_CONFIG,
) -> ConservativeVisibilityResult:
    """Remove direct-observation labels lacking full-cell visibility evidence.

    The operation is deletion-only on observed-free supervision.  Complete
    occupancy, FOV support, Void labels, and semantic values are not edited.
    """

    masked = np.asarray(masked_bev)
    complete = np.asarray(complete_bev)
    if masked.ndim != 2 or complete.shape != masked.shape:
        raise ValueError("masked and complete BEVs must be equal-size 2D rasters")
    allowed_masked = {
        config.occupied_value,
        config.unknown_value,
        config.free_value,
    }
    allowed_complete = {
        config.occupied_value,
        config.unknown_value,
        config.free_value,
    }
    if not set(np.unique(masked).tolist()) <= allowed_masked:
        raise ValueError("masked BEV contains labels outside the configured contract")
    if not set(np.unique(complete).tolist()) <= allowed_complete:
        raise ValueError("complete BEV contains labels outside the configured contract")

    free = masked == config.free_value
    complete_obstacle = complete == config.occupied_value
    raw_reachable = _origin_component(free)
    disconnected_remove = free & ~raw_reachable

    angular_bins = _angular_bin_count(masked.shape, config.angular_oversample)
    first_obstacle = _first_obstacle_radius(
        complete_obstacle,
        angular_bins=angular_bins,
    )
    rows, columns = np.nonzero(raw_reachable)
    partial_remove = np.zeros_like(free, dtype=bool)
    if rows.size:
        starts, ends, entry_radii, _ = _cell_intervals(
            rows,
            columns,
            shape=masked.shape,
        )
        scale = angular_bins / (2.0 * np.pi)
        start_bins = np.floor(starts * scale).astype(np.int64)
        end_bins = np.ceil(ends * scale).astype(np.int64)
        interval_minimum = _circular_range_minimum(
            first_obstacle,
            start_bins,
            end_bins,
        )
        blocked_before_cell = interval_minimum + 1e-6 < entry_radii
        origin = (masked.shape[0] // 2, masked.shape[1] // 2)
        blocked_before_cell[(rows == origin[0]) & (columns == origin[1])] = False
        partial_remove[rows[blocked_before_cell], columns[blocked_before_cell]] = True

    if config.raster_boundary_uncertainty_layers < 0:
        raise ValueError("raster_boundary_uncertainty_layers cannot be negative")
    fully_visible = raw_reachable & ~partial_remove
    if config.raster_boundary_uncertainty_layers:
        boundary_band = ndimage.binary_dilation(
            fully_visible,
            structure=np.ones((3, 3), dtype=bool),
            iterations=config.raster_boundary_uncertainty_layers,
        )
    else:
        boundary_band = fully_visible
    # Partial cells entirely confined to the ordinary raster boundary band are
    # retained.  This prevents a conservative visibility test from shaving a
    # broad valid region.  Long fan/ray interiors extend beyond this band and
    # remain repair candidates.  The band is a discretization tolerance, not a
    # content-dependent branch-width threshold.
    boundary_partial_keep = partial_remove & boundary_band
    visibility_remove = partial_remove & ~boundary_partial_keep
    provisional = raw_reachable & ~visibility_remove
    repaired_reachable = _origin_component(provisional)
    post_disconnected = provisional & ~repaired_reachable
    repair_remove = disconnected_remove | visibility_remove | post_disconnected
    return ConservativeVisibilityResult(
        repair_remove=repair_remove,
        disconnected_remove=disconnected_remove,
        partial_cell_remove=partial_remove,
        boundary_partial_keep=boundary_partial_keep,
        post_visibility_disconnected_remove=post_disconnected,
        raw_origin_reachable=raw_reachable,
        repaired_origin_reachable=repaired_reachable,
        angular_bins=angular_bins,
    )
