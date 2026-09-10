#!/usr/bin/env python3
"""Leak-safe visibility masking for simulator-truth BEV rasters.

The complete occupancy raster remains authoritative.  The leak guard changes
only visibility: narrow radial free-space fingers bracketed by a consistent
occupied surface are converted back to unknown, then free islands not
connected to the robot origin are removed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage


UNKNOWN_VALUE = np.uint8(112)
DEFAULT_MAX_SINGLE_BEV_VOID_RATIO = 0.30
VOID_CHECK_VERSION = 2
VOID_COVERAGE_ALGORITHM = "strict-solid-voxel-or-navmesh-coverage-v4"
VOID_FILTER_CONTRACT = "strict-voxel-outer-void-only-gt-validity-v7"
VOID_REPAIR_ALGORITHM = "outer-connected-void-only-v3"
VISIBILITY_ALGORITHM = (
    "symmetric_grid_shadowcasting_anchored_radial_leak_guard_v2"
)


class SingleBEVVoidRatioExceeded(RuntimeError):
    """Raised when one Single Complete GT has too little geometry coverage."""

    def __init__(
        self,
        *,
        ratio: float,
        threshold: float,
        frame_id: int,
        extent_m: float,
    ) -> None:
        self.ratio = float(ratio)
        self.threshold = float(threshold)
        self.frame_id = int(frame_id)
        self.extent_m = float(extent_m)
        super().__init__(
            "Single Complete GT geometry VOID ratio "
            f"{self.ratio:.9f} exceeds {self.threshold:.9f} "
            f"at frame {self.frame_id}, extent {self.extent_m:g} m"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_type": type(self).__name__,
            "frame_id": self.frame_id,
            "extent_m": self.extent_m,
            "single_complete_gt_void_ratio": self.ratio,
            "max_allowed_single_complete_gt_void_ratio": self.threshold,
            "void_coverage_algorithm": VOID_COVERAGE_ALGORITHM,
            "void_filter_contract": VOID_FILTER_CONTRACT,
        }


def repair_valid_coverage(valid_coverage: np.ndarray) -> np.ndarray:
    """Keep only outer-connected invalid space as VOID.

    This is the NumPy-only form of the verified Strict-VOID v7 scene/output
    repair.  Eight-connected exterior propagation is deliberately conservative
    around diagonal openings.  The input is geometry coverage, never an FOV or
    observation mask.
    """

    valid = np.asarray(valid_coverage, dtype=bool)
    if valid.ndim != 2 or valid.size == 0:
        raise ValueError("geometry-valid coverage must be a non-empty 2D raster")
    return ndimage.binary_fill_holes(
        valid,
        structure=np.ones((3, 3), dtype=bool),
    ).astype(bool, copy=False)


def single_bev_void_ratio(valid_coverage: np.ndarray) -> float:
    """Return the invalid geometry fraction of one Single Complete GT grid.

    Boolean True/1 means that complete simulator geometry covers the cell.
    False/0 means VOID.  Requiring a binary mask makes it impossible to
    accidentally pass a masked BEV whose labels are 0/112/255.
    """

    raster = np.asarray(valid_coverage)
    if raster.ndim != 2 or raster.size == 0:
        raise ValueError("geometry-valid coverage must be a non-empty 2D raster")
    if raster.dtype != np.bool_:
        values = set(int(value) for value in np.unique(raster))
        if not values.issubset({0, 1}):
            raise ValueError(
                "VOID requires a binary simulator-geometry coverage mask; "
                f"received labels {sorted(values)}"
            )
    valid = raster.astype(bool, copy=False)
    return float(np.count_nonzero(~valid) / valid.size)


def enforce_single_bev_void_limit(
    valid_coverage: np.ndarray,
    *,
    threshold: float = DEFAULT_MAX_SINGLE_BEV_VOID_RATIO,
    frame_id: int,
    extent_m: float,
) -> float:
    """Reject when a Single Complete GT geometry VOID ratio exceeds the limit."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("VOID threshold must be within [0, 1]")
    ratio = single_bev_void_ratio(valid_coverage)
    if ratio > threshold:
        raise SingleBEVVoidRatioExceeded(
            ratio=ratio,
            threshold=threshold,
            frame_id=frame_id,
            extent_m=extent_m,
        )
    return ratio


def _shadowcast_visible(
    obstacle: np.ndarray,
    *,
    horizontal_fov_degrees: float,
) -> np.ndarray:
    size = obstacle.shape[0]
    visible = np.zeros_like(obstacle, dtype=bool)
    origin_column = origin_row = size // 2
    visible[origin_row, origin_column] = True

    def cast_octant(
        row: int,
        start_slope: float,
        end_slope: float,
        xx: int,
        xy: int,
        yx: int,
        yy: int,
    ) -> None:
        if start_slope < end_slope:
            return
        next_start_slope = start_slope
        for distance in range(row, size + 1):
            delta_x = -distance - 1
            delta_y = -distance
            blocked = False
            while delta_x <= 0:
                delta_x += 1
                column = origin_column + delta_x * xx + delta_y * xy
                output_row = origin_row + delta_x * yx + delta_y * yy
                left_slope = (delta_x - 0.5) / (delta_y + 0.5)
                right_slope = (delta_x + 0.5) / (delta_y - 0.5)
                if start_slope < right_slope:
                    continue
                if end_slope > left_slope:
                    break
                in_bounds = 0 <= column < size and 0 <= output_row < size
                if in_bounds:
                    visible[output_row, column] = True
                cell_is_obstacle = not in_bounds or obstacle[output_row, column]
                if blocked:
                    if cell_is_obstacle:
                        next_start_slope = right_slope
                        continue
                    blocked = False
                    start_slope = next_start_slope
                elif cell_is_obstacle and distance < size:
                    blocked = True
                    cast_octant(
                        distance + 1,
                        start_slope,
                        left_slope,
                        xx,
                        xy,
                        yx,
                        yy,
                    )
                    next_start_slope = right_slope
            if blocked:
                break

    transforms = (
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0, -1, 1, 0),
        (-1, 0, 0, 1),
        (-1, 0, 0, -1),
        (0, -1, -1, 0),
        (0, 1, -1, 0),
        (1, 0, 0, -1),
    )
    for transform in transforms:
        cast_octant(1, 1.0, 0.0, *transform)

    rows, columns = np.indices(obstacle.shape, dtype=np.float64)
    angle = np.arctan2(columns - origin_column, origin_row - rows)
    visible &= (
        np.abs(angle)
        <= math.radians(horizontal_fov_degrees) / 2.0 + 1e-12
    )
    return visible


def _occupied_first_hit(masked: np.ndarray, bins: int) -> np.ndarray:
    """Return closest visible occupied range for conservative angular bins."""

    size = masked.shape[0]
    center = float(size // 2)
    nearest = np.full(bins, np.inf, dtype=np.float64)
    radians_per_bin = 2.0 * math.pi / bins
    for row, column in np.argwhere(masked == 0):
        right = float(column) - center
        forward = center - float(row)
        radius = math.hypot(right, forward)
        if radius <= math.sqrt(0.5):
            continue
        angle = math.atan2(right, forward)
        center_bin = int(math.floor((angle + math.pi) / (2.0 * math.pi) * bins))
        half_angle = math.asin(min(math.sqrt(0.5) / radius, 1.0))
        half_bins = max(int(math.ceil(half_angle / radians_per_bin)), 1)
        for offset in range(-half_bins, half_bins + 1):
            index = (center_bin + offset) % bins
            nearest[index] = min(nearest[index], radius)
    return nearest


def _anchored_radial_fingers(masked: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Find long narrow observed-free fingers bracketed by one surface."""

    size = masked.shape[0]
    scale = size / 512.0
    bins = max(512, size * 4)
    center = float(size // 2)
    rows, columns = np.indices(masked.shape, dtype=np.float64)
    angles = np.arctan2(columns - center, center - rows)
    radii = np.hypot(columns - center, center - rows)
    indices = (
        np.floor((angles + math.pi) / (2.0 * math.pi) * bins).astype(np.int64)
        % bins
    )
    free = masked == 255
    farthest = np.zeros(bins, dtype=np.float64)
    np.maximum.at(farthest, indices[free], radii[free])
    farthest = ndimage.maximum_filter1d(farthest, size=3, mode="nearest")

    maximum_width_degrees = 3.0
    window = max(int(round(maximum_width_degrees / 360.0 * bins)), 3)
    if window % 2 == 0:
        window += 1
    baseline = ndimage.grey_opening(farthest, size=window, mode="nearest")
    minimum_length = max(4.0, 10.0 * scale)
    candidate_bins = farthest - baseline >= minimum_length
    labels, component_count = ndimage.label(candidate_bins)
    if int(component_count) == 0:
        return np.zeros_like(free), {
            "radial_candidate_runs": 0,
            "radial_approved_runs": 0,
            "radial_removed_free_cells": 0,
        }
    occupied_range = _occupied_first_hit(masked, bins)
    search_bins = max(int(round(2.0 / 360.0 * bins)), 2)
    maximum_throat_width = max(2.0, 6.0 * scale)
    maximum_range_delta = max(2.0, 4.0 * scale)
    radians_per_bin = 2.0 * math.pi / bins
    approved_bins = np.zeros(bins, dtype=bool)
    approved_runs = 0

    for component_id in range(1, int(component_count) + 1):
        component = np.flatnonzero(labels == component_id)
        if component.size == 0:
            continue
        start, end = int(component[0]), int(component[-1])
        root_range = float(np.median(baseline[component]))
        throat_width = component.size * radians_per_bin * root_range
        if root_range <= 0.0 or throat_width > maximum_throat_width:
            continue
        left_indices = np.arange(max(0, start - search_bins), start)
        right_indices = np.arange(end + 1, min(bins, end + 1 + search_bins))
        left_indices = left_indices[np.isfinite(occupied_range[left_indices])]
        right_indices = right_indices[np.isfinite(occupied_range[right_indices])]
        if left_indices.size == 0 or right_indices.size == 0:
            continue
        left_range = float(occupied_range[left_indices[-1]])
        right_range = float(occupied_range[right_indices[0]])
        if abs(left_range - right_range) > maximum_range_delta:
            continue
        if abs(0.5 * (left_range + right_range) - root_range) > maximum_range_delta:
            continue
        approved_bins[component] = True
        approved_runs += 1

    radial_limit = baseline[indices]
    removed = (
        free
        & approved_bins[indices]
        & (radii > radial_limit + max(0.75, 0.75 * scale))
    )
    return removed, {
        "radial_candidate_runs": int(component_count),
        "radial_approved_runs": approved_runs,
        "radial_removed_free_cells": int(np.count_nonzero(removed)),
    }


def stabilize_visibility_mask(masked: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the primary leak guard, then the independent origin post-process."""

    if masked.ndim != 2 or masked.shape[0] != masked.shape[1]:
        raise ValueError("masked must be a square grayscale image")
    result = np.asarray(masked, dtype=np.uint8).copy()
    removed, diagnostics = _anchored_radial_fingers(result)
    result[removed] = UNKNOWN_VALUE

    free = result == 255
    structure = ndimage.generate_binary_structure(2, 1)
    labels, _ = ndimage.label(free, structure=structure)
    center = result.shape[0] // 2
    origin_label = int(labels[center, center])
    if origin_label == 0:
        raise ValueError("robot origin is not free after primary visibility processing")
    isolated = free & (labels != origin_label)
    result[isolated] = UNKNOWN_VALUE
    diagnostics["isolated_removed_free_cells"] = int(np.count_nonzero(isolated))
    return result, diagnostics


def render_visibility_masked_map(
    ground_truth: np.ndarray,
    *,
    horizontal_fov_degrees: float,
) -> np.ndarray:
    """Render exact truth labels through leak-safe grid visibility."""

    if ground_truth.ndim != 2 or ground_truth.shape[0] != ground_truth.shape[1]:
        raise ValueError("ground_truth must be a square grayscale image")
    if not 0.0 < horizontal_fov_degrees <= 360.0:
        raise ValueError("horizontal_fov_degrees must be in (0, 360]")
    visible = _shadowcast_visible(
        ground_truth == 0,
        horizontal_fov_degrees=horizontal_fov_degrees,
    )
    masked = np.full_like(ground_truth, UNKNOWN_VALUE, dtype=np.uint8)
    masked[visible] = ground_truth[visible]
    stabilized, _ = stabilize_visibility_mask(masked)
    known = stabilized != UNKNOWN_VALUE
    if not np.array_equal(stabilized[known], ground_truth[known]):
        raise RuntimeError("visibility output disagrees with complete truth")
    return stabilized
