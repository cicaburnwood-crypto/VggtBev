"""Local-coordinate click-to-plan helpers for the interactive BEV verifier.

The planner has the same runtime boundary as a real robot: a target expressed
in the current ego frame plus the synchronized predicted semantic BEV.  The GT
image may be used as a browser background for a human click, but neither its
pixels nor any simulator pose/extrinsic cross this module's planning boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


FREE_VALUE = 255
OCCUPIED_VALUE = 0
UNKNOWN_VALUE = 112
DEFAULT_INFLATION_RADIUS_M = 0.025


class VerifierPlanningError(RuntimeError):
    """A structured, user-visible verifier planning failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def payload(self) -> dict[str, Any]:
        return {
            "success": False,
            "failure_code": self.code,
            "failure_reason": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class MetricRaster:
    size: int
    extent_m: float

    def __post_init__(self) -> None:
        if self.size <= 0 or not math.isfinite(self.extent_m) or self.extent_m <= 0:
            raise ValueError("raster size and extent must be positive")

    @property
    def cell_size_m(self) -> float:
        return self.extent_m / self.size

    def pixel_to_metric(self, row: int, column: int) -> tuple[float, float]:
        if not (0 <= row < self.size and 0 <= column < self.size):
            raise ValueError("pixel lies outside the raster")
        x_m = -self.extent_m / 2.0 + (column + 0.5) * self.cell_size_m
        z_m = self.extent_m / 2.0 - (row + 0.5) * self.cell_size_m
        return float(x_m), float(z_m)

    def metric_to_pixel(self, x_m: float, z_m: float) -> tuple[int, int]:
        if not (math.isfinite(x_m) and math.isfinite(z_m)):
            raise ValueError("metric coordinates must be finite")
        column = math.floor((x_m + self.extent_m / 2.0) / self.cell_size_m)
        row = math.floor((self.extent_m / 2.0 - z_m) / self.cell_size_m)
        if not (0 <= row < self.size and 0 <= column < self.size):
            raise VerifierPlanningError(
                "target_outside_predicted_extent",
                "The selected metric target lies outside the predicted BEV extent.",
                target_metric_m=[float(x_m), float(z_m)],
                predicted_extent_m=self.extent_m,
            )
        return int(row), int(column)


def _square_semantic(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a square 2-D semantic raster")
    if value.shape[0] < 2:
        raise ValueError(f"{name} is too small")
    return value.astype(np.uint8, copy=False)


_NEIGHBORS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def astar_no_inflation(
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> tuple[tuple[int, int], ...] | None:
    """Run 8-connected A* on the exact semantic raster without dilation."""

    import heapq

    blocked = np.asarray(blocked, dtype=bool)
    if blocked.ndim != 2 or blocked.shape[0] != blocked.shape[1]:
        raise ValueError("blocked must be a square 2-D raster")
    size = blocked.shape[0]
    for point in (start, goal):
        if not (0 <= point[0] < size and 0 <= point[1] < size):
            raise ValueError("A* endpoint lies outside the raster")
        if blocked[point]:
            return None

    def heuristic(point: tuple[int, int]) -> float:
        dr = abs(point[0] - goal[0])
        dc = abs(point[1] - goal[1])
        return min(dr, dc) * math.sqrt(2.0) + abs(dr - dc)

    frontier: list[tuple[float, float, int, tuple[int, int]]] = []
    serial = 0
    heapq.heappush(frontier, (heuristic(start), 0.0, serial, start))
    best = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while frontier:
        _, cost, _, current = heapq.heappop(frontier)
        if cost > best.get(current, math.inf):
            continue
        if current == goal:
            output = [current]
            while output[-1] != start:
                output.append(parent[output[-1]])
            output.reverse()
            return tuple(output)
        row, column = current
        for dr, dc, edge_cost in _NEIGHBORS:
            neighbor = row + dr, column + dc
            if not (0 <= neighbor[0] < size and 0 <= neighbor[1] < size):
                continue
            if blocked[neighbor]:
                continue
            # Do not cut diagonally between two touching occupied/unknown cells.
            if dr and dc and (blocked[row, neighbor[1]] or blocked[neighbor[0], column]):
                continue
            candidate = cost + edge_cost
            if candidate >= best.get(neighbor, math.inf):
                continue
            best[neighbor] = candidate
            parent[neighbor] = current
            serial += 1
            heapq.heappush(
                frontier,
                (candidate + heuristic(neighbor), candidate, serial, neighbor),
            )
    return None


def inflate_occupied_mask(
    occupied: np.ndarray,
    *,
    cell_size_m: float,
    radius_m: float,
) -> tuple[np.ndarray, int, float]:
    """Dilate occupied cells outward by a conservative metric radius.

    ``radius_m`` is a one-sided radius, not a kernel diameter.  Since the A*
    raster is discrete, the radius is rounded upward to a whole number of
    cells so the requested safety clearance is never rounded down.
    """

    source = np.asarray(occupied, dtype=bool)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("occupied must be a square 2-D raster")
    if not math.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    if not math.isfinite(radius_m) or radius_m < 0:
        raise ValueError("radius_m must be non-negative")
    radius_cells = int(math.ceil(radius_m / cell_size_m))
    if radius_cells == 0:
        return source.copy(), 0, 0.0
    inflated = source.copy()
    size = source.shape[0]
    for row_offset in range(-radius_cells, radius_cells + 1):
        for column_offset in range(-radius_cells, radius_cells + 1):
            if math.hypot(row_offset, column_offset) > radius_cells:
                continue
            source_row_start = max(0, -row_offset)
            source_row_stop = min(size, size - row_offset)
            source_column_start = max(0, -column_offset)
            source_column_stop = min(size, size - column_offset)
            target_row_start = source_row_start + row_offset
            target_row_stop = source_row_stop + row_offset
            target_column_start = source_column_start + column_offset
            target_column_stop = source_column_stop + column_offset
            inflated[
                target_row_start:target_row_stop,
                target_column_start:target_column_stop,
            ] |= source[
                source_row_start:source_row_stop,
                source_column_start:source_column_stop,
            ]
    return inflated, radius_cells, radius_cells * cell_size_m


def _nearest_free_start(
    semantic: np.ndarray,
    raster: MetricRaster,
    maximum_anchor_distance_m: float,
) -> tuple[tuple[int, int], float]:
    """Anchor the even-sized raster to the nearest predicted-free ego cell."""

    free_pixels = np.argwhere(semantic == FREE_VALUE)
    if not len(free_pixels):
        raise VerifierPlanningError(
            "prediction_has_no_free_space",
            "The predicted BEV contains no free cell.",
        )
    rows = free_pixels[:, 0].astype(np.float64)
    columns = free_pixels[:, 1].astype(np.float64)
    x_m = -raster.extent_m / 2.0 + (columns + 0.5) * raster.cell_size_m
    z_m = raster.extent_m / 2.0 - (rows + 0.5) * raster.cell_size_m
    distances = np.hypot(x_m, z_m)
    nearest_index = int(np.argmin(distances))
    distance_m = float(distances[nearest_index])
    if distance_m > maximum_anchor_distance_m:
        raise VerifierPlanningError(
            "predicted_start_not_free",
            "No predicted-free cell is close enough to the robot's exact metric origin.",
            nearest_free_distance_m=distance_m,
            maximum_anchor_distance_m=maximum_anchor_distance_m,
        )
    pixel = free_pixels[nearest_index]
    return (int(pixel[0]), int(pixel[1])), distance_m


def _metric_path(
    path: Sequence[tuple[int, int]], raster: MetricRaster
) -> list[list[float]]:
    return [list(raster.pixel_to_metric(row, column)) for row, column in path]


def _path_length_m(path: Sequence[Sequence[float]]) -> float:
    return float(
        sum(
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(path, path[1:])
        )
    )


def relative_camera_motion_metric(
    previous_camera_from_world_vggt: Sequence[Sequence[float]],
    current_camera_from_world_vggt: Sequence[Sequence[float]],
    lambda_m_per_vggt: float,
) -> np.ndarray:
    """Return current-camera-from-previous-camera with metric translation.

    VGGT emits world-to-camera 3x4 extrinsics in an arbitrary shared scale.
    The relative rotation is scale-free; only its translation is multiplied by
    the model's metre-per-VGGT-unit Scale Token.
    """

    if not math.isfinite(lambda_m_per_vggt) or lambda_m_per_vggt <= 0:
        raise VerifierPlanningError(
            "invalid_scale", "The model returned an invalid metric scale factor."
        )
    previous = np.asarray(previous_camera_from_world_vggt, dtype=np.float64)
    current = np.asarray(current_camera_from_world_vggt, dtype=np.float64)
    if previous.shape != (3, 4) or current.shape != (3, 4):
        raise ValueError("VGGT camera extrinsics must each be 3x4")
    if not np.isfinite(previous).all() or not np.isfinite(current).all():
        raise ValueError("VGGT camera extrinsics must be finite")
    previous_rotation = previous[:, :3]
    current_rotation = current[:, :3]
    relative_rotation = current_rotation @ previous_rotation.T
    relative_translation_vggt = (
        current[:, 3] - relative_rotation @ previous[:, 3]
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = relative_rotation
    transform[:3, 3] = relative_translation_vggt * lambda_m_per_vggt
    return transform


def transform_target_by_predicted_motion(
    target_metric_m: Sequence[float],
    current_camera_from_previous_camera_metric: Sequence[Sequence[float]],
) -> list[float]:
    """Express an old ego-frame planar target in the new predicted ego frame."""

    target = np.asarray(target_metric_m, dtype=np.float64)
    transform = np.asarray(
        current_camera_from_previous_camera_metric, dtype=np.float64
    )
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("target_metric_m must be finite [right, forward]")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("predicted relative motion must be a finite 4x4 matrix")
    # VGGT/OpenCV camera axes are x=right, y=down, z=forward.  The robot camera
    # is rigid and level, so a BEV target lies on its x/z plane.
    current = transform @ np.asarray(
        [target[0], 0.0, target[1], 1.0], dtype=np.float64
    )
    return [float(current[0]), float(current[2])]


def local_target_from_normalized_click(
    *,
    local_extent_m: float,
    click_u: float,
    click_v: float,
) -> dict[str, Any]:
    """Convert a display click directly to an ego-local metric coordinate.

    ``click_u`` and ``click_v`` are normalized canvas coordinates.  No image
    array is accepted, deliberately making GT semantic validation impossible.
    The coordinate convention is ``x=right`` and ``z=forward``.
    """

    if not (math.isfinite(click_u) and math.isfinite(click_v)):
        raise VerifierPlanningError(
            "invalid_click", "Click coordinates must be finite."
        )
    if not (0.0 <= click_u <= 1.0 and 0.0 <= click_v <= 1.0):
        raise VerifierPlanningError(
            "invalid_click", "Click coordinates lie outside the target canvas."
        )
    if not math.isfinite(local_extent_m) or local_extent_m <= 0:
        raise VerifierPlanningError(
            "invalid_extent", "The predicted BEV extent is invalid."
        )
    target_x_m = (float(click_u) - 0.5) * float(local_extent_m)
    target_z_m = (0.5 - float(click_v)) * float(local_extent_m)
    return {
        "target_metric_m": [target_x_m, target_z_m],
        "click_uv": [float(click_u), float(click_v)],
        "coordinate_frame": "current ego: x=right, z=forward",
        "local_extent_m": float(local_extent_m),
    }


_GRID_HEADING_BY_STEP: dict[tuple[int, int], int] = {
    (-1, 0): 0,
    (-1, 1): 1,
    (0, 1): 2,
    (1, 1): 3,
    (1, 0): 4,
    (1, -1): 5,
    (0, -1): 6,
    (-1, -1): 7,
}


def compile_grid_path_to_open_loop_actions(
    path_pixels: Sequence[Sequence[int]],
) -> list[str]:
    """Compile an A* pixel path into one immutable, feedback-free action list.

    Heading zero is the initial camera/BEV forward direction (image up).
    Turning is quantized to exact 45-degree simulator actions and every A*
    edge becomes exactly one cardinal or diagonal forward action.  No pose,
    collision map, navmesh, or GT input is accepted by this compiler.
    """

    if not path_pixels:
        return []
    pixels = [(int(point[0]), int(point[1])) for point in path_pixels]
    heading = 0
    actions: list[str] = []
    for first, second in zip(pixels, pixels[1:]):
        step = (second[0] - first[0], second[1] - first[1])
        if step not in _GRID_HEADING_BY_STEP:
            raise ValueError(f"non-adjacent A* grid step: {first} -> {second}")
        target_heading = _GRID_HEADING_BY_STEP[step]
        clockwise = (target_heading - heading) % 8
        if clockwise <= 4:
            actions.extend(["strict_turn_right_45"] * clockwise)
        else:
            actions.extend(["strict_turn_left_45"] * (8 - clockwise))
        actions.append(
            "strict_forward_diagonal"
            if step[0] and step[1]
            else "strict_forward_cardinal"
        )
        heading = target_heading
    return actions


def plan_metric_target(
    *,
    predicted_semantic: np.ndarray,
    predicted_extent_m: float,
    target_metric_m: Sequence[float],
    frame_seq: int,
    model_key: str,
    start_anchor_max_m: float = 0.05,
    inflation_radius_m: float = DEFAULT_INFLATION_RADIUS_M,
) -> dict[str, Any]:
    """Replan to an existing metric target on a newly predicted ego BEV."""

    predicted = _square_semantic(predicted_semantic, "predicted_semantic")
    target = np.asarray(target_metric_m, dtype=np.float64)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise VerifierPlanningError(
            "invalid_target", "The tracked metric target is invalid."
        )
    target_x_m, target_z_m = float(target[0]), float(target[1])
    raster = MetricRaster(predicted.shape[0], float(predicted_extent_m))
    predicted_goal = raster.metric_to_pixel(target_x_m, target_z_m)
    predicted_value = int(predicted[predicted_goal])
    if predicted_value == OCCUPIED_VALUE:
        raise VerifierPlanningError(
            "target_predicted_occupied",
            "The tracked target is occupied in the updated predicted BEV.",
            target_metric_m=[target_x_m, target_z_m],
            predicted_pixel=list(predicted_goal),
        )
    if predicted_value != FREE_VALUE:
        raise VerifierPlanningError(
            "target_predicted_unknown",
            "The tracked target is outside updated predicted free support.",
            target_metric_m=[target_x_m, target_z_m],
            predicted_pixel=list(predicted_goal),
            predicted_value=predicted_value,
        )
    inflated_occupied, inflation_radius_cells, effective_inflation_radius_m = (
        inflate_occupied_mask(
            predicted == OCCUPIED_VALUE,
            cell_size_m=raster.cell_size_m,
            radius_m=inflation_radius_m,
        )
    )
    blocked = (predicted != FREE_VALUE) | inflated_occupied
    if blocked[predicted_goal]:
        raise VerifierPlanningError(
            "target_inside_inflation_margin",
            "The predicted-free target is within the obstacle safety margin.",
            target_metric_m=[target_x_m, target_z_m],
            predicted_pixel=list(predicted_goal),
            inflation_radius_m=float(inflation_radius_m),
            effective_inflation_radius_m=effective_inflation_radius_m,
        )
    traversable_semantic = np.where(
        blocked, OCCUPIED_VALUE, FREE_VALUE
    ).astype(np.uint8)
    predicted_start, start_anchor_distance_m = _nearest_free_start(
        traversable_semantic, raster, start_anchor_max_m
    )
    path = astar_no_inflation(blocked, predicted_start, predicted_goal)
    if path is None:
        raise VerifierPlanningError(
            "no_predicted_astar_path",
            "A* found no connected free path in the updated predicted BEV.",
            predicted_start=list(predicted_start),
            predicted_goal=list(predicted_goal),
            inflation_radius_m=float(inflation_radius_m),
            effective_inflation_radius_m=effective_inflation_radius_m,
        )
    metric_path = _metric_path(path, raster)
    result = {
        "success": True,
        "failure_code": None,
        "failure_reason": None,
        "frame_seq": int(frame_seq),
        "model_key": str(model_key),
        "predicted_start_pixel": list(predicted_start),
        "predicted_target_pixel": list(predicted_goal),
        "path_pixels": [list(point) for point in path],
        "target_metric_m": [target_x_m, target_z_m],
        "path_metric_m": metric_path,
        "path_length_m": _path_length_m(metric_path),
        "path_cell_count": len(path),
        "execution_waypoint_count": 0,
        "start_anchor_distance_m": start_anchor_distance_m,
        "inflation_radius_m": float(inflation_radius_m),
        "inflation_radius_cells": inflation_radius_cells,
        "effective_inflation_radius_m": effective_inflation_radius_m,
        "inflation_semantics": (
            "occupied cells expand outward; unknown stays blocked but is not dilated"
        ),
        "metric_alignment": {
            "predicted_size": raster.size,
            "predicted_extent_m": raster.extent_m,
            "predicted_cell_size_m": raster.cell_size_m,
            "mapping": "ego-local target coordinate -> predicted BEV pixel",
        },
        "planner_runtime_inputs": [
            "predicted_semantic_bev",
            "ego_local_target_x_forward_m",
            "configured_bev_extent_m",
            "configured_inflation_radius_m",
        ],
        "planner_forbidden_inputs": [
            "gt_bev_pixels",
            "simulator_pose",
            "simulator_extrinsic",
            "navmesh",
            "depth_ground_truth",
        ],
    }
    return result
