"""GT-targeted realtime A* planning over a predicted Single BEV.

Target selection is deliberately isolated from model predictions.  A target is
sampled only from the GT-free, geometrically in-FOV, robot-reachable component.
The predicted occupancy and support are consulted only after that target has
been frozen.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass

import numpy as np


GridPoint = tuple[int, int]


class TargetSelectionError(RuntimeError):
    """The GT raster cannot provide a valid target for this frame."""


@dataclass(frozen=True)
class GTPlanningTrial:
    start: GridPoint
    goal: GridPoint
    goal_native_pixel: GridPoint
    goal_x_m: float
    goal_z_m: float
    gt_blocked: np.ndarray
    geometric_fov: np.ndarray
    gt_path: tuple[GridPoint, ...]
    cell_size_m: float

    @property
    def gt_path_length_m(self) -> float:
        return path_length_m(self.gt_path, self.cell_size_m)


@dataclass(frozen=True)
class PredictedPlan:
    success: bool
    failure_reason: str | None
    path: tuple[GridPoint, ...]
    raw_goal_occupancy_probability: float
    raw_goal_support_probability: float
    raw_goal_navigation_confidence: float
    predicted_path_length_m: float | None
    predicted_path_cost_m: float | None
    mean_path_safe_confidence: float | None
    path_length_ratio: float | None
    colliding_path_cells: int
    robot_origin_blocked_before_known_pose_clearance: bool
    predicted_blocked: np.ndarray


def geometric_fov_mask(
    size: int,
    extent_m: float,
    horizontal_fov_degrees: float,
) -> np.ndarray:
    """Return the unobstructed camera FOV in a centered ego BEV.

    Positive z is image-up and the latest robot is at the raster centre.  This
    is geometric FOV support, not sensor visibility, so space occluded behind
    an obstacle remains inside the mask.
    """

    if size <= 0 or extent_m <= 0:
        raise ValueError("size and extent must be positive")
    if not 0.0 < horizontal_fov_degrees < 180.0:
        raise ValueError("horizontal FOV must be inside (0, 180) degrees")
    cell = extent_m / size
    pixel = np.arange(size, dtype=np.float64)
    x = -extent_m / 2.0 + (pixel + 0.5) * cell
    z = extent_m / 2.0 - (pixel + 0.5) * cell
    z_grid, x_grid = np.meshgrid(z, x, indexing="ij")
    tangent = math.tan(math.radians(horizontal_fov_degrees / 2.0))
    return (z_grid >= 0.0) & (np.abs(x_grid) <= z_grid * tangent)


def ego_start_cell(geometric_fov: np.ndarray) -> GridPoint:
    """Choose the raster cell nearest the metric ego origin inside the FOV.

    An even square raster has no cell whose centre is exactly at (0, 0).
    Using ``(size // 2, size // 2)`` therefore selects a cell behind the
    camera and makes the known pose an isolated free island.  Pick the nearest
    cell centre that is actually in the forward geometric FOV instead.
    """

    size = _validate_square(geometric_fov, "geometric_fov")
    candidates = np.argwhere(np.asarray(geometric_fov, dtype=bool))
    if not len(candidates):
        raise ValueError("geometric FOV contains no raster cell")
    # Squared distance in cell coordinates to the metric origin, which lies
    # at (size / 2 - 0.5, size / 2 - 0.5) in centre-index coordinates.
    origin = size / 2.0 - 0.5
    distances = (
        (candidates[:, 0].astype(np.float64) - origin) ** 2
        + (candidates[:, 1].astype(np.float64) - origin) ** 2
    )
    nearest = candidates[int(np.argmin(distances))]
    return int(nearest[0]), int(nearest[1])


def _validate_square(array: np.ndarray, name: str) -> int:
    value = np.asarray(array)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a square 2-D raster")
    return int(value.shape[0])


def block_reduce_any(mask: np.ndarray, output_size: int) -> np.ndarray:
    """Conservatively reduce a binary raster: any blocked source blocks a cell."""

    input_size = _validate_square(mask, "mask")
    if output_size <= 0 or input_size % output_size:
        raise ValueError("output_size must evenly divide the input raster")
    block = input_size // output_size
    return np.asarray(mask, dtype=bool).reshape(
        output_size, block, output_size, block
    ).any(axis=(1, 3))


def block_reduce_all(mask: np.ndarray, output_size: int) -> np.ndarray:
    """Conservatively reduce support: every source pixel must be supported."""

    input_size = _validate_square(mask, "mask")
    if output_size <= 0 or input_size % output_size:
        raise ValueError("output_size must evenly divide the input raster")
    block = input_size // output_size
    return np.asarray(mask, dtype=bool).reshape(
        output_size, block, output_size, block
    ).all(axis=(1, 3))


def block_reduce_max(values: np.ndarray, output_size: int) -> np.ndarray:
    """Reduce a floating raster using the maximum value in each block."""

    input_size = _validate_square(values, "values")
    if output_size <= 0 or input_size % output_size:
        raise ValueError("output_size must evenly divide the input raster")
    block = input_size // output_size
    return np.asarray(values).reshape(
        output_size, block, output_size, block
    ).max(axis=(1, 3))


def inflate_obstacles(blocked: np.ndarray, radius_cells: int) -> np.ndarray:
    """Dilate obstacles with a Euclidean disk without scipy/opencv."""

    size = _validate_square(blocked, "blocked")
    if radius_cells < 0:
        raise ValueError("radius_cells must be non-negative")
    source = np.asarray(blocked, dtype=bool)
    if radius_cells == 0:
        return source.copy()
    output = np.zeros_like(source)
    for delta_row in range(-radius_cells, radius_cells + 1):
        for delta_column in range(-radius_cells, radius_cells + 1):
            if delta_row**2 + delta_column**2 > radius_cells**2:
                continue
            source_rows = slice(
                max(0, -delta_row), min(size, size - delta_row)
            )
            destination_rows = slice(
                max(0, delta_row), min(size, size + delta_row)
            )
            source_columns = slice(
                max(0, -delta_column), min(size, size - delta_column)
            )
            destination_columns = slice(
                max(0, delta_column), min(size, size + delta_column)
            )
            output[destination_rows, destination_columns] |= source[
                source_rows, source_columns
            ]
    return output


def ego_pose_clearance_mask(
    size: int,
    cell_size_m: float,
    radius_m: float,
) -> np.ndarray:
    """Cells intersecting the collision-free footprint at the known ego pose."""

    if size <= 0 or cell_size_m <= 0 or radius_m < 0:
        raise ValueError("size/cell size must be positive and radius non-negative")
    output = np.zeros((size, size), dtype=bool)
    if radius_m == 0:
        return output
    pixel = np.arange(size, dtype=np.float64)
    x = (pixel + 0.5 - size / 2.0) * cell_size_m
    z = (size / 2.0 - pixel - 0.5) * cell_size_m
    z_grid, x_grid = np.meshgrid(z, x, indexing="ij")
    # Include cells touched by the footprint, not only cells whose centres are
    # strictly inside it.  This removes raster pinholes at the circle edge.
    half_diagonal = cell_size_m / math.sqrt(2.0)
    return np.hypot(x_grid, z_grid) <= radius_m + half_diagonal


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


def _neighbors(blocked: np.ndarray, point: GridPoint):
    size = blocked.shape[0]
    row, column = point
    for delta_row, delta_column, cost in _NEIGHBORS:
        next_row = row + delta_row
        next_column = column + delta_column
        if not (0 <= next_row < size and 0 <= next_column < size):
            continue
        if blocked[next_row, next_column]:
            continue
        # A diagonal move may not squeeze through two touching obstacles.
        if delta_row and delta_column:
            if blocked[row, next_column] or blocked[next_row, column]:
                continue
        yield (next_row, next_column), cost


def reachable_component(blocked: np.ndarray, start: GridPoint) -> np.ndarray:
    """Return all cells reachable with the same motion rules used by A*."""

    size = _validate_square(blocked, "blocked")
    if not (0 <= start[0] < size and 0 <= start[1] < size):
        raise ValueError("start lies outside the grid")
    reached = np.zeros((size, size), dtype=bool)
    if blocked[start]:
        return reached
    reached[start] = True
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor, _ in _neighbors(blocked, current):
            if not reached[neighbor]:
                reached[neighbor] = True
                stack.append(neighbor)
    return reached


def _octile_distance(first: GridPoint, second: GridPoint) -> float:
    delta_row = abs(first[0] - second[0])
    delta_column = abs(first[1] - second[1])
    diagonal = min(delta_row, delta_column)
    straight = max(delta_row, delta_column) - diagonal
    return diagonal * math.sqrt(2.0) + straight


def astar_path(
    blocked: np.ndarray,
    start: GridPoint,
    goal: GridPoint,
    traversal_multiplier: np.ndarray | None = None,
) -> tuple[GridPoint, ...] | None:
    """Plan an optimal 8-connected path with optional per-cell soft costs.

    The edge multiplier is the mean of its two endpoint multipliers.  Requiring
    every multiplier to be at least one keeps the ordinary octile heuristic
    admissible.
    """

    size = _validate_square(blocked, "blocked")
    if traversal_multiplier is None:
        multiplier = np.ones((size, size), dtype=np.float64)
    else:
        multiplier = np.asarray(traversal_multiplier, dtype=np.float64)
        if multiplier.shape != (size, size):
            raise ValueError("traversal_multiplier must match blocked")
        if not np.isfinite(multiplier).all() or np.any(multiplier < 1.0):
            raise ValueError("traversal multipliers must be finite and >= 1")
    for name, point in (("start", start), ("goal", goal)):
        if not (0 <= point[0] < size and 0 <= point[1] < size):
            raise ValueError(f"{name} lies outside the grid")
        if blocked[point]:
            return None
    if start == goal:
        return (start,)

    frontier: list[tuple[float, float, int, GridPoint]] = []
    serial = 0
    heapq.heappush(frontier, (_octile_distance(start, goal), 0.0, serial, start))
    best = {start: 0.0}
    parent: dict[GridPoint, GridPoint] = {}
    while frontier:
        _, cost, _, current = heapq.heappop(frontier)
        if cost > best.get(current, math.inf):
            continue
        if current == goal:
            output = [goal]
            while output[-1] != start:
                output.append(parent[output[-1]])
            output.reverse()
            return tuple(output)
        for neighbor, move_cost in _neighbors(blocked, current):
            edge_multiplier = 0.5 * (
                multiplier[current] + multiplier[neighbor]
            )
            candidate = cost + move_cost * edge_multiplier
            if candidate >= best.get(neighbor, math.inf):
                continue
            best[neighbor] = candidate
            parent[neighbor] = current
            serial += 1
            heapq.heappush(
                frontier,
                (
                    candidate + _octile_distance(neighbor, goal),
                    candidate,
                    serial,
                    neighbor,
                ),
            )
    return None


def path_length_m(path: tuple[GridPoint, ...], cell_size_m: float) -> float:
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1]) * cell_size_m
        for first, second in zip(path, path[1:], strict=False)
    )


def path_cost_m(
    path: tuple[GridPoint, ...],
    cell_size_m: float,
    traversal_multiplier: np.ndarray,
) -> float:
    """Return the same confidence-weighted metric cost optimized by A*."""

    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    multiplier = np.asarray(traversal_multiplier, dtype=np.float64)
    _validate_square(multiplier, "traversal_multiplier")
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        * cell_size_m
        * 0.5
        * (float(multiplier[first]) + float(multiplier[second]))
        for first, second in zip(path, path[1:], strict=False)
    )


def safe_confidence_cost_multiplier(
    occupancy_probability: np.ndarray,
    navigation_confidence: np.ndarray,
    *,
    weight: float = 1.0,
    epsilon: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert calibrated BEV confidence into a positive A* cost multiplier.

    ``q_safe = (1 - p_occ) * c_navigation`` and
    ``multiplier = 1 + weight * -log(clamp(q_safe, epsilon, 1))``.
    """

    occupancy = np.asarray(occupancy_probability, dtype=np.float64)
    confidence = np.asarray(navigation_confidence, dtype=np.float64)
    size = _validate_square(occupancy, "occupancy_probability")
    if confidence.shape != (size, size):
        raise ValueError("occupancy and navigation-confidence rasters must match")
    if weight < 0 or not 0.0 < epsilon < 1.0:
        raise ValueError("weight must be non-negative and epsilon inside (0, 1)")
    safe_confidence = (
        (1.0 - np.clip(occupancy, 0.0, 1.0))
        * np.clip(confidence, 0.0, 1.0)
    )
    multiplier = 1.0 + weight * -np.log(
        np.clip(safe_confidence, epsilon, 1.0)
    )
    return safe_confidence, multiplier


def _native_pixel(grid: GridPoint, native_size: int, planning_size: int) -> GridPoint:
    if native_size % planning_size:
        raise ValueError("planning_size must evenly divide native_size")
    scale = native_size // planning_size
    return (
        min(native_size - 1, grid[0] * scale + scale // 2),
        min(native_size - 1, grid[1] * scale + scale // 2),
    )


def _metric_point(grid: GridPoint, size: int, extent_m: float) -> tuple[float, float]:
    cell = extent_m / size
    row, column = grid
    x = -extent_m / 2.0 + (column + 0.5) * cell
    z = extent_m / 2.0 - (row + 0.5) * cell
    return x, z


def select_gt_target(
    complete_gt: np.ndarray,
    *,
    horizontal_fov_degrees: float,
    rng: random.Random,
    planning_size: int = 128,
    extent_m: float = 6.5,
    robot_radius_m: float = 0.1,
    minimum_target_distance_m: float = 0.75,
    free_value: int = 255,
) -> GTPlanningTrial:
    """Freeze one target using GT only; no prediction is accepted as input."""

    native_size = _validate_square(complete_gt, "complete_gt")
    if native_size % planning_size:
        raise ValueError("planning_size must evenly divide complete_gt")
    if robot_radius_m < 0 or minimum_target_distance_m < 0:
        raise ValueError("distances must be non-negative")

    # Anything not explicitly GT-free is conservatively occupied.
    gt_obstacle = block_reduce_any(
        np.asarray(complete_gt) != free_value,
        planning_size,
    )
    cell_size_m = extent_m / planning_size
    radius_cells = math.ceil(robot_radius_m / cell_size_m)
    gt_blocked = inflate_obstacles(gt_obstacle, radius_cells)
    fov = geometric_fov_mask(
        planning_size,
        extent_m,
        horizontal_fov_degrees,
    )
    start = ego_start_cell(fov)
    if gt_blocked[start]:
        raise TargetSelectionError("GT robot origin is blocked after footprint inflation")

    reachable = reachable_component(gt_blocked, start)
    pixel = np.arange(planning_size, dtype=np.float64)
    x = -extent_m / 2.0 + (pixel + 0.5) * cell_size_m
    z = extent_m / 2.0 - (pixel + 0.5) * cell_size_m
    z_grid, x_grid = np.meshgrid(z, x, indexing="ij")
    far_enough = np.hypot(x_grid, z_grid) >= minimum_target_distance_m
    candidates = np.argwhere(reachable & fov & far_enough)
    if not len(candidates):
        raise TargetSelectionError("no GT-free reachable target exists in the FOV")
    chosen = candidates[rng.randrange(len(candidates))]
    goal = (int(chosen[0]), int(chosen[1]))
    gt_path = astar_path(gt_blocked, start, goal)
    if gt_path is None:
        raise AssertionError("reachable-component target has no GT A* path")
    goal_x_m, goal_z_m = _metric_point(goal, planning_size, extent_m)
    return GTPlanningTrial(
        start=start,
        goal=goal,
        goal_native_pixel=_native_pixel(goal, native_size, planning_size),
        goal_x_m=goal_x_m,
        goal_z_m=goal_z_m,
        gt_blocked=gt_blocked,
        geometric_fov=fov,
        gt_path=gt_path,
        cell_size_m=cell_size_m,
    )


def plan_on_prediction(
    occupancy_probability: np.ndarray,
    support_probability: np.ndarray,
    navigation_confidence: np.ndarray,
    trial: GTPlanningTrial,
    *,
    occupancy_threshold: float = 0.5,
    support_threshold: float = 0.5,
    planning_inflation_radius_m: float = 0.1,
    known_ego_pose_clearance_radius_m: float = 0.05,
    confidence_cost_weight: float = 1.0,
    confidence_cost_epsilon: float = 1e-4,
) -> PredictedPlan:
    """Run A* after a GT-selected target has been frozen."""

    native_size = _validate_square(
        occupancy_probability, "occupancy_probability"
    )
    if np.asarray(support_probability).shape != (native_size, native_size):
        raise ValueError("support and occupancy rasters must match")
    if np.asarray(navigation_confidence).shape != (native_size, native_size):
        raise ValueError("navigation confidence and occupancy rasters must match")
    if not 0.0 <= occupancy_threshold <= 1.0:
        raise ValueError("occupancy_threshold must be inside [0, 1]")
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support_threshold must be inside [0, 1]")
    if planning_inflation_radius_m < 0 or known_ego_pose_clearance_radius_m < 0:
        raise ValueError("planning and ego clearance radii must be non-negative")
    planning_size = trial.gt_blocked.shape[0]
    goal_native = trial.goal_native_pixel
    goal_occupancy = float(occupancy_probability[goal_native])
    goal_support = float(support_probability[goal_native])
    goal_navigation_confidence = float(navigation_confidence[goal_native])

    native_safe_confidence, _ = (
        safe_confidence_cost_multiplier(
            occupancy_probability,
            navigation_confidence,
            weight=confidence_cost_weight,
            epsilon=confidence_cost_epsilon,
        )
    )
    planning_safe_confidence = block_reduce_max(
        native_safe_confidence,
        planning_size,
    )
    # Recompute from reduced safe confidence instead of reducing multipliers:
    # the native-to-planning support rule is ANY, so the best-supported native
    # point represents that coarser cell.
    traversal_multiplier = 1.0 + confidence_cost_weight * -np.log(
        np.clip(planning_safe_confidence, confidence_cost_epsilon, 1.0)
    )

    predicted_obstacle = block_reduce_any(
        np.asarray(occupancy_probability) >= occupancy_threshold,
        planning_size,
    )
    # Support is a coverage mask, not an obstacle surface.  Requiring every
    # pixel in a 4x4 planning block to be supported erases the first row at the
    # camera-cone apex and disconnects the robot from otherwise valid space.
    # Any supported source pixel keeps the planning cell known; occupied
    # reduction below remains conservative (ANY occupied blocks the cell).
    predicted_support = block_reduce_any(
        np.asarray(support_probability) >= support_threshold,
        planning_size,
    )
    radius_cells = math.ceil(
        planning_inflation_radius_m / trial.cell_size_m
    )
    # Unknown support is forbidden for planning but is not a physical object:
    # dilating it would make the unknown region behind the camera expand into
    # the FOV apex and systematically cover the robot.  Inflate only occupied
    # predictions, then apply the support boundary.
    predicted_blocked = (
        inflate_obstacles(predicted_obstacle, radius_cells)
        | ~predicted_support
    )
    origin_blocked_before_clearance = bool(predicted_blocked[trial.start])
    # The simulator pose proves that the robot's currently occupied footprint
    # is collision-free.  Clear that local footprint from prediction-derived
    # occupied/unknown values so an even raster or a false positive at the FOV
    # apex cannot imprison the robot.  Nothing outside the current footprint is
    # cleared, and the GT-selected goal is never cleared.
    ego_clearance = ego_pose_clearance_mask(
        planning_size,
        trial.cell_size_m,
        known_ego_pose_clearance_radius_m,
    )
    predicted_blocked[ego_clearance] = False
    # The current pose is known, so confidence there is one by definition.
    traversal_multiplier[ego_clearance] = 1.0
    predicted_blocked[trial.start] = False
    traversal_multiplier[trial.start] = 1.0

    def failure(reason: str) -> PredictedPlan:
        return PredictedPlan(
            success=False,
            failure_reason=reason,
            path=(),
            raw_goal_occupancy_probability=goal_occupancy,
            raw_goal_support_probability=goal_support,
            raw_goal_navigation_confidence=goal_navigation_confidence,
            predicted_path_length_m=None,
            predicted_path_cost_m=None,
            mean_path_safe_confidence=None,
            path_length_ratio=None,
            colliding_path_cells=0,
            robot_origin_blocked_before_known_pose_clearance=(
                origin_blocked_before_clearance
            ),
            predicted_blocked=predicted_blocked,
        )

    if goal_occupancy >= occupancy_threshold:
        return failure("target_predicted_occupied")
    if goal_support < support_threshold:
        return failure("target_predicted_unknown")
    if predicted_blocked[trial.goal]:
        return failure("target_blocked_after_safe_inflation")

    path = astar_path(
        predicted_blocked,
        trial.start,
        trial.goal,
        traversal_multiplier,
    )
    if path is None:
        return failure("no_predicted_astar_path")
    collisions = sum(bool(trial.gt_blocked[cell]) for cell in path)
    predicted_length = path_length_m(path, trial.cell_size_m)
    predicted_cost = path_cost_m(
        path,
        trial.cell_size_m,
        traversal_multiplier,
    )
    mean_safe_confidence = float(
        np.mean([planning_safe_confidence[cell] for cell in path])
    )
    if collisions:
        return PredictedPlan(
            success=False,
            failure_reason="predicted_path_collides_with_gt",
            path=path,
            raw_goal_occupancy_probability=goal_occupancy,
            raw_goal_support_probability=goal_support,
            raw_goal_navigation_confidence=goal_navigation_confidence,
            predicted_path_length_m=predicted_length,
            predicted_path_cost_m=predicted_cost,
            mean_path_safe_confidence=mean_safe_confidence,
            path_length_ratio=None,
            colliding_path_cells=collisions,
            robot_origin_blocked_before_known_pose_clearance=(
                origin_blocked_before_clearance
            ),
            predicted_blocked=predicted_blocked,
        )
    return PredictedPlan(
        success=True,
        failure_reason=None,
        path=path,
        raw_goal_occupancy_probability=goal_occupancy,
        raw_goal_support_probability=goal_support,
        raw_goal_navigation_confidence=goal_navigation_confidence,
        predicted_path_length_m=predicted_length,
        predicted_path_cost_m=predicted_cost,
        mean_path_safe_confidence=mean_safe_confidence,
        path_length_ratio=(
            predicted_length / trial.gt_path_length_m
            if trial.gt_path_length_m > 0
            else 1.0
        ),
        colliding_path_cells=0,
        robot_origin_blocked_before_known_pose_clearance=(
            origin_blocked_before_clearance
        ),
        predicted_blocked=predicted_blocked,
    )
