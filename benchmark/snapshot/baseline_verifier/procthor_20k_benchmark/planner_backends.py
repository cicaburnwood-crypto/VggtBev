#!/usr/bin/env python3
"""Interchangeable planar planning backends for Mode 3.

All backends consume the same predicted VGGTBEV raster and metric point goal.
They never receive the simulator map, navmesh, GT obstacles, or shortest path.
The module intentionally keeps the interface small so the comparison server
can run one shared perception model and swap only the planning component.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage

try:
    from numba import njit
except ImportError:  # pragma: no cover - portable CPU fallback
    njit = None


FREE_VALUE = 255
OCCUPIED_VALUE = 0
UNKNOWN_VALUE = 112
SQRT2 = math.sqrt(2.0)
NEIGHBORS_8 = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, SQRT2),
    (-1, 1, SQRT2),
    (1, -1, SQRT2),
    (1, 1, SQRT2),
)


class PlannerFailure(RuntimeError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class GridProblem:
    blocked: np.ndarray
    weight: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    cell_size_m: float
    extent_m: float
    inflation_radius_m: float
    effective_inflation_radius_m: float

    @property
    def size(self) -> int:
        return int(self.blocked.shape[0])

    def metric_to_pixel(self, right_m: float, forward_m: float) -> tuple[int, int]:
        column = int(math.floor((right_m + self.extent_m / 2.0) / self.cell_size_m))
        row = int(math.floor((self.extent_m / 2.0 - forward_m) / self.cell_size_m))
        return (
            int(np.clip(row, 0, self.size - 1)),
            int(np.clip(column, 0, self.size - 1)),
        )

    def pixel_to_metric(self, point: Sequence[float]) -> list[float]:
        row, column = float(point[0]), float(point[1])
        return [
            -self.extent_m / 2.0 + (column + 0.5) * self.cell_size_m,
            self.extent_m / 2.0 - (row + 0.5) * self.cell_size_m,
        ]


@dataclass
class PlanResult:
    path_pixels: list[list[float]]
    path_metric_m: list[list[float]]
    cost: float
    planning_seconds: float
    expanded: int
    backend_details: dict[str, Any] = field(default_factory=dict)
    control_velocity: tuple[float, float, float] | None = None
    control_sequence: list[list[float]] | None = None
    control_timestep_s: float | None = None
    # Optional native SE(2) headings, expressed in the request ego frame.
    # Graph planners leave this unset and the common differential-drive
    # trajectory generator derives a tangent heading from every native edge.
    path_headings_rad: list[float] | None = None


@dataclass(frozen=True)
class RobotMotionLimits:
    """Deployable differential-drive limits shared by kinematic backends."""

    minimum_linear_velocity_m_s: float = -0.6
    maximum_linear_velocity_m_s: float = 0.6
    minimum_linear_acceleration_m_s2: float = -0.6
    maximum_linear_acceleration_m_s2: float = 0.6
    minimum_angular_velocity_rad_s: float = -1.0
    maximum_angular_velocity_rad_s: float = 1.0
    minimum_angular_acceleration_rad_s2: float = -1.0
    maximum_angular_acceleration_rad_s2: float = 1.0
    track_width_m: float = 0.10

    @property
    def maximum_speed_m_s(self) -> float:
        return max(
            abs(self.minimum_linear_velocity_m_s),
            abs(self.maximum_linear_velocity_m_s),
        )

    @property
    def maximum_angular_speed_rad_s(self) -> float:
        return max(
            abs(self.minimum_angular_velocity_rad_s),
            abs(self.maximum_angular_velocity_rad_s),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "minimum_linear_velocity_m_s": self.minimum_linear_velocity_m_s,
            "maximum_linear_velocity_m_s": self.maximum_linear_velocity_m_s,
            "minimum_linear_acceleration_m_s2": self.minimum_linear_acceleration_m_s2,
            "maximum_linear_acceleration_m_s2": self.maximum_linear_acceleration_m_s2,
            "minimum_angular_velocity_rad_s": self.minimum_angular_velocity_rad_s,
            "maximum_angular_velocity_rad_s": self.maximum_angular_velocity_rad_s,
            "minimum_angular_acceleration_rad_s2": self.minimum_angular_acceleration_rad_s2,
            "maximum_angular_acceleration_rad_s2": self.maximum_angular_acceleration_rad_s2,
            "track_width_m": self.track_width_m,
        }


def _disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def build_problem(
    *,
    semantic: np.ndarray,
    occupancy_probability: np.ndarray,
    navigation_confidence: np.ndarray,
    extent_m: float,
    target_metric_m: Sequence[float],
    inflation_radius_m: float,
    start_anchor_max_m: float = 0.20,
) -> GridProblem:
    semantic = np.asarray(semantic, dtype=np.uint8)
    occupancy = np.asarray(occupancy_probability, dtype=np.float32)
    confidence = np.asarray(navigation_confidence, dtype=np.float32)
    if semantic.ndim != 2 or semantic.shape[0] != semantic.shape[1]:
        raise PlannerFailure("invalid_bev", "semantic BEV must be a square raster")
    if occupancy.shape != semantic.shape or confidence.shape != semantic.shape:
        raise PlannerFailure("invalid_risk", "probability rasters do not match BEV")
    if not math.isfinite(extent_m) or extent_m <= 0.0:
        raise PlannerFailure("invalid_extent", "metric BEV extent is invalid")
    size = semantic.shape[0]
    cell_size_m = float(extent_m) / size
    radius_cells = int(math.ceil(max(0.0, inflation_radius_m) / cell_size_m))
    occupied = semantic == OCCUPIED_VALUE
    inflated = ndimage.binary_dilation(occupied, structure=_disk(radius_cells))
    # Unknown remains a hard barrier. Only occupied is dilated.
    blocked = (semantic != FREE_VALUE) | inflated
    clearance_m = ndimage.distance_transform_edt(~blocked) * cell_size_m
    occupancy = np.clip(occupancy, 0.0, 1.0)
    confidence = np.clip(confidence, 0.0, 1.0)
    # Shared risk field for every backend. It preserves geometric path length
    # while penalising occupied probability, low model confidence, and low
    # obstacle clearance on predicted-free cells.
    weight = (
        1.0
        + 2.0 * occupancy
        + 1.5 * (1.0 - confidence)
        + 1.5 * np.exp(-clearance_m / 0.15)
    ).astype(np.float64)
    weight[blocked] = math.inf

    target = np.asarray(target_metric_m, dtype=np.float64)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise PlannerFailure("invalid_target", "target must be finite [right, forward]")

    def metric_to_pixel(right_m: float, forward_m: float) -> tuple[int, int]:
        column = int(math.floor((right_m + extent_m / 2.0) / cell_size_m))
        row = int(math.floor((extent_m / 2.0 - forward_m) / cell_size_m))
        if not (0 <= row < size and 0 <= column < size):
            raise PlannerFailure("target_outside_bev", "local target lies outside BEV")
        return row, column

    goal = metric_to_pixel(float(target[0]), float(target[1]))
    if blocked[goal]:
        raise PlannerFailure("target_blocked", "target is not predicted traversable")

    free = np.argwhere(~blocked)
    if not len(free):
        raise PlannerFailure("no_free_space", "predicted BEV has no traversable cell")
    rows = free[:, 0].astype(np.float64)
    columns = free[:, 1].astype(np.float64)
    right = -extent_m / 2.0 + (columns + 0.5) * cell_size_m
    forward = extent_m / 2.0 - (rows + 0.5) * cell_size_m
    distances = np.hypot(right, forward)
    anchor_index = int(np.argmin(distances))
    if float(distances[anchor_index]) > start_anchor_max_m:
        raise PlannerFailure(
            "start_not_free",
            "no predicted-free cell is close enough to the robot origin",
            distance_m=float(distances[anchor_index]),
        )
    start = tuple(map(int, free[anchor_index]))
    return GridProblem(
        blocked=blocked,
        weight=weight,
        start=start,
        goal=goal,
        cell_size_m=cell_size_m,
        extent_m=float(extent_m),
        inflation_radius_m=float(inflation_radius_m),
        effective_inflation_radius_m=radius_cells * cell_size_m,
    )


def _octile(a: Sequence[int], b: Sequence[int]) -> float:
    dr = abs(int(a[0]) - int(b[0]))
    dc = abs(int(a[1]) - int(b[1]))
    return min(dr, dc) * SQRT2 + abs(dr - dc)


def _valid_step(problem: GridProblem, current: tuple[int, int], neighbor: tuple[int, int]) -> bool:
    row, column = neighbor
    if not (0 <= row < problem.size and 0 <= column < problem.size):
        return False
    if problem.blocked[neighbor]:
        return False
    dr, dc = row - current[0], column - current[1]
    if dr and dc and (
        problem.blocked[current[0], column]
        or problem.blocked[row, current[1]]
    ):
        return False
    return True


def _edge_cost(problem: GridProblem, a: tuple[int, int], b: tuple[int, int]) -> float:
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    return length * 0.5 * (problem.weight[a] + problem.weight[b])


def _reconstruct(
    parent: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [goal]
    while path[-1] != start:
        if path[-1] not in parent:
            raise PlannerFailure("broken_parent", "planner parent chain is incomplete")
        path.append(parent[path[-1]])
    path.reverse()
    return path


def _result(
    problem: GridProblem,
    path: Sequence[Sequence[float]],
    cost: float,
    started: float,
    expanded: int,
    path_headings_rad: Sequence[float] | None = None,
    **details: Any,
) -> PlanResult:
    pixels = [[float(point[0]), float(point[1])] for point in path]
    metric = [problem.pixel_to_metric(point) for point in pixels]
    return PlanResult(
        path_pixels=pixels,
        path_metric_m=metric,
        cost=float(cost) * problem.cell_size_m,
        planning_seconds=time.monotonic() - started,
        expanded=int(expanded),
        backend_details=details,
        path_headings_rad=(
            None
            if path_headings_rad is None
            else [float(value) for value in path_headings_rad]
        ),
    )


def weighted_astar(
    problem: GridProblem,
    *,
    heuristic_weight: float = 1.0,
    deadline: float | None = None,
) -> tuple[list[tuple[int, int]], float, int]:
    minimum_weight = float(np.min(problem.weight[~problem.blocked]))
    frontier: list[tuple[float, float, int, tuple[int, int]]] = []
    serial = 0
    best = {problem.start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    heapq.heappush(
        frontier,
        (
            heuristic_weight * minimum_weight * _octile(problem.start, problem.goal),
            0.0,
            serial,
            problem.start,
        ),
    )
    expanded = 0
    while frontier:
        if deadline is not None and time.monotonic() >= deadline:
            raise PlannerFailure("planning_budget", "weighted A* exhausted its time budget")
        _, cost, _, current = heapq.heappop(frontier)
        if cost > best.get(current, math.inf):
            continue
        expanded += 1
        if current == problem.goal:
            return _reconstruct(parent, problem.start, problem.goal), cost, expanded
        for dr, dc, _distance in NEIGHBORS_8:
            neighbor = current[0] + dr, current[1] + dc
            if not _valid_step(problem, current, neighbor):
                continue
            candidate = cost + _edge_cost(problem, current, neighbor)
            if candidate >= best.get(neighbor, math.inf):
                continue
            best[neighbor] = candidate
            parent[neighbor] = current
            serial += 1
            heapq.heappush(
                frontier,
                (
                    candidate
                    + heuristic_weight
                    * minimum_weight
                    * _octile(neighbor, problem.goal),
                    candidate,
                    serial,
                    neighbor,
                ),
            )
    raise PlannerFailure("no_path", "weighted A* found no connected path")


def _holonomic_cost_to_go(
    problem: GridProblem, *, deadline: float
) -> tuple[np.ndarray, int]:
    """Exact 2-D obstacle-aware cost-to-go used by Hybrid A*.

    Dolgov et al. explicitly use a dynamic-programming search that ignores
    non-holonomic constraints but respects obstacles.  The resulting table is
    only a heuristic; it is never returned as a path.
    """

    distance = np.full(problem.blocked.shape, math.inf, dtype=np.float64)
    distance[problem.goal] = 0.0
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, problem.goal)]
    expanded = 0
    while queue:
        if time.monotonic() >= deadline:
            raise PlannerFailure(
                "planning_budget",
                "Hybrid A* obstacle-aware heuristic exhausted its time budget",
            )
        cost, node = heapq.heappop(queue)
        if cost > float(distance[node]):
            continue
        expanded += 1
        for dr, dc, _ in NEIGHBORS_8:
            predecessor = node[0] + dr, node[1] + dc
            if not _valid_step(problem, node, predecessor):
                continue
            candidate = cost + _edge_cost(problem, node, predecessor)
            if candidate >= float(distance[predecessor]):
                continue
            distance[predecessor] = candidate
            heapq.heappush(queue, (candidate, predecessor))
    return distance, expanded


def _supercover_line(a: Sequence[float], b: Sequence[float]) -> list[tuple[int, int]]:
    """Conservative raster cells intersected by a segment."""

    y0, x0 = float(a[0]), float(a[1])
    y1, x1 = float(b[0]), float(b[1])
    steps = max(1, int(math.ceil(max(abs(y1 - y0), abs(x1 - x0)) * 2.0)))
    cells: list[tuple[int, int]] = []
    for t in np.linspace(0.0, 1.0, steps + 1):
        cell = int(round(y0 + (y1 - y0) * t)), int(round(x0 + (x1 - x0) * t))
        if not cells or cell != cells[-1]:
            cells.append(cell)
    return cells


def _line_cost(problem: GridProblem, a: tuple[int, int], b: tuple[int, int]) -> float:
    if njit is not None:
        return float(
            _line_cost_numba(
                problem.blocked,
                problem.weight,
                int(a[0]),
                int(a[1]),
                int(b[0]),
                int(b[1]),
            )
        )
    # Vectorised portable fallback.  The previous Python/list implementation
    # made full-resolution 512x512 Lazy Theta* several orders of magnitude
    # slower when numba was not installed on a deployment host.
    dy, dx = int(b[0] - a[0]), int(b[1] - a[1])
    steps = max(1, int(math.ceil(max(abs(dy), abs(dx)) * 2.0)))
    fractions = np.arange(steps + 1, dtype=np.float64) / steps
    rows = np.rint(float(a[0]) + dy * fractions).astype(np.intp)
    columns = np.rint(float(a[1]) + dx * fractions).astype(np.intp)
    keep = np.ones(steps + 1, dtype=bool)
    keep[1:] = (rows[1:] != rows[:-1]) | (columns[1:] != columns[:-1])
    rows, columns = rows[keep], columns[keep]
    if (
        np.any(rows < 0)
        or np.any(rows >= problem.size)
        or np.any(columns < 0)
        or np.any(columns >= problem.size)
        or np.any(problem.blocked[rows, columns])
    ):
        return math.inf
    if len(rows) > 1:
        row_delta = np.diff(rows)
        column_delta = np.diff(columns)
        diagonals = (row_delta != 0) & (column_delta != 0)
        indices = np.flatnonzero(diagonals)
        if len(indices) and (
            np.any(problem.blocked[rows[indices], columns[indices + 1]])
            or np.any(problem.blocked[rows[indices + 1], columns[indices]])
        ):
            return math.inf
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    return length * float(np.mean(problem.weight[rows, columns]))


def _line_cost_numba_impl(
    blocked: np.ndarray,
    weight: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
) -> float:
    """Allocation-free conservative segment cost used by Lazy Theta*."""

    dy = y1 - y0
    dx = x1 - x0
    steps = max(1, int(math.ceil(max(abs(dy), abs(dx)) * 2.0)))
    previous_y = -1
    previous_x = -1
    total_weight = 0.0
    count = 0
    rows, columns = blocked.shape
    for index in range(steps + 1):
        fraction = index / steps
        row = int(round(y0 + dy * fraction))
        column = int(round(x0 + dx * fraction))
        if row == previous_y and column == previous_x:
            continue
        if row < 0 or row >= rows or column < 0 or column >= columns:
            return math.inf
        if blocked[row, column]:
            return math.inf
        if previous_y >= 0:
            delta_row = row - previous_y
            delta_column = column - previous_x
            if delta_row != 0 and delta_column != 0:
                if blocked[previous_y, column] or blocked[row, previous_x]:
                    return math.inf
        total_weight += weight[row, column]
        count += 1
        previous_y = row
        previous_x = column
    if count == 0:
        return math.inf
    return math.hypot(dy, dx) * total_weight / count


_line_cost_numba = (
    njit(cache=True)(_line_cost_numba_impl) if njit is not None else _line_cost_numba_impl
)
_NUMBA_WARMED = False


class PlannerBackend:
    key = "base"
    label = "Base"
    family = "graph"

    def reset(self) -> None:
        pass

    def plan(self, problem: GridProblem, *, budget_s: float = 0.35) -> PlanResult:
        raise NotImplementedError


class AStarBackend(PlannerBackend):
    key = "astar"
    label = "Risk-aware A*"

    def plan(self, problem: GridProblem, *, budget_s: float = 0.35) -> PlanResult:
        started = time.monotonic()
        path, cost, expanded = weighted_astar(problem, deadline=started + budget_s)
        return _result(
            problem,
            path,
            cost,
            started,
            expanded,
            algorithm="A* (Hart, Nilsson & Raphael, 1968)",
            heuristic="admissible octile distance times minimum traversal weight",
            heuristic_weight=1.0,
            optimal_on_grid=True,
        )


class LazyThetaBackend(PlannerBackend):
    key = "lazy_theta"
    label = "Risk-aware Lazy Theta*"

    def plan(self, problem: GridProblem, *, budget_s: float = 0.35) -> PlanResult:
        """Nash, Koenig & Tovey's Lazy Theta* (AAAI 2010).

        In particular, line of sight is checked once in ``SetVertex`` when a
        vertex is expanded.  This is not eager Theta* and does not use a
        weighted heuristic.
        """

        started = time.monotonic()
        deadline = started + budget_s
        minimum_weight = float(np.min(problem.weight[~problem.blocked]))
        frontier: list[tuple[float, float, int, tuple[int, int]]] = []
        serial = 0
        best = {problem.start: 0.0}
        parent = {problem.start: problem.start}
        closed: set[tuple[int, int]] = set()
        heapq.heappush(
            frontier,
            (
                minimum_weight * _octile(problem.start, problem.goal),
                0.0,
                0,
                problem.start,
            ),
        )
        expanded = 0
        while frontier:
            if time.monotonic() >= deadline:
                raise PlannerFailure("planning_budget", "Lazy Theta* exhausted its time budget")
            _, cost, _, current = heapq.heappop(frontier)
            if cost > best.get(current, math.inf):
                continue

            # Lazy Theta*: validate the optimistic parent only now. If it is
            # not visible, reconnect through the best already-expanded grid
            # neighbor (SetVertex in the paper).
            if current != problem.start:
                ancestor = parent[current]
                if not math.isfinite(_line_cost(problem, ancestor, current)):
                    replacement: tuple[float, tuple[int, int]] | None = None
                    for dr, dc, _ in NEIGHBORS_8:
                        candidate_parent = current[0] + dr, current[1] + dc
                        if candidate_parent not in closed:
                            continue
                        if not _valid_step(problem, candidate_parent, current):
                            continue
                        candidate_cost = best[candidate_parent] + _edge_cost(
                            problem, candidate_parent, current
                        )
                        if replacement is None or candidate_cost < replacement[0]:
                            replacement = candidate_cost, candidate_parent
                    if replacement is None:
                        continue
                    best[current], parent[current] = replacement
                    cost = replacement[0]

            expanded += 1
            if current == problem.goal:
                path = _reconstruct(parent, problem.start, problem.goal)
                return _result(
                    problem,
                    path,
                    cost,
                    started,
                    expanded,
                    any_angle=True,
                    algorithm="Lazy Theta* (Nash, Koenig & Tovey, 2010)",
                    lazy_line_of_sight=True,
                    heuristic_weight=1.0,
                )
            closed.add(current)
            for dr, dc, _ in NEIGHBORS_8:
                neighbor = current[0] + dr, current[1] + dc
                if neighbor in closed or not _valid_step(problem, current, neighbor):
                    continue
                # ComputeCost from Lazy Theta*: assume the parent's visibility
                # and defer the one LOS check until the neighbor is expanded.
                ancestor = parent[current]
                direct = math.hypot(
                    neighbor[0] - ancestor[0], neighbor[1] - ancestor[1]
                ) * 0.5 * (problem.weight[ancestor] + problem.weight[neighbor])
                candidate_parent = ancestor
                candidate = best[ancestor] + direct
                if candidate >= best.get(neighbor, math.inf):
                    continue
                best[neighbor] = candidate
                parent[neighbor] = candidate_parent
                serial += 1
                heapq.heappush(
                    frontier,
                    (
                        candidate
                        + minimum_weight * _octile(neighbor, problem.goal),
                        candidate,
                        serial,
                        neighbor,
                    ),
                )
        raise PlannerFailure("no_path", "Lazy Theta* found no connected path")


class DStarLiteBackend(PlannerBackend):
    """Incremental D* Lite on the current ego raster.

    The consistency algorithm is native D* Lite. Pixel state is deliberately
    reinitialized for every independently recentered rolling ego BEV because
    no deployable, non-GT transform is available to align those grid vertices.
    """

    key = "dstar_lite"
    label = "Risk-aware D* Lite"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.g: np.ndarray | None = None
        self.rhs: np.ndarray | None = None
        self.problem: GridProblem | None = None
        self.queue: list[tuple[float, float, int, tuple[int, int]]] = []
        self.serial = 0
        self.last_start: tuple[int, int] | None = None
        self.km = 0.0
        self.reused_updates = 0

    def _key(self, node: tuple[int, int]) -> tuple[float, float]:
        assert self.g is not None and self.rhs is not None and self.problem is not None
        value = min(float(self.g[node]), float(self.rhs[node]))
        return value + _octile(self.problem.start, node) + self.km, value

    def _push(self, node: tuple[int, int]) -> None:
        first, second = self._key(node)
        self.serial += 1
        heapq.heappush(self.queue, (first, second, self.serial, node))

    def _neighbors(self, node: tuple[int, int]) -> Iterable[tuple[int, int]]:
        assert self.problem is not None
        for dr, dc, _ in NEIGHBORS_8:
            candidate = node[0] + dr, node[1] + dc
            if _valid_step(self.problem, node, candidate):
                yield candidate

    def _update_vertex(self, node: tuple[int, int]) -> None:
        assert self.problem is not None and self.g is not None and self.rhs is not None
        if node != self.problem.goal:
            values = [
                _edge_cost(self.problem, node, successor) + float(self.g[successor])
                for successor in self._neighbors(node)
            ]
            self.rhs[node] = min(values, default=math.inf)
        if not math.isclose(float(self.g[node]), float(self.rhs[node]), rel_tol=0.0, abs_tol=1e-12):
            self._push(node)

    def _initialize(self, problem: GridProblem) -> None:
        self.problem = problem
        self.g = np.full(problem.blocked.shape, math.inf, dtype=np.float64)
        self.rhs = np.full(problem.blocked.shape, math.inf, dtype=np.float64)
        self.rhs[problem.goal] = 0.0
        self.queue = []
        self.serial = 0
        self.km = 0.0
        self.last_start = problem.start
        self._push(problem.goal)

    def _apply_changed_grid(self, problem: GridProblem) -> None:
        assert self.problem is not None
        old = self.problem
        changed = np.argwhere(
            (old.blocked != problem.blocked)
            | (np.abs(np.nan_to_num(old.weight, nan=1e9, posinf=1e9) - np.nan_to_num(problem.weight, nan=1e9, posinf=1e9)) > 1e-3)
        )
        self.km += _octile(self.last_start or old.start, problem.start)
        self.last_start = problem.start
        self.problem = problem
        affected: set[tuple[int, int]] = set()
        for row, column in changed:
            node = int(row), int(column)
            affected.add(node)
            for dr, dc, _ in NEIGHBORS_8:
                other = node[0] + dr, node[1] + dc
                if 0 <= other[0] < problem.size and 0 <= other[1] < problem.size:
                    affected.add(other)
        for node in affected:
            self._update_vertex(node)
        self.reused_updates += 1

    def plan(self, problem: GridProblem, *, budget_s: float = 0.35) -> PlanResult:
        started = time.monotonic()
        # Each model output is a newly translated and rotated ego raster.
        # Reusing pixel-indexed D* state across those frames silently treats
        # different metric locations as the same vertex.  Without feeding GT
        # pose or another localization oracle to the backend, the correct
        # deployable operation is a fresh D* Lite solve per ego frame.
        self._initialize(problem)
        reused = False
        assert self.g is not None and self.rhs is not None and self.problem is not None
        expanded = 0
        deadline = started + budget_s
        while self.queue:
            top = self.queue[0]
            start_key = self._key(problem.start)
            if (top[0], top[1]) >= start_key and math.isclose(
                float(self.rhs[problem.start]), float(self.g[problem.start]), abs_tol=1e-12
            ):
                break
            if time.monotonic() >= deadline:
                raise PlannerFailure("planning_budget", "D* Lite exhausted its time budget")
            old_first, old_second, _, node = heapq.heappop(self.queue)
            new_key = self._key(node)
            if (old_first, old_second) < new_key:
                self._push(node)
                continue
            if (old_first, old_second) > new_key:
                # A newer, better lazy-queue entry already represents this
                # vertex. The larger stale key must not invalidate it.
                continue
            if math.isclose(
                float(self.g[node]),
                float(self.rhs[node]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                # Lazy deletion: an older duplicate can remain after the
                # vertex has become locally consistent.
                continue
            if float(self.g[node]) > float(self.rhs[node]):
                self.g[node] = self.rhs[node]
                for predecessor in list(self._neighbors(node)):
                    self._update_vertex(predecessor)
            else:
                self.g[node] = math.inf
                self._update_vertex(node)
                for predecessor in list(self._neighbors(node)):
                    self._update_vertex(predecessor)
            expanded += 1
        if not math.isfinite(float(self.g[problem.start])):
            raise PlannerFailure("no_path", "D* Lite found no connected path")
        path = [problem.start]
        visited = {problem.start}
        cost = 0.0
        while path[-1] != problem.goal:
            current = path[-1]
            candidates = [
                (_edge_cost(problem, current, node) + float(self.g[node]), node)
                for node in self._neighbors(current)
            ]
            if not candidates:
                raise PlannerFailure("no_path", "D* Lite path extraction failed")
            edge_and_tail, next_node = min(candidates)
            if next_node in visited:
                raise PlannerFailure("cycle", "D* Lite path extraction formed a cycle")
            cost += _edge_cost(problem, current, next_node)
            path.append(next_node)
            visited.add(next_node)
            if len(path) > problem.size * problem.size:
                raise PlannerFailure("cycle", "D* Lite path exceeded grid size")
        return _result(
            problem,
            path,
            cost,
            started,
            expanded,
            algorithm="D* Lite (Koenig & Likhachev, 2002)",
            incremental_state_reused=reused,
            reuse_count=0,
            rolling_ego_reinitialization=True,
            reason=(
                "pixel-indexed state cannot be reused across independently "
                "recentered predicted ego BEVs without an external pose oracle"
            ),
        )


class ADStarBackend(PlannerBackend):
    key = "adstar"
    label = "Anytime Dynamic A*"

    def reset(self) -> None:
        pass

    def plan(self, problem: GridProblem, *, budget_s: float = 0.35) -> PlanResult:
        """Canonical AD* OPEN/INCONS repair on the current predicted graph.

        Every independently recentered ego BEV is a new graph, so graph state
        is initialized at the start of this call.  The anytime improvements
        inside the call reuse ``g``, ``rhs``, OPEN and INCONS exactly as AD*
        specifies; no independent Weighted-A* fallbacks are run.
        """

        started = time.monotonic()
        deadline = started + budget_s
        minimum_weight = float(np.min(problem.weight[~problem.blocked]))
        g = np.full(problem.blocked.shape, math.inf, dtype=np.float64)
        rhs = np.full(problem.blocked.shape, math.inf, dtype=np.float64)
        rhs[problem.goal] = 0.0
        epsilon = 2.5
        epsilon_step = 0.5
        open_queue: list[tuple[float, float, int, tuple[int, int]]] = []
        incons: set[tuple[int, int]] = set()
        closed: set[tuple[int, int]] = set()
        serial = 0
        expanded = 0

        def neighbors(node: tuple[int, int]) -> Iterable[tuple[int, int]]:
            for dr, dc, _ in NEIGHBORS_8:
                candidate = node[0] + dr, node[1] + dc
                if _valid_step(problem, node, candidate):
                    yield candidate

        def key(node: tuple[int, int]) -> tuple[float, float]:
            value = min(float(g[node]), float(rhs[node]))
            return (
                value
                + epsilon
                * minimum_weight
                * _octile(problem.start, node),
                value,
            )

        def push(node: tuple[int, int]) -> None:
            nonlocal serial
            first, second = key(node)
            serial += 1
            heapq.heappush(open_queue, (first, second, serial, node))

        def update_state(node: tuple[int, int]) -> None:
            if node != problem.goal:
                rhs[node] = min(
                    (
                        _edge_cost(problem, node, successor)
                        + float(g[successor])
                        for successor in neighbors(node)
                    ),
                    default=math.inf,
                )
            if not math.isclose(
                float(g[node]), float(rhs[node]), rel_tol=0.0, abs_tol=1e-12
            ):
                if node not in closed:
                    push(node)
                else:
                    incons.add(node)

        def pop_valid() -> tuple[tuple[float, float], tuple[int, int]] | None:
            while open_queue:
                first, second, _, node = heapq.heappop(open_queue)
                current = key(node)
                if (first, second) > current:
                    continue
                return (first, second), node
            return None

        def top_key() -> tuple[float, float]:
            while open_queue:
                first, second, _, node = open_queue[0]
                current = key(node)
                if (first, second) > current or math.isclose(
                    float(g[node]), float(rhs[node]), rel_tol=0.0, abs_tol=1e-12
                ):
                    heapq.heappop(open_queue)
                    continue
                return first, second
            return math.inf, math.inf

        def improve_path() -> bool:
            nonlocal expanded
            while top_key() < key(problem.start) or not math.isclose(
                float(rhs[problem.start]),
                float(g[problem.start]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                if time.monotonic() >= deadline:
                    return False
                item = pop_valid()
                if item is None:
                    return True
                old_key, node = item
                if old_key < key(node):
                    push(node)
                elif float(g[node]) > float(rhs[node]):
                    g[node] = rhs[node]
                    closed.add(node)
                    for predecessor in neighbors(node):
                        update_state(predecessor)
                else:
                    g[node] = math.inf
                    update_state(node)
                    for predecessor in neighbors(node):
                        update_state(predecessor)
                expanded += 1
            return True

        push(problem.goal)
        completed_epsilons: list[float] = []
        best_solution: tuple[list[tuple[int, int]], float, float] | None = None
        while True:
            completed = improve_path()
            if math.isfinite(float(g[problem.start])):
                path = [problem.start]
                visited = {problem.start}
                path_cost = 0.0
                while path[-1] != problem.goal:
                    current = path[-1]
                    candidates = [
                        (
                            _edge_cost(problem, current, successor)
                            + float(g[successor]),
                            successor,
                        )
                        for successor in neighbors(current)
                    ]
                    if not candidates:
                        break
                    _, successor = min(candidates)
                    if successor in visited:
                        break
                    path_cost += _edge_cost(problem, current, successor)
                    path.append(successor)
                    visited.add(successor)
                if path[-1] == problem.goal:
                    best_solution = path, path_cost, epsilon
                    completed_epsilons.append(epsilon)
            if not completed or epsilon <= 1.0 or time.monotonic() >= deadline:
                break
            epsilon = max(1.0, epsilon - epsilon_step)
            # AD*: OPEN <- OPEN union INCONS, recompute priorities, then clear
            # CLOSED. Rebuilding from all inconsistent states is exactly the
            # same set operation while avoiding stale heap entries.
            inconsistent = np.argwhere(~np.isclose(g, rhs, rtol=0.0, atol=1e-12))
            open_queue.clear()
            incons.clear()
            closed.clear()
            for row, column in inconsistent:
                push((int(row), int(column)))

        if best_solution is None:
            code = "planning_budget" if time.monotonic() >= deadline else "no_path"
            raise PlannerFailure(code, "AD* did not produce a path")
        path, cost, final_epsilon = best_solution
        return _result(
            problem,
            path,
            cost,
            started,
            expanded,
            algorithm="AD* (Likhachev et al., 2005)",
            final_epsilon=final_epsilon,
            anytime=True,
            dynamic_reuse="disabled_across_unaligned_ego_frames",
            epsilon_start=2.5,
            epsilon_decrement=epsilon_step,
            completed_epsilons=completed_epsilons,
            open_incons_repair=True,
            optimal_on_grid=final_epsilon == 1.0,
        )


class HybridAStarBackend(PlannerBackend):
    key = "hybrid_astar"
    label = "Hybrid A* (SE(2))"
    family = "kinematic"
    heading_bins = 16
    primitive_step_m = 0.18
    steering_indices = (-1, 0, 1)
    steering_bin_fraction = 0.5
    heuristic_weight = 1.0
    motion_limits = RobotMotionLimits()

    def plan(self, problem: GridProblem, *, budget_s: float = 0.55) -> PlanResult:
        started = time.monotonic()
        deadline = started + budget_s
        bins = self.heading_bins
        step = max(2.0, self.primitive_step_m / problem.cell_size_m)
        step_m = step * problem.cell_size_m
        maximum_speed = max(self.motion_limits.maximum_speed_m_s, 1e-6)
        maximum_curvature = (
            self.motion_limits.maximum_angular_speed_rad_s / maximum_speed
        )
        maximum_steering_index = max(abs(item) for item in self.steering_indices)
        steering_step_angle = (
            maximum_curvature * step_m / max(maximum_steering_index, 1)
        )
        start = (float(problem.start[0]), float(problem.start[1]), -math.pi / 2.0)
        goal = problem.goal
        minimum_weight = float(np.min(problem.weight[~problem.blocked]))
        holonomic_cost, heuristic_expanded = _holonomic_cost_to_go(
            problem, deadline=deadline
        )
        if not math.isfinite(float(holonomic_cost[problem.start])):
            raise PlannerFailure("no_path", "Hybrid A* 2-D heuristic found no path")

        def heuristic_at(point: Sequence[int]) -> float:
            return float(holonomic_cost[int(point[0]), int(point[1])])

        def key(state: tuple[float, float, float]) -> tuple[int, int, int]:
            heading = int(round((state[2] % (2 * math.pi)) / (2 * math.pi) * bins)) % bins
            return int(round(state[0])), int(round(state[1])), heading

        def segment_valid(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
            cells = _supercover_line(a[:2], b[:2])
            return all(
                0 <= p[0] < problem.size
                and 0 <= p[1] < problem.size
                and not problem.blocked[p]
                for p in cells
            )

        start_key = key(start)
        frontier: list[tuple[float, float, int, tuple[float, float, float]]] = []
        heapq.heappush(
            frontier,
            (
                self.heuristic_weight * heuristic_at(start_key),
                0.0,
                0,
                start,
            ),
        )
        best = {start_key: 0.0}
        states = {start_key: start}
        parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        serial = 0
        expanded = 0
        goal_key: tuple[int, int, int] | None = None
        while frontier:
            if time.monotonic() >= deadline:
                raise PlannerFailure(
                    "planning_budget", "Hybrid A* exhausted its time budget"
                )
            _, cost, _, state = heapq.heappop(frontier)
            state_key = key(state)
            if cost > best.get(state_key, math.inf):
                continue
            expanded += 1
            if math.hypot(state[0] - goal[0], state[1] - goal[1]) <= step:
                goal_key = state_key
                break
            # Differential-drive-compatible primitives: rotate in place or
            # advance with bounded heading change.
            candidates: list[tuple[tuple[float, float, float], float]] = []
            for turn in (-1, 1):
                candidates.append(((state[0], state[1], state[2] + turn * 2 * math.pi / bins), 0.35 * step))
            for direction in (1.0, -1.0):
                for steering in self.steering_indices:
                    heading = state[2] + direction * steering * steering_step_angle
                    candidate = (
                        # Image rows decrease in the robot-forward direction.
                        # Negative direction is a physically valid reverse
                        # differential-drive primitive, never lateral motion.
                        state[0] + direction * step * math.sin(heading),
                        state[1] + direction * step * math.cos(heading),
                        heading,
                    )
                    reverse_penalty = 1.20 if direction < 0.0 else 1.0
                    candidates.append(
                        (
                            candidate,
                            step
                            * reverse_penalty
                            * (1.0 + 0.12 * abs(steering)),
                        )
                    )
            for candidate, primitive_cost in candidates:
                candidate_key = key(candidate)
                if not (0 <= candidate_key[0] < problem.size and 0 <= candidate_key[1] < problem.size):
                    continue
                if problem.blocked[candidate_key[:2]] or not segment_valid(state, candidate):
                    continue
                risk = float(problem.weight[candidate_key[:2]])
                new_cost = cost + primitive_cost * risk
                if new_cost >= best.get(candidate_key, math.inf):
                    continue
                best[candidate_key] = new_cost
                states[candidate_key] = candidate
                parent[candidate_key] = state_key
                serial += 1
                heuristic = heuristic_at(candidate_key)
                heapq.heappush(
                    frontier,
                    (
                        new_cost + self.heuristic_weight * heuristic,
                        new_cost,
                        serial,
                        candidate,
                    ),
                )
        if goal_key is None:
            raise PlannerFailure("no_path", "Hybrid A* found no SE(2) path")
        keys = [goal_key]
        while keys[-1] != start_key:
            keys.append(parent[keys[-1]])
        keys.reverse()
        pixels = [[states[item][0], states[item][1]] for item in keys]
        # Search headings live in image (row, column) coordinates, where row
        # increases backward. Convert them to the verifier's yaw-left metric
        # convention: body forward is [-sin(yaw), cos(yaw)] in [right,forward].
        headings = [-(states[item][2] + math.pi / 2.0) for item in keys]
        # Do not append a straight chord to the exact goal cell: that chord is
        # not one of the searched motion primitives and can contradict the
        # final body heading.  The native search terminates within one
        # primitive step, inside the verifier's 20 cm success radius.
        return _result(
            problem,
            pixels,
            best[goal_key],
            started,
            expanded,
            path_headings_rad=headings,
            heading_bins=bins,
            primitive_step_m=self.primitive_step_m,
            steering_indices=list(self.steering_indices),
            heuristic_weight=self.heuristic_weight,
            maximum_curvature_rad_m=maximum_curvature,
            minimum_turning_radius_m=1.0 / max(maximum_curvature, 1e-9),
            reverse_primitives=True,
            in_place_rotation_primitives=True,
            terminal_goal_tolerance_m=step_m,
            robot_motion_limits=self.motion_limits.as_dict(),
            algorithm="Hybrid A* (Dolgov et al., 2008)",
            obstacle_aware_holonomic_heuristic=True,
            holonomic_heuristic_expanded=heuristic_expanded,
            cross_backend_fallback=False,
        )


class StateLatticeAStarBackend(PlannerBackend):
    key = "state_lattice_astar"
    label = "State Lattice A*"
    family = "kinematic"
    heading_bins = 8
    motion_limits = RobotMotionLimits()

    def plan(self, problem: GridProblem, *, budget_s: float = 0.70) -> PlanResult:
        """A* over a repeating, feasible differential-drive state lattice."""

        started = time.monotonic()
        deadline = started + budget_s
        heuristic, heuristic_expanded = _holonomic_cost_to_go(
            problem, deadline=deadline
        )
        if not math.isfinite(float(heuristic[problem.start])):
            raise PlannerFailure("no_path", "State lattice has no holonomic route")

        # Heading 6 is negative image-row: the robot's initial forward axis.
        start = (problem.start[0], problem.start[1], 6)
        frontier: list[tuple[float, float, int, tuple[int, int, int]]] = [
            (float(heuristic[problem.start]), 0.0, 0, start)
        ]
        best = {start: 0.0}
        parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        serial = 0
        expanded = 0
        goal_state: tuple[int, int, int] | None = None

        def successors(
            state: tuple[int, int, int]
        ) -> Iterable[tuple[tuple[int, int, int], float]]:
            row, column, heading = state
            angle = 2.0 * math.pi * heading / self.heading_bins
            dr = int(round(math.sin(angle)))
            dc = int(round(math.cos(angle)))
            for direction in (1, -1):
                endpoint = row + direction * dr, column + direction * dc
                if _valid_step(problem, (row, column), endpoint):
                    yield (
                        (endpoint[0], endpoint[1], heading),
                        _edge_cost(problem, (row, column), endpoint),
                    )
            # An in-place differential-drive rotation is a native feasible
            # primitive. Its cost is the physical track travel expressed in
            # grid cells; it does not translate the robot.
            turn_angle = 2.0 * math.pi / self.heading_bins
            turn_distance_cells = (
                0.5 * self.motion_limits.track_width_m * turn_angle
                / problem.cell_size_m
            )
            turn_cost = max(turn_distance_cells, 1e-6) * float(
                problem.weight[row, column]
            )
            yield (row, column, (heading - 1) % self.heading_bins), turn_cost
            yield (row, column, (heading + 1) % self.heading_bins), turn_cost

        while frontier:
            if time.monotonic() >= deadline:
                raise PlannerFailure(
                    "planning_budget", "State Lattice A* exhausted its time budget"
                )
            _, cost, _, state = heapq.heappop(frontier)
            if cost > best.get(state, math.inf):
                continue
            expanded += 1
            if state[:2] == problem.goal:
                goal_state = state
                break
            for successor, edge_cost in successors(state):
                candidate = cost + edge_cost
                if candidate >= best.get(successor, math.inf):
                    continue
                best[successor] = candidate
                parent[successor] = state
                serial += 1
                heapq.heappush(
                    frontier,
                    (
                        candidate + float(heuristic[successor[:2]]),
                        candidate,
                        serial,
                        successor,
                    ),
                )
        if goal_state is None:
            raise PlannerFailure("no_path", "State Lattice A* found no path")
        states = [goal_state]
        while states[-1] != start:
            states.append(parent[states[-1]])
        states.reverse()
        headings = [
            -(2.0 * math.pi * state[2] / self.heading_bins + math.pi / 2.0)
            for state in states
        ]
        return _result(
            problem,
            [state[:2] for state in states],
            best[goal_state],
            started,
            expanded,
            path_headings_rad=headings,
            algorithm="State Lattice A* (Pivtoraiko, Knepper & Kelly, 2009)",
            motion_primitives=(
                "8-heading repeating lattice: in-place rotations plus "
                "forward/reverse straight differential-drive edges"
            ),
            heading_bins=self.heading_bins,
            search="A*",
            heuristic_weight=1.0,
            holonomic_heuristic_expanded=heuristic_expanded,
            robot_motion_limits=self.motion_limits.as_dict(),
        )


class MPPIBackend(PlannerBackend):
    key = "mppi"
    label = "MPPI local control"
    family = "sampling-control"

    def __init__(self) -> None:
        self.rng = np.random.default_rng(20260827)
        self.motion_limits = RobotMotionLimits()
        self.nominal_controls: np.ndarray | None = None
        self.initial_linear_velocity_m_s = 0.0
        self.initial_angular_velocity_rad_s = 0.0

    def reset(self) -> None:
        self.rng = np.random.default_rng(20260827)
        self.nominal_controls = None
        self.initial_linear_velocity_m_s = 0.0
        self.initial_angular_velocity_rad_s = 0.0

    def set_motion_state(self, linear_velocity_m_s: float, angular_velocity_rad_s: float) -> None:
        self.initial_linear_velocity_m_s = float(
            np.clip(
                linear_velocity_m_s,
                self.motion_limits.minimum_linear_velocity_m_s,
                self.motion_limits.maximum_linear_velocity_m_s,
            )
        )
        self.initial_angular_velocity_rad_s = float(
            np.clip(
                angular_velocity_rad_s,
                self.motion_limits.minimum_angular_velocity_rad_s,
                self.motion_limits.maximum_angular_velocity_rad_s,
            )
        )

    def plan(self, problem: GridProblem, *, budget_s: float = 0.35) -> PlanResult:
        started = time.monotonic()
        count, horizon, dt, iterations = 2048, 36, 0.10, 3
        target = np.asarray(problem.pixel_to_metric(problem.goal), dtype=np.float64)
        anchor = np.asarray(problem.pixel_to_metric(problem.start), dtype=np.float64)
        guide_used = False
        guide_path_cells = 0
        tracking_target = target.copy()
        try:
            guide, _, _ = weighted_astar(
                problem,
                heuristic_weight=1.0,
                deadline=started + min(0.06, max(0.02, budget_s * 0.30)),
            )
            guide_metric = np.asarray(
                [problem.pixel_to_metric(point) for point in guide], dtype=np.float64
            )
            guide_path_cells = len(guide_metric)
            if len(guide_metric) > 1:
                cumulative = np.concatenate(
                    ([0.0], np.cumsum(np.linalg.norm(np.diff(guide_metric, axis=0), axis=1)))
                )
                lookahead_index = int(np.searchsorted(cumulative, 0.90, side="left"))
                tracking_target = guide_metric[min(lookahead_index, len(guide_metric) - 1)]
                guide_used = True
        except PlannerFailure:
            pass
        target_delta = tracking_target - anchor
        desired_bearing = math.atan2(
            float(target_delta[0]), max(float(target_delta[1]), 1e-6)
        )
        if self.nominal_controls is None or self.nominal_controls.shape != (horizon, 2):
            nominal = np.zeros((horizon, 2), dtype=np.float64)
        else:
            nominal = np.vstack(
                (self.nominal_controls[1:], self.nominal_controls[-1:])
            )
        guidance = np.zeros_like(nominal)
        guidance[:, 0] = min(
            self.motion_limits.maximum_linear_velocity_m_s,
            max(0.30, float(np.linalg.norm(target_delta)) / (horizon * dt)),
        )
        guidance[:, 1] = np.clip(
            -1.8 * desired_bearing,
            self.motion_limits.minimum_angular_velocity_rad_s,
            self.motion_limits.maximum_angular_velocity_rad_s,
        )
        nominal = 0.25 * nominal + 0.75 * guidance

        def enforce_dynamics(raw: np.ndarray) -> np.ndarray:
            result = np.empty_like(raw)
            previous_v = np.full(
                raw.shape[0], self.initial_linear_velocity_m_s, dtype=np.float64
            )
            previous_w = np.full(
                raw.shape[0], self.initial_angular_velocity_rad_s, dtype=np.float64
            )
            for index in range(horizon):
                desired_v = np.clip(
                    raw[:, index, 0],
                    self.motion_limits.minimum_linear_velocity_m_s,
                    self.motion_limits.maximum_linear_velocity_m_s,
                )
                desired_w = np.clip(
                    raw[:, index, 1],
                    self.motion_limits.minimum_angular_velocity_rad_s,
                    self.motion_limits.maximum_angular_velocity_rad_s,
                )
                previous_v += np.clip(
                    desired_v - previous_v,
                    self.motion_limits.minimum_linear_acceleration_m_s2 * dt,
                    self.motion_limits.maximum_linear_acceleration_m_s2 * dt,
                )
                previous_w += np.clip(
                    desired_w - previous_w,
                    self.motion_limits.minimum_angular_acceleration_rad_s2 * dt,
                    self.motion_limits.maximum_angular_acceleration_rad_s2 * dt,
                )
                result[:, index, 0] = previous_v
                result[:, index, 1] = previous_w
            return result

        def evaluate(controls: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            sample_count = controls.shape[0]
            right = np.full(sample_count, anchor[0])
            forward = np.full(sample_count, anchor[1])
            yaw = np.zeros(sample_count)
            total = np.zeros(sample_count)
            trajectories = np.zeros((sample_count, horizon, 2), dtype=np.float64)
            headings = np.zeros((sample_count, horizon), dtype=np.float64)
            valid = np.ones(sample_count, dtype=bool)
            previous_controls = np.empty((sample_count, 2), dtype=np.float64)
            previous_controls[:, 0] = self.initial_linear_velocity_m_s
            previous_controls[:, 1] = self.initial_angular_velocity_rad_s
            for step_index in range(horizon):
                # Controls are interval-end body velocities. Integrate their
                # midpoint so the rollout matches the verifier's continuous
                # acceleration-limited differential-drive execution.
                velocity = 0.5 * (
                    previous_controls[:, 0] + controls[:, step_index, 0]
                )
                omega = 0.5 * (
                    previous_controls[:, 1] + controls[:, step_index, 1]
                )
                yaw_mid = yaw + 0.5 * omega * dt
                right += -velocity * np.sin(yaw_mid) * dt
                forward += velocity * np.cos(yaw_mid) * dt
                yaw += omega * dt
                trajectories[:, step_index, 0] = right
                trajectories[:, step_index, 1] = forward
                headings[:, step_index] = yaw
                columns = np.floor(
                    (right + problem.extent_m / 2.0) / problem.cell_size_m
                ).astype(int)
                rows = np.floor(
                    (problem.extent_m / 2.0 - forward) / problem.cell_size_m
                ).astype(int)
                inside = (
                    (rows >= 0)
                    & (rows < problem.size)
                    & (columns >= 0)
                    & (columns < problem.size)
                )
                safe_rows = np.clip(rows, 0, problem.size - 1)
                safe_columns = np.clip(columns, 0, problem.size - 1)
                valid &= inside & ~problem.blocked[safe_rows, safe_columns]
                sampled_weight = problem.weight[safe_rows, safe_columns]
                total += np.where(
                    valid & np.isfinite(sampled_weight),
                    np.nan_to_num(sampled_weight, posinf=0.0)
                    * np.abs(velocity)
                    * dt,
                    1e5,
                )
                delta_control = controls[:, step_index] - previous_controls
                total += 0.02 * omega * omega
                total += 0.04 * np.sum(delta_control * delta_control, axis=1)
                total += 0.015 * np.maximum(0.0, -velocity)
                previous_controls = controls[:, step_index]
            tracking_distance = np.hypot(
                right - tracking_target[0], forward - tracking_target[1]
            )
            final_distance = np.hypot(right - target[0], forward - target[1])
            terminal_bearing = np.arctan2(
                -(tracking_target[0] - right),
                np.maximum(tracking_target[1] - forward, 1e-6),
            )
            heading_error = np.arctan2(
                np.sin(yaw - terminal_bearing), np.cos(yaw - terminal_bearing)
            )
            total += (
                24.0 * tracking_distance
                + 4.0 * final_distance
                + 0.50 * np.abs(heading_error)
            )
            total[~valid] = math.inf
            return total, trajectories, headings, valid

        noise_sigma = np.asarray([0.24, 0.48], dtype=np.float64)
        temperature = 1.0
        best_controls: np.ndarray | None = None
        best_cost = math.inf
        best_trajectory: np.ndarray | None = None
        best_headings: np.ndarray | None = None
        completed_iterations = 0
        for _ in range(iterations):
            noise = self.rng.normal(
                0.0, noise_sigma, (count, horizon, 2)
            )
            raw = nominal[None, :, :] + noise
            # Deterministic members prevent a finite random batch from missing
            # essential differential-drive behaviours. This is an explicit
            # engineering robustness enhancement, not native MPPI semantics.
            raw[0] = nominal
            raw[1] = guidance
            raw[2, :, 0] = 0.0
            raw[2, :, 1] = -math.copysign(
                self.motion_limits.maximum_angular_velocity_rad_s,
                desired_bearing if abs(desired_bearing) > 1e-6 else 1.0,
            )
            raw[3, :, 0] = 0.0
            raw[3, :, 1] = -raw[2, :, 1]
            raw[4, :, 0] = -0.25
            raw[4, :, 1] = guidance[:, 1]
            # The importance-sampling correction must use the perturbation of
            # the sequence actually evaluated. Keeping the pre-override random
            # noise here biases deterministic members by an unrelated cost.
            noise[:5] = raw[:5] - nominal[None, :, :]
            controls = enforce_dynamics(raw)
            total, trajectories, headings, valid = evaluate(controls)
            # Use the state/control objective directly for the path-integral
            # weights. The former unbounded ``u Sigma^-1 epsilon`` term was
            # numerically dominant at this short horizon: a sufficiently
            # negative perturbation could beat metres of terminal error and
            # make a forward goal produce reverse commands. The regularisation
            # inside ``evaluate`` already prices control effort and slew.
            finite = np.flatnonzero(np.isfinite(total) & valid)
            if not len(finite):
                continue
            winner = int(finite[np.argmin(total[finite])])
            if float(total[winner]) < best_cost:
                best_cost = float(total[winner])
                best_controls = controls[winner].copy()
                best_trajectory = trajectories[winner].copy()
                best_headings = headings[winner].copy()
            baseline = float(np.min(total[finite]))
            temperature = max(0.50, float(np.std(total[finite])) * 0.25)
            weights = np.exp(
                -np.clip((total[finite] - baseline) / temperature, 0.0, 60.0)
            )
            weights /= max(float(weights.sum()), 1e-12)
            weighted_delta = np.tensordot(weights, noise[finite], axes=(0, 0))
            # Trust-region the batch update so one noisy finite sample cannot
            # replace a useful warm start. Receding-horizon replanning can
            # still move the sequence quickly over successive frames.
            weighted_delta[:, 0] = np.clip(weighted_delta[:, 0], -0.08, 0.08)
            weighted_delta[:, 1] = np.clip(weighted_delta[:, 1], -0.16, 0.16)
            nominal += weighted_delta
            nominal = enforce_dynamics(nominal[None, :, :])[0]
            completed_iterations += 1
            if time.monotonic() - started >= budget_s:
                break
        if completed_iterations == 0 or best_controls is None:
            raise PlannerFailure("no_rollout", "MPPI found no collision-free rollout")
        # Select the strongest safe sequence among the optimised mean, best
        # sampled rollout, and deterministic graph-guide sequence. This is a
        # constrained-MPPI guard: all candidates use the same differential-
        # drive rollout and objective and only the first control is executed.
        guide_controls = enforce_dynamics(guidance[None, :, :])[0]
        candidate_controls = np.stack((nominal, best_controls, guide_controls), axis=0)
        candidate_total, candidate_trajectories, candidate_headings, candidate_valid = evaluate(
            candidate_controls
        )
        start_tracking_distance = float(np.linalg.norm(anchor - tracking_target))
        candidate_progress = start_tracking_distance - np.linalg.norm(
            candidate_trajectories[:, -1, :] - tracking_target[None, :], axis=1
        )
        # Translating sequences must make measurable progress. Pure rotation
        # remains admissible when the local target is substantially off-axis.
        translating = np.max(np.abs(candidate_controls[:, :, 0]), axis=1) > 0.05
        progress_ok = (candidate_progress >= 0.02) | ~translating
        admissible = candidate_valid & np.isfinite(candidate_total) & progress_ok
        if not np.any(admissible):
            admissible = candidate_valid & np.isfinite(candidate_total)
        if not np.any(admissible):
            raise PlannerFailure("no_rollout", "MPPI found no safe executable control sequence")
        # Prefer the weighted mean whenever it is admissible. A single sampled
        # winner has lower Monte-Carlo cost surprisingly often but causes
        # frame-to-frame steering jitter when executed receding-horizon. The
        # deterministic guide is the next choice; a sampled rollout is the
        # final safe fallback.
        if bool(admissible[0]):
            selected_index = 0
        elif bool(admissible[2]):
            selected_index = 2
        else:
            selected_index = 1
        selection_labels = ("weighted_mean", "best_safe_rollout", "graph_guide")
        controls_out = candidate_controls[selected_index]
        trajectory_out = candidate_trajectories[selected_index]
        headings_out = candidate_headings[selected_index]
        cost_out = float(candidate_total[selected_index])
        sampled_rollout_fallback = selected_index == 1
        guide_sequence_selected = selected_index == 2
        self.nominal_controls = np.vstack((controls_out[1:], controls_out[-1:]))
        path_metric = np.vstack(
            (anchor.reshape(1, 2), trajectory_out)
        ).tolist()
        path_headings = np.concatenate(([0.0], headings_out)).tolist()
        pixels = [problem.metric_to_pixel(point[0], point[1]) for point in path_metric]
        result = _result(
            problem,
            pixels,
            cost_out / max(problem.cell_size_m, 1e-9),
            started,
            count * max(completed_iterations, 1),
            path_headings_rad=path_headings,
            rollout_count=count,
            horizon_seconds=horizon * dt,
            optimization_iterations=completed_iterations,
            algorithm="MPPI (Williams, Aldrich & Theodorou, 2017)",
            control_update="exponential-cost-weighted Gaussian perturbation",
            importance_sampling_cross_term="disabled_after_reverse-motion_instability",
            control_update_trust_region=[0.08, 0.16],
            temperature=temperature,
            noise_standard_deviation=noise_sigma.tolist(),
            warm_started_nominal_sequence=True,
            deterministic_behavior_rollouts=True,
            graph_guide_used=guide_used,
            graph_guide_cells=guide_path_cells,
            graph_guide_lookahead_m=0.90,
            sampled_rollout_fallback=sampled_rollout_fallback,
            guide_sequence_selected=guide_sequence_selected,
            selected_sequence=selection_labels[selected_index],
            selected_sequence_progress_m=float(candidate_progress[selected_index]),
            target_bearing_sign_fix=True,
            globally_complete=False,
            robot_motion_limits=self.motion_limits.as_dict(),
        )
        result.path_metric_m = [[float(a), float(b)] for a, b in path_metric]
        result.control_velocity = (
            float(controls_out[0, 0]),
            0.0,
            float(controls_out[0, 1]),
        )
        result.control_sequence = controls_out.astype(float).tolist()
        result.control_timestep_s = dt
        return result


class BITStarBackend(PlannerBackend):
    key = "bitstar"
    label = "BIT*"
    family = "sampling"

    def __init__(self) -> None:
        self.rng = np.random.default_rng(20260827)

    def reset(self) -> None:
        self.rng = np.random.default_rng(20260827)

    def plan(self, problem: GridProblem, *, budget_s: float = 0.55) -> PlanResult:
        """Batch Informed Trees over a continuous 2-D implicit RGG.

        This keeps BIT*'s explicit search tree, separate vertex/edge queues,
        informed batches, pruning and rewiring.  It is not the former batched
        PRM approximation.
        """

        started = time.monotonic()
        deadline = started + budget_s
        free = np.argwhere(~problem.blocked)
        if len(free) < 2:
            raise PlannerFailure("no_free_space", "BIT* has insufficient free samples")
        minimum_weight = float(np.min(problem.weight[~problem.blocked]))
        points: dict[int, np.ndarray] = {
            0: np.asarray(problem.start, dtype=np.float64),
            1: np.asarray(problem.goal, dtype=np.float64),
        }
        vertices: set[int] = {0}
        samples: set[int] = {1}
        parent: dict[int, int] = {}
        children: dict[int, set[int]] = {0: set()}
        g: dict[int, float] = {0: 0.0, 1: math.inf}
        next_id = 2
        best_cost = math.inf
        solution_found_at: float | None = None
        post_solution_refinement_s = min(0.12, budget_s * 0.20)
        vertex_queue: list[tuple[float, int, int]] = []
        edge_queue: list[tuple[float, int, int, int]] = []
        serial = 0
        expanded = 0
        batches = 0

        start_point = points[0]
        goal_point = points[1]
        c_min = float(np.linalg.norm(goal_point - start_point))
        centre = 0.5 * (start_point + goal_point)
        direction = goal_point - start_point
        orientation = math.atan2(float(direction[1]), float(direction[0]))
        rotation = np.asarray(
            [
                [math.cos(orientation), -math.sin(orientation)],
                [math.sin(orientation), math.cos(orientation)],
            ],
            dtype=np.float64,
        )

        def h_hat(node: int) -> float:
            return minimum_weight * float(np.linalg.norm(points[node] - goal_point))

        def g_hat(node: int) -> float:
            return minimum_weight * float(np.linalg.norm(points[node] - start_point))

        def sample_batch(count: int) -> None:
            nonlocal next_id
            accepted = 0
            attempts = 0
            max_attempts = count * 50
            while accepted < count and attempts < max_attempts:
                attempts += 1
                if math.isfinite(best_cost):
                    c_max = best_cost / max(minimum_weight, 1e-9)
                    if c_max <= c_min:
                        break
                    radius = math.sqrt(self.rng.random())
                    angle = self.rng.uniform(0.0, 2.0 * math.pi)
                    unit = np.asarray(
                        [radius * math.cos(angle), radius * math.sin(angle)]
                    )
                    axes = np.asarray(
                        [0.5 * c_max, 0.5 * math.sqrt(c_max * c_max - c_min * c_min)]
                    )
                    point = centre + rotation @ (axes * unit)
                else:
                    point = (
                        free[self.rng.integers(0, len(free))].astype(np.float64)
                        + self.rng.uniform(-0.499, 0.499, size=2)
                    )
                cell = tuple(map(int, np.rint(point)))
                if not (
                    0 <= cell[0] < problem.size
                    and 0 <= cell[1] < problem.size
                    and not problem.blocked[cell]
                ):
                    continue
                points[next_id] = point
                g[next_id] = math.inf
                samples.add(next_id)
                next_id += 1
                accepted += 1

        def prune() -> None:
            nonlocal vertices, samples
            if not math.isfinite(best_cost):
                return
            samples = {
                node for node in samples if g_hat(node) + h_hat(node) < best_cost
            }
            removed = {
                node
                for node in vertices
                if node != 0 and g_hat(node) + h_hat(node) >= best_cost
            }
            for node in removed:
                if math.isfinite(g.get(node, math.inf)):
                    g[node] = math.inf
                    samples.add(node)
                old_parent = parent.pop(node, None)
                if old_parent is not None:
                    children.setdefault(old_parent, set()).discard(node)
                children.pop(node, None)
            vertices -= removed
            for node, predecessor in list(parent.items()):
                if node not in vertices or predecessor not in vertices:
                    parent.pop(node, None)

        def radius() -> float:
            q = max(2, len(vertices) + len(samples))
            free_measure = float(len(free))
            # BIT*/RRT* connection radius in R^2, with eta=1.1.
            value = 1.1 * 2.0 * math.sqrt(
                1.5 * free_measure / math.pi * math.log(q) / q
            )
            return max(2.0, min(value, problem.size * math.sqrt(2.0)))

        def push_vertex(node: int) -> None:
            nonlocal serial
            serial += 1
            heapq.heappush(vertex_queue, (g[node] + h_hat(node), serial, node))

        def expand_vertex(node: int, connection_radius: float) -> None:
            nonlocal serial, expanded
            expanded += 1
            point = points[node]
            for candidate in tuple(samples):
                estimate = float(np.linalg.norm(point - points[candidate]))
                if estimate > connection_radius:
                    continue
                key_value = g[node] + minimum_weight * estimate + h_hat(candidate)
                if key_value >= best_cost:
                    continue
                serial += 1
                heapq.heappush(
                    edge_queue, (key_value, serial, node, candidate)
                )
            for candidate in tuple(vertices):
                if candidate == node:
                    continue
                estimate = float(np.linalg.norm(point - points[candidate]))
                if estimate > connection_radius:
                    continue
                if g[node] + minimum_weight * estimate >= g[candidate]:
                    continue
                key_value = g[node] + minimum_weight * estimate + h_hat(candidate)
                if key_value >= best_cost:
                    continue
                serial += 1
                heapq.heappush(
                    edge_queue, (key_value, serial, node, candidate)
                )

        while time.monotonic() < deadline:
            if (
                solution_found_at is not None
                and time.monotonic() - solution_found_at >= post_solution_refinement_s
            ):
                break
            if not vertex_queue and not edge_queue:
                prune()
                sample_batch(256 if batches == 0 else 512)
                batches += 1
                for vertex in vertices:
                    push_vertex(vertex)
                if not vertex_queue:
                    break
            connection_radius = radius()
            while vertex_queue and (
                not edge_queue or vertex_queue[0][0] <= edge_queue[0][0]
            ):
                if time.monotonic() >= deadline:
                    break
                _, _, vertex = heapq.heappop(vertex_queue)
                if vertex not in vertices or not math.isfinite(g[vertex]):
                    continue
                expand_vertex(vertex, connection_radius)
            if time.monotonic() >= deadline:
                break
            if not edge_queue:
                continue
            _, _, source, target = heapq.heappop(edge_queue)
            if source not in vertices:
                continue
            estimate = minimum_weight * float(
                np.linalg.norm(points[source] - points[target])
            )
            if g[source] + estimate + h_hat(target) >= best_cost:
                continue
            edge_cost = _line_cost(
                problem,
                tuple(map(int, np.rint(points[source]))),
                tuple(map(int, np.rint(points[target]))),
            )
            if not math.isfinite(edge_cost):
                continue
            candidate_cost = g[source] + edge_cost
            if candidate_cost + h_hat(target) >= best_cost:
                continue
            if candidate_cost >= g.get(target, math.inf):
                continue
            old_cost = g.get(target, math.inf)
            old_parent = parent.get(target)
            if old_parent is not None:
                children.setdefault(old_parent, set()).discard(target)
            parent[target] = source
            children.setdefault(source, set()).add(target)
            children.setdefault(target, set())
            g[target] = candidate_cost
            if target in samples:
                samples.remove(target)
                vertices.add(target)
                push_vertex(target)
            elif math.isfinite(old_cost):
                # A BIT* rewire changes the cost-to-come of the full subtree.
                delta = candidate_cost - old_cost
                stack = list(children.get(target, ()))
                while stack:
                    descendant = stack.pop()
                    g[descendant] += delta
                    push_vertex(descendant)
                    stack.extend(children.get(descendant, ()))
                push_vertex(target)
            if target == 1:
                best_cost = candidate_cost
                if solution_found_at is None:
                    solution_found_at = time.monotonic()
            elif math.isfinite(g.get(1, math.inf)):
                best_cost = g[1]

        if not math.isfinite(best_cost) or 1 not in parent:
            raise PlannerFailure("no_path", "BIT* found no path inside its budget")
        path_ids = [1]
        while path_ids[-1] != 0:
            path_ids.append(parent[path_ids[-1]])
        path_ids.reverse()
        return _result(
            problem,
            [points[node].tolist() for node in path_ids],
            best_cost,
            started,
            expanded,
            algorithm="BIT* (Gammell, Srinivasa & Barfoot, 2015)",
            sample_count=len(points),
            batch_count=batches,
            separate_vertex_and_edge_queues=True,
            informed_sampling=True,
            pruning=True,
            rewiring=True,
            anytime=True,
            first_solution_early_stop=True,
            post_solution_refinement_seconds=post_solution_refinement_s,
        )


BACKEND_TYPES = (
    AStarBackend,
    LazyThetaBackend,
    DStarLiteBackend,
    ADStarBackend,
    HybridAStarBackend,
    StateLatticeAStarBackend,
    MPPIBackend,
    BITStarBackend,
)


def make_backends(
    motion_limits: RobotMotionLimits | None = None,
) -> dict[str, PlannerBackend]:
    global _NUMBA_WARMED
    if njit is not None and not _NUMBA_WARMED:
        # Keep one-time JIT compilation outside the first measured online plan.
        _line_cost_numba(
            np.zeros((2, 2), dtype=np.bool_),
            np.ones((2, 2), dtype=np.float64),
            0,
            0,
            1,
            1,
        )
        _NUMBA_WARMED = True
    limits = motion_limits or RobotMotionLimits()
    backends = {backend.key: backend for backend in (kind() for kind in BACKEND_TYPES)}
    for backend in backends.values():
        if isinstance(
            backend,
            (HybridAStarBackend, StateLatticeAStarBackend, MPPIBackend),
        ):
            backend.motion_limits = limits
    return backends



def _select_engineered_subgoal(
    *,
    semantic: np.ndarray,
    occupancy_probability: np.ndarray,
    navigation_confidence: np.ndarray,
    extent_m: float,
    requested_target_metric_m: np.ndarray,
    inflation_radius_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair a local target using only the predicted BEV.

    The selector clips remote goals, computes the predicted-free component
    connected to the robot, and chooses its safest cell closest to the clipped
    target.  This is an intentional engineering enhancement shared by every
    backend; it is not claimed as part of any native planning algorithm.
    """

    semantic = np.asarray(semantic, dtype=np.uint8)
    occupancy = np.asarray(occupancy_probability, dtype=np.float32)
    confidence = np.asarray(navigation_confidence, dtype=np.float32)
    size = int(semantic.shape[0])
    cell_size_m = float(extent_m) / size
    radius_cells = int(math.ceil(max(0.0, inflation_radius_m) / cell_size_m))
    inflated = ndimage.binary_dilation(
        semantic == OCCUPIED_VALUE, structure=_disk(radius_cells)
    )
    blocked = (semantic != FREE_VALUE) | inflated
    free = np.argwhere(~blocked)
    if not len(free):
        raise PlannerFailure("no_free_space", "predicted BEV has no free cell")

    rows = free[:, 0].astype(np.float64)
    columns = free[:, 1].astype(np.float64)
    right = -extent_m / 2.0 + (columns + 0.5) * cell_size_m
    forward = extent_m / 2.0 - (rows + 0.5) * cell_size_m
    origin_distance = np.hypot(right, forward)
    anchor_index = int(np.argmin(origin_distance))
    if float(origin_distance[anchor_index]) > 0.20:
        raise PlannerFailure(
            "start_not_free",
            "no predicted-free cell is close enough to the robot",
            distance_m=float(origin_distance[anchor_index]),
        )
    start = tuple(map(int, free[anchor_index]))
    seed = np.zeros_like(blocked, dtype=bool)
    seed[start] = True
    reachable = ndimage.binary_propagation(
        seed,
        structure=ndimage.generate_binary_structure(2, 1),
        mask=~blocked,
    )
    pixels = np.argwhere(reachable)
    if len(pixels) <= 1:
        raise PlannerFailure("no_path", "predicted free component is empty")

    distance = float(np.linalg.norm(requested_target_metric_m))
    horizon_m = float(extent_m) * 0.45
    desired = requested_target_metric_m.copy()
    if distance > horizon_m:
        desired *= horizon_m / distance
    pr = pixels[:, 0].astype(np.float64)
    pc = pixels[:, 1].astype(np.float64)
    candidate_right = -extent_m / 2.0 + (pc + 0.5) * cell_size_m
    candidate_forward = extent_m / 2.0 - (pr + 0.5) * cell_size_m
    candidate_radius = np.hypot(candidate_right, candidate_forward)
    minimum_progress = min(0.20, max(0.06, float(np.linalg.norm(desired)) * 0.20))
    eligible = (candidate_radius >= minimum_progress) & (candidate_radius <= horizon_m)
    if not np.any(eligible):
        eligible = candidate_radius <= horizon_m
    pixels = pixels[eligible]
    candidate_right = candidate_right[eligible]
    candidate_forward = candidate_forward[eligible]
    distance_to_goal = np.hypot(
        candidate_right - float(desired[0]),
        candidate_forward - float(desired[1]),
    )
    clearance = ndimage.distance_transform_edt(~blocked) * cell_size_m
    candidate_clearance = clearance[pixels[:, 0], pixels[:, 1]]
    candidate_occupancy = np.clip(
        occupancy[pixels[:, 0], pixels[:, 1]], 0.0, 1.0
    )
    candidate_confidence = np.clip(
        confidence[pixels[:, 0], pixels[:, 1]], 0.0, 1.0
    )
    score = (
        distance_to_goal
        + 0.06 * candidate_occupancy
        + 0.05 * (1.0 - candidate_confidence)
        + 0.03 * np.exp(-candidate_clearance / 0.10)
    )
    selected = int(np.argmin(score))
    result = np.asarray(
        [candidate_right[selected], candidate_forward[selected]], dtype=np.float64
    )
    repair_distance = float(np.linalg.norm(result - desired))
    return result, {
        "subgoal_selection": "predicted_reachable_risk_aware",
        "requested_target_distance_m": distance,
        "local_horizon_m": horizon_m,
        "desired_clipped_target_metric_m": desired.tolist(),
        "planned_goal_repaired": repair_distance > cell_size_m * 1.5,
        "planned_goal_repair_distance_m": repair_distance,
        "selected_goal_clearance_m": float(candidate_clearance[selected]),
        "reachable_candidate_count": int(len(pixels)),
    }


def _lift_graph_path_for_diffdrive(
    path: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[float]]:
    lifted = [[float(path[0][0]), float(path[0][1])]]
    headings = [0.0]
    current_heading = 0.0
    for first, second in zip(path, path[1:]):
        dr = float(second[0] - first[0])
        dc = float(second[1] - first[1])
        segment_heading = -math.atan2(dc, -dr)
        heading_error = math.atan2(
            math.sin(segment_heading - current_heading),
            math.cos(segment_heading - current_heading),
        )
        if abs(heading_error) > 1e-6:
            lifted.append([float(first[0]), float(first[1])])
            headings.append(segment_heading)
        lifted.append([float(second[0]), float(second[1])])
        headings.append(segment_heading)
        current_heading = segment_heading
    return lifted, headings


def plan_engineered(
    backend: PlannerBackend,
    *,
    semantic: np.ndarray,
    occupancy_probability: np.ndarray,
    navigation_confidence: np.ndarray,
    extent_m: float,
    target_metric_m: Sequence[float],
    inflation_radius_m: float,
    budget_s: float,
    allow_cross_backend_fallback: bool = True,
) -> tuple[PlanResult, GridProblem, list[float]]:
    target = np.asarray(target_metric_m, dtype=np.float64)
    distance = float(np.linalg.norm(target))
    if target.shape != (2,) or not np.isfinite(target).all() or distance <= 1e-9:
        raise PlannerFailure("invalid_target", "target must be a finite nonzero point")
    horizon_m = float(extent_m) * 0.45
    if distance > horizon_m:
        target = target * (horizon_m / distance)
        selection_name = "radial_metric_horizon_clip"
    else:
        selection_name = "direct_metric_goal"
    selection_details: dict[str, Any] = {
        "subgoal_selection": selection_name,
        "planned_goal_repaired": False,
        "planned_goal_repair_distance_m": 0.0,
        "reachable_target_repair_computed": False,
    }
    candidate = target.tolist()
    try:
        problem = build_problem(
            semantic=semantic,
            occupancy_probability=occupancy_probability,
            navigation_confidence=navigation_confidence,
            extent_m=extent_m,
            target_metric_m=candidate,
            inflation_radius_m=inflation_radius_m,
        )
    except PlannerFailure as error:
        if error.code != "target_blocked":
            raise
        target, selection_details = _select_engineered_subgoal(
            semantic=semantic,
            occupancy_probability=occupancy_probability,
            navigation_confidence=navigation_confidence,
            extent_m=extent_m,
            requested_target_metric_m=np.asarray(target_metric_m, dtype=np.float64),
            inflation_radius_m=inflation_radius_m,
        )
        selection_details["reachable_target_repair_computed"] = True
        candidate = target.tolist()
        problem = build_problem(
            semantic=semantic,
            occupancy_probability=occupancy_probability,
            navigation_confidence=navigation_confidence,
            extent_m=extent_m,
            target_metric_m=candidate,
            inflation_radius_m=inflation_radius_m,
        )
    planning_started = time.monotonic()
    primary_failure: PlannerFailure | None = None
    try:
        result = backend.plan(problem, budget_s=budget_s)
    except PlannerFailure as error:
        primary_failure = error
        if not allow_cross_backend_fallback:
            raise PlannerFailure(
                error.code,
                str(error),
                **error.details,
                primary_algorithm=backend.label,
                engineering_profile="realtime_reliable_v2_no_cross_backend_fallback",
                cross_backend_fallback=False,
                planned_subgoal_metric_m=[float(x) for x in candidate],
                **selection_details,
            ) from error
        fallback_deadline = time.monotonic() + max(0.20, min(0.60, budget_s))
        try:
            path, cost, expanded = weighted_astar(
                problem, heuristic_weight=1.0, deadline=fallback_deadline
            )
        except PlannerFailure as fallback_error:
            if (
                fallback_error.code != "no_path"
                or selection_details.get("reachable_target_repair_computed")
            ):
                raise
            target, selection_details = _select_engineered_subgoal(
                semantic=semantic,
                occupancy_probability=occupancy_probability,
                navigation_confidence=navigation_confidence,
                extent_m=extent_m,
                requested_target_metric_m=np.asarray(target_metric_m, dtype=np.float64),
                inflation_radius_m=inflation_radius_m,
            )
            selection_details["reachable_target_repair_computed"] = True
            candidate = target.tolist()
            problem = build_problem(
                semantic=semantic,
                occupancy_probability=occupancy_probability,
                navigation_confidence=navigation_confidence,
                extent_m=extent_m,
                target_metric_m=candidate,
                inflation_radius_m=inflation_radius_m,
            )
            path, cost, expanded = weighted_astar(
                problem,
                heuristic_weight=1.0,
                deadline=time.monotonic() + max(0.20, min(0.60, budget_s)),
            )
        headings: list[float] | None = None
        output_path: Sequence[Sequence[float]] = path
        if isinstance(backend, (HybridAStarBackend, StateLatticeAStarBackend)):
            output_path, headings = _lift_graph_path_for_diffdrive(path)
        result = _result(
            problem,
            output_path,
            cost,
            planning_started,
            expanded,
            path_headings_rad=headings,
            algorithm=f"{backend.label} + shared reliability fallback",
            primary_algorithm=backend.label,
            primary_failure_code=error.code,
            primary_failure_message=str(error),
            engineering_fallback_used=True,
            fallback_algorithm="Risk-aware A*",
            fallback_path_lifted_to_differential_drive=headings is not None,
        )
    result.backend_details.update(
        requested_target_metric_m=[float(x) for x in target_metric_m],
        planned_subgoal_metric_m=[float(x) for x in candidate],
        **selection_details,
        engineering_profile="realtime_reliable_v2",
        engineering_fallback_used=bool(primary_failure),
        cross_backend_fallback=bool(primary_failure),
        risk_cost=(
            "length * [1 + 2*Pocc + 1.5*(1-confidence) + "
            "1.5*exp(-clearance/0.15m)]"
        ),
    )
    return result, problem, candidate


# Compatibility name used by older tests and launchers. The implementation is
# now explicitly the engineering-enhanced policy above.
plan_direct = plan_engineered
