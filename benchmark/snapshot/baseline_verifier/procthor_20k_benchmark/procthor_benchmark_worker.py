#!/usr/bin/env python3
"""Resumable ProcTHOR oracle-localized local-navigation benchmark.

Short and long paths are independent task pools.  Every attempt uniformly
samples one of all 10,000 official train houses with replacement.  Long tasks
freeze one privileged GT A* route into fixed local endpoints; every method
receives only that current endpoint. Point-conditioned methods receive its
ego-local metric coordinates, while NoMaD receives its native ImageGoal RGB.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import heapq
import io
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if not (ROOT / "planner_backends.py").is_file():
    # Local staging layout; deployments place planner_backends.py beside this
    # worker so the benchmark remains self-contained on each server.
    sys.path.insert(0, str(ROOT / "source_5090"))

from planner_backends import (  # noqa: E402
    BACKEND_TYPES,
    MPPIBackend,
    PlannerFailure,
    RobotMotionLimits,
    make_backends,
    plan_engineered,
)


SEED = 20260829
TRAIN_HOUSES = 10_000
SHARD_COUNT = 8
TARGET_PATHS_PER_KIND_PER_SHARD = 10_000 // SHARD_COUNT
GRID_SIZE_M = 0.25
CAMERA_HEIGHT_M = 0.55
HORIZONTAL_FOV_DEGREES = 90.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SUCCESS_RADIUS_M = 0.20
SAFETY_MARGIN_M = 0.10
ROBOT_SIDE_M = 0.10
TOTAL_INFLATION_M = SAFETY_MARGIN_M + ROBOT_SIDE_M / math.sqrt(2.0)
ROBOT_COLLISION_RADIUS_M = ROBOT_SIDE_M / math.sqrt(2.0)
GEOMETRY_VOXEL_SIZE_M = 0.025
GEOMETRY_OBSTACLE_MIN_HEIGHT_M = 0.03
GEOMETRY_OBSTACLE_MAX_HEIGHT_M = 1.40
GEOMETRY_CACHE_SCHEMA = "procthor-complete-geometry-collision-v1"
GEOMETRIC_PATH_SAMPLE_SPACING_M = GEOMETRY_VOXEL_SIZE_M / 2.0
BEV_INFERENCE_HZ = 3.0
MOTION_LIMITS = RobotMotionLimits(
    minimum_linear_velocity_m_s=-0.6,
    maximum_linear_velocity_m_s=0.6,
    minimum_linear_acceleration_m_s2=-0.6,
    maximum_linear_acceleration_m_s2=0.6,
    minimum_angular_velocity_rad_s=-1.0,
    maximum_angular_velocity_rad_s=1.0,
    minimum_angular_acceleration_rad_s2=-1.0,
    maximum_angular_acceleration_rad_s2=1.0,
    track_width_m=0.10,
)
PLANNER_BUDGET_SECONDS = {
    "astar": 0.35,
    "lazy_theta": 0.40,
    "dstar_lite": 0.40,
    "adstar": 0.40,
    "hybrid_astar": 0.55,
    "state_lattice_astar": 0.65,
    "mppi": 0.40,
    "bitstar": 0.55,
}
EXTERNAL_METHOD_HZ = {
    "straight_line": 10.0,
    "limo_tel": 5.0,
    "limo_aug": 5.0,
    "omnivla": 3.0,
    "mbra_logonav": 5.0,
    "nomad": 4.0,
    "genie_samtp": 5.0,
}
EXTERNAL_METHODS = tuple(EXTERNAL_METHOD_HZ)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def post_json(url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if result.get("success") is False or "error" in result:
        raise RuntimeError(str(result.get("error", "runtime request failed")))
    return result


def encode_jpeg(rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=90
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_png(encoded: str, probability: bool = False) -> np.ndarray:
    image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
    array = np.asarray(image)
    if probability:
        return array.astype(np.float32) / 65535.0
    return array.astype(np.uint8)


def distance_xz(first: Sequence[float], second: Sequence[float]) -> float:
    return float(math.hypot(float(first[0]) - float(second[0]), float(first[2]) - float(second[2])))


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def local_target(position: Sequence[float], yaw_degrees: float, goal: Sequence[float]) -> list[float]:
    yaw = math.radians(float(yaw_degrees))
    delta_x = float(goal[0]) - float(position[0])
    delta_z = float(goal[2]) - float(position[2])
    right = math.cos(yaw) * delta_x - math.sin(yaw) * delta_z
    forward = math.sin(yaw) * delta_x + math.cos(yaw) * delta_z
    return [right, forward]


def path_length(points: Sequence[Sequence[float]]) -> float:
    return float(sum(distance_xz(a, b) for a, b in zip(points, points[1:])))


def path_length_2d(points: Sequence[Sequence[float]]) -> float:
    return float(
        sum(
            math.hypot(
                float(second[0]) - float(first[0]),
                float(second[1]) - float(first[1]),
            )
            for first, second in zip(points, points[1:])
        )
    )


def local_point_to_world(
    position: Sequence[float], yaw_degrees: float, point: Sequence[float]
) -> list[float]:
    """Convert one [right, forward] point without consulting simulator GT."""

    yaw = math.radians(float(yaw_degrees))
    right, forward = float(point[0]), float(point[1])
    return [
        float(position[0]) + math.cos(yaw) * right + math.sin(yaw) * forward,
        float(position[1]),
        float(position[2]) - math.sin(yaw) * right + math.cos(yaw) * forward,
    ]


def executable_path_prefix(
    path: Sequence[Sequence[float]], maximum_arc_m: float
) -> tuple[list[list[float]], float]:
    """Return the native path prefix physically reachable before next replan.

    No smoothing, shortcutting or waypoint downsampling is performed.  The
    only inserted point is the exact arc-length boundary at the execution
    horizon.
    """

    if maximum_arc_m <= 0:
        raise ValueError("maximum executable arc must be positive")
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 1:
        raise PlannerFailure("planner returned no geometric path")
    if not np.isfinite(values).all():
        raise PlannerFailure("planner returned a non-finite geometric path")
    points = [[float(value[0]), float(value[1])] for value in values]
    if math.hypot(points[0][0], points[0][1]) > 1e-6:
        points.insert(0, [0.0, 0.0])
    compact = [points[0]]
    for point in points[1:]:
        if math.hypot(
            point[0] - compact[-1][0], point[1] - compact[-1][1]
        ) > 1e-8:
            compact.append(point)
    if len(compact) < 2:
        raise PlannerFailure("planner returned a zero-length geometric path")
    output = [compact[0]]
    accumulated = 0.0
    for first, second in zip(compact, compact[1:]):
        edge = math.hypot(second[0] - first[0], second[1] - first[1])
        remaining = maximum_arc_m - accumulated
        if edge <= remaining + 1e-12:
            output.append(second)
            accumulated += edge
            if accumulated >= maximum_arc_m - 1e-12:
                break
            continue
        ratio = remaining / edge
        output.append(
            [
                first[0] + ratio * (second[0] - first[0]),
                first[1] + ratio * (second[1] - first[1]),
            ]
        )
        accumulated = maximum_arc_m
        break
    return output, accumulated


def constant_control_path(
    forward_m_s: float,
    angular_left_rad_s: float,
    duration_s: float,
    *,
    spacing_m: float = GEOMETRIC_PATH_SAMPLE_SPACING_M,
) -> tuple[list[list[float]], float]:
    """Integrate a differential-drive command in ego [right, forward]."""

    velocity = float(np.clip(forward_m_s, -0.6, 0.6))
    angular = float(np.clip(angular_left_rad_s, -1.0, 1.0))
    duration = max(0.0, float(duration_s))
    arc = abs(velocity) * duration
    steps = max(1, int(math.ceil(arc / max(spacing_m, 1e-4))))
    points = [[0.0, 0.0]]
    for index in range(1, steps + 1):
        elapsed = duration * index / steps
        if abs(angular) <= 1e-8:
            right, forward = 0.0, velocity * elapsed
        else:
            right = velocity / angular * (math.cos(angular * elapsed) - 1.0)
            forward = velocity / angular * math.sin(angular * elapsed)
        points.append([right, forward])
    # Simulator yaw is clockwise-positive, whereas angular_left is CCW.
    return points, -math.degrees(angular * duration)


@dataclass
class CollisionCheck:
    safe: bool
    checked_samples: int
    minimum_clearance_m: float
    collision_world: list[float] | None = None


@dataclass
class GroundTruthGeometry:
    """Evaluator-only complete simulator geometry; never planner input."""

    truth: np.ndarray
    lower_bound: np.ndarray
    voxel_size_m: float
    clearance_m: np.ndarray

    @classmethod
    def create(
        cls, truth: np.ndarray, lower_bound: np.ndarray, voxel_size_m: float
    ) -> "GroundTruthGeometry":
        from scipy import ndimage

        truth = np.asarray(truth, dtype=np.uint8)
        if truth.ndim != 2 or not truth.size:
            raise RuntimeError("invalid complete simulator truth raster")
        lower_bound = np.asarray(lower_bound, dtype=np.float64)
        if lower_bound.shape != (3,):
            raise RuntimeError("invalid complete simulator truth lower bound")
        clearance = ndimage.distance_transform_edt(truth == 255) * float(
            voxel_size_m
        )
        return cls(truth, lower_bound, float(voxel_size_m), clearance)

    def _clearance_at(self, point: Sequence[float]) -> float:
        column = int(
            round((float(point[0]) - float(self.lower_bound[0])) / self.voxel_size_m)
        )
        unflipped_row = int(
            round((float(point[2]) - float(self.lower_bound[2])) / self.voxel_size_m)
        )
        row = self.truth.shape[0] - 1 - unflipped_row
        if not (0 <= row < self.truth.shape[0] and 0 <= column < self.truth.shape[1]):
            return 0.0
        return float(self.clearance_m[row, column])

    def check_path(
        self,
        points: Sequence[Sequence[float]],
        *,
        robot_radius_m: float = ROBOT_COLLISION_RADIUS_M,
        sample_spacing_m: float = GEOMETRIC_PATH_SAMPLE_SPACING_M,
    ) -> CollisionCheck:
        if not points:
            return CollisionCheck(False, 0, 0.0, None)
        samples: list[list[float]] = [list(map(float, points[0]))]
        for first, second in zip(points, points[1:]):
            length = distance_xz(first, second)
            count = max(1, int(math.ceil(length / max(sample_spacing_m, 1e-4))))
            for index in range(1, count + 1):
                ratio = index / count
                samples.append(
                    [
                        float(first[0]) + ratio * (float(second[0]) - float(first[0])),
                        float(first[1]) + ratio * (float(second[1]) - float(first[1])),
                        float(first[2]) + ratio * (float(second[2]) - float(first[2])),
                    ]
                )
        minimum = math.inf
        for sample in samples:
            clearance = self._clearance_at(sample)
            minimum = min(minimum, clearance)
            if clearance + 1e-9 < robot_radius_m:
                return CollisionCheck(False, len(samples), minimum, sample)
        return CollisionCheck(True, len(samples), minimum, None)


def distance_to_polyline_xz(
    point: Sequence[float], polyline: Sequence[Sequence[float]]
) -> float:
    if not polyline:
        return math.inf
    if len(polyline) == 1:
        return distance_xz(point, polyline[0])
    query = np.asarray([float(point[0]), float(point[2])], dtype=np.float64)
    best = math.inf
    for first, second in zip(polyline, polyline[1:]):
        a = np.asarray([float(first[0]), float(first[2])], dtype=np.float64)
        b = np.asarray([float(second[0]), float(second[2])], dtype=np.float64)
        direction = b - a
        denominator = float(direction @ direction)
        ratio = 0.0 if denominator <= 1e-12 else float((query - a) @ direction / denominator)
        projection = a + np.clip(ratio, 0.0, 1.0) * direction
        best = min(best, float(np.linalg.norm(query - projection)))
    return best


@dataclass
class ReachableGraph:
    points: list[list[float]]
    keys: list[tuple[int, int]]
    by_key: dict[tuple[int, int], int]
    edges: list[list[tuple[int, float]]]

    @classmethod
    def build(cls, reachable: Sequence[dict[str, Any]]) -> "ReachableGraph":
        candidates: dict[tuple[int, int], list[float]] = {}
        for point in reachable:
            normalized = [float(point["x"]), float(point["y"]), float(point["z"])]
            key = (round(normalized[0] / GRID_SIZE_M), round(normalized[2] / GRID_SIZE_M))
            candidates.setdefault(key, normalized)
        keys = sorted(candidates)
        points = [candidates[key] for key in keys]
        by_key = {key: index for index, key in enumerate(keys)}
        edges: list[list[tuple[int, float]]] = [[] for _ in points]
        for index, key in enumerate(keys):
            for dx, dz in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ):
                neighbor = by_key.get((key[0] + dx, key[1] + dz))
                if neighbor is None:
                    continue
                # Diagonal motion is valid only when it cannot cut through a
                # blocked grid corner.  This gives a materially tighter GT
                # shortest-path reference than a Manhattan-only graph while
                # remaining conservative with respect to AI2-THOR reachability.
                if dx and dz and (
                    (key[0] + dx, key[1]) not in by_key
                    or (key[0], key[1] + dz) not in by_key
                ):
                    continue
                length = distance_xz(points[index], points[neighbor])
                maximum = (math.sqrt(2.0) + 0.15) * GRID_SIZE_M
                if 0.5 * GRID_SIZE_M <= length <= maximum:
                    edges[index].append((neighbor, length))
        graph = cls(points=points, keys=keys, by_key=by_key, edges=edges)
        if not points or max((len(component) for component in graph.components()), default=0) < 20:
            raise RuntimeError("reachable graph has no useful connected component")
        return graph

    def components(self) -> list[list[int]]:
        unseen = set(range(len(self.points)))
        output: list[list[int]] = []
        while unseen:
            seed = unseen.pop()
            component = [seed]
            queue = deque([seed])
            while queue:
                current = queue.popleft()
                for neighbor, _ in self.edges[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.append(neighbor)
                        queue.append(neighbor)
            output.append(component)
        return output

    def nearest(self, point: Sequence[float]) -> int:
        key = (round(float(point[0]) / GRID_SIZE_M), round(float(point[2]) / GRID_SIZE_M))
        direct = self.by_key.get(key)
        if direct is not None:
            return direct
        return min(range(len(self.points)), key=lambda i: distance_xz(self.points[i], point))

    def dijkstra(self, start: int) -> tuple[list[float], list[int | None]]:
        distances = [math.inf] * len(self.points)
        parent: list[int | None] = [None] * len(self.points)
        distances[start] = 0.0
        queue: list[tuple[float, int]] = [(0.0, start)]
        while queue:
            current_distance, current = heapq.heappop(queue)
            if current_distance != distances[current]:
                continue
            for neighbor, edge in self.edges[current]:
                candidate = current_distance + edge
                if candidate + 1e-9 < distances[neighbor]:
                    distances[neighbor] = candidate
                    parent[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
        return distances, parent

    def shortest(self, start: int, goal: int) -> tuple[list[int], float]:
        distances = [math.inf] * len(self.points)
        parent: list[int | None] = [None] * len(self.points)
        distances[start] = 0.0
        goal_point = self.points[goal]
        queue: list[tuple[float, float, int]] = [(distance_xz(self.points[start], goal_point), 0.0, start)]
        while queue:
            _priority, current_distance, current = heapq.heappop(queue)
            if current_distance != distances[current]:
                continue
            if current == goal:
                break
            for neighbor, edge in self.edges[current]:
                candidate = current_distance + edge
                if candidate + 1e-9 < distances[neighbor]:
                    distances[neighbor] = candidate
                    parent[neighbor] = current
                    heuristic = distance_xz(self.points[neighbor], goal_point)
                    heapq.heappush(queue, (candidate + heuristic, candidate, neighbor))
        if not math.isfinite(distances[goal]):
            raise RuntimeError("GT graph has no path")
        path = []
        cursor: int | None = goal
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()
        return path, distances[goal]

    def waypoint(self, current_position: Sequence[float], goal: int, lookahead_m: float = 2.5) -> tuple[list[float], float]:
        indices, remaining = self.shortest(self.nearest(current_position), goal)
        travelled = 0.0
        selected = indices[-1]
        for first, second in zip(indices, indices[1:]):
            travelled += distance_xz(self.points[first], self.points[second])
            selected = second
            if travelled >= lookahead_m:
                break
        return self.points[selected], remaining


def reconstruct(parent: Sequence[int | None], goal: int) -> list[int]:
    output = []
    cursor: int | None = goal
    while cursor is not None:
        output.append(cursor)
        cursor = parent[cursor]
    output.reverse()
    return output


def _grid_line_is_navigable(
    graph: ReachableGraph, first_index: int, second_index: int
) -> bool:
    """Conservatively validate a straight chord on the GT reachable grid."""

    first = graph.keys[first_index]
    second = graph.keys[second_index]
    span = max(abs(second[0] - first[0]), abs(second[1] - first[1]))
    if span == 0:
        return True
    # Four samples per grid interval avoid skipping a cell at steep slopes.
    for sample in range(4 * span + 1):
        ratio = sample / (4 * span)
        key = (
            round(first[0] + ratio * (second[0] - first[0])),
            round(first[1] + ratio * (second[1] - first[1])),
        )
        if key not in graph.by_key:
            return False
    return True


def segment_gt_route(
    graph: ReachableGraph,
    path_indices: Sequence[int],
    *,
    maximum_arc_m: float = 2.0,
) -> list[int]:
    """Freeze a GT route into local, collision-free waypoint chords.

    The upper planner is allowed GT.  A local method receives only the next
    endpoint, never this route, the GT graph, or future endpoints.
    """

    if len(path_indices) < 2:
        raise RuntimeError("GT route is too short to segment")
    anchors = [0]
    cursor = 0
    while cursor < len(path_indices) - 1:
        arc = 0.0
        best = cursor + 1
        candidate = cursor + 1
        while candidate < len(path_indices):
            arc += distance_xz(
                graph.points[path_indices[candidate - 1]],
                graph.points[path_indices[candidate]],
            )
            if arc > maximum_arc_m + 1e-9:
                break
            if _grid_line_is_navigable(
                graph, path_indices[cursor], path_indices[candidate]
            ):
                best = candidate
            candidate += 1
        if best <= cursor:
            best = cursor + 1
        anchors.append(best)
        cursor = best
    if anchors[-1] != len(path_indices) - 1:
        anchors.append(len(path_indices) - 1)
    return [path_indices[index] for index in anchors[1:]]


def sample_task(
    graph: ReachableGraph,
    rng: random.Random,
    scene_index: int,
    attempt_index: int,
    path_kind: str,
) -> dict[str, Any]:
    if path_kind not in {"short", "long"}:
        raise ValueError(f"invalid path kind: {path_kind}")
    component = max(graph.components(), key=len)
    starts = list(component)
    rng.shuffle(starts)
    for start in starts[: min(128, len(starts))]:
        distances, parent = graph.dijkstra(start)
        if path_kind == "short":
            candidates = [
                index
                for index in component
                if 1.5 <= distances[index] <= 3.0
                and distance_xz(graph.points[start], graph.points[index]) <= 3.0
            ]
        else:
            candidates = [
                index
                for index in component
                if math.isfinite(distances[index]) and distances[index] >= 5.0
            ]
        if not candidates:
            continue
        if path_kind == "long":
            candidates.sort(key=lambda index: distances[index], reverse=True)
            candidates = candidates[: max(1, min(20, len(candidates)))]
        goal = rng.choice(candidates)
        route = reconstruct(parent, goal)
        start_point = graph.points[start]
        heading_point = graph.points[route[min(1, len(route) - 1)]]
        yaw = math.degrees(
            math.atan2(
                heading_point[0] - start_point[0],
                heading_point[2] - start_point[2],
            )
        )
        yaw = (yaw + rng.uniform(-20.0, 20.0)) % 360.0
        subgoal_indices = (
            [goal]
            if path_kind == "short"
            else segment_gt_route(graph, route, maximum_arc_m=2.0)
        )
        return {
            "schema": "procthor-oracle-localized-nav-task-v2",
            "attempt_index": attempt_index,
            "scene_index": scene_index,
            "path_kind": path_kind,
            "start_index": start,
            "start_world": start_point,
            "start_yaw_degrees": yaw,
            "goal_index": goal,
            "goal_world": graph.points[goal],
            "gt_shortest_path_m": distances[goal],
            "gt_path_world": [graph.points[index] for index in route],
            "subgoals_world": [graph.points[index] for index in subgoal_indices],
            "subgoal_count": len(subgoal_indices),
            "target_service": (
                "one current frozen-route endpoint; ego-local metric PointGoal "
                "for point-conditioned methods or endpoint RGB for NoMaD"
            ),
            "upper_guide": (
                "direct final point"
                if path_kind == "short"
                else "one frozen GT A* route; <=2.0 m fixed visible chords; no online global replan"
            ),
            "oracle_permissions": {
                "gt_pose_used_only_by_common_target_service": True,
                "gt_route_visible_to_local_method": False,
                "predicted_extrinsic_target_tracking": False,
            },
        }
    raise RuntimeError(f"scene cannot provide a valid {path_kind} path")


def candidate_scenes(
    shard_id: int, attempt_index: int, path_kind: str, count: int = 24
) -> list[int]:
    """Uniform independent draws from all 10K scenes, with no scene quotas."""

    kind_salt = 0x51A7 if path_kind == "short" else 0x10A9
    rng = random.Random(
        SEED ^ kind_salt ^ (shard_id * 1_000_003) ^ (attempt_index * 9_176)
    )
    # The first draw defines the task's random scene. Remaining independent
    # draws are deterministic load/path fallbacks, not ordered shard ranges.
    output: list[int] = []
    while len(output) < min(count, TRAIN_HOUSES):
        candidate = rng.randrange(TRAIN_HOUSES)
        if candidate not in output:
            output.append(candidate)
    return output


def load_houses(dataset_dir: Path, indices: set[int]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    with gzip.open(dataset_dir / "train.jsonl.gz", "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in indices:
                output[index] = json.loads(line)
                if len(output) == len(indices):
                    break
    missing = indices - output.keys()
    if missing:
        raise RuntimeError(f"missing ProcTHOR houses: {sorted(missing)[:10]}")
    return output


def vertical_fov(horizontal_degrees: float, width: int, height: int) -> float:
    horizontal = math.radians(horizontal_degrees)
    return math.degrees(2.0 * math.atan(math.tan(horizontal / 2.0) * height / width))


def load_or_build_scene_geometry(
    *,
    controller: Any,
    house: dict[str, Any],
    scene_index: int,
    floor_y: float,
    cache_root: Path,
) -> tuple[GroundTruthGeometry, float, bool, dict[str, Any]]:
    """Build complete GT once and atomically share it across all five shards."""

    from procthor_bev import build_complete_truth

    started = time.monotonic()
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_root = cache_root / ".locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    stem = (
        f"scene_{scene_index:05d}_v{round(GEOMETRY_VOXEL_SIZE_M * 1000):03d}_"
        f"h{round(GEOMETRY_OBSTACLE_MIN_HEIGHT_M * 100):03d}_"
        f"{round(GEOMETRY_OBSTACLE_MAX_HEIGHT_M * 100):03d}"
    )
    cache_path = cache_root / f"{stem}.npz"
    lock_path = lock_root / f"{stem}.lock"
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as payload:
                    schema = str(payload["schema"].item())
                    cached_floor = float(payload["floor_y"].item())
                    voxel_size = float(payload["voxel_size_m"].item())
                    if schema != GEOMETRY_CACHE_SCHEMA:
                        raise RuntimeError(f"unsupported geometry cache schema {schema}")
                    if abs(cached_floor - floor_y) > 0.02:
                        raise RuntimeError("geometry cache floor does not match scene")
                    if abs(voxel_size - GEOMETRY_VOXEL_SIZE_M) > 1e-9:
                        raise RuntimeError("geometry cache voxel size does not match")
                    truth = np.asarray(payload["truth"], dtype=np.uint8)
                    lower_bound = np.asarray(payload["lower_bound"], dtype=np.float64)
                    statistics_value = json.loads(str(payload["statistics_json"].item()))
                return (
                    GroundTruthGeometry.create(truth, lower_bound, voxel_size),
                    time.monotonic() - started,
                    True,
                    statistics_value,
                )
            except Exception as error:
                print(
                    f"discarding invalid geometry cache {cache_path}: {error}",
                    flush=True,
                )
                cache_path.unlink(missing_ok=True)
        truth, _valid, lower_bound, statistics_value = build_complete_truth(
            controller,
            house,
            floor_y=floor_y,
            voxel_size=GEOMETRY_VOXEL_SIZE_M,
            obstacle_min_height=GEOMETRY_OBSTACLE_MIN_HEIGHT_M,
            obstacle_max_height=GEOMETRY_OBSTACLE_MAX_HEIGHT_M,
        )
        temporary = cache_path.with_name(cache_path.name + ".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray(GEOMETRY_CACHE_SCHEMA),
                floor_y=np.asarray(floor_y, dtype=np.float64),
                voxel_size_m=np.asarray(GEOMETRY_VOXEL_SIZE_M, dtype=np.float64),
                truth=np.asarray(truth, dtype=np.uint8),
                lower_bound=np.asarray(lower_bound, dtype=np.float64),
                statistics_json=np.asarray(json.dumps(statistics_value)),
            )
        temporary.replace(cache_path)
        return (
            GroundTruthGeometry.create(
                truth, lower_bound, GEOMETRY_VOXEL_SIZE_M
            ),
            time.monotonic() - started,
            False,
            statistics_value,
        )


class Scene:
    def __init__(
        self, house: dict[str, Any], args: argparse.Namespace, scene_index: int
    ) -> None:
        from procthor_collection_core import (
            assert_process_gpu_binding,
            dominant_floor_positions,
            floor_levels,
            isolated_cloud_controller,
        )

        started = time.monotonic()
        self.controller = None
        try:
            self.controller = isolated_cloud_controller(
                args.procthor_runtime_root,
                physical_gpu_index=args.gpu_index,
                scene=house,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fieldOfView=vertical_fov(HORIZONTAL_FOV_DEGREES, CAMERA_WIDTH, CAMERA_HEIGHT),
                agentMode="locobot",
                gridSize=GRID_SIZE_M,
                snapToGrid=False,
                rotateStepDegrees=1,
                renderDepthImage=False,
                makeAgentsVisible=False,
                quality="Low",
            )
            assert_process_gpu_binding(
                int(self.controller.unity_pid), args.resolved_nvidia_gpu_uuid
            )
            event = self.controller.step(
                action="GetReachablePositions", renderImage=False
            )
            if not event.metadata.get("lastActionSuccess", False):
                raise RuntimeError(
                    event.metadata.get("errorMessage", "GetReachablePositions failed")
                )
            self.floor_y, reachable = dominant_floor_positions(
                event.metadata.get("actionReturn") or [], floor_levels(house)
            )
            self.graph = ReachableGraph.build(reachable)
            first = self.graph.points[0]
            event = self.controller.step(
                action="AddThirdPartyCamera",
                position={
                    "x": first[0],
                    "y": self.floor_y + CAMERA_HEIGHT_M,
                    "z": first[2],
                },
                rotation={"x": 0.0, "y": 0.0, "z": 0.0},
                fieldOfView=vertical_fov(
                    HORIZONTAL_FOV_DEGREES, CAMERA_WIDTH, CAMERA_HEIGHT
                ),
                renderImage=False,
            )
            if not event.metadata.get("lastActionSuccess", False):
                raise RuntimeError("AddThirdPartyCamera failed")
            self.renderer_load_seconds = time.monotonic() - started
            (
                self.geometry,
                self.geometry_load_seconds,
                self.geometry_cache_hit,
                self.geometry_statistics,
            ) = load_or_build_scene_geometry(
                controller=self.controller,
                house=house,
                scene_index=scene_index,
                floor_y=self.floor_y,
                cache_root=args.geometry_cache_root,
            )
            self.load_seconds = time.monotonic() - started
        except Exception:
            if self.controller is not None:
                self.controller.stop()
            raise

    def stop(self) -> None:
        if self.controller is not None:
            self.controller.stop()

    def rgb(self, position: Sequence[float], yaw_degrees: float) -> np.ndarray:
        event = self.controller.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=0,
            position={
                "x": float(position[0]),
                "y": self.floor_y + CAMERA_HEIGHT_M,
                "z": float(position[2]),
            },
            rotation={"x": 0.0, "y": float(yaw_degrees) % 360.0, "z": 0.0},
            fieldOfView=vertical_fov(
                HORIZONTAL_FOV_DEGREES, CAMERA_WIDTH, CAMERA_HEIGHT
            ),
        )
        if (
            not event.metadata.get("lastActionSuccess", False)
            or not event.third_party_camera_frames
        ):
            raise RuntimeError("third-party RGB camera update failed")
        return np.asarray(
            event.third_party_camera_frames[0], dtype=np.uint8
        )[..., :3]

    def validate_task_geometry(self, task: dict[str, Any]) -> None:
        check = self.geometry.check_path(task["gt_path_world"])
        if not check.safe:
            raise RuntimeError(
                "AI2-THOR reachable GT route disagrees with complete geometry "
                f"at {check.collision_world}; minimum clearance "
                f"{check.minimum_clearance_m:.3f} m"
            )


@dataclass
class Motion:
    position: list[float]
    yaw_degrees: float
    forward_m_s: float = 0.0
    angular_left_rad_s: float = 0.0


@dataclass
class GeometricAdvance:
    safe: bool
    travelled_m: float
    world_path: list[list[float]]
    collision: CollisionCheck


def native_heading_at_arc(
    path: Sequence[Sequence[float]],
    headings_left_rad: Sequence[float] | None,
    arc_m: float,
) -> float | None:
    if headings_left_rad is None or len(headings_left_rad) != len(path):
        return None
    accumulated = 0.0
    selected = float(headings_left_rad[0])
    for index, (first, second) in enumerate(zip(path, path[1:]), start=1):
        edge = math.hypot(
            float(second[0]) - float(first[0]),
            float(second[1]) - float(first[1]),
        )
        selected = float(headings_left_rad[index])
        accumulated += edge
        if accumulated >= arc_m - 1e-9:
            break
    return selected


def apply_geometric_path(
    scene: Scene,
    motion: Motion,
    local_path: Sequence[Sequence[float]],
    *,
    maximum_arc_m: float,
    native_headings_left_rad: Sequence[float] | None = None,
    forced_yaw_delta_degrees: float | None = None,
) -> GeometricAdvance:
    """Evaluator-check one execution horizon, then advance without Teleport.

    The complete simulator geometry is deliberately consulted only after the
    model and planner have returned a path.  It can declare collision, but it
    cannot repair, truncate, steer, or trigger an early replan.
    """

    values = np.asarray(local_path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 1:
        raise PlannerFailure("missing_path", "planner returned no geometric path")
    spatial_length = path_length_2d(values.tolist())
    if spatial_length <= 1e-8:
        if forced_yaw_delta_degrees is None or abs(forced_yaw_delta_degrees) <= 1e-8:
            raise PlannerFailure("zero_path", "planner returned a zero-length path")
        world_path = [list(motion.position)]
        collision = scene.geometry.check_path(world_path)
        if collision.safe:
            motion.yaw_degrees = (
                motion.yaw_degrees + float(forced_yaw_delta_degrees)
            ) % 360.0
        return GeometricAdvance(collision.safe, 0.0, world_path, collision)

    prefix, travelled = executable_path_prefix(local_path, maximum_arc_m)
    world_path = [
        local_point_to_world(motion.position, motion.yaw_degrees, point)
        for point in prefix
    ]
    collision = scene.geometry.check_path(world_path)
    if not collision.safe:
        return GeometricAdvance(False, travelled, world_path, collision)

    starting_yaw = motion.yaw_degrees
    motion.position = list(world_path[-1])
    if forced_yaw_delta_degrees is not None:
        ratio = min(1.0, travelled / max(spatial_length, 1e-9))
        motion.yaw_degrees = (
            starting_yaw + float(forced_yaw_delta_degrees) * ratio
        ) % 360.0
    else:
        native_heading = native_heading_at_arc(
            local_path, native_headings_left_rad, travelled
        )
        if native_heading is not None:
            # Planner headings use CCW/left-positive; simulator yaw is
            # clockwise/right-positive.
            motion.yaw_degrees = (
                starting_yaw - math.degrees(native_heading)
            ) % 360.0
        else:
            tangent = np.asarray(prefix[-1]) - np.asarray(prefix[-2])
            motion.yaw_degrees = (
                starting_yaw
                + math.degrees(math.atan2(float(tangent[0]), float(tangent[1])))
            ) % 360.0
    return GeometricAdvance(True, travelled, world_path, collision)


@dataclass
class OracleTargetService:
    """The one localization/waypoint interface shared by every method.

    It owns the privileged GT world endpoints.  A local method can request
    only the current endpoint expressed as [right, forward] from its current
    simulator pose.  Predicted extrinsics, odometry, maps and future endpoints
    are deliberately absent from this API.
    """

    subgoals_world: list[list[float]]
    current_index: int = 0
    refresh_count: int = 0
    switch_count: int = 0

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> "OracleTargetService":
        subgoals = [list(map(float, point)) for point in task["subgoals_world"]]
        if not subgoals:
            raise RuntimeError("task contains no subgoals")
        return cls(subgoals_world=subgoals)

    @property
    def complete(self) -> bool:
        return self.current_index >= len(self.subgoals_world)

    @property
    def completed_subgoals(self) -> int:
        return min(self.current_index, len(self.subgoals_world))

    def advance_if_reached(self, position: Sequence[float]) -> bool:
        advanced = False
        # Duplicate anchors are not expected, but accepting all already-reached
        # anchors makes the immutable manifest robust without route projection.
        while not self.complete and distance_xz(
            position, self.subgoals_world[self.current_index]
        ) <= SUCCESS_RADIUS_M:
            self.current_index += 1
            self.switch_count += 1
            advanced = True
        return advanced

    def query(
        self, position: Sequence[float], yaw_degrees: float
    ) -> list[float]:
        if self.complete:
            raise RuntimeError("target service queried after final subgoal completion")
        self.refresh_count += 1
        world = self.subgoals_world[self.current_index]
        # Deliberately return no world point, pose, route index or future goal.
        # The inference-facing API contains exactly one ego-local metric point.
        return local_target(position, yaw_degrees, world)


def subgoal_image_yaw_degrees(task: dict[str, Any], subgoal_index: int) -> float:
    """Return the frozen-route camera heading for a NoMaD ImageGoal.

    NoMaD is natively conditioned on a goal image rather than a metric point.
    The target image is rendered at the *current* common pre-planned subgoal,
    facing along the frozen route. No future image or final-goal image is sent.
    """

    subgoals = task["subgoals_world"]
    if not 0 <= subgoal_index < len(subgoals):
        raise IndexError("subgoal image index is outside the frozen route")
    target = subgoals[subgoal_index]
    route = task["gt_path_world"]
    route_index = min(
        range(len(route)), key=lambda index: distance_xz(route[index], target)
    )
    if route_index + 1 < len(route):
        first, second = route[route_index], route[route_index + 1]
    elif route_index > 0:
        first, second = route[route_index - 1], route[route_index]
    else:
        return float(task["start_yaw_degrees"])
    return math.degrees(
        math.atan2(float(second[0]) - float(first[0]), float(second[2]) - float(first[2]))
    ) % 360.0


def decode_bev_response(response: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    semantic = decode_png(response["model_single_semantic_png_base64"])
    occupancy = decode_png(response["planner_occupancy_probability_u16_png_base64"], probability=True)
    confidence = decode_png(response["planner_navigation_confidence_u16_png_base64"], probability=True)
    if "single_metric_extent_m" not in response:
        raise RuntimeError("BEV runtime omitted required predicted metric extent")
    extent = float(response["single_metric_extent_m"])
    if semantic.ndim != 2 or occupancy.ndim != 2 or confidence.ndim != 2:
        raise RuntimeError("BEV runtime returned a non-2D planner tensor")
    if semantic.shape != occupancy.shape or semantic.shape != confidence.shape:
        raise RuntimeError(
            "BEV runtime returned inconsistent semantic/occupancy/confidence shapes"
        )
    if not math.isfinite(extent) or not 0.5 <= extent <= 30.0:
        raise RuntimeError(f"invalid predicted BEV metric extent: {extent!r}")
    if not np.isfinite(occupancy).all() or not np.isfinite(confidence).all():
        raise RuntimeError("BEV runtime returned non-finite planner probabilities")
    return semantic, occupancy, confidence, extent


def episode_timeout(path_kind: str, gt_length: float) -> float:
    if path_kind == "short":
        return 45.0
    return min(180.0, max(75.0, 3.5 * gt_length / 0.4))


def run_bev_episode(
    scene: Scene,
    task: dict[str, Any],
    model_name: str,
    model_url: str,
    backend_key: str,
    attempt_id: str,
) -> dict[str, Any]:
    path_kind = str(task["path_kind"])
    final_goal = task["goal_world"]
    gt_length = float(task["gt_shortest_path_m"])
    target_service = OracleTargetService.from_task(task)
    motion = Motion(position=list(task["start_world"]), yaw_degrees=float(task["start_yaw_degrees"]))
    segment = f"{attempt_id}-{path_kind}-{model_name}-{backend_key}"
    post_json(model_url, "/reset", {"segment_id": segment}, 30.0)
    backend = make_backends(MOTION_LIMITS)[backend_key]
    image_frame = 0
    simulated = 0.0
    inference_period = 1.0 / BEV_INFERENCE_HZ
    actual_length = 0.0
    inference_seconds = 0.0
    planner_seconds = 0.0
    replans = 0
    global_replans = 0
    collisions = 0
    planner_errors = 0
    last_path_length = 0.0
    model_request_wall_seconds = 0.0
    metric_extent_samples: list[float] = []
    route_errors: list[float] = [0.0]
    executed_world_path: list[list[float]] = [list(motion.position)]
    segment_completion_events: list[dict[str, Any]] = []
    target_repair_count = 0
    cross_backend_fallback_count = 0
    last_scale_diagnostics: dict[str, Any] = {}
    started = time.monotonic()
    status, reason = "timeout", "simulation time limit"
    while simulated <= episode_timeout(path_kind, gt_length):
        step_simulated_seconds = inference_period
        previous_completed = target_service.completed_subgoals
        target_service.advance_if_reached(motion.position)
        if target_service.completed_subgoals > previous_completed:
            for completed in range(
                previous_completed + 1, target_service.completed_subgoals + 1
            ):
                segment_completion_events.append(
                    {
                        "subgoal_number": completed,
                        "simulated_seconds": simulated,
                        "executed_path_length_m": actual_length,
                    }
                )
        terminal_distance = distance_xz(motion.position, final_goal)
        if target_service.complete and terminal_distance <= SUCCESS_RADIUS_M:
            status, reason = "success", None
            break
        try:
            point_goal = target_service.query(
                motion.position, motion.yaw_degrees
            )
            global_replans = target_service.switch_count
            rgb = scene.rgb(motion.position, motion.yaw_degrees)
            image_frame += 1
            request_started = time.monotonic()
            response = post_json(
                model_url,
                "/predict",
                {
                    "segment_id": segment,
                    "frame_seq": image_frame,
                    "image_png_base64": encode_jpeg(rgb),
                    "physical_camera_height_m": CAMERA_HEIGHT_M,
                    "threshold": 0.5,
                },
                300.0,
            )
            model_request_wall_seconds += time.monotonic() - request_started
            # Account for inference even when the following planner fails.
            inference_seconds += float(response.get("inference_seconds", 0.0))
            semantic, occupancy, confidence, extent = decode_bev_response(response)
            metric_extent_samples.append(extent)
            last_scale_diagnostics = {
                key: response.get(key)
                for key in (
                    "model_scale_token_bev_per_vggt",
                    "camera_metric_scale_m_per_vggt",
                    "bev_metric_scale_m_per_bev",
                    "vggt_camera_height",
                    "ground_inlier_fraction",
                    "ground_fallback_used",
                )
                if key in response
            }
            if isinstance(backend, MPPIBackend):
                backend.set_motion_state(motion.forward_m_s, motion.angular_left_rad_s)
            planning_started = time.monotonic()
            try:
                result, _problem, _subgoal = plan_engineered(
                    backend,
                    semantic=semantic,
                    occupancy_probability=occupancy,
                    navigation_confidence=confidence,
                    extent_m=extent,
                    target_metric_m=point_goal,
                    inflation_radius_m=TOTAL_INFLATION_M,
                    budget_s=PLANNER_BUDGET_SECONDS[backend_key],
                    allow_cross_backend_fallback=False,
                )
            finally:
                planner_seconds += time.monotonic() - planning_started
            last_path_length = path_length_2d(result.path_metric_m)
            backend_details = dict(result.backend_details)
            target_repair_count += int(
                bool(backend_details.get("planned_goal_repaired"))
                or bool(backend_details.get("reachable_target_repair_computed"))
            )
            cross_backend_fallback_count += int(
                bool(backend_details.get("cross_backend_fallback"))
            )
            old_yaw = motion.yaw_degrees
            advance = apply_geometric_path(
                scene,
                motion,
                result.path_metric_m,
                maximum_arc_m=max(last_path_length, 1e-6),
                native_headings_left_rad=result.path_headings_rad,
            )
            replans += 1
            planner_errors = 0
            if not advance.safe:
                collisions += 1
                status = "collision"
                reason = (
                    "planned full native path intersects complete simulator "
                    f"geometry at {advance.collision.collision_world}; minimum "
                    f"clearance {advance.collision.minimum_clearance_m:.3f} m"
                )
                break
            actual_length += advance.travelled_m
            executed_world_path.extend(advance.world_path[1:])
            step_simulated_seconds = max(
                inference_period,
                advance.travelled_m / MOTION_LIMITS.maximum_speed_m_s,
            )
            route_errors.extend(
                distance_to_polyline_xz(point, task["gt_path_world"])
                for point in advance.world_path[1:]
            )
            motion.forward_m_s = min(
                MOTION_LIMITS.maximum_speed_m_s,
                advance.travelled_m / max(step_simulated_seconds, 1e-9),
            )
            yaw_delta_clockwise = wrap_angle(
                math.radians(motion.yaw_degrees - old_yaw)
            )
            motion.angular_left_rad_s = float(
                np.clip(
                    -yaw_delta_clockwise / max(step_simulated_seconds, 1e-9),
                    -MOTION_LIMITS.maximum_angular_speed_rad_s,
                    MOTION_LIMITS.maximum_angular_speed_rad_s,
                )
            )
        except Exception as error:
            planner_errors += 1
            motion.forward_m_s = 0.0
            motion.angular_left_rad_s = 0.0
            if planner_errors >= 5:
                status, reason = "planner_failure", f"{type(error).__name__}: {error}"
                break
        simulated += step_simulated_seconds
    terminal_distance = distance_xz(motion.position, final_goal)
    completion_adjusted = max(1.0, (actual_length + terminal_distance) / max(gt_length, 1e-6))
    return {
        "status": status,
        "success": status == "success",
        "reason": reason,
        "model": model_name,
        "backend": backend_key,
        "path_kind": path_kind,
        "gt_shortest_path_m": gt_length,
        "executed_path_length_m": actual_length,
        "executed_world_path": executed_world_path,
        "executed_path_ratio": max(1.0, actual_length / max(gt_length, 1e-6)) if status == "success" else None,
        "completion_adjusted_path_ratio": completion_adjusted,
        "latest_planned_path_length_m": last_path_length,
        "latest_planned_path_ratio": max(1.0, (actual_length + last_path_length) / max(gt_length, 1e-6)),
        "terminal_distance_m": terminal_distance,
        "simulated_seconds": simulated,
        "wall_seconds": time.monotonic() - started,
        "model_inference_seconds": inference_seconds,
        "model_request_wall_seconds": model_request_wall_seconds,
        "planner_seconds": planner_seconds,
        "last_backend_details": locals().get("backend_details", {}),
        "target_repair_count": target_repair_count,
        "cross_backend_fallback_count": cross_backend_fallback_count,
        "metric_extent_mean_m": (
            statistics.fmean(metric_extent_samples) if metric_extent_samples else None
        ),
        "metric_extent_min_m": min(metric_extent_samples, default=None),
        "metric_extent_max_m": max(metric_extent_samples, default=None),
        "last_scale_diagnostics": last_scale_diagnostics,
        "replans": replans,
        "gt_upper_replans": global_replans,
        "target_service_refreshes": target_service.refresh_count,
        "subgoals_total": len(target_service.subgoals_world),
        "subgoals_completed": target_service.completed_subgoals,
        "segment_completion_events": segment_completion_events,
        "gt_route_cross_track_error_mean_m": statistics.fmean(route_errors),
        "gt_route_cross_track_error_max_m": max(route_errors),
        "collisions": collisions,
        "inference_hz": replans / max(simulated, 1e-6),
        "execution_mode": "online_gt_geometry_checked_full_native_path_v1",
        "simulator_teleport_steps": 0,
        "geometry_collision_voxel_size_m": scene.geometry.voxel_size_m,
        "geometry_collision_robot_radius_m": ROBOT_COLLISION_RADIUS_M,
        "planner_safety_inflation_m": SAFETY_MARGIN_M,
        "planner_total_inflation_m": TOTAL_INFLATION_M,
        "target_localization": "common continuously refreshed simulator-GT ego point",
        "predicted_extrinsic_used_for_target_tracking": False,
        "forbidden_inference_inputs": ["GT BEV", "GT depth", "GT obstacles", "GT route"],
        "evaluator_only_gt": ["complete simulator geometry collision", "terminal distance", "path metrics"],
    }


def run_external_episode(
    scene: Scene,
    task: dict[str, Any],
    method: str,
    baseline_url: str,
) -> dict[str, Any]:
    path_kind = str(task["path_kind"])
    final_goal = task["goal_world"]
    gt_length = float(task["gt_shortest_path_m"])
    target_service = OracleTargetService.from_task(task)
    motion = Motion(position=list(task["start_world"]), yaw_degrees=float(task["start_yaw_degrees"]))
    post_json(baseline_url, "/reset", {}, 30.0)
    images: deque[str] = deque(maxlen=11)
    nomad_goal_images: dict[int, str] = {}
    simulated = 0.0
    inference_period = 1.0 / EXTERNAL_METHOD_HZ[method]
    actual_length = 0.0
    inference_seconds = 0.0
    replans = 0
    global_replans = 0
    collisions = 0
    planner_errors = 0
    last_path_length = 0.0
    model_request_wall_seconds = 0.0
    route_errors: list[float] = [0.0]
    executed_world_path: list[list[float]] = [list(motion.position)]
    segment_completion_events: list[dict[str, Any]] = []
    started = time.monotonic()
    status, reason = "timeout", "simulation time limit"
    while simulated <= episode_timeout(path_kind, gt_length):
        step_simulated_seconds = inference_period
        previous_completed = target_service.completed_subgoals
        target_service.advance_if_reached(motion.position)
        if target_service.completed_subgoals > previous_completed:
            for completed in range(
                previous_completed + 1, target_service.completed_subgoals + 1
            ):
                segment_completion_events.append(
                    {
                        "subgoal_number": completed,
                        "simulated_seconds": simulated,
                        "executed_path_length_m": actual_length,
                    }
                )
        terminal_distance = distance_xz(motion.position, final_goal)
        if target_service.complete and terminal_distance <= SUCCESS_RADIUS_M:
            status, reason = "success", None
            break
        try:
            point_goal = target_service.query(
                motion.position, motion.yaw_degrees
            )
            global_replans = target_service.switch_count
            if method != "straight_line":
                images.append(encode_jpeg(scene.rgb(motion.position, motion.yaw_degrees)))
            goal_image_base64 = None
            if method == "nomad":
                subgoal_index = target_service.current_index
                if subgoal_index not in nomad_goal_images:
                    goal_position = target_service.subgoals_world[subgoal_index]
                    goal_yaw = subgoal_image_yaw_degrees(task, subgoal_index)
                    nomad_goal_images[subgoal_index] = encode_jpeg(
                        scene.rgb(goal_position, goal_yaw)
                    )
                goal_image_base64 = nomad_goal_images[subgoal_index]
            request_started = time.monotonic()
            request_payload = {
                "method": method,
                "target_metric_m": point_goal,
                "images_base64": list(images),
                # No method receives GT trajectory history. The only common
                # privileged runtime input is the current pre-planned target.
                "gt_trajectory_history_metric_m": [],
            }
            if goal_image_base64 is not None:
                request_payload["goal_image_base64"] = goal_image_base64
            if method == "genie_samtp":
                focal_px = (CAMERA_WIDTH * 0.5) / math.tan(
                    math.radians(HORIZONTAL_FOV_DEGREES) * 0.5
                )
                request_payload["camera_calibration"] = {
                    "intrinsics": [
                        [focal_px, 0.0, (CAMERA_WIDTH - 1) * 0.5],
                        [0.0, focal_px, (CAMERA_HEIGHT - 1) * 0.5],
                        [0.0, 0.0, 1.0],
                    ],
                    # Optical axes: +x image-right, +y image-down, +z forward.
                    # Ground axes: +x robot-right, +y robot-forward, +z up.
                    # This is fixed camera calibration, not simulator pose GT.
                    "T_ground_camera": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, -1.0, 0.0, CAMERA_HEIGHT_M],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "ground_z_m": 0.0,
                }
            response = post_json(
                baseline_url,
                "/predict",
                request_payload,
                900.0 if method == "omnivla" else 300.0,
            )
            model_request_wall_seconds += time.monotonic() - request_started
            inference_seconds += float(response.get("inference_seconds", 0.0))
            raw_path = response.get("path_metric_m") or []
            control = response.get("control_velocity") or {}
            execution_preference = str(
                response.get("execution_preference", "native_path")
            )
            if execution_preference == "control_velocity":
                execution_path, forced_yaw_delta = constant_control_path(
                    float(control.get("forward_m_s", 0.0)),
                    float(control.get("angular_left_rad_s", 0.0)),
                    inference_period,
                )
                last_path_length = path_length_2d(execution_path)
            elif execution_preference == "native_path_1m_receding_horizon":
                if len(raw_path) < 2:
                    raise RuntimeError("GeNIE returned no executable native path")
                # GeNIE III-E aligns to its selected path, advances by 1 m,
                # and observes/replans.  executable_path_prefix preserves every
                # released planner vertex and only interpolates the exact 1 m
                # endpoint; it is not path downsampling or GT repair.
                execution_path, last_path_length = executable_path_prefix(
                    raw_path, 1.0
                )
                forced_yaw_delta = None
            elif len(raw_path) >= 2:
                execution_path = raw_path
                forced_yaw_delta = None
                last_path_length = path_length_2d(raw_path)
            else:
                execution_path, forced_yaw_delta = constant_control_path(
                    float(control.get("forward_m_s", 0.0)),
                    float(control.get("angular_left_rad_s", 0.0)),
                    inference_period,
                )
                last_path_length = path_length_2d(execution_path)
            old_yaw = motion.yaw_degrees
            advance = apply_geometric_path(
                scene,
                motion,
                execution_path,
                maximum_arc_m=max(last_path_length, 1e-6),
                forced_yaw_delta_degrees=forced_yaw_delta,
            )
            replans += 1
            planner_errors = 0
            if not advance.safe:
                collisions += 1
                status = "collision"
                reason = (
                    "predicted full native path intersects complete simulator "
                    f"geometry at {advance.collision.collision_world}; minimum "
                    f"clearance {advance.collision.minimum_clearance_m:.3f} m"
                )
                break
            actual_length += advance.travelled_m
            executed_world_path.extend(advance.world_path[1:])
            step_simulated_seconds = max(
                inference_period,
                advance.travelled_m / MOTION_LIMITS.maximum_speed_m_s,
            )
            route_errors.extend(
                distance_to_polyline_xz(point, task["gt_path_world"])
                for point in advance.world_path[1:]
            )
            motion.forward_m_s = min(
                MOTION_LIMITS.maximum_speed_m_s,
                advance.travelled_m / max(step_simulated_seconds, 1e-9),
            )
            yaw_delta_clockwise = wrap_angle(
                math.radians(motion.yaw_degrees - old_yaw)
            )
            motion.angular_left_rad_s = float(
                np.clip(
                    -yaw_delta_clockwise / max(step_simulated_seconds, 1e-9),
                    -MOTION_LIMITS.maximum_angular_speed_rad_s,
                    MOTION_LIMITS.maximum_angular_speed_rad_s,
                )
            )
        except Exception as error:
            planner_errors += 1
            motion.forward_m_s = 0.0
            motion.angular_left_rad_s = 0.0
            if planner_errors >= 5:
                status, reason = "inference_failure", f"{type(error).__name__}: {error}"
                break
        simulated += step_simulated_seconds
    terminal_distance = distance_xz(motion.position, final_goal)
    return {
        "status": status,
        "success": status == "success",
        "reason": reason,
        "method": method,
        "path_kind": path_kind,
        "gt_shortest_path_m": gt_length,
        "executed_path_length_m": actual_length,
        "executed_world_path": executed_world_path,
        "executed_path_ratio": max(1.0, actual_length / max(gt_length, 1e-6)) if status == "success" else None,
        "completion_adjusted_path_ratio": max(1.0, (actual_length + terminal_distance) / max(gt_length, 1e-6)),
        "latest_planned_path_length_m": last_path_length,
        "latest_planned_path_ratio": max(1.0, (actual_length + last_path_length) / max(gt_length, 1e-6)),
        "terminal_distance_m": terminal_distance,
        "simulated_seconds": simulated,
        "wall_seconds": time.monotonic() - started,
        "model_inference_seconds": inference_seconds,
        "model_request_wall_seconds": model_request_wall_seconds,
        "planner_seconds": 0.0,
        "replans": replans,
        "gt_upper_replans": global_replans,
        "target_service_refreshes": target_service.refresh_count,
        "subgoals_total": len(target_service.subgoals_world),
        "subgoals_completed": target_service.completed_subgoals,
        "segment_completion_events": segment_completion_events,
        "gt_route_cross_track_error_mean_m": statistics.fmean(route_errors),
        "gt_route_cross_track_error_max_m": max(route_errors),
        "collisions": collisions,
        "inference_hz": replans / max(simulated, 1e-6),
        "execution_preference": locals().get(
            "execution_preference", "native_path"
        ),
        "execution_mode": (
            "official_receding_horizon_velocity_control_v2"
            if locals().get("execution_preference") == "control_velocity"
            else "official_genie_align_advance_1m_replan_v1"
            if locals().get("execution_preference")
            == "native_path_1m_receding_horizon"
            else "online_gt_geometry_checked_full_native_path_v1"
        ),
        "simulator_teleport_steps": 0,
        "geometry_collision_voxel_size_m": scene.geometry.voxel_size_m,
        "geometry_collision_robot_radius_m": ROBOT_COLLISION_RADIUS_M,
        "target_localization": (
            "current frozen-route subgoal RGB rendered for native NoMaD ImageGoal"
            if method == "nomad"
            else "common continuously refreshed simulator-GT ego point"
        ),
        "native_target_input": locals().get("response", {}).get("target_input"),
        "predicted_extrinsic_used_for_target_tracking": False,
        "forbidden_inference_inputs": ["GT BEV", "GT depth", "GT obstacles", "GT route"],
        "evaluator_only_gt": ["complete simulator geometry collision", "terminal distance", "path metrics"],
    }


def task_path(output: Path, path_kind: str, attempt: int) -> Path:
    return output / "tasks" / path_kind / f"attempt_{attempt:06d}.json"


def result_path(output: Path, path_kind: str, phase: str, attempt: int) -> Path:
    return output / "results" / path_kind / phase / f"attempt_{attempt:06d}.json"


def _bev_system_names(args: argparse.Namespace) -> list[str]:
    return [
        f"{model}+{backend.key}"
        for model in args.models
        for backend in BACKEND_TYPES
        if backend.key in args.backends
    ]


def ensure_tasks_and_run_bev(args: argparse.Namespace) -> None:
    path_kind = str(args.path_kind)
    attempts = list(range(args.attempt_start, args.attempt_start + args.attempt_count))
    candidate_map = {
        attempt: candidate_scenes(args.shard_id, attempt, path_kind)
        for attempt in attempts
    }
    required = {index for values in candidate_map.values() for index in values}
    # A replay may intentionally bind attempt_000000 to an existing task whose
    # scene was sampled by another shard/attempt.  Load that explicit scene in
    # addition to the fallback candidates so the fixed task is reproducible.
    for attempt in attempts:
        existing_task = task_path(args.output_root, path_kind, attempt)
        if existing_task.is_file():
            required.add(int(read_json(existing_task)["scene_index"]))
    houses = load_houses(args.dataset_dir, required)
    expected_systems = _bev_system_names(args)
    for attempt in attempts:
        attempt_lock_path = (
            args.output_root
            / "locks"
            / "attempts"
            / path_kind
            / f"attempt_{attempt:06d}.lock"
        )
        attempt_lock_path.parent.mkdir(parents=True, exist_ok=True)
        attempt_lock = attempt_lock_path.open("a+")
        fcntl.flock(attempt_lock.fileno(), fcntl.LOCK_EX)
        output_file = result_path(args.output_root, path_kind, "bev", attempt)
        existing = read_json(output_file) if output_file.is_file() else {}
        systems: dict[str, Any] = dict(existing.get("systems", {}))
        external_output_file = result_path(
            args.output_root, path_kind, "external", attempt
        )
        external_existing = (
            read_json(external_output_file)
            if external_output_file.is_file()
            else {}
        )
        external_systems: dict[str, Any] = dict(
            external_existing.get("systems", {})
        )
        bev_complete_before = all(
            isinstance(systems.get(name), dict) for name in expected_systems
        )
        external_complete_before = all(
            isinstance(external_systems.get(method), dict)
            for method in args.methods
        )
        if bev_complete_before and (
            args.phase != "combined" or external_complete_before
        ):
            fcntl.flock(attempt_lock.fileno(), fcntl.LOCK_UN)
            attempt_lock.close()
            continue
        task_file = task_path(args.output_root, path_kind, attempt)
        task = read_json(task_file) if task_file.is_file() else None
        scene: Scene | None = None
        rejections: list[dict[str, Any]] = []
        try:
            if task is None:
                for scene_index in candidate_map[attempt]:
                    try:
                        scene = Scene(houses[scene_index], args, scene_index)
                        task = sample_task(
                            scene.graph,
                            random.Random(
                                SEED
                                ^ (0x51A7 if path_kind == "short" else 0x10A9)
                                ^ (args.shard_id * 1_000_003)
                                ^ attempt
                            ),
                            scene_index,
                            attempt,
                            path_kind,
                        )
                        scene.validate_task_geometry(task)
                        task.update(
                            shard_id=args.shard_id,
                            task_rejections=rejections,
                            scene_sampling="uniform random with replacement over all 10,000 train houses",
                        )
                        atomic_json(task_file, task)
                        break
                    except Exception as error:
                        rejections.append(
                            {
                                "scene_index": scene_index,
                                "reason": f"{type(error).__name__}: {error}",
                            }
                        )
                        if scene is not None:
                            scene.stop()
                            scene = None
                if task is None:
                    failure = {
                        "attempt_index": attempt,
                        "path_kind": path_kind,
                        "task_failure": rejections,
                    }
                    atomic_json(output_file, {**failure, "phase": "bev"})
                    if args.phase == "combined":
                        atomic_json(
                            external_output_file,
                            {**failure, "phase": "external"},
                        )
                    continue
            if str(task["path_kind"]) != path_kind:
                raise RuntimeError("task path kind does not match worker pool")
            if scene is None:
                scene = Scene(
                    houses[int(task["scene_index"])],
                    args,
                    int(task["scene_index"]),
                )
                scene.validate_task_geometry(task)
            model_urls = {
                "m03_merged": args.m03_url,
                "single_baseline": args.single_url,
            }
            for model_name in args.models:
                model_url = model_urls[model_name]
                for backend_type in BACKEND_TYPES:
                    if backend_type.key not in args.backends:
                        continue
                    system = f"{model_name}+{backend_type.key}"
                    if isinstance(systems.get(system), dict):
                        continue
                    systems[system] = run_bev_episode(
                        scene,
                        task,
                        model_name,
                        model_url,
                        backend_type.key,
                        f"shard{args.shard_id}-{path_kind}-attempt{attempt}",
                    )
                    atomic_json(
                        output_file,
                        {
                            "schema": "procthor-nav-results-v2",
                            "phase": "bev",
                            "path_kind": path_kind,
                            "attempt_index": attempt,
                            "scene_index": task["scene_index"],
                            "scene_load_seconds": scene.load_seconds,
                            "renderer_load_seconds": scene.renderer_load_seconds,
                            "geometry_load_seconds": scene.geometry_load_seconds,
                            "geometry_cache_hit": scene.geometry_cache_hit,
                            "systems": systems,
                        },
                    )
            if args.phase == "combined":
                for method in args.methods:
                    if isinstance(external_systems.get(method), dict):
                        continue
                    external_systems[method] = run_external_episode(
                        scene, task, method, args.baseline_url
                    )
                    # A fresh combined attempt owns one scene initialization,
                    # recorded by the BEV result.  For resumed attempts whose
                    # BEV was already complete, this is a real second load and
                    # is therefore recorded on the external result.
                    external_load = scene.load_seconds if bev_complete_before else 0.0
                    atomic_json(
                        external_output_file,
                        {
                            "schema": "procthor-nav-results-v2",
                            "phase": "external",
                            "path_kind": path_kind,
                            "attempt_index": attempt,
                            "scene_index": task["scene_index"],
                            "scene_load_seconds": external_load,
                            "renderer_load_seconds": (
                                scene.renderer_load_seconds
                                if bev_complete_before
                                else 0.0
                            ),
                            "geometry_load_seconds": (
                                scene.geometry_load_seconds
                                if bev_complete_before
                                else 0.0
                            ),
                            "geometry_cache_hit": scene.geometry_cache_hit,
                            "shared_scene_initialization_with_bev": (
                                not bev_complete_before
                            ),
                            "systems": external_systems,
                        },
                    )
        finally:
            if scene is not None:
                scene.stop()
            fcntl.flock(attempt_lock.fileno(), fcntl.LOCK_UN)
            attempt_lock.close()


def run_external(args: argparse.Namespace) -> None:
    path_kind = str(args.path_kind)
    attempts = list(range(args.attempt_start, args.attempt_start + args.attempt_count))
    tasks = {
        attempt: read_json(task_path(args.output_root, path_kind, attempt))
        for attempt in attempts
        if task_path(args.output_root, path_kind, attempt).is_file()
    }
    if not tasks:
        return
    houses = load_houses(
        args.dataset_dir, {int(task["scene_index"]) for task in tasks.values()}
    )
    for attempt, task in tasks.items():
        output_file = result_path(args.output_root, path_kind, "external", attempt)
        existing = read_json(output_file) if output_file.is_file() else {}
        systems: dict[str, Any] = dict(existing.get("systems", {}))
        if all(isinstance(systems.get(method), dict) for method in args.methods):
            continue
        scene = Scene(
            houses[int(task["scene_index"])], args, int(task["scene_index"])
        )
        try:
            scene.validate_task_geometry(task)
            for method in args.methods:
                if isinstance(systems.get(method), dict):
                    continue
                systems[method] = run_external_episode(
                    scene, task, method, args.baseline_url
                )
                atomic_json(
                    output_file,
                    {
                        "schema": "procthor-nav-results-v2",
                        "phase": "external",
                        "path_kind": path_kind,
                        "attempt_index": attempt,
                        "scene_index": task["scene_index"],
                        "scene_load_seconds": scene.load_seconds,
                        "renderer_load_seconds": scene.renderer_load_seconds,
                        "geometry_load_seconds": scene.geometry_load_seconds,
                        "geometry_cache_hit": scene.geometry_cache_hit,
                        "systems": systems,
                    },
                )
        finally:
            scene.stop()


def _metric_summary(rows: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> dict[str, Any]:
    absolute_success = sum(bool(row.get("success")) for row in rows)
    relative_success = sum(bool(row.get("success")) for row in accepted)
    successful = [row for row in accepted if row.get("success")]
    ratios = [
        float(row["executed_path_ratio"])
        for row in successful
        if row.get("executed_path_ratio") is not None
    ]
    planned_ratios = [
        float(row["latest_planned_path_ratio"])
        for row in successful
        if row.get("latest_planned_path_ratio") is not None
    ]
    terminal = [float(row.get("terminal_distance_m", 0.0)) for row in rows]
    completion_adjusted = [
        float(row["completion_adjusted_path_ratio"])
        for row in rows
        if row.get("completion_adjusted_path_ratio") is not None
    ]
    inference_hz = [
        float(row["inference_hz"])
        for row in rows
        if row.get("inference_hz") is not None
    ]
    return {
        "absolute_successes": absolute_success,
        "absolute_denominator_all_attempted_paths": len(rows),
        "absolute_success_rate": absolute_success / max(1, len(rows)),
        "relative_successes": relative_success,
        "relative_denominator_at_least_one_success_paths": len(accepted),
        "relative_success_rate": relative_success / max(1, len(accepted)),
        "successful_path_ratio_mean": statistics.fmean(ratios) if ratios else None,
        "successful_path_ratio_median": statistics.median(ratios) if ratios else None,
        "successful_path_ratio_sum": sum(ratios),
        "successful_path_ratio_count": len(ratios),
        "successful_latest_planned_path_ratio_mean": (
            statistics.fmean(planned_ratios) if planned_ratios else None
        ),
        "successful_latest_planned_path_ratio_sum": sum(planned_ratios),
        "successful_latest_planned_path_ratio_count": len(planned_ratios),
        "terminal_distance_mean_m": statistics.fmean(terminal) if terminal else None,
        "terminal_distance_sum_m": sum(terminal),
        "terminal_distance_count": len(terminal),
        "completion_adjusted_path_ratio_mean": (
            statistics.fmean(completion_adjusted) if completion_adjusted else None
        ),
        "completion_adjusted_path_ratio_sum": sum(completion_adjusted),
        "completion_adjusted_path_ratio_count": len(completion_adjusted),
        "inference_hz_mean": statistics.fmean(inference_hz) if inference_hz else None,
        "inference_hz_sum": sum(inference_hz),
        "inference_hz_count": len(inference_hz),
        "total_wall_seconds": sum(float(row.get("wall_seconds", 0.0)) for row in rows),
        "total_model_request_wall_seconds": sum(
            float(row.get("model_request_wall_seconds", 0.0)) for row in rows
        ),
        "total_model_inference_seconds": sum(
            float(row.get("model_inference_seconds", 0.0)) for row in rows
        ),
        "total_planner_seconds": sum(
            float(row.get("planner_seconds", 0.0)) for row in rows
        ),
        "total_collisions": sum(int(row.get("collisions", 0)) for row in rows),
        "total_replans": sum(int(row.get("replans", 0)) for row in rows),
        "total_gt_upper_subgoal_switches": sum(
            int(row.get("gt_upper_replans", 0)) for row in rows
        ),
        "status_counts": {
            status: sum(str(row.get("status")) == status for row in rows)
            for status in sorted({str(row.get("status")) for row in rows})
        },
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    # Aggregate exactly the comparison contract selected by the caller.  The
    # legacy runner still gets the historical full matrix through argparse's
    # defaults, while focused/formal runners can request a strict subset.
    selected_models = getattr(args, "models", ("m03_merged", "single_baseline"))
    selected_backends = getattr(
        args, "backends", tuple(kind.key for kind in BACKEND_TYPES)
    )
    selected_methods = getattr(args, "methods", EXTERNAL_METHODS)
    expected_systems = {
        f"{model}+{backend.key}"
        for model in selected_models
        for backend in BACKEND_TYPES
        if backend.key in selected_backends
    } | set(selected_methods)
    paths_summary: dict[str, Any] = {}
    for path_kind in ("short", "long"):
        attempts: list[dict[str, Any]] = []
        directory = args.output_root / "tasks" / path_kind
        for task_file in sorted(directory.glob("attempt_*.json")):
            attempt = int(task_file.stem.split("_")[-1])
            bev_file = result_path(args.output_root, path_kind, "bev", attempt)
            external_file = result_path(
                args.output_root, path_kind, "external", attempt
            )
            if not bev_file.is_file() or not external_file.is_file():
                continue
            bev, external = read_json(bev_file), read_json(external_file)
            if bev.get("task_failure") or external.get("task_failure"):
                continue
            available_systems = {
                **bev.get("systems", {}),
                **external.get("systems", {}),
            }
            if not all(
                isinstance(available_systems.get(name), dict)
                for name in expected_systems
            ):
                continue
            # Ignore stale results from a broader earlier invocation rather
            # than silently changing the formal comparison denominator.
            systems = {name: available_systems[name] for name in expected_systems}
            any_success = any(bool(result.get("success")) for result in systems.values())
            task = read_json(task_file)
            attempts.append(
                {
                    "attempt_index": attempt,
                    "scene_index": task["scene_index"],
                    "eligible": any_success,
                    "all_failed": not any_success,
                    "systems": systems,
                    "shared_scene_load_seconds": float(
                        bev.get("scene_load_seconds", 0.0)
                    )
                    + float(external.get("scene_load_seconds", 0.0)),
                }
            )
        attempts.sort(key=lambda row: row["attempt_index"])
        accepted_attempts = [row for row in attempts if row["eligible"]][
            : args.target_paths
        ]
        accepted_ids = {row["attempt_index"] for row in accepted_attempts}
        system_names = sorted({key for row in attempts for key in row["systems"]})
        systems_summary = {
            system: _metric_summary(
                [row["systems"][system] for row in attempts],
                [row["systems"][system] for row in accepted_attempts],
            )
            for system in system_names
        }
        paths_summary[path_kind] = {
            "target_relative_paths": args.target_paths,
            "complete_attempted_paths": len(attempts),
            "accepted_relative_paths": len(accepted_attempts),
            "all_system_failed_attempts": sum(row["all_failed"] for row in attempts),
            "accepted_attempt_indices": sorted(accepted_ids),
            "shared_scene_load_seconds": sum(
                row["shared_scene_load_seconds"] for row in attempts
            ),
            "systems": systems_summary,
        }
    summary = {
        "schema": "procthor-20k-oracle-localized-navigation-summary-v2",
        "shard_id": args.shard_id,
        "paths": paths_summary,
        "contracts": {
            "short_and_long_pools_are_independent": True,
            "scene_sampling": "uniform random with replacement over all 10,000 ProcTHOR train scenes; no ordering, quotas or balancing",
            "absolute_denominator": "all complete attempted paths of that path kind, including all-system failures",
            "relative_denominator": "first target_paths of that kind on which at least one non-oracle system succeeds",
            "long_upper_guide": "one frozen privileged GT A* route segmented into <=2.0 m fixed local endpoints; no online global replan",
            "target_service": (
                "one current frozen-route endpoint; metric PointGoal for point models "
                "or the same endpoint rendered as ImageGoal for NoMaD"
            ),
            "predicted_extrinsic_target_tracking": False,
            "gt_trajectory_history_input": False,
            "oracle_is_not_a_competing_method": True,
        },
    }
    atomic_json(args.output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("bev", "external", "combined", "aggregate"),
        required=True,
    )
    parser.add_argument(
        "--path-kind",
        choices=("short", "long"),
        help="required for bev/external; aggregate always reports both independent pools",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--procthor-runtime-root", type=Path)
    parser.add_argument("--databuilder-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--geometry-cache-root",
        type=Path,
        help="shared complete-geometry cache (must be common to all shards)",
    )
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--attempt-start", type=int, default=0)
    parser.add_argument("--attempt-count", type=int, default=0)
    parser.add_argument(
        "--target-paths", type=int, default=TARGET_PATHS_PER_KIND_PER_SHARD
    )
    parser.add_argument("--m03-url", default="http://127.0.0.1:21001")
    parser.add_argument("--single-url", default="http://127.0.0.1:21002")
    parser.add_argument("--baseline-url", default="http://127.0.0.1:21003")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("m03_merged", "single_baseline"),
        default=("m03_merged", "single_baseline"),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=tuple(kind.key for kind in BACKEND_TYPES),
        default=tuple(kind.key for kind in BACKEND_TYPES),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=EXTERNAL_METHODS,
        default=EXTERNAL_METHODS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_id < SHARD_COUNT:
        raise ValueError("shard-id must be in [0, 7]")
    if args.gpu_index < 0 or args.attempt_start < 0 or args.attempt_count < 0:
        raise ValueError("GPU and attempt values must be non-negative")
    if args.target_paths < 1:
        raise ValueError("target-paths must be positive")
    if args.phase in {"bev", "external", "combined"} and args.path_kind is None:
        raise ValueError("--path-kind is required for execution phases")
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.geometry_cache_root = (
        args.geometry_cache_root.expanduser().resolve()
        if args.geometry_cache_root is not None
        else (args.output_root.parent / "geometry_cache").resolve()
    )
    args.geometry_cache_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.databuilder_root.expanduser().resolve()))
    if args.phase in {"bev", "external", "combined"}:
        from procthor_collection_core import (
            expose_bundled_vulkaninfo,
            locate_ai2thor_runtime,
            nvidia_gpu_inventory,
            refresh_ai2thor_cuda_vulkan_mapping,
        )

        runtime_root, _executable = locate_ai2thor_runtime()
        args.procthor_runtime_root = runtime_root
        expose_bundled_vulkaninfo(runtime_root)
        mapping = refresh_ai2thor_cuda_vulkan_mapping(runtime_root)
        inventory = nvidia_gpu_inventory()
        if args.gpu_index not in mapping or args.gpu_index not in inventory:
            raise RuntimeError(f"physical GPU {args.gpu_index} missing from AI2-THOR mapping")
        args.resolved_nvidia_gpu_uuid = inventory[args.gpu_index]
    if args.phase in {"bev", "combined"}:
        ensure_tasks_and_run_bev(args)
    elif args.phase == "external":
        run_external(args)
    else:
        print(json.dumps(aggregate(args), indent=2))


if __name__ == "__main__":
    main()
