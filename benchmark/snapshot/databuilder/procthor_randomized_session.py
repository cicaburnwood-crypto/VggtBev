#!/usr/bin/env python3
"""Generate randomized RGB-D navigation sessions from offline ProcTHOR houses.

The output intentionally mirrors the camera/trajectory portion of the Habitat
collector while keeping ProcTHOR's Unity coordinate convention explicit.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
import gzip
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from bev_visibility import (
    DEFAULT_MAX_SINGLE_BEV_VOID_RATIO,
    VOID_CHECK_VERSION,
    VOID_FILTER_CONTRACT,
    VOID_REPAIR_ALGORITHM,
    enforce_single_bev_void_limit,
    repair_valid_coverage,
)
from procthor_bev import (
    BEVAccumulator,
    UNKNOWN_VALUE,
    VISIBILITY_ALGORITHM,
    build_complete_truth,
    extent_key,
    merged_modality,
    render_ego_obstacle_map,
    render_visibility_masked_map,
    validate_bev_frame,
)


SPLIT_COUNTS = {"train": 10_000, "val": 1_000, "test": 1_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="directory containing split JSONL files (not needed with --fresh-houses)",
    )
    parser.add_argument(
        "--fresh-houses",
        action="store_true",
        help="generate a new ProcTHOR house online for each session",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=sorted(SPLIT_COUNTS), default="train")
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260810)
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--nvidia-gpu-index",
        type=int,
        help="physical index reported by nvidia-smi; automatically mapped to Unity/Vulkan",
    )
    gpu_group.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="direct AI2-THOR Vulkan device index (prefer --nvidia-gpu-index)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--capture-hz", type=float, default=2.0)
    parser.add_argument("--camera-height-range", type=float, nargs=2, default=(0.3, 0.8))
    parser.add_argument("--fov-range", type=float, nargs=2, default=(60.0, 120.0))
    parser.add_argument("--speed-range", type=float, nargs=2, default=(0.6, 2.0))
    parser.add_argument("--path-length-range", type=float, nargs=2, default=(3.0, 10.0))
    parser.add_argument("--bev-size", type=int, default=512)
    parser.add_argument("--bev-extents", type=float, nargs="+", default=(6.5,))
    parser.add_argument("--merged-extents", type=float, nargs="+", default=(10.0,))
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--obstacle-min-height", type=float, default=0.0)
    parser.add_argument("--obstacle-max-height", type=float, default=1.4)
    parser.add_argument("--house-attempts", type=int, default=12)
    parser.add_argument("--path-attempts", type=int, default=200)
    parser.add_argument(
        "--max-single-bev-void-ratio",
        type=float,
        default=DEFAULT_MAX_SINGLE_BEV_VOID_RATIO,
        help=(
            "reject when any Single Complete GT geometry-valid mask contains "
            "a larger VOID fraction"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sessions < 1:
        raise ValueError("--sessions must be positive")
    if args.width < 16 or args.height < 16:
        raise ValueError("camera dimensions are too small")
    if args.capture_hz <= 0:
        raise ValueError("--capture-hz must be positive")
    if args.bev_size < 16:
        raise ValueError("--bev-size must be at least 16")
    if not args.bev_extents or not all(value > 0 for value in args.bev_extents):
        raise ValueError("--bev-extents must contain positive values")
    if not args.merged_extents or not all(value > 0 for value in args.merged_extents):
        raise ValueError("--merged-extents must contain positive values")
    if not 0.001 <= args.voxel_size <= 0.2:
        raise ValueError("--voxel-size must be within [0.001, 0.2] metres")
    if not 0.0 <= args.max_single_bev_void_ratio <= 1.0:
        raise ValueError("--max-single-bev-void-ratio must be within [0, 1]")
    if not 0 <= args.obstacle_min_height < args.obstacle_max_height:
        raise ValueError("obstacle heights must satisfy 0 <= minimum < maximum")
    for name in ("camera_height_range", "fov_range", "speed_range", "path_length_range"):
        low, high = getattr(args, name)
        if not (0 < low <= high):
            raise ValueError(f"invalid --{name.replace('_', '-')}: {low}, {high}")
    if not args.fresh_houses:
        if args.dataset_dir is None:
            raise ValueError("--dataset-dir is required unless --fresh-houses is used")
        split_file = args.dataset_dir / f"{args.split}.jsonl.gz"
        if not split_file.is_file():
            raise FileNotFoundError(split_file)


def resolve_gpu(args: argparse.Namespace) -> tuple[int, int | None, str | None]:
    if args.nvidia_gpu_index is None:
        return int(args.gpu_device), None, None
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    nvidia_uuids: dict[int, str] = {}
    for line in query.splitlines():
        index_text, uuid_text = [field.strip() for field in line.split(",", 1)]
        nvidia_uuids[int(index_text)] = uuid_text.removeprefix("GPU-").lower()
    if args.nvidia_gpu_index not in nvidia_uuids:
        raise RuntimeError(f"nvidia-smi has no GPU index {args.nvidia_gpu_index}")
    wanted = nvidia_uuids[args.nvidia_gpu_index]
    summary = subprocess.run(
        ["vulkaninfo", "--summary"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    matches = re.findall(
        r"GPU(\d+):.*?deviceUUID\s*=\s*([0-9a-fA-F-]+)", summary, flags=re.DOTALL
    )
    for vulkan_text, uuid_text in matches:
        if uuid_text.lower() == wanted:
            return int(vulkan_text), args.nvidia_gpu_index, wanted
    raise RuntimeError(
        f"physical NVIDIA GPU {args.nvidia_gpu_index} UUID {wanted} was not found by Vulkan"
    )


def isolated_cloud_controller(
    runtime_root: Path, *, physical_gpu_index: int, **controller_kwargs: Any
) -> Any:
    """Create a CloudRendering controller backed by the isolated runtime cache.

    Passing ``local_executable_path`` to AI2-THOR creates an ``ExternalBuild``
    whose platform is hard-coded to Linux/X11, even when the executable itself
    is a CloudRendering build.  Override only ``base_dir`` instead, so the
    normal CloudRendering build and CUDA-UUID-to-Vulkan mapping stay active
    without changing HOME for this process or any other process.
    """

    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    isolated_base = str(Path(runtime_root) / "runtime_home/.ai2thor")

    class IsolatedCloudController(Controller):
        @property
        def base_dir(self) -> str:
            return isolated_base

    return IsolatedCloudController(
        platform=CloudRendering,
        gpu_device=int(physical_gpu_index),
        **controller_kwargs,
    )


def load_house(dataset_dir: Path, split: str, index: int) -> dict[str, Any]:
    path = dataset_dir / f"{split}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for current, line in enumerate(stream):
            if current == index:
                return json.loads(line)
    raise IndexError(index)


def upgrade_generated_house_schema(house: dict[str, Any]) -> None:
    """Adapt the legacy ProcTHOR generator material schema to current AI2-THOR."""
    house.setdefault("metadata", {})["schema"] = "1.0.0"
    metadata = house["metadata"]
    agent_pose = metadata.get("agent")
    if agent_pose is not None:
        poses = metadata.setdefault("agentPoses", {})
        for embodiment in ("default", "arm", "stretch"):
            poses[embodiment] = copy.deepcopy(agent_pose)
        locobot_pose = copy.deepcopy(agent_pose)
        locobot_pose.pop("standing", None)
        poses["locobot"] = locobot_pose
    for room in house.get("rooms", []):
        if isinstance(room.get("floorMaterial"), str):
            room["floorMaterial"] = {"name": room["floorMaterial"]}
    for wall in house.get("walls", []):
        if isinstance(wall.get("material"), str):
            material: dict[str, Any] = {"name": wall["material"]}
            if "color" in wall:
                material["color"] = wall["color"]
            wall["material"] = material
    parameters = house.get("proceduralParameters") or {}
    if isinstance(parameters.get("ceilingMaterial"), str):
        material = {"name": parameters["ceilingMaterial"]}
        if "ceilingColor" in parameters:
            material["color"] = parameters["ceilingColor"]
        parameters["ceilingMaterial"] = material
    legacy_openings = [
        opening
        for key in ("doors", "windows")
        for opening in house.get(key, [])
        if "boundingBox" in opening
    ]
    if legacy_openings:
        from procthor.databases import asset_id_database

        for opening in legacy_openings:
            bounding_box = opening.pop("boundingBox")
            offset = opening.pop("assetOffset")
            asset = asset_id_database[opening["assetId"]]
            opening["holePolygon"] = [bounding_box["min"], bounding_box["max"]]
            opening["assetPosition"] = {
                "x": (
                    bounding_box["min"]["x"]
                    + offset["x"]
                    + asset["boundingBox"]["x"] / 2
                ),
                "y": (
                    bounding_box["min"]["y"]
                    + offset["y"]
                    + asset["boundingBox"]["y"] / 2
                ),
                "z": 0,
            }


def generate_fresh_house(
    args: argparse.Namespace, generation_seed: int
) -> tuple[dict[str, Any], dict[str, str]]:
    """Generate and simulator-validate a house not present in ProcTHOR-10K."""
    from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator

    controller = isolated_cloud_controller(
        args.procthor_runtime_root,
        physical_gpu_index=args.gpu_index,
        scene="Procedural",
        width=300,
        height=300,
        quality="Low",
        renderDepthImage=False,
        makeAgentsVisible=False,
    )
    try:
        original_step = controller.step

        def checked_step(*step_args: Any, **step_kwargs: Any) -> Any:
            action = step_kwargs.get("action")
            if action is None and step_args:
                action = step_args[0]
            if action == "CreateHouse" and isinstance(step_kwargs.get("house"), dict):
                upgrade_generated_house_schema(step_kwargs["house"])
            event = original_step(*step_args, **step_kwargs)
            if not event:
                print(
                    "ProcTHOR generator simulator failure: "
                    f"action={action} error={event.metadata.get('errorMessage')}",
                    file=sys.stderr,
                    flush=True,
                )
            return event

        controller.step = checked_step
        generator = HouseGenerator(
            split=args.split,
            seed=generation_seed,
            room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
            controller=controller,
        )
        house, _ = generator.sample()
        upgrade_generated_house_schema(house.data)
        warnings = house.validate(controller)
        if warnings:
            # ProcTHOR reports room-level navigability issues as warnings.  A
            # generated house can still contain a large valid navmesh component
            # and produce a fully reachable trajectory.  Keep these warnings in
            # the session metadata and let the path/BEV quality gates decide
            # whether the sampled session is usable.
            print(
                f"Fresh-house validation warnings (recorded): {warnings}",
                file=sys.stderr,
                flush=True,
            )
        return house.data, warnings
    finally:
        controller.stop()


def xyz(value: dict[str, Any]) -> np.ndarray:
    return np.asarray([value["x"], value["y"], value["z"]], dtype=np.float64)


def path_length(points: Iterable[dict[str, Any]]) -> float:
    arrays = [xyz(p) for p in points]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(arrays, arrays[1:])))


def deduplicate_path(points: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    clean: list[dict[str, float]] = []
    for point in points:
        normalized = {axis: float(point[axis]) for axis in ("x", "y", "z")}
        if not clean or np.linalg.norm(xyz(normalized) - xyz(clean[-1])) > 1e-5:
            clean.append(normalized)
    return clean


def choose_shortest_path(
    reachable: list[dict[str, Any]],
    rng: random.Random,
    length_range: tuple[float, float],
    attempts: int,
    grid_size: float = 0.25,
) -> tuple[list[dict[str, float]], float]:
    if len(reachable) < 2:
        raise RuntimeError("house has fewer than two reachable positions")
    low, high = length_range
    points = [
        {axis: float(point[axis]) for axis in ("x", "y", "z")} for point in reachable
    ]
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(points):
        key = (round(point["x"] / grid_size), round(point["z"] / grid_size))
        buckets.setdefault(key, []).append(index)

    def neighbors(index: int) -> Iterable[tuple[int, float]]:
        point = points[index]
        key = (round(point["x"] / grid_size), round(point["z"] / grid_size))
        for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for candidate in buckets.get((key[0] + dx, key[1] + dz), []):
                delta = xyz(points[candidate]) - xyz(point)
                distance = float(np.linalg.norm(delta))
                if abs(float(delta[1])) <= 0.15 and 0.5 * grid_size <= distance <= 1.3 * grid_size:
                    yield candidate, distance

    start_indices = list(range(len(points)))
    rng.shuffle(start_indices)
    for start_index in start_indices[: min(attempts, len(start_indices))]:
        parent: dict[int, int | None] = {start_index: None}
        distance_from_start: dict[int, float] = {start_index: 0.0}
        queue: deque[int] = deque([start_index])
        candidates: list[int] = []
        while queue:
            current = queue.popleft()
            current_distance = distance_from_start[current]
            if low <= current_distance <= high:
                candidates.append(current)
            if current_distance >= high:
                continue
            for neighbor, edge_length in neighbors(current):
                if neighbor in parent:
                    continue
                next_distance = current_distance + edge_length
                if next_distance <= high + grid_size:
                    parent[neighbor] = current
                    distance_from_start[neighbor] = next_distance
                    queue.append(neighbor)
        if candidates:
            goal_index = rng.choice(candidates)
            indices: list[int] = []
            cursor: int | None = goal_index
            while cursor is not None:
                indices.append(cursor)
                cursor = parent[cursor]
            indices.reverse()
            path = [points[index] for index in indices]
            length = path_length(path)
            if low <= length <= high:
                return path, length
    raise RuntimeError(
        f"reachable grid has {len(points)} points but no connected {low:.1f}-{high:.1f} m path"
    )


def interpolate_path(points: list[dict[str, float]], spacing: float) -> list[dict[str, float]]:
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    cumulative = [0.0]
    arrays = [xyz(p) for p in points]
    for first, second in zip(arrays, arrays[1:]):
        cumulative.append(cumulative[-1] + float(np.linalg.norm(second - first)))
    total = cumulative[-1]
    distances = list(np.arange(0.0, total, spacing, dtype=np.float64))
    if not distances or total - distances[-1] > 1e-6:
        distances.append(total)
    samples: list[dict[str, float]] = []
    segment = 0
    for distance in distances:
        while segment + 1 < len(cumulative) - 1 and distance > cumulative[segment + 1]:
            segment += 1
        segment_length = cumulative[segment + 1] - cumulative[segment]
        alpha = 0.0 if segment_length <= 1e-9 else (distance - cumulative[segment]) / segment_length
        point = arrays[segment] * (1.0 - alpha) + arrays[segment + 1] * alpha
        samples.append({axis: float(value) for axis, value in zip(("x", "y", "z"), point)})
    return samples


def yaw_toward(current: dict[str, float], following: dict[str, float]) -> float:
    dx = following["x"] - current["x"]
    dz = following["z"] - current["z"]
    return float(math.degrees(math.atan2(dx, dz)) % 360.0)


def floor_levels(house: dict[str, Any]) -> list[float]:
    """Return the distinct physical floor heights encoded by ProcTHOR rooms."""
    values: list[float] = []
    for room in house.get("rooms", []):
        for point in room.get("floorPolygon", []):
            if "y" in point:
                values.append(float(point["y"]))
    if not values:
        raise RuntimeError("house has no room floorPolygon heights")
    return sorted({round(value, 6) for value in values})


def floor_below(position_y: float, levels: list[float]) -> float:
    """Select the highest modeled floor not above the agent reference point."""
    candidates = [level for level in levels if level <= position_y + 1e-4]
    if not candidates:
        return min(levels, key=lambda level: abs(level - position_y))
    return max(candidates)


def camera_intrinsics(width: int, height: int, vertical_fov_degrees: float) -> dict[str, Any]:
    vertical = math.radians(vertical_fov_degrees)
    fy = 0.5 * height / math.tan(0.5 * vertical)
    fx = fy
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    horizontal = math.degrees(2.0 * math.atan(0.5 * width / fx))
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "skew": 0.0,
        "vertical_fov_degrees": vertical_fov_degrees,
        "horizontal_fov_degrees": horizontal,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "fov_parameter_axis": "vertical (Unity Camera.fieldOfView)",
        "pixel_convention": "integer pixel centers; principal point at ((W-1)/2, (H-1)/2)",
    }


def camera_matrices(
    camera_position: dict[str, Any], yaw_degrees: float, horizon_degrees: float
) -> tuple[list[list[float]], list[list[float]]]:
    yaw = math.radians(yaw_degrees)
    horizon = math.radians(horizon_degrees)
    right = np.asarray([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float64)
    up = np.asarray(
        [math.sin(yaw) * math.sin(horizon), math.cos(horizon), math.cos(yaw) * math.sin(horizon)],
        dtype=np.float64,
    )
    forward = np.asarray(
        [math.sin(yaw) * math.cos(horizon), -math.sin(horizon), math.cos(yaw) * math.cos(horizon)],
        dtype=np.float64,
    )
    rotation = np.column_stack([right, up, forward])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz(camera_position)
    inverse = np.linalg.inv(transform)
    return transform.tolist(), inverse.tolist()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def depth_preview(depth: np.ndarray) -> Image.Image:
    finite = depth[np.isfinite(depth) & (depth > 0)]
    ceiling = float(np.percentile(finite, 95)) if finite.size else 1.0
    normalized = np.clip(depth / max(ceiling, 1e-6), 0.0, 1.0)
    gray = np.asarray((1.0 - normalized) * 255.0, dtype=np.uint8)
    return Image.fromarray(gray).convert("RGB")


def make_preview(rgb_paths: list[Path], depth_paths: list[Path], output: Path) -> None:
    indices = sorted(set([0, len(rgb_paths) // 2, len(rgb_paths) - 1]))
    cells: list[Image.Image] = []
    for index in indices:
        rgb = Image.open(rgb_paths[index]).convert("RGB")
        with np.load(depth_paths[index]) as archive:
            depth = archive["depth_m"]
        pair = Image.new("RGB", (rgb.width, rgb.height * 2))
        pair.paste(rgb, (0, 0))
        pair.paste(depth_preview(depth), (0, rgb.height))
        cells.append(pair)
    montage = Image.new("RGB", (sum(cell.width for cell in cells), max(cell.height for cell in cells)))
    offset = 0
    for cell in cells:
        montage.paste(cell, (offset, 0))
        offset += cell.width
    montage.thumbnail((1920, 960), Image.Resampling.LANCZOS)
    montage.save(output, compress_level=6)


def render_session(
    args: argparse.Namespace,
    rng: random.Random,
    session_index: int,
    house_index: int | None,
    house: dict[str, Any],
    generation_seed: int | None = None,
    generation_warnings: dict[str, str] | None = None,
) -> dict[str, Any]:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    fov = rng.uniform(*args.fov_range)
    camera_height = rng.uniform(*args.camera_height_range)
    speed = rng.uniform(*args.speed_range)
    controller = None
    temp_dir: Path | None = None
    try:
        controller = Controller(
            scene=house,
            platform=CloudRendering,
            gpu_device=args.resolved_gpu_device,
            width=args.width,
            height=args.height,
            fieldOfView=fov,
            agentMode="locobot",
            gridSize=0.25,
            snapToGrid=False,
            rotateStepDegrees=30,
            renderDepthImage=True,
            makeAgentsVisible=False,
            quality="Low",
        )
        event = controller.step(action="GetReachablePositions")
        if not event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(event.metadata.get("errorMessage", "GetReachablePositions failed"))
        reachable = event.metadata.get("actionReturn") or []
        path, geodesic_length = choose_shortest_path(
            reachable,
            rng,
            tuple(args.path_length_range),
            args.path_attempts,
        )
        samples = interpolate_path(path, speed / args.capture_hz)
        if len(samples) < 2:
            raise RuntimeError("interpolated trajectory has fewer than two frames")

        levels = floor_levels(house)
        initial_yaw = yaw_toward(samples[0], samples[1])
        initial_floor_y = floor_below(samples[0]["y"], levels)
        event = controller.step(
            action="AddThirdPartyCamera",
            position={
                "x": samples[0]["x"],
                "y": initial_floor_y + camera_height,
                "z": samples[0]["z"],
            },
            rotation={"x": 0.0, "y": initial_yaw, "z": 0.0},
            fieldOfView=fov,
        )
        if not event.metadata.get("lastActionSuccess", False):
            raise RuntimeError(
                event.metadata.get("errorMessage", "AddThirdPartyCamera failed")
            )

        if generation_seed is None:
            source_name = f"{args.split}_{house_index:05d}"
        else:
            source_name = f"fresh_{generation_seed:010d}"
        session_name = f"session_{session_index:06d}_procthor_{source_name}"
        final_dir = args.output / session_name
        temp_dir = args.output / f".{session_name}.tmp"
        if final_dir.exists():
            if not args.overwrite:
                raise FileExistsError(final_dir)
            shutil.rmtree(final_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
        camera_dir = temp_dir / "camera"
        depth_dir = temp_dir / "depth"
        camera_dir.mkdir(parents=True)
        depth_dir.mkdir(parents=True)
        if generation_seed is not None:
            write_json(temp_dir / "house.json", house)

        intrinsics = camera_intrinsics(args.width, args.height, fov)
        write_json(temp_dir / "camera_intrinsics.json", intrinsics)
        extent_keys = [extent_key(extent) for extent in args.bev_extents]
        for key in extent_keys:
            for modality in ("masked", "complete"):
                (temp_dir / key / modality).mkdir(parents=True)
            for merged_extent in args.merged_extents:
                for kind in ("masked", "complete"):
                    (temp_dir / key / merged_modality(kind, merged_extent)).mkdir(
                        parents=True
                    )
        print(
            "Building complete ProcTHOR mesh-voxel ground truth "
            f"at {args.voxel_size:g} m resolution...",
            flush=True,
        )
        (
            full_truth,
            full_valid_coverage,
            truth_lower_bound,
            voxel_statistics,
        ) = build_complete_truth(
            controller,
            house,
            floor_y=initial_floor_y,
            voxel_size=args.voxel_size,
            obstacle_min_height=args.obstacle_min_height,
            obstacle_max_height=args.obstacle_max_height,
        )
        truth_dir = temp_dir / "bev_truth"
        truth_dir.mkdir()
        Image.fromarray(full_truth).save(
            truth_dir / "complete_global.png", compress_level=6
        )
        accumulators = {
            key: BEVAccumulator(extent, args.bev_size)
            for key, extent in zip(extent_keys, args.bev_extents)
        }
        extrinsics_stream = (temp_dir / "camera_extrinsics.jsonl").open("w", encoding="utf-8")
        trajectory_stream = (temp_dir / "ground_truth_trajectory.jsonl").open("w", encoding="utf-8")
        rgb_paths: list[Path] = []
        depth_paths: list[Path] = []
        measured_heights: list[float] = []
        single_complete_gt_void_ratios: list[float] = []
        void_coverage_algorithm = str(
            voxel_statistics["strict_void_coverage"]["algorithm"]
        )
        try:
            last_yaw = 0.0
            for frame_index, position in enumerate(samples):
                if frame_index + 1 < len(samples):
                    last_yaw = yaw_toward(position, samples[frame_index + 1])
                event = controller.step(
                    action="TeleportFull",
                    x=position["x"],
                    y=position["y"],
                    z=position["z"],
                    rotation={"x": 0.0, "y": last_yaw, "z": 0.0},
                    horizon=0.0,
                    forceAction=False,
                )
                if not event.metadata.get("lastActionSuccess", False):
                    raise RuntimeError(
                        f"frame {frame_index} teleport failed: {event.metadata.get('errorMessage')}"
                    )
                local_floor_y = floor_below(position["y"], levels)
                event = controller.step(
                    action="UpdateThirdPartyCamera",
                    thirdPartyCameraId=0,
                    position={
                        "x": position["x"],
                        "y": local_floor_y + camera_height,
                        "z": position["z"],
                    },
                    rotation={"x": 0.0, "y": last_yaw, "z": 0.0},
                    fieldOfView=fov,
                )
                if not event.metadata.get("lastActionSuccess", False):
                    raise RuntimeError(
                        f"frame {frame_index} camera update failed: "
                        f"{event.metadata.get('errorMessage')}"
                    )
                if not event.third_party_camera_frames or not event.third_party_depth_frames:
                    raise RuntimeError("third-party RGB-D camera returned no image")
                rgb = np.asarray(event.third_party_camera_frames[0], dtype=np.uint8)
                depth = np.asarray(event.third_party_depth_frames[0], dtype=np.float32)
                if rgb.shape == (args.height, args.width, 4):
                    rgb = rgb[:, :, :3]
                if rgb.shape != (args.height, args.width, 3):
                    raise RuntimeError(f"unexpected RGB shape {rgb.shape}")
                if depth.shape != (args.height, args.width):
                    raise RuntimeError(f"unexpected depth shape {depth.shape}")
                finite_ratio = float(np.isfinite(depth).mean())
                if finite_ratio < 0.99:
                    raise RuntimeError(f"depth finite ratio is only {finite_ratio:.4f}")

                rgb_path = camera_dir / f"frame_{frame_index:06d}.png"
                depth_path = depth_dir / f"frame_{frame_index:06d}.npz"
                Image.fromarray(rgb).save(rgb_path, compress_level=4)
                np.savez_compressed(depth_path, depth_m=depth)
                rgb_paths.append(rgb_path)
                depth_paths.append(depth_path)

                agent = event.metadata["agent"]
                cameras = event.metadata.get("thirdPartyCameras") or []
                if not cameras:
                    raise RuntimeError("third-party camera metadata is missing")
                camera = cameras[0]
                camera_position = camera["position"]
                camera_rotation = camera["rotation"]
                yaw = float(camera_rotation["y"])
                horizon = float(camera_rotation["x"])
                world_from_camera, camera_from_world = camera_matrices(
                    camera_position, yaw, horizon
                )
                measured_height = float(camera_position["y"] - local_floor_y)
                measured_heights.append(measured_height)
                yaw_radians = math.radians(yaw)
                bev_forward = np.asarray(
                    [math.sin(yaw_radians), math.cos(yaw_radians)],
                    dtype=np.float64,
                )
                bev_right = np.asarray(
                    [math.cos(yaw_radians), -math.sin(yaw_radians)],
                    dtype=np.float64,
                )
                agent_position_world = [
                    float(agent["position"][axis]) for axis in ("x", "y", "z")
                ]
                bev_extrinsic = {
                    "agent_position_world_m": agent_position_world,
                    "bev_forward_xz": bev_forward.tolist(),
                    "bev_right_xz": bev_right.tolist(),
                    "world_from_bev_planar": [
                        [
                            float(bev_right[0]),
                            float(bev_forward[0]),
                            agent_position_world[0],
                        ],
                        [
                            float(bev_right[1]),
                            float(bev_forward[1]),
                            agent_position_world[2],
                        ],
                        [0.0, 0.0, 1.0],
                    ],
                }
                common = {
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index / args.capture_hz,
                    "agent_position": {k: float(agent["position"][k]) for k in ("x", "y", "z")},
                    "agent_rotation_degrees": {k: float(agent["rotation"][k]) for k in ("x", "y", "z")},
                    "local_floor_y": local_floor_y,
                    "camera_position": {k: float(camera_position[k]) for k in ("x", "y", "z")},
                    "camera_rotation_degrees": {
                        k: float(camera_rotation[k]) for k in ("x", "y", "z")
                    },
                    "camera_horizon_degrees": horizon,
                    "world_from_camera_unity": world_from_camera,
                    "camera_from_world_unity": camera_from_world,
                    **bev_extrinsic,
                }
                frame_name = f"frame_{frame_index:06d}.png"
                bev_record: dict[str, Any] = {}
                for key, extent in zip(extent_keys, args.bev_extents):
                    complete = render_ego_obstacle_map(
                        full_scene_map=full_truth,
                        lower_bound=truth_lower_bound,
                        source_meters_per_pixel=args.voxel_size,
                        position=agent_position_world,
                        forward=bev_forward,
                        right=bev_right,
                        size=args.bev_size,
                        extent=extent,
                    )
                    masked = render_visibility_masked_map(
                        complete,
                        horizontal_fov_degrees=intrinsics[
                            "horizontal_fov_degrees"
                        ],
                    )
                    validate_bev_frame(complete, masked)
                    single_valid_coverage = render_ego_obstacle_map(
                        full_scene_map=full_valid_coverage.astype(
                            np.uint8, copy=False
                        ),
                        lower_bound=truth_lower_bound,
                        source_meters_per_pixel=args.voxel_size,
                        position=agent_position_world,
                        forward=bev_forward,
                        right=bev_right,
                        size=args.bev_size,
                        extent=extent,
                    ).astype(bool, copy=False)
                    single_valid_coverage = repair_valid_coverage(
                        single_valid_coverage
                    )
                    void_ratio = enforce_single_bev_void_limit(
                        single_valid_coverage,
                        threshold=args.max_single_bev_void_ratio,
                        frame_id=frame_index,
                        extent_m=extent,
                    )
                    single_complete_gt_void_ratios.append(void_ratio)
                    Image.fromarray(complete).save(
                        temp_dir / key / "complete" / frame_name,
                        compress_level=6,
                    )
                    Image.fromarray(masked).save(
                        temp_dir / key / "masked" / frame_name,
                        compress_level=6,
                    )
                    accumulator = accumulators[key]
                    accumulator.update(complete, masked, bev_extrinsic)
                    merged_record: dict[str, Any] = {}
                    for merged_extent in args.merged_extents:
                        images: dict[str, np.ndarray] = {}
                        extent_record: dict[str, Any] = {}
                        for kind in ("masked", "complete"):
                            merged, merged_meta = accumulator.render_ego(
                                kind, bev_extrinsic, merged_extent
                            )
                            images[kind] = merged
                            Image.fromarray(merged).save(
                                temp_dir
                                / key
                                / merged_modality(kind, merged_extent)
                                / frame_name,
                                compress_level=6,
                            )
                            extent_record[kind] = merged_meta
                        validate_bev_frame(
                            images["complete"],
                            images["masked"],
                            allow_complete_unknown=True,
                        )
                        merged_record[f"{merged_extent:g}m"] = extent_record
                    bev_record[key] = {
                        "current_shape": [args.bev_size, args.bev_size],
                        "current_extent_m": extent,
                        "current_meters_per_pixel": extent / args.bev_size,
                        "single_complete_gt_void_ratio": void_ratio,
                        "single_complete_gt_valid_ratio": 1.0 - void_ratio,
                        "void_coverage_algorithm": void_coverage_algorithm,
                        "files": {
                            "complete": f"{key}/complete/{frame_name}",
                            "masked": f"{key}/masked/{frame_name}",
                        },
                        "merged": merged_record,
                    }
                extrinsics_stream.write(json.dumps(common) + "\n")
                trajectory_stream.write(
                    json.dumps(
                        {
                            **common,
                            "rgb": str(rgb_path.relative_to(temp_dir)),
                            "depth": str(depth_path.relative_to(temp_dir)),
                            "depth_min_m": float(np.min(depth)),
                            "depth_max_m": float(np.max(depth)),
                            "bev": bev_record,
                        }
                    )
                    + "\n"
                )
        finally:
            extrinsics_stream.close()
            trajectory_stream.close()

        median_height = float(np.median(measured_heights))
        if abs(median_height - camera_height) > 0.03:
            raise RuntimeError(
                f"requested camera height {camera_height:.4f} m but measured {median_height:.4f} m"
            )
        make_preview(rgb_paths, depth_paths, temp_dir / "preview.png")
        metadata = {
            "schema_version": 1,
            "dataset": "procthor-generated" if generation_seed is not None else "procthor-10k",
            "split": args.split,
            "house_index": house_index,
            "fresh_generation_seed": generation_seed,
            "fresh_generation_validation_warnings": generation_warnings,
            "fresh_house_schema_adapter": (
                "legacy generator output upgraded to current material, wall-opening, "
                "schema-version, and embodiment-pose fields"
                if generation_seed is not None
                else None
            ),
            "scene_name": event.metadata.get("sceneName"),
            "random_seed": args.seed,
            "ai2thor_vulkan_gpu_device": args.resolved_gpu_device,
            "nvidia_gpu_index": args.resolved_nvidia_gpu_index,
            "nvidia_gpu_uuid": args.resolved_nvidia_gpu_uuid,
            "reachable_positions": len(reachable),
            "requested_camera_height_m": camera_height,
            "measured_camera_height_m": median_height,
            "camera_height_reference": "local ProcTHOR room floorPolygon to optical center",
            "camera_source": "AI2-THOR synchronized third-party RGB-D camera",
            "agent_rendering": "disabled via Initialize.makeAgentsVisible=false",
            "scene_floor_levels_y": levels,
            "vertical_fov_degrees": fov,
            "robot_speed_mps": speed,
            "capture_hz": args.capture_hz,
            "path_length_m": geodesic_length,
            "path_planner": "BFS shortest path over simulator GetReachablePositions grid",
            "reachable_grid_path": path,
            "frames": len(rgb_paths),
            "bev": {
                "size": args.bev_size,
                "extent_classes_m": list(args.bev_extents),
                "merged_normalized_extents_m": list(args.merged_extents),
                "obstacle_height_band_m": [
                    args.obstacle_min_height,
                    args.obstacle_max_height,
                ],
                "voxel_size_m": args.voxel_size,
                "truth_source": voxel_statistics["truth_source"],
                "global_truth_file": "bev_truth/complete_global.png",
                "values": {"occupied": 0, "unknown": 112, "free": 255},
                "orientation": "ego-centric; robot centered; forward is up",
                "masking": (
                    "exact grid shadowcasting plus anchored radial leak guard and "
                    "origin-connected free-space cleanup"
                ),
                "visibility_algorithm": VISIBILITY_ALGORITHM,
                "void_check": {
                    "version": VOID_CHECK_VERSION,
                    "modality": "single/current Complete GT geometry-valid mask",
                    "coverage_algorithm": void_coverage_algorithm,
                    "filter_contract": VOID_FILTER_CONTRACT,
                    "repair": VOID_REPAIR_ALGORITHM,
                    "ratio_definition": (
                        "count(not geometry_valid) / total_pixels after complete-"
                        "scene and ego-grid enclosed-hole repair"
                    ),
                    "max_allowed_ratio": args.max_single_bev_void_ratio,
                    "comparison": "reject when ratio > max_allowed_ratio",
                },
                "void_statistics": {
                    "single_complete_gt_bevs_checked": len(
                        single_complete_gt_void_ratios
                    ),
                    "session_max_single_complete_gt_void_ratio": max(
                        single_complete_gt_void_ratios, default=0.0
                    ),
                    "max_allowed_single_complete_gt_void_ratio": (
                        args.max_single_bev_void_ratio
                    ),
                    "coverage_algorithm": void_coverage_algorithm,
                    "filter_contract": VOID_FILTER_CONTRACT,
                    "passed": True,
                },
                "fusion_rule": (
                    "temporal union of observed cells; merged masked and complete "
                    "values always come from the same accumulated static truth"
                ),
                "merged_fusion_version": 2,
                "robot_origin_pixel_row_column": [
                    args.bev_size // 2,
                    args.bev_size // 2,
                ],
            },
            "voxel_statistics": voxel_statistics,
            "depth_format": "lossless npz containing float32 metric depth under key depth_m",
            "coordinate_convention": (
                "Unity world and camera-local axes: +x right, +y up, +z forward. "
                "world_from_camera_unity maps camera-local [x_right,y_up,z_forward,1] to world."
            ),
        }
        write_json(temp_dir / "metadata.json", metadata)
        validation = {
            "valid": True,
            "rgb_frames": len(rgb_paths),
            "depth_frames": len(depth_paths),
            "extrinsics_lines": sum(1 for _ in (temp_dir / "camera_extrinsics.jsonl").open()),
            "trajectory_lines": sum(1 for _ in (temp_dir / "ground_truth_trajectory.jsonl").open()),
            "bev_frame_counts": {
                key: {
                    modality: len(list((temp_dir / key / modality).glob("*.png")))
                    for modality in (
                        ["masked", "complete"]
                        + [
                            merged_modality(kind, merged_extent)
                            for merged_extent in args.merged_extents
                            for kind in ("masked", "complete")
                        ]
                    )
                }
                for key in extent_keys
            },
        }
        write_json(temp_dir / "validation.json", validation)
        (temp_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
        os.replace(temp_dir, final_dir)
        return {"session": session_name, "path": str(final_dir), **metadata, **validation}
    finally:
        if controller is not None:
            controller.stop()
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    (
        args.resolved_gpu_device,
        args.resolved_nvidia_gpu_index,
        args.resolved_nvidia_gpu_uuid,
    ) = resolve_gpu(args)
    print(
        "GPU mapping: "
        f"nvidia={args.resolved_nvidia_gpu_index} uuid={args.resolved_nvidia_gpu_uuid} "
        f"vulkan={args.resolved_gpu_device}",
        flush=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    reports: list[dict[str, Any]] = []
    used_houses: set[int] = set()
    for session_index in range(args.sessions):
        errors: list[str] = []
        for _ in range(args.house_attempts):
            generation_seed: int | None = None
            generation_warnings: dict[str, str] | None = None
            if args.fresh_houses:
                generation_seed = rng.randrange(2**31)
                house_index = None
                print(f"Generating fresh ProcTHOR house seed={generation_seed}", flush=True)
                try:
                    house, generation_warnings = generate_fresh_house(
                        args, generation_seed
                    )
                except Exception as exc:
                    errors.append(
                        f"fresh seed {generation_seed}: {type(exc).__name__}: {exc}"
                    )
                    print(errors[-1], file=sys.stderr, flush=True)
                    continue
            else:
                house_index = rng.randrange(SPLIT_COUNTS[args.split])
                if house_index in used_houses:
                    continue
                used_houses.add(house_index)
                house = load_house(args.dataset_dir, args.split, house_index)
            try:
                reports.append(
                    render_session(
                        args,
                        rng,
                        session_index,
                        house_index,
                        house,
                        generation_seed=generation_seed,
                        generation_warnings=generation_warnings,
                    )
                )
                break
            except Exception as exc:
                source = (
                    f"fresh seed {generation_seed}"
                    if generation_seed is not None
                    else f"house {house_index}"
                )
                errors.append(f"{source}: {type(exc).__name__}: {exc}")
                print(errors[-1], file=sys.stderr, flush=True)
        else:
            raise RuntimeError(
                f"session {session_index} failed after {args.house_attempts} houses:\n"
                + "\n".join(errors)
            )
    manifest = {
        "schema_version": 1,
        "collector": "procthor_randomized_session.py",
        "created_unix_seconds": time.time(),
        "seed": args.seed,
        "sessions_requested": args.sessions,
        "sessions_completed": len(reports),
        "sessions": reports,
    }
    write_json(args.output / "collection_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
