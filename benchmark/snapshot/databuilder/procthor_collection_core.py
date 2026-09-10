#!/usr/bin/env python3
"""Shared 25-path ProcTHOR collection core with the Habitat schema-v4 contract."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from gpu_render_startup import (
    assert_process_gpu_binding,
    nvidia_gpu_inventory,
    refresh_ai2thor_cuda_vulkan_mapping,
    renderer_startup_gate,
)
from bev_visibility import (
    DEFAULT_MAX_SINGLE_BEV_VOID_RATIO,
    VOID_CHECK_VERSION,
    VOID_FILTER_CONTRACT,
    VOID_REPAIR_ALGORITHM,
    SingleBEVVoidRatioExceeded,
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
)
from procthor_randomized_session import (
    SPLIT_COUNTS,
    camera_matrices,
    choose_shortest_path,
    floor_below,
    floor_levels,
    generate_fresh_house,
    isolated_cloud_controller,
    interpolate_path,
    load_house,
    yaw_toward,
)


GT_QUALITY_VERSION = 6
SCHEMA_VERSION = GT_QUALITY_VERSION
DEFAULT_SESSIONS_PER_SCENE = 25
DEFAULT_MAX_INITIALIZATION_ATTEMPTS = 200
DEFAULT_AGENT_RADIUS_M = 0.10
RUNNER_RESERVATION_MARKER = ".PROCTHOR_RUNNER_RESERVED"


def default_output_root() -> Path:
    medical = Path("/data/disk_7t/diwen/VGGT_DATA/data_build_unlimited")
    return medical if medical.is_dir() else Path.home() / "data" / "BEV"


def discover_builtin_dataset() -> Path | None:
    candidates = []
    configured = os.environ.get("PROCTHOR_DATASET_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path.home() / "data/ProcTHOR/dataset/procthor-10k",
            Path("/home/liudiwen/data/ProcTHOR/runtime_home/.prior/datasets/allenai/procthor-10k"),
            Path("/data/disk_7t/diwen/VGGT_DATA/ProcTHOR/runtime_home/.prior/datasets/allenai/procthor-10k"),
        ]
    )
    for candidate in candidates:
        if (candidate / "train.jsonl.gz").is_file():
            return candidate
        if candidate.is_dir():
            matches = sorted(candidate.glob("*/train.jsonl.gz"))
            if matches:
                return matches[0].parent
    return None


def locate_ai2thor_runtime() -> tuple[Path, Path]:
    """Locate the isolated runtime without changing HOME or global configuration."""

    roots = (
        Path("/home/liudiwen/data/ProcTHOR"),
        Path("/data/disk_7t/diwen/VGGT_DATA/ProcTHOR"),
    )
    for root in roots:
        releases = root / "runtime_home/.ai2thor/releases"
        if not releases.is_dir():
            continue
        executables = sorted(
            path
            for path in releases.glob("*/thor-CloudRendering-*")
            if path.is_file() and os.access(path, os.X_OK)
        )
        if executables:
            return root, executables[-1]
    raise FileNotFoundError("the isolated ProcTHOR AI2-THOR runtime was not found")


def expose_bundled_vulkaninfo(runtime_root: Path) -> Path:
    """Make the isolated Vulkan probe visible to AI2-THOR itself.

    AI2-THOR invokes ``vulkaninfo`` by name whenever ``gpu_device`` is used,
    even after the collector has already mapped the physical NVIDIA UUID to a
    Vulkan device.  The servers intentionally keep the tool inside the
    ProcTHOR runtime instead of installing it system-wide, so prepend only that
    private binary directory to this collector process' PATH.
    """

    system = shutil.which("vulkaninfo")
    if system is not None:
        return Path(system)
    bundled = runtime_root / "tools/vulkan-tools/root/usr/bin/vulkaninfo"
    if not bundled.is_file() or not os.access(bundled, os.X_OK):
        raise FileNotFoundError("vulkaninfo was not found in PATH or ProcTHOR runtime")
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{bundled.parent}{os.pathsep}{current}"
    return bundled


def resolve_physical_gpu(gpu_index: int, runtime_root: Path) -> tuple[int, str]:
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
    uuids: dict[int, str] = {}
    for line in query.splitlines():
        index_text, uuid_text = [value.strip() for value in line.split(",", 1)]
        uuids[int(index_text)] = uuid_text.removeprefix("GPU-").lower()
    if gpu_index not in uuids:
        raise RuntimeError(f"nvidia-smi has no physical GPU {gpu_index}")
    wanted = uuids[gpu_index]
    vulkaninfo = shutil.which("vulkaninfo")
    bundled = runtime_root / "tools/vulkan-tools/root/usr/bin/vulkaninfo"
    executable = vulkaninfo or (str(bundled) if bundled.is_file() else None)
    if executable is None:
        raise FileNotFoundError("vulkaninfo was not found")
    summary = subprocess.run(
        [executable, "--summary"], check=True, text=True, capture_output=True
    ).stdout
    for vulkan_text, uuid_text in re.findall(
        r"GPU(\d+):.*?deviceUUID\s*=\s*([0-9a-fA-F-]+)",
        summary,
        flags=re.DOTALL,
    ):
        if uuid_text.lower() == wanted:
            return int(vulkan_text), wanted
    raise RuntimeError(
        f"physical NVIDIA GPU {gpu_index} UUID {wanted} was not found by Vulkan"
    )


def build_parser(mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fresh ProcTHOR scenes and collect 25 paths per scene"
            if mode == "generated"
            else "Collect 25 paths per stored ProcTHOR-10K scene"
        )
    )
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--scene-count", type=int, default=1)
    parser.add_argument(
        "--sessions-per-scene",
        type=int,
        default=DEFAULT_SESSIONS_PER_SCENE,
        help="production default and required collection unit: 25",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--split", choices=sorted(SPLIT_COUNTS), default="train")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--capture-hz", type=float, default=2.0)
    parser.add_argument("--camera-height-range", type=float, nargs=2, default=(0.3, 0.8))
    parser.add_argument(
        "--horizontal-fov-range", type=float, nargs=2, default=(60.0, 120.0)
    )
    parser.add_argument("--speed-range", type=float, nargs=2, default=(0.6, 2.0))
    parser.add_argument("--path-length-range", type=float, nargs=2, default=(3.0, 10.0))
    parser.add_argument("--bev-size", type=int, default=512)
    parser.add_argument("--bev-extents", type=float, nargs="+", default=(6.5,))
    parser.add_argument("--merged-extents", type=float, nargs="+", default=(10.0,))
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--obstacle-min-height", type=float, default=0.0)
    parser.add_argument("--obstacle-max-height", type=float, default=1.4)
    parser.add_argument("--path-attempts", type=int, default=200)
    parser.add_argument(
        "--max-initialization-attempts",
        "--max-session-attempts",
        dest="max_initialization_attempts",
        type=int,
        default=DEFAULT_MAX_INITIALIZATION_ATTEMPTS,
        help=(
            "maximum total session attempts in one initialized scene; "
            "successes and failures both count (legacy alias: "
            "--max-session-attempts)"
        ),
    )
    parser.add_argument(
        "--max-single-bev-void-ratio",
        type=float,
        default=DEFAULT_MAX_SINGLE_BEV_VOID_RATIO,
        help=(
            "abort and delete a session attempt when any current Single "
            "Complete GT has a larger geometry-VOID fraction"
        ),
    )
    parser.add_argument("--max-scene-attempts", type=int, default=0)
    parser.add_argument(
        "--allow-partial-scene-groups",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "commit successful sessions when a scene does not reach its path "
            "target (enabled by default)"
        ),
    )
    parser.add_argument(
        "--allow-partial-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "write COMPLETE even when fewer scene initializations finish "
            "(enabled by default)"
        ),
    )
    if mode == "generated":
        parser.add_argument("--house-generation-attempts", type=int, default=12)
    else:
        parser.add_argument("--dataset-dir", type=Path, default=discover_builtin_dataset())
    return parser


def validate_args(args: argparse.Namespace, mode: str) -> None:
    if args.scene_count <= 0 or args.sessions_per_scene <= 0:
        raise ValueError("scene and session counts must be positive")
    if args.width <= 0 or args.height <= 0 or args.bev_size <= 0:
        raise ValueError("image dimensions must be positive")
    if args.capture_hz <= 0:
        raise ValueError("capture rate must be positive")
    for name in (
        "camera_height_range",
        "horizontal_fov_range",
        "speed_range",
        "path_length_range",
    ):
        low, high = getattr(args, name)
        if not 0 < low <= high:
            raise ValueError(f"invalid {name}: {low}, {high}")
    if args.horizontal_fov_range[1] >= 179.0:
        raise ValueError("horizontal FOV must stay below 179 degrees")
    if not 0.001 <= args.voxel_size <= 0.2:
        raise ValueError("voxel size must be within [0.001, 0.2] metres")
    if not 0 <= args.obstacle_min_height < args.obstacle_max_height:
        raise ValueError("invalid obstacle height band")
    if not args.bev_extents or min(args.bev_extents) <= 0:
        raise ValueError("BEV extents must be positive")
    if not args.merged_extents or min(args.merged_extents) <= 0:
        raise ValueError("merged extents must be positive")
    if args.max_initialization_attempts <= 0:
        raise ValueError("--max-initialization-attempts must be positive")
    if not 0.0 <= args.max_single_bev_void_ratio <= 1.0:
        raise ValueError("--max-single-bev-void-ratio must be within [0, 1]")
    if mode == "builtin":
        if args.dataset_dir is None:
            raise FileNotFoundError(
                "ProcTHOR-10K was not found; pass --dataset-dir containing split JSONL files"
            )
        args.dataset_dir = args.dataset_dir.expanduser().resolve()
        if not (args.dataset_dir / f"{args.split}.jsonl.gz").is_file():
            raise FileNotFoundError(args.dataset_dir / f"{args.split}.jsonl.gz")


def vertical_fov(horizontal_degrees: float, width: int, height: int) -> float:
    horizontal = math.radians(horizontal_degrees)
    return math.degrees(
        2.0 * math.atan(math.tan(horizontal * 0.5) * float(height) / float(width))
    )


def camera_intrinsics(width: int, height: int, horizontal_degrees: float) -> dict[str, Any]:
    horizontal = math.radians(horizontal_degrees)
    focal = 0.5 * width / math.tan(0.5 * horizontal)
    vertical = 2.0 * math.atan(0.5 * height / focal)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return {
        "model": "pinhole",
        "width": width,
        "height": height,
        "fx": focal,
        "fy": focal,
        "cx": cx,
        "cy": cy,
        "K": [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        "horizontal_fov_degrees": horizontal_degrees,
        "vertical_fov_degrees": math.degrees(vertical),
        "pixel_coordinate_convention": (
            "integer coordinates address pixel centers; origin is top-left"
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "scene"


def allocate_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        target = args.run_dir.expanduser().resolve()
        try:
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            marker = target / RUNNER_RESERVATION_MARKER
            entries = list(target.iterdir()) if target.is_dir() else []
            if entries != [marker] or not marker.is_file():
                raise
            # Only the unlimited runner may pre-create a run directory. The
            # one-shot marker is consumed before any collector output exists;
            # arbitrary existing/non-empty paths remain protected.
            marker.unlink()
        return target
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"GPU{args.gpu_index}-PROCTHOR-(\d+)$")
    existing = [
        int(match.group(1))
        for child in root.iterdir()
        if (match := pattern.fullmatch(child.name)) is not None
    ]
    index = max(existing, default=0) + 1
    while True:
        target = root / f"GPU{args.gpu_index}-PROCTHOR-{index}"
        try:
            target.mkdir()
            return target
        except FileExistsError:
            index += 1


def quaternion_yaw(yaw_degrees: float) -> list[float]:
    half = math.radians(yaw_degrees) * 0.5
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def dominant_floor_positions(
    reachable: Sequence[dict[str, Any]], levels: Sequence[float]
) -> tuple[float, list[dict[str, Any]]]:
    groups: dict[float, list[dict[str, Any]]] = {}
    for point in reachable:
        floor = floor_below(float(point["y"]), list(levels))
        groups.setdefault(floor, []).append(point)
    if not groups:
        raise RuntimeError("GetReachablePositions returned no positions")
    floor_y, positions = max(groups.items(), key=lambda item: len(item[1]))
    return floor_y, positions


def strict_validate_bev(
    complete: np.ndarray,
    masked: np.ndarray,
    *,
    extent: float,
    allow_complete_unknown: bool = False,
) -> dict[str, int]:
    if complete.ndim != 2 or complete.shape != masked.shape:
        raise RuntimeError("BEV pair has invalid shape")
    allowed_complete = {0, 112, 255} if allow_complete_unknown else {0, 255}
    if not set(map(int, np.unique(complete))).issubset(allowed_complete):
        raise RuntimeError("complete BEV has invalid labels")
    if not set(map(int, np.unique(masked))).issubset({0, 112, 255}):
        raise RuntimeError("masked BEV has invalid labels")
    known = masked != UNKNOWN_VALUE
    if not known.any():
        raise RuntimeError("masked BEV is entirely unknown")
    if not np.array_equal(masked[known], complete[known]):
        raise RuntimeError("masked known values disagree with complete truth")
    center = complete.shape[0] // 2
    radius = int(math.ceil(DEFAULT_AGENT_RADIUS_M / (extent / complete.shape[0])))
    rows, columns = np.ogrid[: complete.shape[0], : complete.shape[1]]
    footprint = (rows - center) ** 2 + (columns - center) ** 2 <= radius**2
    if np.any(complete[footprint] != 255):
        raise RuntimeError("robot footprint is not completely free")
    if masked[center, center] != 255:
        raise RuntimeError("robot origin is not known-free")
    return {
        "known_cell_count": int(known.sum()),
        "free_cell_count": int((complete == 255).sum()),
        "footprint_radius_pixels": radius,
    }


def create_session_directories(
    root: Path, extent_keys: Sequence[str], merged_extents: Sequence[float]
) -> None:
    (root / "camera").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    for key in extent_keys:
        for modality in ("masked", "complete"):
            (root / key / modality).mkdir(parents=True)
        for merged_extent in merged_extents:
            for kind in ("masked", "complete"):
                (root / key / merged_modality(kind, merged_extent)).mkdir(parents=True)


def render_trajectory_image(
    full_truth: np.ndarray,
    lower_bound: np.ndarray,
    voxel_size: float,
    path: Sequence[dict[str, float]],
    sampled_positions: Sequence[Sequence[float]],
    output: Path,
) -> None:
    rgb = np.repeat(full_truth[..., None], 3, axis=2)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    def pixel(point: Sequence[float] | dict[str, float]) -> tuple[int, int]:
        if isinstance(point, dict):
            x, z = float(point["x"]), float(point["z"])
        else:
            x, z = float(point[0]), float(point[2])
        column = int(round((x - lower_bound[0]) / voxel_size))
        row_unflipped = int(round((z - lower_bound[2]) / voxel_size))
        return column, image.height - 1 - row_unflipped

    path_pixels = [pixel(point) for point in path]
    draw.line(path_pixels, fill=(255, 181, 45), width=max(2, int(0.04 / voxel_size)))
    for point in sampled_positions:
        x, y = pixel(point)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(64, 201, 255))
    for point, color in ((path[0], (49, 233, 129)), (path[-1], (255, 77, 94))):
        x, y = pixel(point)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white")
    image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
    image.save(output)


def make_preview(
    session_dir: Path,
    extent_keys: Sequence[str],
    merged_extents: Sequence[float],
    final_name: str,
    parameters: dict[str, float],
) -> None:
    modalities = ["masked", "complete"]
    for extent in merged_extents:
        modalities.extend(
            [merged_modality("masked", extent), merged_modality("complete", extent)]
        )
    tile, gap, header = 240, 10, 64
    columns = max(4, len(modalities))
    rows = len(extent_keys) + 1
    canvas = Image.new(
        "RGB",
        (columns * tile + (columns + 1) * gap, header + rows * tile + (rows + 1) * gap),
        (12, 16, 22),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 10),
        (
            f"FOV {parameters['horizontal_fov_degrees']:.1f} deg | "
            f"camera {parameters['camera_height_m']:.2f} m | "
            f"speed {parameters['robot_speed_mps']:.2f} m/s"
        ),
        fill=(238, 243, 248),
    )

    def paste(path: Path, column: int, row: int, label: str) -> None:
        contained = ImageOps.contain(Image.open(path).convert("RGB"), (tile, tile - 20))
        x = gap + column * (tile + gap) + (tile - contained.width) // 2
        y = header + gap + row * (tile + gap)
        canvas.paste(contained, (x, y + 18))
        draw.text((gap + column * (tile + gap), y), label, fill=(142, 155, 169))

    paste(session_dir / "camera" / final_name, 0, 0, "final camera")
    paste(session_dir / "ground_truth_trajectory.png", 1, 0, "ground-truth trajectory")
    paste(session_dir / "camera/frame_000000.png", 2, 0, "initial camera")
    for row, key in enumerate(extent_keys, start=1):
        for column, modality in enumerate(modalities):
            paste(session_dir / key / modality / final_name, column, row, f"{key}: {modality}")
    canvas.save(session_dir / "preview.png")


def validate_written_session(
    session_dir: Path,
    frame_count: int,
    extent_keys: Sequence[str],
    merged_extents: Sequence[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected_png = [f"frame_{index:06d}.png" for index in range(frame_count)]
    if sorted(path.name for path in (session_dir / "camera").glob("*.png")) != expected_png:
        raise RuntimeError("camera sequence is incomplete")
    with Image.open(session_dir / "camera" / expected_png[-1]) as camera:
        if camera.size != (args.width, args.height) or camera.mode != "RGB":
            raise RuntimeError("camera frame shape or mode mismatch")
    expected_npz = [name.replace(".png", ".npz") for name in expected_png]
    if sorted(path.name for path in (session_dir / "depth").glob("*.npz")) != expected_npz:
        raise RuntimeError("depth sequence is incomplete")
    with np.load(session_dir / "depth" / expected_npz[-1], allow_pickle=False) as data:
        if data.files != ["depth"]:
            raise RuntimeError(f"depth NPZ key mismatch: {data.files}")
        depth = np.asarray(data["depth"])
    if depth.shape != (args.height, args.width) or depth.dtype != np.float32:
        raise RuntimeError("depth shape or dtype mismatch")
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise RuntimeError("depth contains non-finite or negative values")
    intrinsics = json.loads(
        (session_dir / "camera_intrinsics.json").read_text(encoding="utf-8")
    )
    if intrinsics.get("width") != args.width or intrinsics.get("height") != args.height:
        raise RuntimeError("camera intrinsic dimensions mismatch")
    extrinsics = [
        json.loads(line)
        for line in (session_dir / "camera_extrinsics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if [row.get("frame_id") for row in extrinsics] != list(range(frame_count)):
        raise RuntimeError("extrinsic sequence is incomplete")
    trajectories = [
        json.loads(line)
        for line in (session_dir / "ground_truth_trajectory.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if [row.get("frame_id") for row in trajectories] != list(range(frame_count)):
        raise RuntimeError("trajectory sequence is incomplete")
    value_sets: dict[str, list[int]] = {}
    checked_single_void_ratios: list[float] = []
    for key, current_extent in zip(extent_keys, args.bev_extents):
        modalities = ["masked", "complete"]
        for merged_extent in merged_extents:
            modalities.extend(
                [merged_modality("masked", merged_extent), merged_modality("complete", merged_extent)]
            )
        for modality in modalities:
            files = sorted((session_dir / key / modality).glob("*.png"))
            if [path.name for path in files] != expected_png:
                raise RuntimeError(f"incomplete {key}/{modality}")
            final = None
            for frame_id, path in enumerate(files):
                with Image.open(path) as image:
                    if image.size != (args.bev_size, args.bev_size) or image.mode != "L":
                        raise RuntimeError(f"invalid {key}/{modality}/{path.name}")
                    raster = np.asarray(image).copy()
                allowed = {0, 255} if modality == "complete" else {0, 112, 255}
                if not set(map(int, np.unique(raster))).issubset(allowed):
                    raise RuntimeError(f"invalid labels in {key}/{modality}/{path.name}")
                center = args.bev_size // 2
                if raster[center, center] != 255:
                    raise RuntimeError(f"robot origin is not free in {key}/{modality}/{path.name}")
                final = raster
            if final is None:
                raise RuntimeError(f"empty {key}/{modality}")
            value_sets[f"{key}/{modality}"] = sorted(map(int, np.unique(final)))
    void_algorithms: set[str] = set()
    for frame_id, row in enumerate(trajectories):
        frame_bev = row.get("bev", {})
        for key, current_extent in zip(extent_keys, args.bev_extents):
            record = frame_bev.get(key, {})
            algorithm = str(record.get("void_coverage_algorithm", ""))
            if not algorithm:
                raise RuntimeError(
                    f"frame {frame_id} {key} lacks geometry VOID provenance"
                )
            void_algorithms.add(algorithm)
            try:
                ratio = float(record["single_complete_gt_void_ratio"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"frame {frame_id} {key} has no valid geometry VOID ratio"
                ) from error
            if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise RuntimeError(
                    f"frame {frame_id} {key} has invalid geometry VOID ratio {ratio}"
                )
            if ratio > args.max_single_bev_void_ratio:
                raise SingleBEVVoidRatioExceeded(
                    ratio=ratio,
                    threshold=args.max_single_bev_void_ratio,
                    frame_id=frame_id,
                    extent_m=current_extent,
                )
            checked_single_void_ratios.append(ratio)
    if len(void_algorithms) != 1:
        raise RuntimeError(
            f"session mixes geometry VOID algorithms: {sorted(void_algorithms)}"
        )
    required = (
        "camera_intrinsics.json",
        "camera_extrinsics.jsonl",
        "ground_truth_trajectory.jsonl",
        "ground_truth_trajectory.csv",
        "ground_truth_trajectory.png",
        "metadata.json",
        "preview.png",
    )
    missing = [name for name in required if not (session_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"session contract files are missing: {missing}")
    return {
        "passed": True,
        "frame_count": frame_count,
        "synchronized_modalities_per_frame": 2
        + (2 + 2 * len(merged_extents)) * len(extent_keys),
        "final_frame_value_sets": value_sets,
        "depth_format": "lossless float32 metres in compressed NumPy .npz",
        "camera_intrinsics_file": "camera_intrinsics.json",
        "camera_extrinsics_file": "camera_extrinsics.jsonl",
        "void_check_version": VOID_CHECK_VERSION,
        "void_coverage_algorithm": next(iter(void_algorithms)),
        "void_filter_contract": VOID_FILTER_CONTRACT,
        "all_single_complete_gt_bevs_void_checked": True,
        "single_complete_gt_bevs_checked": len(checked_single_void_ratios),
        "max_allowed_single_complete_gt_void_ratio": (
            args.max_single_bev_void_ratio
        ),
        "session_max_single_complete_gt_void_ratio": max(
            checked_single_void_ratios, default=0.0
        ),
    }


def collect_one_session(
    *,
    controller: Any,
    full_truth: np.ndarray,
    full_valid_coverage: np.ndarray,
    truth_lower_bound: np.ndarray,
    voxel_statistics: dict[str, Any],
    house: dict[str, Any],
    source: dict[str, Any],
    scene_group_id: str,
    scene_session_ordinal: int,
    scene_sampling_cycle: int,
    session_index: int,
    path: list[dict[str, float]],
    geodesic_length: float,
    horizontal_fov: float,
    camera_height: float,
    speed: float,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    samples = interpolate_path(path, speed / args.capture_hz)
    if len(samples) < 2:
        raise RuntimeError("path interpolation produced fewer than two frames")
    sample_positions = [np.asarray([p["x"], p["y"], p["z"]]) for p in samples]
    distances = [0.0]
    for first, second in zip(sample_positions, sample_positions[1:]):
        distances.append(distances[-1] + float(np.linalg.norm(second - first)))
    times = [distance / speed for distance in distances]
    session_name = (
        f"session_{session_index:06d}_{source['dataset']}_"
        f"{sanitize(source['scene_id'])}"
    )
    final_dir = output_root / session_name
    partial_dir = output_root / f".{session_name}.partial"
    if final_dir.exists() or partial_dir.exists():
        raise FileExistsError(session_name)
    extent_keys = [extent_key(value) for value in args.bev_extents]
    create_session_directories(partial_dir, extent_keys, args.merged_extents)
    write_json(partial_dir / "house.json", house)
    intrinsics = camera_intrinsics(args.width, args.height, horizontal_fov)
    write_json(partial_dir / "camera_intrinsics.json", intrinsics)
    accumulators = {
        key: BEVAccumulator(extent, args.bev_size)
        for key, extent in zip(extent_keys, args.bev_extents)
    }
    levels = floor_levels(house)
    void_coverage_algorithm = str(
        voxel_statistics["strict_void_coverage"]["algorithm"]
    )
    trajectory_rows: list[dict[str, Any]] = []
    measured_heights: list[float] = []
    try:
        last_yaw = yaw_toward(samples[0], samples[1])
        for frame_id, (position, path_distance, sim_time) in enumerate(
            zip(samples, distances, times)
        ):
            if frame_id + 1 < len(samples):
                last_yaw = yaw_toward(position, samples[frame_id + 1])
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
                raise RuntimeError(f"frame {frame_id} teleport failed")
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
                fieldOfView=intrinsics["vertical_fov_degrees"],
            )
            if not event.metadata.get("lastActionSuccess", False):
                raise RuntimeError(f"frame {frame_id} camera update failed")
            if not event.third_party_camera_frames or not event.third_party_depth_frames:
                raise RuntimeError("third-party RGB-D camera returned no frame")
            rgb = np.asarray(event.third_party_camera_frames[0], dtype=np.uint8)[..., :3]
            depth = np.asarray(event.third_party_depth_frames[0], dtype=np.float32)
            if rgb.shape != (args.height, args.width, 3):
                raise RuntimeError(f"unexpected RGB shape {rgb.shape}")
            if depth.shape != (args.height, args.width) or not np.isfinite(depth).all():
                raise RuntimeError("invalid metric depth")
            frame_name = f"frame_{frame_id:06d}.png"
            depth_name = f"frame_{frame_id:06d}.npz"
            Image.fromarray(rgb).save(partial_dir / "camera" / frame_name, compress_level=4)
            np.savez_compressed(partial_dir / "depth" / depth_name, depth=depth)

            agent = event.metadata["agent"]
            camera = event.metadata["thirdPartyCameras"][0]
            camera_position = camera["position"]
            camera_rotation = camera["rotation"]
            yaw = float(camera_rotation["y"])
            horizon = float(camera_rotation["x"])
            camera_to_world, world_to_camera = camera_matrices(
                camera_position, yaw, horizon
            )
            measured_heights.append(float(camera_position["y"] - local_floor_y))
            yaw_radians = math.radians(yaw)
            forward = np.asarray(
                [math.sin(yaw_radians), math.cos(yaw_radians)], dtype=np.float64
            )
            right = np.asarray(
                [math.cos(yaw_radians), -math.sin(yaw_radians)], dtype=np.float64
            )
            agent_position = [float(agent["position"][axis]) for axis in ("x", "y", "z")]
            extrinsic = {
                "agent_position_world_m": agent_position,
                "agent_rotation_xyzw": quaternion_yaw(float(agent["rotation"]["y"])),
                "camera_position_world_m": [
                    float(camera_position[axis]) for axis in ("x", "y", "z")
                ],
                "camera_rotation_xyzw": quaternion_yaw(yaw),
                "camera_to_world_matrix": camera_to_world,
                "world_to_camera_matrix": world_to_camera,
                "bev_forward_xz": forward.tolist(),
                "bev_right_xz": right.tolist(),
                "world_from_bev_planar": [
                    [float(right[0]), float(forward[0]), agent_position[0]],
                    [float(right[1]), float(forward[1]), agent_position[2]],
                    [0.0, 0.0, 1.0],
                ],
                "coordinate_convention": (
                    "Unity world +x right, +y up, +z forward; BEV local axes "
                    "are right/forward; camera matrices use local +x/+y/+z"
                ),
            }
            bev_record: dict[str, Any] = {}
            for key, extent in zip(extent_keys, args.bev_extents):
                complete = render_ego_obstacle_map(
                    full_scene_map=full_truth,
                    lower_bound=truth_lower_bound,
                    source_meters_per_pixel=args.voxel_size,
                    position=agent_position,
                    forward=forward,
                    right=right,
                    size=args.bev_size,
                    extent=extent,
                )
                masked = render_visibility_masked_map(
                    complete, horizontal_fov_degrees=horizontal_fov
                )
                strict_validate_bev(complete, masked, extent=extent)
                single_valid_coverage = render_ego_obstacle_map(
                    full_scene_map=full_valid_coverage.astype(
                        np.uint8, copy=False
                    ),
                    lower_bound=truth_lower_bound,
                    source_meters_per_pixel=args.voxel_size,
                    position=agent_position,
                    forward=forward,
                    right=right,
                    size=args.bev_size,
                    extent=extent,
                ).astype(bool, copy=False)
                single_valid_coverage = repair_valid_coverage(
                    single_valid_coverage
                )
                void_ratio = enforce_single_bev_void_limit(
                    single_valid_coverage,
                    threshold=args.max_single_bev_void_ratio,
                    frame_id=frame_id,
                    extent_m=extent,
                )
                Image.fromarray(masked).save(partial_dir / key / "masked" / frame_name)
                Image.fromarray(complete).save(partial_dir / key / "complete" / frame_name)
                accumulator = accumulators[key]
                accumulator.update(complete, masked, extrinsic)
                merged_record: dict[str, Any] = {}
                for merged_extent in args.merged_extents:
                    images: dict[str, np.ndarray] = {}
                    extent_record: dict[str, Any] = {}
                    for kind in ("masked", "complete"):
                        image, image_meta = accumulator.render_ego(
                            kind, extrinsic, merged_extent
                        )
                        images[kind] = image
                        Image.fromarray(image).save(
                            partial_dir
                            / key
                            / merged_modality(kind, merged_extent)
                            / frame_name
                        )
                        extent_record[kind] = image_meta
                    strict_validate_bev(
                        images["complete"],
                        images["masked"],
                        extent=merged_extent,
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
                    "merged": merged_record,
                }
            row = {
                "frame_id": frame_id,
                "sim_time_s": sim_time,
                "path_distance_m": path_distance,
                "path_progress": path_distance / geodesic_length,
                "agent_position_world_m": extrinsic["agent_position_world_m"],
                "bev_forward_xz": extrinsic["bev_forward_xz"],
                "extrinsic": extrinsic,
                "camera_file": f"camera/{frame_name}",
                "depth_file": f"depth/{depth_name}",
                "bev": bev_record,
            }
            trajectory_rows.append(row)

        with (partial_dir / "ground_truth_trajectory.jsonl").open("w", encoding="utf-8") as stream:
            for row in trajectory_rows:
                stream.write(json.dumps(row) + "\n")
        with (partial_dir / "camera_extrinsics.jsonl").open("w", encoding="utf-8") as stream:
            for row in trajectory_rows:
                stream.write(
                    json.dumps(
                        {
                            "frame_id": row["frame_id"],
                            "sim_time_s": row["sim_time_s"],
                            "camera_file": row["camera_file"],
                            "depth_file": row["depth_file"],
                            "extrinsic": row["extrinsic"],
                        }
                    )
                    + "\n"
                )
        with (partial_dir / "ground_truth_trajectory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            fields = (
                "frame_id",
                "sim_time_s",
                "path_distance_m",
                "path_progress",
                "agent_x",
                "agent_y",
                "agent_z",
                "forward_x",
                "forward_z",
            )
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in trajectory_rows:
                position = row["agent_position_world_m"]
                forward = row["bev_forward_xz"]
                writer.writerow(
                    {
                        "frame_id": row["frame_id"],
                        "sim_time_s": row["sim_time_s"],
                        "path_distance_m": row["path_distance_m"],
                        "path_progress": row["path_progress"],
                        "agent_x": position[0],
                        "agent_y": position[1],
                        "agent_z": position[2],
                        "forward_x": forward[0],
                        "forward_z": forward[1],
                    }
                )
        render_trajectory_image(
            full_truth,
            truth_lower_bound,
            args.voxel_size,
            path,
            [row["agent_position_world_m"] for row in trajectory_rows],
            partial_dir / "ground_truth_trajectory.png",
        )
        random_parameters = {
            "horizontal_fov_degrees": horizontal_fov,
            "camera_height_m": camera_height,
            "robot_speed_mps": speed,
        }
        final_error = float(
            np.linalg.norm(
                np.asarray(trajectory_rows[-1]["agent_position_world_m"])
                - np.asarray([path[-1][axis] for axis in ("x", "y", "z")])
            )
        )
        speed_errors = [
            abs(
                (current["path_distance_m"] - previous["path_distance_m"])
                / (current["sim_time_s"] - previous["sim_time_s"])
                - speed
            )
            for previous, current in zip(trajectory_rows, trajectory_rows[1:])
        ]
        camera_height_error = max(abs(value - camera_height) for value in measured_heights)
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "session_id": session_name,
            "session_seed": source["session_seed"],
            "scene_sampling_cycle": scene_sampling_cycle,
            "scene_group_id": scene_group_id,
            "scene_session_ordinal": scene_session_ordinal,
            "sessions_per_scene_initialization": args.sessions_per_scene,
            "dataset": source["dataset"],
            "scene_split": source["scene_split"],
            "dataset_root": source["dataset_root"],
            "scene_id": source["scene_id"],
            "scene_file": source["scene_file"],
            "scene_dataset_config": None,
            "navmesh_file": None,
            "navmesh_source": "AI2-THOR GetReachablePositions",
            "asset_load_verified": True,
            "path_verified_before_capture": True,
            "path_verified_against_collision_truth": True,
            "random_parameters": random_parameters,
            "camera_intrinsics": intrinsics,
            "camera_intrinsics_file": "camera_intrinsics.json",
            "camera_extrinsics_file": "camera_extrinsics.jsonl",
            "depth": {
                "directory": "depth",
                "filename_pattern": "frame_INDEX.npz",
                "format": "compressed NumPy .npz (lossless DEFLATE)",
                "dtype": "float32",
                "units": "metres",
                "source": "AI2-THOR synchronized third-party metric depth ground truth",
                "near_m": 0.01,
                "far_m": None,
            },
            "capture_hz": args.capture_hz,
            "gpu_device_id": args.gpu_index,
            "ai2thor_vulkan_gpu_device": args.resolved_gpu_device,
            "nvidia_gpu_uuid": args.resolved_nvidia_gpu_uuid,
            "motion_model": "constant-speed kinematic replay over a reachable-grid BFS path",
            "bev": {
                "size": args.bev_size,
                "extent_classes_m": list(args.bev_extents),
                "obstacle_height_band_m": [
                    args.obstacle_min_height,
                    args.obstacle_max_height,
                ],
                "truth_source": voxel_statistics["truth_source"],
                "gt_quality_version": GT_QUALITY_VERSION,
                "ground_handling": "room floor polygons are free support; in-band entity meshes are occupied",
                "robot_origin_pixel_row_column": [args.bev_size // 2, args.bev_size // 2],
                "robot_origin_convention": "one shared integer pixel centre for all BEVs",
                "path_footprint_radius_m": DEFAULT_AGENT_RADIUS_M,
                "voxel_size_m": args.voxel_size,
                "masked_values": {"occupied": 0, "unknown": 112, "free": 255},
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
                    "explicitly_excluded_inputs": [
                        "horizontal_fov",
                        "masked_bev",
                        "masked_unknown_value_112",
                        "rgb",
                        "depth",
                        "model_prediction",
                    ],
                    "max_allowed_ratio": args.max_single_bev_void_ratio,
                    "comparison": "reject when ratio > max_allowed_ratio",
                    "rejection_action": (
                        "abort session attempt and recursively delete its "
                        "transactional partial directory"
                    ),
                },
                "fusion_rule": (
                    "temporal union of observed cells; merged masked and complete "
                    "values come from the same static simulator truth"
                ),
                "merged_fusion_version": 2,
                "merged_orientation": "ego-centric; latest robot centered and forward up",
                "merged_normalized_extents_m": list(args.merged_extents),
                "merged_normalized_size": [args.bev_size, args.bev_size],
            },
            "voxel_statistics": voxel_statistics,
            "estimated_voxel_grid_cells": int(full_truth.size),
            "voxel_grid_cell_limit": None,
            "path": {
                "start": [path[0][axis] for axis in ("x", "y", "z")],
                "goal": [path[-1][axis] for axis in ("x", "y", "z")],
                "reachable": True,
                "geodesic_distance_m": geodesic_length,
                "polyline_distance_m": geodesic_length,
                "waypoints": path,
                "duration_s": times[-1],
                "collision_truth_validation": {"passed": True},
            },
            "frame_count": len(trajectory_rows),
            "vggt_estimates": {"status": "skipped"},
            "completed_at_unix_s": time.time(),
            "wall_time_s": time.time() - started,
            "source_details": source,
        }
        write_json(partial_dir / "metadata.json", metadata)
        make_preview(
            partial_dir,
            extent_keys,
            args.merged_extents,
            f"frame_{len(trajectory_rows) - 1:06d}.png",
            random_parameters,
        )
        validation = validate_written_session(
            partial_dir,
            len(trajectory_rows),
            extent_keys,
            args.merged_extents,
            args,
        )
        validation.update(
            {
                "final_goal_error_m": final_error,
                "maximum_speed_error_mps": max(speed_errors, default=0.0),
                "maximum_camera_height_error_m": camera_height_error,
                "gt_quality_version": GT_QUALITY_VERSION,
                "all_frames_collision_truth_validated": True,
            }
        )
        metadata["bev"]["void_statistics"] = {
            "single_complete_gt_bevs_checked": validation[
                "single_complete_gt_bevs_checked"
            ],
            "session_max_single_complete_gt_void_ratio": validation[
                "session_max_single_complete_gt_void_ratio"
            ],
            "max_allowed_single_complete_gt_void_ratio": (
                args.max_single_bev_void_ratio
            ),
            "coverage_algorithm": void_coverage_algorithm,
            "filter_contract": VOID_FILTER_CONTRACT,
            "passed": True,
        }
        metadata["validation"] = validation
        write_json(partial_dir / "metadata.json", metadata)
        write_json(partial_dir / "validation.json", validation)
        (partial_dir / "COMPLETE").write_text("validated\n", encoding="utf-8")
        partial_dir.rename(final_dir)
        return metadata | {"output_directory": str(final_dir)}
    except Exception:
        if partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)
        raise


def initialize_scene(
    house: dict[str, Any], horizontal_fov: float, camera_height: float, args: argparse.Namespace
) -> tuple[
    Any,
    list[dict[str, Any]],
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    with renderer_startup_gate(args.gpu_index, label="procthor:scene-init"):
        controller = isolated_cloud_controller(
            args.procthor_runtime_root,
            physical_gpu_index=args.gpu_index,
            scene=house,
            width=args.width,
            height=args.height,
            fieldOfView=vertical_fov(horizontal_fov, args.width, args.height),
            agentMode="locobot",
            gridSize=0.25,
            snapToGrid=False,
            rotateStepDegrees=30,
            renderDepthImage=True,
            makeAgentsVisible=False,
            quality="Low",
        )
        try:
            assert_process_gpu_binding(
                int(controller.unity_pid), args.resolved_nvidia_gpu_uuid
            )
            event = controller.step(action="GetReachablePositions")
        except Exception:
            controller.stop()
            raise
    if not event.metadata.get("lastActionSuccess", False):
        controller.stop()
        raise RuntimeError(event.metadata.get("errorMessage", "GetReachablePositions failed"))
    floor_y, reachable = dominant_floor_positions(
        event.metadata.get("actionReturn") or [], floor_levels(house)
    )
    if len(reachable) < 2:
        controller.stop()
        raise RuntimeError("dominant floor has fewer than two reachable positions")
    first = reachable[0]
    event = controller.step(
        action="AddThirdPartyCamera",
        position={"x": first["x"], "y": floor_y + camera_height, "z": first["z"]},
        rotation={"x": 0.0, "y": 0.0, "z": 0.0},
        fieldOfView=vertical_fov(horizontal_fov, args.width, args.height),
    )
    if not event.metadata.get("lastActionSuccess", False):
        controller.stop()
        raise RuntimeError("AddThirdPartyCamera failed")
    print(
        f"Building one shared complete truth for {args.sessions_per_scene} sessions...",
        flush=True,
    )
    try:
        full_truth, full_valid_coverage, lower_bound, statistics = build_complete_truth(
            controller,
            house,
            floor_y=floor_y,
            voxel_size=args.voxel_size,
            obstacle_min_height=args.obstacle_min_height,
            obstacle_max_height=args.obstacle_max_height,
        )
    except Exception:
        controller.stop()
        raise
    return (
        controller,
        reachable,
        floor_y,
        full_truth,
        full_valid_coverage,
        lower_bound,
        statistics,
    )


def collect_scene_group(
    *,
    house: dict[str, Any],
    source: dict[str, Any],
    scene_sampling_cycle: int,
    first_session_index: int,
    run_dir: Path,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    horizontal_fov = rng.uniform(*args.horizontal_fov_range)
    camera_height = rng.uniform(*args.camera_height_range)
    group_id = f"scene_{scene_sampling_cycle:06d}_{sanitize(source['scene_id'])}"
    staging = run_dir / f".{group_id}.partial"
    staging.mkdir()
    controller = None
    sessions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        (
            controller,
            reachable,
            floor_y,
            full_truth,
            full_valid_coverage,
            lower_bound,
            statistics,
        ) = initialize_scene(house, horizontal_fov, camera_height, args)
        attempt = 0
        while len(sessions) < args.sessions_per_scene:
            if attempt >= args.max_initialization_attempts:
                message = (
                    f"scene produced only {len(sessions)}/{args.sessions_per_scene} "
                    f"sessions after the maximum {attempt} total initialization "
                    "attempts (successful and failed attempts both count)"
                )
                if not args.allow_partial_scene_groups:
                    raise RuntimeError(message)
                print(f"  partial scene group accepted: {message}", flush=True)
                failures.append(
                    {
                        "scene_group_id": group_id,
                        "attempt": attempt,
                        "max_initialization_attempts": (
                            args.max_initialization_attempts
                        ),
                        "completed_in_group": len(sessions),
                        "reason": message,
                        "partial_group_committed": True,
                    }
                )
                break
            attempt += 1
            speed = rng.uniform(*args.speed_range)
            session_seed = rng.randrange(0, 2**31 - 1)
            source["session_seed"] = session_seed
            try:
                path, length = choose_shortest_path(
                    list(reachable),
                    random.Random(session_seed),
                    tuple(args.path_length_range),
                    args.path_attempts,
                )
                session = collect_one_session(
                    controller=controller,
                    full_truth=full_truth,
                    full_valid_coverage=full_valid_coverage,
                    truth_lower_bound=lower_bound,
                    voxel_statistics=statistics,
                    house=house,
                    source=dict(source),
                    scene_group_id=group_id,
                    scene_session_ordinal=len(sessions),
                    scene_sampling_cycle=scene_sampling_cycle,
                    session_index=first_session_index + len(sessions),
                    path=path,
                    geodesic_length=length,
                    horizontal_fov=horizontal_fov,
                    camera_height=camera_height,
                    speed=speed,
                    output_root=staging,
                    args=args,
                )
            except Exception as error:
                failure = {
                        "scene_group_id": group_id,
                        "attempt": attempt,
                        "completed_in_group": len(sessions),
                        "reason": f"{type(error).__name__}: {error}",
                    }
                if isinstance(error, SingleBEVVoidRatioExceeded):
                    failure.update(error.as_dict())
                    failure["partial_session_cache_deleted"] = True
                failures.append(failure)
                print(
                    f"  path attempt {attempt} rejected: {error} "
                    f"({len(sessions)}/{args.sessions_per_scene})",
                    flush=True,
                )
                continue
            sessions.append(session)
            print(
                f"  completed {len(sessions):02d}/{args.sessions_per_scene:02d}: "
                f"{session['session_id']} ({session['frame_count']} frames)",
                flush=True,
            )
        for session in sessions:
            staged = Path(session["output_directory"])
            final = run_dir / staged.name
            staged.rename(final)
            session["output_directory"] = str(final)
        write_json(
            run_dir / f"{group_id}.json",
            {
                "scene_group_id": group_id,
                "source": source,
                "dominant_floor_y": floor_y,
                "horizontal_fov_degrees": horizontal_fov,
                "camera_height_m": camera_height,
                "target_sessions": args.sessions_per_scene,
                "completed_sessions": len(sessions),
                "initialization_attempts": attempt,
                "max_initialization_attempts": args.max_initialization_attempts,
                "attempt_counting": "every session attempt, successful or failed",
                "max_single_bev_void_ratio": args.max_single_bev_void_ratio,
                "partial_group": len(sessions) != args.sessions_per_scene,
                "sessions": [item["session_id"] for item in sessions],
                "failed_path_attempts": failures,
            },
        )
        return sessions, failures
    finally:
        if controller is not None:
            controller.stop()
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def generated_house(args: argparse.Namespace, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    errors = []
    helper_args = SimpleNamespace(
        split=args.split,
        gpu_index=args.gpu_index,
        procthor_runtime_root=args.procthor_runtime_root,
        resolved_gpu_device=args.resolved_gpu_device,
        local_ai2thor_executable=args.local_ai2thor_executable,
    )
    for _ in range(args.house_generation_attempts):
        seed = rng.randrange(0, 2**31 - 1)
        try:
            house, warnings = generate_fresh_house(helper_args, seed)
            return house, {
                "dataset": "procthor-generated",
                "dataset_root": "ProcTHOR HouseGenerator",
                "scene_id": f"fresh_{seed:010d}",
                "scene_file": "Procedural",
                "generation_seed": seed,
                "generation_validation_warnings": warnings,
                "scene_split": {
                    "role": args.split,
                    "unit": "generated scene",
                    "train_fraction": None,
                    "split_seed": args.seed,
                    "manifest": "../scene_split_manifest.json",
                },
            }
        except Exception as error:
            errors.append(f"seed {seed}: {type(error).__name__}: {error}")
            print(errors[-1], flush=True)
    raise RuntimeError("fresh-house generation failed:\n" + "\n".join(errors))


def builtin_house(
    args: argparse.Namespace, rng: random.Random, used: set[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(used) >= SPLIT_COUNTS[args.split]:
        used.clear()
    while True:
        index = rng.randrange(SPLIT_COUNTS[args.split])
        if index not in used:
            used.add(index)
            break
    house = load_house(args.dataset_dir, args.split, index)
    return house, {
        "dataset": "procthor-10k",
        "dataset_root": str(args.dataset_dir),
        "scene_id": f"{args.split}_{index:05d}",
        "scene_file": f"{args.dataset_dir / (args.split + '.jsonl.gz')}#{index}",
        "house_index": index,
        "scene_split": {
            "role": args.split,
            "unit": "stored ProcTHOR scene",
            "train_fraction": None,
            "split_seed": args.seed,
            "manifest": "../scene_split_manifest.json",
        },
    }


def write_inspection_html(run_dir: Path, sessions: Sequence[dict[str, Any]]) -> None:
    cards = []
    for session in sessions:
        directory = Path(session["output_directory"]).name
        parameters = session["random_parameters"]
        cards.append(
            f'''<article><h2>{html.escape(session['session_id'])}</h2>
            <p>{html.escape(session['dataset'])}/{html.escape(session['scene_id'])}
            · FOV {parameters['horizontal_fov_degrees']:.1f}°
            · camera {parameters['camera_height_m']:.2f} m
            · speed {parameters['robot_speed_mps']:.2f} m/s
            · {session['frame_count']} frames</p>
            <a href="{html.escape(directory)}/metadata.json">metadata.json</a>
            <img src="{html.escape(directory)}/preview.png"></article>'''
        )
    document = f'''<!doctype html><meta charset="utf-8"><title>ProcTHOR collection</title>
    <style>body{{margin:0;padding:24px;color:#eef3f8;background:#090c10;font:14px system-ui}}
    article{{max-width:1100px;margin:0 auto 28px;padding:16px;background:#151b23;border:1px solid #2b3542;border-radius:12px}}
    h2{{margin-top:0}}a{{color:#40c9ff}}img{{display:block;width:100%;margin-top:12px;border-radius:8px}}</style>
    <h1>ProcTHOR 25-path collection</h1>{''.join(cards)}'''
    (run_dir / "inspection.html").write_text(document, encoding="utf-8")


def run(mode: str, argv: Sequence[str] | None = None) -> None:
    parser = build_parser(mode)
    args = parser.parse_args(argv)
    validate_args(args, mode)
    (
        args.procthor_runtime_root,
        args.local_ai2thor_executable,
    ) = locate_ai2thor_runtime()
    expose_bundled_vulkaninfo(args.procthor_runtime_root)
    current_mapping = refresh_ai2thor_cuda_vulkan_mapping(
        args.procthor_runtime_root
    )
    current_inventory = nvidia_gpu_inventory()
    if args.gpu_index not in current_mapping or args.gpu_index not in current_inventory:
        raise RuntimeError(f"physical GPU {args.gpu_index} is absent from current mapping")
    args.resolved_gpu_device = current_mapping[args.gpu_index]
    args.resolved_nvidia_gpu_uuid = current_inventory[
        args.gpu_index
    ].removeprefix("GPU-")
    run_dir = allocate_run_dir(args)
    rng = random.Random(args.seed)
    used_builtin: set[int] = set()
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    successful_scenes = 0
    scene_attempt = 0
    max_scene_attempts = args.max_scene_attempts or max(args.scene_count * 10, 10)
    write_json(
        run_dir / "scene_split_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "collector_mode": mode,
            "split": args.split,
            "unit": "scene",
            "sessions_per_scene_initialization": args.sessions_per_scene,
        },
    )
    try:
        while successful_scenes < args.scene_count and scene_attempt < max_scene_attempts:
            scene_attempt += 1
            try:
                if mode == "generated":
                    house, source = generated_house(args, rng)
                else:
                    house, source = builtin_house(args, rng, used_builtin)
                print(
                    f"Scene attempt {scene_attempt}: {source['dataset']}/{source['scene_id']} "
                    f"target={args.sessions_per_scene}",
                    flush=True,
                )
                group, group_failures = collect_scene_group(
                    house=house,
                    source=source,
                    scene_sampling_cycle=successful_scenes,
                    first_session_index=len(completed),
                    run_dir=run_dir,
                    rng=rng,
                    args=args,
                )
            except Exception as error:
                failure = {
                    "scene_attempt": scene_attempt,
                    "reason": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                print(f"Scene rejected: {failure['reason']}", flush=True)
                continue
            completed.extend(group)
            failures.extend(group_failures)
            successful_scenes += 1
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "collector": (
                "collect_procthor_generated.py"
                if mode == "generated"
                else "collect_procthor_builtin.py"
            ),
            "collector_mode": mode,
            "collection_seed": args.seed,
            "gpu_device_id": args.gpu_index,
            "ai2thor_vulkan_gpu_device": args.resolved_gpu_device,
            "nvidia_gpu_uuid": args.resolved_nvidia_gpu_uuid,
            "requested_scene_initializations": args.scene_count,
            "completed_scene_initializations": successful_scenes,
            "sessions_per_scene_initialization": args.sessions_per_scene,
            "requested_sessions": args.scene_count * args.sessions_per_scene,
            "completed_sessions": len(completed),
            "allow_partial_scene_groups": args.allow_partial_scene_groups,
            "allow_partial_run": args.allow_partial_run,
            "max_initialization_attempts": args.max_initialization_attempts,
            "initialization_attempt_counting": (
                "every attempted session counts, successful or failed"
            ),
            "max_single_bev_void_ratio": args.max_single_bev_void_ratio,
            "void_check_version": VOID_CHECK_VERSION,
            "void_check_modality": (
                "single/current Complete GT geometry-valid coverage only"
            ),
            "void_filter_contract": VOID_FILTER_CONTRACT,
            "bev_extent_classes_m": list(args.bev_extents),
            "merged_normalized_extents_m": list(args.merged_extents),
            "capture_hz": args.capture_hz,
            "camera_resolution": [args.width, args.height],
            "camera_height_range_m": list(args.camera_height_range),
            "horizontal_fov_range_degrees": list(args.horizontal_fov_range),
            "robot_speed_range_mps": list(args.speed_range),
            "path_length_range_m": list(args.path_length_range),
            "scene_attempts": scene_attempt,
            "sessions": [
                {
                    "session_id": item["session_id"],
                    "dataset": item["dataset"],
                    "scene_id": item["scene_id"],
                    "scene_group_id": item["scene_group_id"],
                    "scene_session_ordinal": item["scene_session_ordinal"],
                    "output_directory": item["output_directory"],
                    "random_parameters": item["random_parameters"],
                    "frame_count": item["frame_count"],
                    "validation": item["validation"],
                    "vggt_estimates": item["vggt_estimates"],
                }
                for item in completed
            ],
            "rejected_attempts": failures,
        }
        write_json(run_dir / "collection_manifest.json", manifest)
        write_inspection_html(run_dir, completed)
        if successful_scenes != args.scene_count and not args.allow_partial_run:
            raise RuntimeError(
                f"only {successful_scenes}/{args.scene_count} scene groups completed"
            )
        (run_dir / "COMPLETE").write_text("collector-complete\n", encoding="utf-8")
        print(f"Collection complete: {run_dir}")
    except Exception:
        write_json(
            run_dir / "FAILED.json",
            {
                "collector_mode": mode,
                "completed_scene_initializations": successful_scenes,
                "completed_sessions": len(completed),
                "failures": failures,
            },
        )
        raise
