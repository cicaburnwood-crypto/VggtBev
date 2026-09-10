#!/usr/bin/env python3
"""Collect synchronized random Habitat navigation sessions.

Each completed session contains one HM3D, HSSD, or MP3D scene, one verified random
reachable shortest path, randomized camera/speed parameters, RGB frames,
exact simulator trajectory/extrinsics, and synchronized BEV scale classes.
By default, collection uses a stable, dataset-specific 95% scene split and
holds out the remaining 5% of scenes for validation.  BEV
truth is derived from the solid collision-mesh voxel map, never from RGB or
depth.  After simulation, the original VGGT model estimates per-frame camera
intrinsics, extrinsics, and depth for every RGB sequence.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import html
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import habitat_sim
import numpy as np
import quaternion
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from bev_visibility import (
    DEFAULT_MAX_SINGLE_BEV_VOID_RATIO,
    VOID_COVERAGE_ALGORITHM,
    VOID_CHECK_VERSION,
    VOID_FILTER_CONTRACT,
    VOID_REPAIR_ALGORITHM,
    SingleBEVVoidRatioExceeded,
    enforce_single_bev_void_limit,
    repair_valid_coverage,
)
from gpu_render_startup import (
    physical_gpu_from_visible_device,
    renderer_startup_gate,
)
from point_navigation import PointNavigationError
from web_navigation import (
    VISIBILITY_ALGORITHM,
    capture_frame_extrinsic,
    render_full_obstacle_map,
    render_ego_obstacle_map,
    render_visibility_masked_map,
)


DEFAULT_DATASET_ROOTS = (
    Path("/home/liudiwen/VGGT/scenes/hm3d"),
    Path("/home/liudiwen/VGGT/scenes/hssd-hab"),
    Path("/home/liudiwen/VGGT/scenes/mp3d"),
)
DEFAULT_OUTPUT = Path("output/random_session_collection")
DEFAULT_VGGT_ROOTS = (
    Path("/home/wolfie/Project/vggt"),
    Path("/data/disk_14t/diwen/vggt"),
)
DEFAULT_BEV_EXTENTS = (3.5, 5.0, 6.5, 8.0)
DEFAULT_MERGED_EXTENTS = (8.0, 15.0)
DEFAULT_CAMERA_HEIGHT_RANGE_M = (0.30, 0.80)
DEFAULT_HORIZONTAL_FOV_RANGE_DEGREES = (60.0, 120.0)
DEFAULT_ROBOT_SPEED_RANGE_MPS = (0.6, 2.0)
DEFAULT_SESSIONS_PER_INITIALIZATION = 25
DEFAULT_MAX_INITIALIZATION_ATTEMPTS = 200
GT_QUALITY_VERSION = 5
DEFAULT_OBSTACLE_MIN_HEIGHT_M = 0.0
DEFAULT_OBSTACLE_MAX_HEIGHT_M = 1.4
DEFAULT_AGENT_RADIUS_M = 0.10
SUPPORTED_DATASETS = ("hm3d", "replica", "hssd", "mp3d")
# Keep explicit Replica support only for auditing/replaying legacy data. New
# builds never select it unless the caller deliberately opts in.
DEFAULT_ENABLED_DATASETS = ("hm3d", "hssd", "mp3d")
DEFAULT_SCENE_TRAIN_FRACTION = 0.95
DEFAULT_SCENE_SPLIT_SEED = 20_260_727
UNKNOWN_VALUE = np.uint8(112)


class AssetRejected(RuntimeError):
    """Raised when an asset cannot safely produce a valid session."""


@dataclass(frozen=True)
class SceneAsset:
    dataset: str
    scene_id: str
    scene_file: Path
    dataset_root: Path
    scene_dataset_config: Optional[Path] = None
    navmesh_file: Optional[Path] = None


@dataclass(frozen=True)
class SessionParameters:
    seed: int
    horizontal_fov_degrees: float
    camera_height_m: float
    robot_speed_mps: float


@dataclass(frozen=True)
class PlannedPath:
    start: np.ndarray
    goal: np.ndarray
    waypoints: np.ndarray
    geodesic_distance_m: float
    polyline_distance_m: float
    floor_height_m: float


@dataclass(frozen=True)
class InitializedScene:
    navmesh: Path
    navmesh_source: str
    lower_bound: np.ndarray
    full_truth: np.ndarray
    full_valid_coverage: np.ndarray
    voxel_statistics: Dict[str, Any]
    estimated_voxel_cells: int
    floor_height_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect randomized synchronized Habitat RGB/BEV sessions"
    )
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        default=DEFAULT_DATASET_ROOTS,
        metavar="PATH",
        help=(
            "one or more HM3D, HSSD, or MP3D dataset roots; Replica is only "
            "used when explicitly included with --enabled-datasets"
        ),
    )
    parser.add_argument(
        "--enabled-datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=DEFAULT_ENABLED_DATASETS,
        help=(
            "datasets eligible for collection "
            "(default: hm3d hssd mp3d; Replica disabled)"
        ),
    )
    parser.add_argument(
        "--scene-train-fraction",
        type=float,
        default=DEFAULT_SCENE_TRAIN_FRACTION,
        help=(
            "fraction of scenes assigned to the collection split independently "
            "for each enabled dataset (default: 0.95)"
        ),
    )
    parser.add_argument(
        "--scene-split-role",
        choices=("train", "validation"),
        default="train",
        help=(
            "collect from the 95%% scene split or the held-out validation "
            "scenes (default: train)"
        ),
    )
    parser.add_argument(
        "--scene-split-seed",
        type=int,
        default=DEFAULT_SCENE_SPLIT_SEED,
        help="stable scene-level split seed shared by every GPU",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        dest="legacy_dataset_roots",
        help="backward-compatible repeatable alias overriding --dataset-roots",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        dest="scene_ids",
        default=[],
        help=(
            "limit discovery to an exact scene ID; repeat to select multiple "
            "scenes (default: no filtering)"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sessions",
        type=int,
        default=None,
        help=(
            "number of completed sessions; when omitted, collect the configured "
            "number of paths from every scene in the selected split"
        ),
    )
    parser.add_argument(
        "--hm3d-sessions",
        type=int,
        default=None,
        help="exact number of completed HM3D sessions",
    )
    parser.add_argument(
        "--hssd-sessions",
        type=int,
        default=None,
        help="exact number of completed HSSD sessions",
    )
    parser.add_argument(
        "--replica-sessions",
        type=int,
        default=None,
        help="exact number of completed Replica sessions",
    )
    parser.add_argument(
        "--mp3d-sessions",
        type=int,
        default=None,
        help="exact number of completed MP3D sessions",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--capture-hz", type=float, default=2.0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
        help="Habitat-Sim GPU device index visible to this process",
    )
    parser.add_argument("--bev-size", type=int, default=512)
    parser.add_argument(
        "--bev-extents",
        type=float,
        nargs="+",
        default=DEFAULT_BEV_EXTENTS,
        metavar="METRES",
    )
    parser.add_argument(
        "--camera-height-range",
        type=float,
        nargs=2,
        default=DEFAULT_CAMERA_HEIGHT_RANGE_M,
        metavar=("MIN_METRES", "MAX_METRES"),
    )
    parser.add_argument(
        "--fov-range",
        type=float,
        nargs=2,
        default=DEFAULT_HORIZONTAL_FOV_RANGE_DEGREES,
        metavar=("MIN_DEGREES", "MAX_DEGREES"),
    )
    parser.add_argument(
        "--robot-speed-range",
        type=float,
        nargs=2,
        default=DEFAULT_ROBOT_SPEED_RANGE_MPS,
        metavar=("MIN_MPS", "MAX_MPS"),
    )
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument(
        "--obstacle-min-height",
        type=float,
        default=DEFAULT_OBSTACLE_MIN_HEIGHT_M,
    )
    parser.add_argument(
        "--obstacle-max-height",
        type=float,
        default=DEFAULT_OBSTACLE_MAX_HEIGHT_M,
    )
    parser.add_argument("--min-path-length", type=float, default=3.0)
    parser.add_argument("--max-path-length", type=float, default=10.0)
    parser.add_argument(
        "--path-attempts",
        type=int,
        default=600,
        help="NavMesh start/goal candidates tried by one path search",
    )
    parser.add_argument(
        "--max-initialization-attempts",
        "--max-session-path-attempts",
        dest="max_initialization_attempts",
        type=int,
        default=DEFAULT_MAX_INITIALIZATION_ATTEMPTS,
        help=(
            "maximum total session attempts in one loaded/voxelized scene; "
            "successful and rejected attempts both count (default: 200; "
            "legacy alias: --max-session-path-attempts)"
        ),
    )
    parser.add_argument(
        "--max-single-bev-void-ratio",
        type=float,
        default=DEFAULT_MAX_SINGLE_BEV_VOID_RATIO,
        help=(
            "reject and delete a session attempt immediately when any current "
            "Single Complete GT has more than this fraction of geometry VOID "
            "(default: 0.30)"
        ),
    )
    parser.add_argument(
        "--sessions-per-initialization",
        type=int,
        default=DEFAULT_SESSIONS_PER_INITIALIZATION,
        help=(
            "independent path sessions collected from one loaded/voxelized "
            "scene initialization; rejected path candidates are resampled in "
            "that same initialization until the target is reached "
            "(default: 25)"
        ),
    )
    parser.add_argument("--max-floor-variation", type=float, default=0.25)
    parser.add_argument(
        "--max-voxel-grid-cells",
        type=int,
        default=0,
        help=(
            "optional maximum rows*columns*height-layers before rejecting an "
            "asset; 0 disables this rejection (default: 0)"
        ),
    )
    parser.add_argument(
        "--merged-extents",
        type=float,
        nargs="+",
        default=DEFAULT_MERGED_EXTENTS,
        metavar="METRES",
        help="ego-centric merged output spans rendered in parallel",
    )
    parser.add_argument(
        "--navmesh-cache",
        type=Path,
        default=None,
        help="cache for generated HSSD navmeshes (default: OUTPUT/_navmesh_cache)",
    )
    parser.add_argument(
        "--skip-vggt",
        action="store_true",
        help="collect simulator data only, without VGGT estimates",
    )
    parser.add_argument(
        "--vggt-root",
        type=Path,
        default=None,
        help="original VGGT source tree (auto-detected when omitted)",
    )
    parser.add_argument(
        "--vggt-python",
        type=Path,
        default=None,
        help="Python executable containing torch for VGGT (auto-detected when omitted)",
    )
    parser.add_argument(
        "--vggt-checkpoint",
        type=Path,
        default=None,
        help="VGGT model.pt checkpoint (auto-detected under --vggt-root)",
    )
    parser.add_argument(
        "--vggt-image-resolution",
        type=int,
        default=512,
        help="VGGT balanced preprocessing resolution; must be divisible by 16",
    )
    parser.add_argument(
        "--vggt-device",
        default="cuda",
        help="device passed to the VGGT estimator (default: cuda)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sessions is not None and args.sessions <= 0:
        raise ValueError("--sessions must be positive")
    if (
        not args.enabled_datasets
        or len(set(args.enabled_datasets)) != len(args.enabled_datasets)
    ):
        raise ValueError("--enabled-datasets must contain unique dataset names")
    if not 0.0 < args.scene_train_fraction < 1.0:
        raise ValueError("--scene-train-fraction must be strictly between 0 and 1")
    quota_by_dataset = {
        "hm3d": args.hm3d_sessions,
        "replica": args.replica_sessions,
        "hssd": args.hssd_sessions,
        "mp3d": args.mp3d_sessions,
    }
    if any(value is not None for value in quota_by_dataset.values()):
        if args.sessions is not None:
            raise ValueError(
                "--sessions cannot be combined with per-dataset session quotas"
            )
        missing_enabled = [
            dataset
            for dataset in args.enabled_datasets
            if quota_by_dataset[dataset] is None
        ]
        if missing_enabled:
            raise ValueError(
                "session quotas must be provided for every enabled dataset: "
                + ", ".join(missing_enabled)
            )
        disabled_nonzero = [
            dataset
            for dataset, value in quota_by_dataset.items()
            if dataset not in args.enabled_datasets and value not in (None, 0)
        ]
        if disabled_nonzero:
            raise ValueError(
                "non-zero quota provided for a disabled dataset: "
                + ", ".join(disabled_nonzero)
            )
        enabled_values = [
            int(quota_by_dataset[dataset])
            for dataset in args.enabled_datasets
        ]
        if any(value < 0 for value in enabled_values):
            raise ValueError("per-dataset session quotas cannot be negative")
        if sum(enabled_values) <= 0:
            raise ValueError("at least one per-dataset session quota must be positive")
    if args.capture_hz <= 0:
        raise ValueError("--capture-hz must be positive")
    if args.camera_width <= 0 or args.camera_height <= 0 or args.bev_size <= 0:
        raise ValueError("camera and BEV dimensions must be positive")
    if args.gpu_device_id < 0:
        raise ValueError("--gpu-device-id must be non-negative")
    if args.sessions_per_initialization <= 0:
        raise ValueError("--sessions-per-initialization must be positive")
    if args.max_voxel_grid_cells < 0:
        raise ValueError("--max-voxel-grid-cells cannot be negative")
    if not args.bev_extents or any(value <= 0 for value in args.bev_extents):
        raise ValueError("all --bev-extents must be positive")
    if len(set(args.bev_extents)) != len(args.bev_extents):
        raise ValueError("--bev-extents cannot contain duplicates")
    camera_height_min, camera_height_max = args.camera_height_range
    if not 0 < camera_height_min <= camera_height_max:
        raise ValueError(
            "--camera-height-range must satisfy 0 < minimum <= maximum"
        )
    fov_min, fov_max = args.fov_range
    if not 0 < fov_min <= fov_max < 180:
        raise ValueError("--fov-range must satisfy 0 < minimum <= maximum < 180")
    speed_min, speed_max = args.robot_speed_range
    if not 0 < speed_min <= speed_max:
        raise ValueError(
            "--robot-speed-range must satisfy 0 < minimum <= maximum"
        )
    if not 0.001 <= args.voxel_size <= 0.2:
        raise ValueError("--voxel-size must be in [0.001, 0.2]")
    if not 0 <= args.obstacle_min_height < args.obstacle_max_height:
        raise ValueError("obstacle heights must satisfy 0 <= minimum < maximum")
    if not 0 < args.min_path_length <= args.max_path_length:
        raise ValueError("path lengths must satisfy 0 < minimum <= maximum")
    if args.path_attempts <= 0:
        raise ValueError("path attempts must be positive")
    if args.max_initialization_attempts <= 0:
        raise ValueError("--max-initialization-attempts must be positive")
    if not 0.0 <= args.max_single_bev_void_ratio <= 1.0:
        raise ValueError("--max-single-bev-void-ratio must be within [0, 1]")
    if not args.merged_extents or any(value <= 0 for value in args.merged_extents):
        raise ValueError("all --merged-extents must be positive")
    if len(set(args.merged_extents)) != len(args.merged_extents):
        raise ValueError("--merged-extents cannot contain duplicates")
    if args.vggt_image_resolution <= 0 or args.vggt_image_resolution % 16:
        raise ValueError("--vggt-image-resolution must be positive and divisible by 16")


def sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "scene"


def discover_scenes(dataset_root: Path) -> List[SceneAsset]:
    """Discover dataset-specific scene entrypoints without cross-labeling assets."""
    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        return []

    hssd_config = root / "hssd-hab.scene_dataset_config.json"
    if hssd_config.is_file():
        return [
            SceneAsset(
                dataset="hssd",
                scene_id=scene.name.removesuffix(".scene_instance.json"),
                scene_file=scene,
                dataset_root=root,
                scene_dataset_config=hssd_config,
            )
            for scene in sorted((root / "scenes").glob("*.scene_instance.json"))
        ]

    replica_config = root / "replica.scene_dataset_config.json"
    replica_stages = sorted(root.glob("*/habitat/replica_stage.stage_config.json"))
    if replica_stages:
        assets: List[SceneAsset] = []
        for stage in replica_stages:
            habitat_dir = stage.parent
            navmesh_candidates = (
                habitat_dir / "mesh_semantic.navmesh",
                habitat_dir / "mesh_preseg_semantic.navmesh",
            )
            assets.append(
                SceneAsset(
                    dataset="replica",
                    scene_id=stage.parent.parent.name,
                    scene_file=stage,
                    dataset_root=root,
                    scene_dataset_config=(
                        replica_config if replica_config.is_file() else None
                    ),
                    navmesh_file=next(
                        (path for path in navmesh_candidates if path.is_file()), None
                    ),
                )
            )
        return assets

    # Matterport3D Habitat assets use one directory per building with matching
    # <scene>.glb, <scene>.navmesh, <scene>.house, and semantic PLY files.  The
    # GLB is the rendering/collision stage and the supplied navmesh is loaded
    # explicitly.  Requiring the HOUSE file prevents arbitrary plain GLBs from
    # being mislabeled as MP3D.
    mp3d_assets: List[SceneAsset] = []
    for scene in sorted(root.glob("*/*.glb")):
        scene_id = scene.parent.name
        navmesh = scene.with_suffix(".navmesh")
        house = scene.with_suffix(".house")
        semantic = scene.with_name(f"{scene_id}_semantic.ply")
        if (
            scene.name == f"{scene_id}.glb"
            and navmesh.is_file()
            and house.is_file()
            and semantic.is_file()
        ):
            mp3d_assets.append(
                SceneAsset(
                    dataset="mp3d",
                    scene_id=scene_id,
                    scene_file=scene,
                    dataset_root=root,
                    navmesh_file=navmesh,
                )
            )
    if mp3d_assets:
        return mp3d_assets

    assets_by_scene_id: Dict[str, SceneAsset] = {}
    for scene in sorted(root.rglob("*.basis.glb")):
        navmesh = scene.with_suffix(".navmesh")
        if navmesh.is_file():
            asset = SceneAsset(
                dataset="hm3d",
                scene_id=scene.parent.name,
                scene_file=scene,
                dataset_root=root,
                navmesh_file=navmesh,
            )
            existing = assets_by_scene_id.get(asset.scene_id)
            if existing is None:
                assets_by_scene_id[asset.scene_id] = asset
                continue

            # HM3D minival is an official subset copied from val. Treat those
            # files as one logical scene so the same environment can never
            # enter both the training and validation partitions. Prefer the
            # non-minival copy, then use the absolute path as a deterministic
            # tie-breaker.
            def canonical_rank(candidate: SceneAsset) -> Tuple[bool, str]:
                relative_parts = candidate.scene_file.relative_to(root).parts
                return "minival" in relative_parts, str(candidate.scene_file)

            assets_by_scene_id[asset.scene_id] = min(
                existing,
                asset,
                key=canonical_rank,
            )
    return sorted(
        assets_by_scene_id.values(),
        key=lambda asset: (asset.scene_id, str(asset.scene_file)),
    )


def partition_scene_assets(
    scenes: Sequence[SceneAsset],
    *,
    enabled_datasets: Sequence[str],
    train_fraction: float,
    split_seed: int,
) -> Dict[str, Dict[str, List[SceneAsset]]]:
    """Create an exact, deterministic, non-overlapping scene-level split."""

    partitions: Dict[str, Dict[str, List[SceneAsset]]] = {}
    for dataset in enabled_datasets:
        dataset_scenes = [scene for scene in scenes if scene.dataset == dataset]
        scene_ids = [scene.scene_id for scene in dataset_scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError(
                f"duplicate {dataset} scene IDs prevent a stable split"
            )

        def score(scene: SceneAsset) -> Tuple[bytes, str]:
            identity = (
                f"vggnav-scene-split-v1\0{split_seed}\0"
                f"{dataset}\0{scene.scene_id}"
            )
            return hashlib.sha256(identity.encode("utf-8")).digest(), scene.scene_id

        ordered = sorted(dataset_scenes, key=score)
        train_count = math.floor(train_fraction * len(ordered))
        partitions[dataset] = {
            "train": ordered[:train_count],
            "validation": ordered[train_count:],
        }
    return partitions


def scene_split_manifest(
    partitions: Dict[str, Dict[str, List[SceneAsset]]],
    *,
    train_fraction: float,
    split_seed: int,
    selected_role: str,
) -> Dict[str, Any]:
    datasets: Dict[str, Any] = {}
    for dataset, roles in partitions.items():
        train = roles["train"]
        validation = roles["validation"]
        datasets[dataset] = {
            "total_scene_count": len(train) + len(validation),
            "train_scene_count": len(train),
            "validation_scene_count": len(validation),
            "train_scenes": [
                {
                    "scene_id": scene.scene_id,
                    "scene_file": str(scene.scene_file),
                }
                for scene in train
            ],
            "validation_scenes": [
                {
                    "scene_id": scene.scene_id,
                    "scene_file": str(scene.scene_file),
                }
                for scene in validation
            ],
        }
    return {
        "schema_version": 1,
        "algorithm": "sha256-ranked exact scene-count split v1",
        "split_unit": "scene",
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "validation_fraction": 1.0 - train_fraction,
        "selected_role": selected_role,
        "enabled_datasets": list(partitions),
        "replica_enabled": "replica" in partitions,
        "mp3d_enabled": "mp3d" in partitions,
        "datasets": datasets,
    }


def extent_key(extent: float) -> str:
    return f"bev_{extent:g}m".replace(".", "p")


def merged_modality(kind: str, extent: float) -> str:
    if kind not in {"masked", "complete"}:
        raise ValueError(f"unsupported merged BEV kind: {kind}")
    return f"merged_{kind}_{extent:g}m".replace(".", "p")


def bev_modalities(merged_extents: Sequence[float]) -> List[str]:
    modalities = ["masked", "complete"]
    for extent in merged_extents:
        modalities.extend(
            [
                merged_modality("masked", extent),
                merged_modality("complete", extent),
            ]
        )
    return modalities


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=json_ready) + "\n",
        encoding="utf-8",
    )


def make_simulator(
    asset: SceneAsset,
    *,
    camera_width: int,
    camera_height: int,
    horizontal_fov_degrees: float,
    sensor_height_m: float,
    gpu_device_id: int,
) -> habitat_sim.Simulator:
    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(asset.scene_file)
    if asset.scene_dataset_config is not None:
        simulator_config.scene_dataset_config_file = str(
            asset.scene_dataset_config
        )
    # HSSD furniture is instantiated as rigid objects. Bullet-backed managed
    # wrappers are required to enumerate their exact collision assets and
    # transforms; no physics stepping occurs during deterministic replay.
    simulator_config.enable_physics = asset.dataset == "hssd"
    simulator_config.gpu_device_id = gpu_device_id

    camera = habitat_sim.CameraSensorSpec()
    camera.uuid = "camera_sensor"
    camera.sensor_type = habitat_sim.SensorType.COLOR
    camera.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    camera.resolution = [camera_height, camera_width]
    camera.position = [0.0, sensor_height_m, 0.0]
    camera.orientation = [0.0, 0.0, 0.0]
    camera.hfov = horizontal_fov_degrees
    camera.near = 0.01
    camera.far = 1000.0

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth_sensor"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth.resolution = [camera_height, camera_width]
    depth.position = [0.0, sensor_height_m, 0.0]
    depth.orientation = [0.0, 0.0, 0.0]
    depth.hfov = horizontal_fov_degrees
    depth.near = camera.near
    depth.far = camera.far

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [camera, depth]
    physical_gpu = physical_gpu_from_visible_device(gpu_device_id)
    with renderer_startup_gate(
        physical_gpu,
        label=f"habitat:{asset.dataset}:{asset.scene_id}",
    ):
        return habitat_sim.Simulator(
            habitat_sim.Configuration(simulator_config, [agent_config])
        )


def camera_intrinsics(
    width: int, height: int, horizontal_fov_degrees: float
) -> Dict[str, Any]:
    horizontal_radians = math.radians(horizontal_fov_degrees)
    focal_length = 0.5 * width / math.tan(0.5 * horizontal_radians)
    vertical_fov = 2.0 * math.atan(0.5 * height / focal_length)
    return {
        "model": "pinhole",
        "width": width,
        "height": height,
        "fx": focal_length,
        "fy": focal_length,
        "cx": (width - 1) / 2.0,
        "cy": (height - 1) / 2.0,
        "K": [
            [focal_length, 0.0, (width - 1) / 2.0],
            [0.0, focal_length, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        "horizontal_fov_degrees": horizontal_fov_degrees,
        "vertical_fov_degrees": math.degrees(vertical_fov),
        "pixel_coordinate_convention": (
            "integer coordinates address pixel centers; origin is top-left"
        ),
    }


def prepare_navmesh(
    simulator: habitat_sim.Simulator,
    asset: SceneAsset,
    navmesh_cache: Path,
) -> Tuple[Path, str]:
    """Load a supplied navmesh or generate one from the fully instantiated scene."""
    if simulator.pathfinder.is_loaded:
        source = asset.navmesh_file or Path("<scene-configuration>")
        return source, "loaded_by_scene_configuration"
    if asset.navmesh_file is not None and asset.navmesh_file.is_file():
        if not simulator.pathfinder.load_nav_mesh(str(asset.navmesh_file)):
            raise AssetRejected(f"failed to load navmesh: {asset.navmesh_file}")
        return asset.navmesh_file, "dataset_file"

    cache_file = (
        navmesh_cache
        / asset.dataset
        / f"{sanitize_identifier(asset.scene_id)}.navmesh"
    )
    if cache_file.is_file():
        if simulator.pathfinder.load_nav_mesh(str(cache_file)):
            return cache_file, "generated_cache"
        cache_file.unlink()

    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = 0.10
    settings.agent_height = 1.50
    settings.include_static_objects = True
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"  generating navmesh from complete {asset.dataset} scene "
        f"(including static objects)",
        flush=True,
    )
    if not simulator.recompute_navmesh(simulator.pathfinder, settings):
        raise AssetRejected("Habitat failed to generate a scene navmesh")
    if not simulator.pathfinder.is_loaded:
        raise AssetRejected("generated navmesh did not become ready")
    if not simulator.pathfinder.save_nav_mesh(str(cache_file)):
        raise AssetRejected(f"failed to cache generated navmesh: {cache_file}")
    return cache_file, "generated_complete_scene"


def validate_loaded_asset(
    simulator: habitat_sim.Simulator,
    asset: SceneAsset,
    navmesh_cache: Path,
    camera_width: int,
    camera_height: int,
) -> Tuple[Path, str]:
    navmesh, navmesh_source = prepare_navmesh(simulator, asset, navmesh_cache)
    if not simulator.pathfinder.is_loaded:
        raise AssetRejected("navmesh did not become ready")
    stage = simulator.get_stage_initialization_template()
    if stage is None or not Path(stage.collision_asset_fullpath).is_file():
        raise AssetRejected("active collision asset is unavailable")
    observations = simulator.get_sensor_observations()
    image = np.asarray(observations.get("camera_sensor"))
    if image.shape != (camera_height, camera_width, 4) or image.dtype != np.uint8:
        raise AssetRejected(
            f"camera readiness check failed: shape={image.shape}, dtype={image.dtype}"
        )
    return navmesh, navmesh_source


def clean_waypoints(points: Sequence[Sequence[float]]) -> np.ndarray:
    cleaned: List[np.ndarray] = []
    for point in points:
        coordinate = np.asarray(point, dtype=np.float64)
        if not cleaned or float(np.linalg.norm(coordinate - cleaned[-1])) > 1e-6:
            cleaned.append(coordinate)
    if len(cleaned) < 2:
        raise AssetRejected("shortest path contains fewer than two unique waypoints")
    return np.stack(cleaned)


def random_reachable_path(
    simulator: habitat_sim.Simulator,
    *,
    seed: int,
    min_length: float,
    max_length: float,
    max_floor_variation: float,
    attempts: int,
    target_floor_height: Optional[float] = None,
) -> PlannedPath:
    pathfinder = simulator.pathfinder
    pathfinder.seed(seed)
    for _ in range(attempts):
        start = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        goal = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
            continue
        if target_floor_height is not None and (
            abs(float(start[1]) - target_floor_height) > max_floor_variation
            or abs(float(goal[1]) - target_floor_height) > max_floor_variation
        ):
            continue
        if abs(float(start[1] - goal[1])) > max_floor_variation:
            continue
        query = habitat_sim.ShortestPath()
        query.requested_start = start
        query.requested_end = goal
        if not pathfinder.find_path(query):
            continue
        distance = float(query.geodesic_distance)
        if not min_length <= distance <= max_length:
            continue
        waypoints = clean_waypoints(query.points)
        if float(np.ptp(waypoints[:, 1])) > max_floor_variation:
            continue
        if target_floor_height is not None and np.max(
            np.abs(waypoints[:, 1] - target_floor_height)
        ) > max_floor_variation:
            continue
        segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        polyline_distance = float(np.sum(segment_lengths))
        if not min_length <= polyline_distance <= max_length * 1.05:
            continue
        return PlannedPath(
            start=waypoints[0].astype(np.float32),
            goal=waypoints[-1].astype(np.float32),
            waypoints=waypoints,
            geodesic_distance_m=distance,
            polyline_distance_m=polyline_distance,
            floor_height_m=(
                target_floor_height
                if target_floor_height is not None
                else float(waypoints[0, 1])
            ),
        )
    raise AssetRejected(
        f"no reachable {min_length:g}-{max_length:g} m same-floor path "
        f"was found in {attempts} attempts"
        + (
            f" near floor {target_floor_height:.3f} m"
            if target_floor_height is not None
            else ""
        )
    )


def path_samples(
    plan: PlannedPath, speed_mps: float, capture_hz: float
) -> List[Tuple[float, float, np.ndarray, np.ndarray]]:
    segment_vectors = np.diff(plan.waypoints, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    duration = plan.polyline_distance_m / speed_mps
    times = list(np.arange(0.0, duration, 1.0 / capture_hz))
    if not times or duration - times[-1] > 1e-9:
        times.append(duration)
    samples = []
    for sample_time in times:
        distance = min(sample_time * speed_mps, plan.polyline_distance_m)
        segment = min(
            int(np.searchsorted(cumulative, distance, side="right") - 1),
            len(segment_lengths) - 1,
        )
        length = float(segment_lengths[segment])
        fraction = 0.0 if length <= 1e-12 else (distance - cumulative[segment]) / length
        position = (
            plan.waypoints[segment]
            + fraction * segment_vectors[segment]
        )
        forward = segment_vectors[segment][[0, 2]].astype(np.float64)
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-9:
            for candidate in range(segment, -1, -1):
                forward = segment_vectors[candidate][[0, 2]].astype(np.float64)
                norm = float(np.linalg.norm(forward))
                if norm > 1e-9:
                    break
        forward /= max(norm, 1e-12)
        samples.append((float(sample_time), float(distance), position, forward))
    return samples


def validate_path_against_collision_truth(
    plan: PlannedPath,
    *,
    full_truth: np.ndarray,
    lower_bound: np.ndarray,
    voxel_size: float,
    agent_radius_m: float = DEFAULT_AGENT_RADIUS_M,
) -> Dict[str, Any]:
    """Require the complete robot footprint to remain free along the path."""

    if full_truth.ndim != 2 or full_truth.dtype != np.uint8:
        raise ValueError("full_truth must be a 2-D uint8 occupancy image")
    if voxel_size <= 0 or agent_radius_m <= 0:
        raise ValueError("voxel_size and agent_radius_m must be positive")

    # Sample more densely than the truth grid so a long capture interval cannot
    # jump over a collision.  Repeated source cells are checked only once.
    check_step_m = voxel_size * 0.5
    world_samples: List[np.ndarray] = []
    for start, end in zip(plan.waypoints, plan.waypoints[1:]):
        segment = np.asarray(end - start, dtype=np.float64)
        length = float(np.linalg.norm(segment))
        steps = max(1, int(math.ceil(length / check_step_m)))
        world_samples.extend(
            np.asarray(start, dtype=np.float64)
            + segment * (index / steps)
            for index in range(steps)
        )
    world_samples.append(np.asarray(plan.waypoints[-1], dtype=np.float64))

    # A source pixel represents an area, not an infinitesimal point.  Including
    # half of its diagonal makes this a conservative robot-footprint test.
    conservative_radius_m = (
        float(agent_radius_m) + voxel_size * math.sqrt(0.5)
    )
    radius_pixels = int(math.ceil(conservative_radius_m / voxel_size))
    footprint_offsets = [
        (delta_row, delta_column)
        for delta_row in range(-radius_pixels, radius_pixels + 1)
        for delta_column in range(-radius_pixels, radius_pixels + 1)
        if math.hypot(delta_row, delta_column) * voxel_size
        <= conservative_radius_m + 1e-12
    ]

    checked_cells: set[Tuple[int, int]] = set()
    for position in world_samples:
        column = int(
            math.floor(
                (float(position[0]) - float(lower_bound[0])) / voxel_size
            )
        )
        unflipped_row = int(
            math.floor(
                (float(position[2]) - float(lower_bound[2])) / voxel_size
            )
        )
        row = full_truth.shape[0] - 1 - unflipped_row
        cell = (row, column)
        if cell in checked_cells:
            continue
        checked_cells.add(cell)
        if not (
            0 <= row < full_truth.shape[0]
            and 0 <= column < full_truth.shape[1]
        ):
            raise AssetRejected(
                "planned path leaves the collision-truth raster bounds"
            )
        for delta_row, delta_column in footprint_offsets:
            test_row = row + delta_row
            test_column = column + delta_column
            if not (
                0 <= test_row < full_truth.shape[0]
                and 0 <= test_column < full_truth.shape[1]
            ):
                raise AssetRejected(
                    "planned robot footprint leaves collision-truth bounds"
                )
            if full_truth[test_row, test_column] != 255:
                raise AssetRejected(
                    "planned path conflicts with complete collision truth at "
                    f"world=({position[0]:.3f}, {position[1]:.3f}, "
                    f"{position[2]:.3f}), truth_pixel=({row}, {column})"
                )

    return {
        "passed": True,
        "checked_unique_truth_cells": len(checked_cells),
        "dense_check_step_m": check_step_m,
        "agent_radius_m": float(agent_radius_m),
        "conservative_footprint_radius_m": conservative_radius_m,
    }


def set_agent_pose(
    simulator: habitat_sim.Simulator,
    position: np.ndarray,
    forward_xz: np.ndarray,
) -> None:
    yaw = math.atan2(-float(forward_xz[0]), -float(forward_xz[1]))
    rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
    agent = simulator.get_agent(0)
    state = agent.get_state()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = rotation
    agent.set_state(state, reset_sensors=True)


def crop_corners(
    position: Sequence[float],
    forward: Sequence[float],
    right: Sequence[float],
    extent: float,
) -> np.ndarray:
    half = extent / 2.0
    corners = []
    for local_right in (-half, half):
        for local_forward in (-half, half):
            corners.append(
                [
                    position[0] + right[0] * local_right + forward[0] * local_forward,
                    position[2] + right[1] * local_right + forward[1] * local_forward,
                ]
            )
    return np.asarray(corners, dtype=np.float64)


class BEVAccumulator:
    """Accumulate visibility and render static truth in the latest ego frame.

    Occupancy is never fused from the masked inputs.  Complete simulator truth
    and its coverage are accumulated independently, while masked inputs
    contribute only a monotonic observed/unobserved mask.  Both merged products
    therefore sample their values from the same static truth map.
    """

    def __init__(self, extent: float, size: int) -> None:
        self.extent = float(extent)
        self.size = int(size)
        self.meters_per_pixel = self.extent / self.size
        self.minimum_x: Optional[float] = None
        self.maximum_z: Optional[float] = None
        self.complete = np.full((1, 1), UNKNOWN_VALUE, dtype=np.uint8)
        self.complete_known = np.zeros((1, 1), dtype=bool)
        self.observed = np.zeros((1, 1), dtype=bool)
        self.all_crop_corners: List[np.ndarray] = []

    def _expand(self, corners: np.ndarray) -> None:
        mpp = self.meters_per_pixel
        requested_min_x = math.floor(float(np.min(corners[:, 0])) / mpp) * mpp
        requested_max_x = math.ceil(float(np.max(corners[:, 0])) / mpp) * mpp
        requested_min_z = math.floor(float(np.min(corners[:, 1])) / mpp) * mpp
        requested_max_z = math.ceil(float(np.max(corners[:, 1])) / mpp) * mpp
        if self.minimum_x is None or self.maximum_z is None:
            columns = int(round((requested_max_x - requested_min_x) / mpp)) + 1
            rows = int(round((requested_max_z - requested_min_z) / mpp)) + 1
            self.minimum_x = requested_min_x
            self.maximum_z = requested_max_z
            shape = (rows, columns)
            self.complete = np.full(shape, UNKNOWN_VALUE, dtype=np.uint8)
            self.complete_known = np.zeros(shape, dtype=bool)
            self.observed = np.zeros(shape, dtype=bool)
            return

        old_min_x = self.minimum_x
        old_max_z = self.maximum_z
        old_max_x = old_min_x + (self.complete.shape[1] - 1) * mpp
        old_min_z = old_max_z - (self.complete.shape[0] - 1) * mpp
        new_min_x = min(old_min_x, requested_min_x)
        new_max_x = max(old_max_x, requested_max_x)
        new_min_z = min(old_min_z, requested_min_z)
        new_max_z = max(old_max_z, requested_max_z)
        if (
            abs(new_min_x - old_min_x) < mpp * 0.1
            and abs(new_max_x - old_max_x) < mpp * 0.1
            and abs(new_min_z - old_min_z) < mpp * 0.1
            and abs(new_max_z - old_max_z) < mpp * 0.1
        ):
            return
        new_columns = int(round((new_max_x - new_min_x) / mpp)) + 1
        new_rows = int(round((new_max_z - new_min_z) / mpp)) + 1
        column_offset = int(round((old_min_x - new_min_x) / mpp))
        row_offset = int(round((new_max_z - old_max_z) / mpp))
        slices = (
            slice(row_offset, row_offset + self.complete.shape[0]),
            slice(column_offset, column_offset + self.complete.shape[1]),
        )
        for name, fill in (
            ("complete", UNKNOWN_VALUE),
            ("complete_known", False),
            ("observed", False),
        ):
            old = getattr(self, name)
            expanded = np.full((new_rows, new_columns), fill, dtype=old.dtype)
            expanded[slices] = old
            setattr(self, name, expanded)
        self.minimum_x = new_min_x
        self.maximum_z = new_max_z

    def _ego_to_world_transform(
        self,
        position: Sequence[float],
        forward: Sequence[float],
        right: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.minimum_x is None or self.maximum_z is None:
            raise RuntimeError("accumulator bounds are not initialized")
        center = float(self.size // 2)
        mpp = self.meters_per_pixel
        matrix = np.asarray(
            [[forward[1], -forward[0]], [-right[1], right[0]]],
            dtype=np.float64,
        )
        offset = np.asarray(
            [
                center
                - (self.minimum_x - position[0]) * forward[0] / mpp
                - (self.maximum_z - position[2]) * forward[1] / mpp,
                center
                + (self.minimum_x - position[0]) * right[0] / mpp
                + (self.maximum_z - position[2]) * right[1] / mpp,
            ],
            dtype=np.float64,
        )
        return matrix, offset

    def update(
        self,
        complete: np.ndarray,
        masked: np.ndarray,
        extrinsic: Dict[str, Any],
    ) -> None:
        position = extrinsic["agent_position_world_m"]
        forward = extrinsic["bev_forward_xz"]
        right = extrinsic["bev_right_xz"]
        corners = crop_corners(position, forward, right, self.extent)
        self._expand(corners)
        self.all_crop_corners.append(corners)
        matrix, offset = self._ego_to_world_transform(position, forward, right)
        output_shape = self.complete.shape

        complete_warped = ndimage.affine_transform(
            complete,
            matrix,
            offset,
            output_shape=output_shape,
            output=np.uint8,
            order=0,
            mode="constant",
            cval=int(UNKNOWN_VALUE),
            prefilter=False,
        )
        complete_mask = ndimage.affine_transform(
            np.ones_like(complete, dtype=np.uint8),
            matrix,
            offset,
            output_shape=output_shape,
            output=np.uint8,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).astype(bool)
        self.complete[complete_mask] = complete_warped[complete_mask]
        self.complete_known |= complete_mask

        source_known = (masked != UNKNOWN_VALUE).astype(np.uint8)
        observed_warped = ndimage.affine_transform(
            source_known,
            matrix,
            offset,
            output_shape=output_shape,
            output=np.uint8,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).astype(bool)
        self.observed |= observed_warped

    def render_ego(
        self, kind: str, extrinsic: Dict[str, Any], merged_extent: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.all_crop_corners:
            raise RuntimeError("cannot render an empty accumulator")
        if self.minimum_x is None or self.maximum_z is None:
            raise RuntimeError("accumulator bounds are unavailable")
        position = extrinsic["agent_position_world_m"]
        forward = np.asarray(extrinsic["bev_forward_xz"], dtype=np.float64)
        right = np.asarray(extrinsic["bev_right_xz"], dtype=np.float64)
        output_size = self.size
        output_mpp = merged_extent / output_size
        center = float(output_size // 2)
        scale = output_mpp / self.meters_per_pixel
        matrix = np.asarray(
            [
                [forward[1] * scale, -right[1] * scale],
                [-forward[0] * scale, right[0] * scale],
            ],
            dtype=np.float64,
        )
        offset = np.asarray(
            [
                (
                    self.maximum_z
                    - position[2]
                    - center * output_mpp * (forward[1] - right[1])
                )
                / self.meters_per_pixel,
                (
                    position[0]
                    - self.minimum_x
                    + center * output_mpp * (forward[0] - right[0])
                )
                / self.meters_per_pixel,
            ],
            dtype=np.float64,
        )
        if kind not in ("masked", "complete"):
            raise ValueError("kind must be 'masked' or 'complete'")
        source = self.complete
        rendered = ndimage.affine_transform(
            source,
            matrix,
            offset,
            output_shape=(output_size, output_size),
            output=np.uint8,
            order=0,
            mode="constant",
            cval=int(UNKNOWN_VALUE),
            prefilter=False,
        )
        source_known = (
            self.observed & self.complete_known
            if kind == "masked"
            else self.complete_known
        )
        rendered_known = ndimage.affine_transform(
            source_known.astype(np.uint8),
            matrix,
            offset,
            output_shape=(output_size, output_size),
            output=np.uint8,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).astype(bool)
        rendered[~rendered_known] = UNKNOWN_VALUE
        return rendered, {
            "shape": [output_size, output_size],
            "extent_m": merged_extent,
            "meters_per_pixel": output_mpp,
            "orientation": "ego-centric; latest robot at center; forward is up",
            "history_frame_count": len(self.all_crop_corners),
            "normalization": (
                "fixed square; unavailable coverage is 112 unknown; "
                "history outside the square is cropped"
            ),
        }


def estimate_voxel_grid(
    simulator: habitat_sim.Simulator,
    floor_height: float,
    voxel_size: float,
    obstacle_min_height: float,
    obstacle_max_height: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    lower_bound, _ = simulator.pathfinder.get_bounds()
    lower_bound = np.asarray(lower_bound, dtype=np.float64)
    navigable = simulator.pathfinder.get_topdown_view(voxel_size, floor_height)
    sampled_min_height = float(obstacle_min_height)
    layers = int(
        math.ceil((obstacle_max_height - sampled_min_height) / voxel_size)
    )
    estimated_cells = int(navigable.shape[0] * navigable.shape[1] * layers)
    return lower_bound, navigable, estimated_cells


def save_grayscale(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8)[..., :3]).save(path)


def save_depth(path: Path, depth_m: np.ndarray) -> None:
    """Write losslessly compressed float32 metric depth.

    The archive keeps the original float32 values (no quantization), while
    DEFLATE removes the substantial spatial redundancy in indoor depth maps.
    Existing sessions containing ``.npy`` depth are still accepted by the
    validation/audit code; only newly collected sessions use ``.npz``.
    """
    np.savez_compressed(
        path,
        depth=np.asarray(depth_m, dtype=np.float32),
    )


def load_depth(path: Path) -> np.ndarray:
    """Load either a legacy uncompressed NPY or a new compressed NPZ depth."""
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["depth"]:
                raise RuntimeError(
                    f"compressed depth archive has unexpected arrays: "
                    f"{archive.files}"
                )
            return np.asarray(archive["depth"])
    raise RuntimeError(f"unsupported depth suffix: {path}")


def validate_bev_frame_quality(
    complete: np.ndarray,
    masked: np.ndarray,
    *,
    extent_m: float,
    agent_radius_m: float = DEFAULT_AGENT_RADIUS_M,
    complete_may_contain_unknown: bool = False,
) -> Dict[str, int]:
    """Validate aligned GT labels and the complete footprint at one frame."""

    if (
        complete.ndim != 2
        or complete.shape[0] != complete.shape[1]
        or masked.shape != complete.shape
    ):
        raise RuntimeError("BEV pair must contain aligned square images")
    complete_values = set(int(value) for value in np.unique(complete))
    masked_values = set(int(value) for value in np.unique(masked))
    allowed_complete = (
        {0, 112, 255} if complete_may_contain_unknown else {0, 255}
    )
    if not complete_values.issubset(allowed_complete):
        raise RuntimeError(f"invalid complete BEV labels: {complete_values}")
    if not masked_values.issubset({0, 112, 255}):
        raise RuntimeError(f"invalid masked BEV labels: {masked_values}")
    if not complete_may_contain_unknown and 255 not in complete_values:
        raise AssetRejected("complete BEV contains no free cells")

    known = masked != UNKNOWN_VALUE
    known_count = int(np.count_nonzero(known))
    if known_count == 0:
        raise AssetRejected("masked BEV is entirely unknown")
    if not np.array_equal(masked[known], complete[known]):
        raise RuntimeError("masked BEV known cells disagree with complete truth")

    size = complete.shape[0]
    center = size // 2
    metres_per_pixel = float(extent_m) / size
    radius_pixels = int(math.ceil(agent_radius_m / metres_per_pixel))
    rows, columns = np.ogrid[:size, :size]
    footprint = (
        (rows - center) ** 2 + (columns - center) ** 2
        <= radius_pixels**2
    )
    if np.any(complete[footprint] != 255):
        raise AssetRejected(
            "robot footprint is not completely free in complete BEV"
        )
    if int(masked[center, center]) != 255:
        raise AssetRejected(
            "robot origin is not known-free in masked BEV"
        )
    return {
        "known_cell_count": known_count,
        "free_cell_count": int(np.count_nonzero(complete == 255)),
        "footprint_radius_pixels": radius_pixels,
    }


def render_trajectory_image(
    simulator: habitat_sim.Simulator,
    plan: PlannedPath,
    sampled_positions: Sequence[Sequence[float]],
    output: Path,
    meters_per_pixel: float = 0.05,
) -> None:
    topdown = simulator.pathfinder.get_topdown_view(
        meters_per_pixel, plan.floor_height_m
    )
    lower_bound, _ = simulator.pathfinder.get_bounds()
    lower_bound = np.asarray(lower_bound, dtype=np.float64)
    rgb = np.zeros((*topdown.shape, 3), dtype=np.uint8)
    rgb[topdown] = [236, 240, 242]
    rgb[~topdown] = [25, 30, 36]
    image = Image.fromarray(np.flipud(rgb))
    draw = ImageDraw.Draw(image)

    def pixel(point: Sequence[float]) -> Tuple[int, int]:
        column = int(round((point[0] - lower_bound[0]) / meters_per_pixel))
        unflipped_row = int(round((point[2] - lower_bound[2]) / meters_per_pixel))
        row = image.height - 1 - unflipped_row
        return column, row

    path_pixels = [pixel(point) for point in plan.waypoints]
    draw.line(path_pixels, fill=(255, 181, 45), width=4)
    for position in sampled_positions:
        x, y = pixel(position)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(64, 201, 255))
    for point, color in ((plan.start, (49, 233, 129)), (plan.goal, (255, 77, 94))):
        x, y = pixel(point)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white")
    image.save(output)


def make_preview(
    session_dir: Path,
    extent_keys: Sequence[str],
    merged_extents: Sequence[float],
    final_frame_name: str,
    parameters: SessionParameters,
) -> None:
    tile = 240
    gap = 10
    header = 64
    rows = len(extent_keys) + 1
    modalities = bev_modalities(merged_extents)
    columns = len(modalities)
    width = columns * tile + (columns + 1) * gap
    height = header + rows * tile + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), (12, 16, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 10),
        (
            f"FOV {parameters.horizontal_fov_degrees:.1f} deg | "
            f"camera {parameters.camera_height_m:.2f} m | "
            f"speed {parameters.robot_speed_mps:.2f} m/s"
        ),
        fill=(238, 243, 248),
    )

    def paste(path: Path, column: int, row: int, label: str) -> None:
        image = Image.open(path).convert("RGB")
        contained = ImageOps.contain(image, (tile, tile - 20))
        x = gap + column * (tile + gap) + (tile - contained.width) // 2
        y = header + gap + row * (tile + gap)
        canvas.paste(contained, (x, y + 18))
        draw.text((gap + column * (tile + gap), y), label, fill=(142, 155, 169))

    paste(session_dir / "camera" / final_frame_name, 0, 0, "final camera")
    paste(session_dir / "ground_truth_trajectory.png", 1, 0, "ground-truth trajectory")
    paste(session_dir / "camera" / "frame_000000.png", 2, 0, "initial camera")
    for row, key in enumerate(extent_keys, start=1):
        root = session_dir / key
        for column, modality in enumerate(modalities):
            paste(
                root / modality / final_frame_name,
                column,
                row,
                f"{key}: {modality}",
            )
    canvas.save(session_dir / "preview.png")


def validate_session(
    session_dir: Path,
    frame_count: int,
    extent_keys: Sequence[str],
    bev_extents: Sequence[float],
    merged_extents: Sequence[float],
    camera_size: Tuple[int, int],
    bev_size: int,
    max_single_bev_void_ratio: float,
) -> Dict[str, Any]:
    expected_names = [f"frame_{index:06d}.png" for index in range(frame_count)]
    camera_names = sorted(path.name for path in (session_dir / "camera").glob("*.png"))
    if camera_names != expected_names:
        raise RuntimeError("camera frame sequence is incomplete")
    with Image.open(session_dir / "camera" / expected_names[-1]) as image:
        if image.size != camera_size or image.mode != "RGB":
            raise RuntimeError("camera frame dimensions or mode are invalid")
    expected_depth_stems = [
        f"frame_{index:06d}" for index in range(frame_count)
    ]
    depth_dir = session_dir / "depth"
    depth_names_by_suffix = {
        suffix: sorted(path.name for path in depth_dir.glob(f"*.{suffix}"))
        for suffix in ("npy", "npz")
    }
    depth_suffix = next(
        (
            suffix
            for suffix, names in depth_names_by_suffix.items()
            if [Path(name).stem for name in names] == expected_depth_stems
        ),
        None,
    )
    if depth_suffix is None:
        raise RuntimeError("depth frame sequence is incomplete")
    depth_names = depth_names_by_suffix[depth_suffix]
    final_depth = load_depth(session_dir / "depth" / depth_names[-1])
    if final_depth.shape != (camera_size[1], camera_size[0]):
        raise RuntimeError(f"invalid depth shape: {final_depth.shape}")
    if final_depth.dtype != np.float32:
        raise RuntimeError(f"invalid depth dtype: {final_depth.dtype}")
    if np.isnan(final_depth).any() or (final_depth < 0).any():
        raise RuntimeError("depth contains NaN or negative values")
    extrinsic_path = session_dir / "camera_extrinsics.jsonl"
    extrinsic_rows = [
        json.loads(line)
        for line in extrinsic_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [row.get("frame_id") for row in extrinsic_rows] != list(range(frame_count)):
        raise RuntimeError("camera extrinsic sequence is incomplete")
    trajectory_rows = [
        json.loads(line)
        for line in (session_dir / "ground_truth_trajectory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if [row.get("frame_id") for row in trajectory_rows] != list(range(frame_count)):
        raise RuntimeError("ground-truth trajectory sequence is incomplete")
    intrinsics = json.loads(
        (session_dir / "camera_intrinsics.json").read_text(encoding="utf-8")
    )
    if intrinsics.get("width") != camera_size[0] or intrinsics.get(
        "height"
    ) != camera_size[1]:
        raise RuntimeError("camera intrinsic dimensions are invalid")
    value_sets: Dict[str, List[int]] = {}
    checked_single_void_ratios: List[float] = []
    modalities = bev_modalities(merged_extents)
    for key, current_extent in zip(extent_keys, bev_extents):
        for modality in modalities:
            folder = session_dir / key / modality
            names = sorted(path.name for path in folder.glob("*.png"))
            if names != expected_names:
                raise RuntimeError(f"incomplete {key}/{modality} sequence")
            values: List[int] = []
            for frame_id, name in enumerate(names):
                with Image.open(folder / name) as image:
                    if image.size != (bev_size, bev_size) or image.mode != "L":
                        raise RuntimeError(
                            f"invalid {key}/{modality} shape or mode: "
                            f"{image.size}, {image.mode}"
                        )
                    raster = np.asarray(image).copy()
                    values = sorted(
                        int(value) for value in np.unique(raster)
                    )
                allowed = (
                    {0, 255}
                    if modality == "complete"
                    else {0, 112, 255}
                )
                if not set(values).issubset(allowed):
                    raise RuntimeError(
                        f"unexpected values in {key}/{modality}: {values}"
                    )
                if modality == "complete" and 255 not in values:
                    raise AssetRejected(
                        f"{key}/{modality}/{name} has no free cells"
                    )
                if modality != "complete" and values == [int(UNKNOWN_VALUE)]:
                    raise AssetRejected(
                        f"{key}/{modality}/{name} is entirely unknown"
                    )
                center = bev_size // 2
                if int(raster[center, center]) != 255:
                    raise AssetRejected(
                        f"{key}/{modality}/{name} robot origin is not free"
                    )
            value_sets[f"{key}/{modality}"] = values
    for frame_id, row in enumerate(trajectory_rows):
        frame_bev = row.get("bev", {})
        for key, current_extent in zip(extent_keys, bev_extents):
            record = frame_bev.get(key, {})
            if record.get("void_coverage_algorithm") != VOID_COVERAGE_ALGORITHM:
                raise RuntimeError(
                    f"frame {frame_id} {key} lacks Strict-VOID geometry provenance"
                )
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
            if ratio > max_single_bev_void_ratio:
                raise SingleBEVVoidRatioExceeded(
                    ratio=ratio,
                    threshold=max_single_bev_void_ratio,
                    frame_id=frame_id,
                    extent_m=current_extent,
                )
            checked_single_void_ratios.append(ratio)
    return {
        "passed": True,
        "frame_count": frame_count,
        "synchronized_modalities_per_frame": 2
        + len(modalities) * len(extent_keys),
        "final_frame_value_sets": value_sets,
        "depth_format": "lossless float32 metres in compressed NumPy .npz",
        "camera_intrinsics_file": "camera_intrinsics.json",
        "camera_extrinsics_file": "camera_extrinsics.jsonl",
        "void_check_version": VOID_CHECK_VERSION,
        "void_coverage_algorithm": VOID_COVERAGE_ALGORITHM,
        "void_filter_contract": VOID_FILTER_CONTRACT,
        "all_single_complete_gt_bevs_void_checked": True,
        "single_complete_gt_bevs_checked": len(checked_single_void_ratios),
        "max_allowed_single_complete_gt_void_ratio": max_single_bev_void_ratio,
        "session_max_single_complete_gt_void_ratio": max(
            checked_single_void_ratios, default=0.0
        ),
    }


def create_session_directories(
    session_dir: Path,
    extent_keys: Sequence[str],
    merged_extents: Sequence[float],
) -> None:
    (session_dir / "camera").mkdir(parents=True)
    (session_dir / "depth").mkdir(parents=True)
    modalities = bev_modalities(merged_extents)
    for key in extent_keys:
        for modality in modalities:
            (session_dir / key / modality).mkdir(parents=True)


def collect_session(
    *,
    asset: SceneAsset,
    session_index: int,
    scene_sampling_cycle: int,
    parameters: SessionParameters,
    output_root: Path,
    args: argparse.Namespace,
    simulator: Optional[habitat_sim.Simulator] = None,
    initialized_scene: Optional[InitializedScene] = None,
    plan: Optional[PlannedPath] = None,
    started_at: Optional[float] = None,
) -> Dict[str, Any]:
    scene_id = asset.scene_id
    session_name = (
        f"session_{session_index:06d}_{asset.dataset}_"
        f"{sanitize_identifier(scene_id)}"
    )
    final_dir = output_root / session_name
    partial_dir = output_root / f".{session_name}.partial"
    if final_dir.exists():
        raise RuntimeError(f"session output already exists: {final_dir}")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)

    if simulator is None and initialized_scene is not None:
        raise ValueError("initialized_scene requires a reusable simulator")
    if simulator is not None and (initialized_scene is None or plan is None):
        raise ValueError(
            "a reusable simulator requires initialized_scene and a planned path"
        )
    started_at = time.time() if started_at is None else started_at
    simulator_context = (
        make_simulator(
            asset,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            horizontal_fov_degrees=parameters.horizontal_fov_degrees,
            sensor_height_m=parameters.camera_height_m,
            gpu_device_id=args.gpu_device_id,
        )
        if simulator is None
        else contextlib.nullcontext(simulator)
    )
    with simulator_context as active_simulator:
        simulator = active_simulator
        if initialized_scene is None:
            navmesh, navmesh_source = validate_loaded_asset(
                simulator,
                asset,
                args.navmesh_cache,
                args.camera_width,
                args.camera_height,
            )
            plan = random_reachable_path(
                simulator,
                seed=parameters.seed,
                min_length=args.min_path_length,
                max_length=args.max_path_length,
                max_floor_variation=args.max_floor_variation,
                attempts=args.path_attempts,
            )
            lower_bound, navigable_map, estimated_voxel_cells = (
                estimate_voxel_grid(
                    simulator,
                    plan.floor_height_m,
                    args.voxel_size,
                    args.obstacle_min_height,
                    args.obstacle_max_height,
                )
            )
            if (
                args.max_voxel_grid_cells > 0
                and estimated_voxel_cells > args.max_voxel_grid_cells
            ):
                raise AssetRejected(
                    f"estimated voxel grid {estimated_voxel_cells:,} exceeds "
                    f"limit {args.max_voxel_grid_cells:,}"
                )
            print(
                f"[{session_index + 1}/{args.sessions}] "
                f"{asset.dataset}/{scene_id}: asset/path ready; "
                f"voxelizing {estimated_voxel_cells:,} cells",
                flush=True,
            )
            full_truth, full_valid_coverage, voxel_statistics = render_full_obstacle_map(
                simulator,
                floor_height=plan.floor_height_m,
                meters_per_pixel=args.voxel_size,
                rows=navigable_map.shape[0],
                columns=navigable_map.shape[1],
                lower_bound=lower_bound,
                obstacle_min_height=args.obstacle_min_height,
                obstacle_max_height=args.obstacle_max_height,
                navigable_map=navigable_map,
            )
        else:
            navmesh = initialized_scene.navmesh
            navmesh_source = initialized_scene.navmesh_source
            lower_bound = initialized_scene.lower_bound
            full_truth = initialized_scene.full_truth
            full_valid_coverage = initialized_scene.full_valid_coverage
            voxel_statistics = initialized_scene.voxel_statistics
            estimated_voxel_cells = initialized_scene.estimated_voxel_cells
            print(
                f"[{session_index + 1}/{args.sessions}] "
                f"{asset.dataset}/{scene_id}: path ready; "
                "using shared loaded scene/navmesh/voxel truth",
                flush=True,
            )

        if plan is None:
            raise RuntimeError("session path was not initialized")
        path_truth_validation = validate_path_against_collision_truth(
            plan,
            full_truth=full_truth,
            lower_bound=lower_bound,
            voxel_size=args.voxel_size,
        )
        samples = path_samples(plan, parameters.robot_speed_mps, args.capture_hz)
        set_agent_pose(simulator, samples[0][2], samples[0][3])
        ready_image = np.asarray(
            simulator.get_sensor_observations()["camera_sensor"]
        )
        if ready_image.shape != (args.camera_height, args.camera_width, 4):
            raise AssetRejected("camera failed after verified path placement")

        extent_keys = [extent_key(value) for value in args.bev_extents]
        create_session_directories(partial_dir, extent_keys, args.merged_extents)
        accumulators = {
            key: BEVAccumulator(extent, args.bev_size)
            for key, extent in zip(extent_keys, args.bev_extents)
        }
        intrinsics = camera_intrinsics(
            args.camera_width,
            args.camera_height,
            parameters.horizontal_fov_degrees,
        )
        write_json(partial_dir / "camera_intrinsics.json", intrinsics)
        metadata: Dict[str, Any] = {
            "schema_version": GT_QUALITY_VERSION,
            "status": "collecting",
            "session_id": session_name,
            "session_seed": parameters.seed,
            "scene_sampling_cycle": scene_sampling_cycle,
            "dataset": asset.dataset,
            "scene_split": {
                "role": args.scene_split_role,
                "unit": "scene",
                "train_fraction": args.scene_train_fraction,
                "split_seed": args.scene_split_seed,
                "manifest": "../scene_split_manifest.json",
            },
            "dataset_root": str(asset.dataset_root),
            "scene_id": scene_id,
            "scene_file": str(asset.scene_file),
            "scene_dataset_config": (
                str(asset.scene_dataset_config)
                if asset.scene_dataset_config is not None
                else None
            ),
            "navmesh_file": str(navmesh),
            "navmesh_source": navmesh_source,
            "asset_load_verified": True,
            "path_verified_before_capture": True,
            "path_verified_against_collision_truth": True,
            "random_parameters": {
                "horizontal_fov_degrees": parameters.horizontal_fov_degrees,
                "camera_height_m": parameters.camera_height_m,
                "robot_speed_mps": parameters.robot_speed_mps,
            },
            "camera_intrinsics": intrinsics,
            "camera_intrinsics_file": "camera_intrinsics.json",
            "camera_extrinsics_file": "camera_extrinsics.jsonl",
            "depth": {
                "directory": "depth",
                "filename_pattern": "frame_INDEX.npz",
                "format": "compressed NumPy .npz (lossless DEFLATE)",
                "dtype": "float32",
                "units": "metres",
                "source": "Habitat-Sim pinhole depth sensor ground truth",
                "near_m": 0.01,
                "far_m": 1000.0,
            },
            "capture_hz": args.capture_hz,
            "gpu_device_id": args.gpu_device_id,
            "motion_model": (
                "constant-speed kinematic replay along the verified Habitat "
                "ShortestPath polyline"
            ),
            "bev": {
                "size": args.bev_size,
                "extent_classes_m": list(args.bev_extents),
                "obstacle_height_band_m": [
                    args.obstacle_min_height,
                    args.obstacle_max_height,
                ],
                "truth_source": "solid Habitat collision-mesh voxel occupancy",
                "gt_quality_version": GT_QUALITY_VERSION,
                "ground_handling": (
                    "all geometry in the configured height band except "
                    "near-horizontal support-floor triangles"
                ),
                "robot_origin_pixel_row_column": [
                    args.bev_size // 2,
                    args.bev_size // 2,
                ],
                "robot_origin_convention": (
                    "one shared integer pixel centre for current, visibility, "
                    "and merged BEVs"
                ),
                "path_footprint_radius_m": DEFAULT_AGENT_RADIUS_M,
                "voxel_size_m": args.voxel_size,
                "masked_values": {"occupied": 0, "unknown": 112, "free": 255},
                "visibility_algorithm": VISIBILITY_ALGORITHM,
                "void_check": {
                    "version": VOID_CHECK_VERSION,
                    "modality": "single/current Complete GT geometry-valid mask",
                    "coverage_algorithm": VOID_COVERAGE_ALGORITHM,
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
                    "values come from the same accumulated static collision truth; "
                    "never-observed cells remain unknown"
                ),
                "merged_fusion_version": 2,
                "merged_orientation": (
                    "ego-centric in every output; latest robot centered and forward up"
                ),
                "merged_normalized_extents_m": list(args.merged_extents),
                "merged_normalized_size": [args.bev_size, args.bev_size],
            },
            "voxel_statistics": voxel_statistics,
            "estimated_voxel_grid_cells": estimated_voxel_cells,
            "voxel_grid_cell_limit": (
                args.max_voxel_grid_cells
                if args.max_voxel_grid_cells > 0
                else None
            ),
            "path": {
                "start": plan.start.tolist(),
                "goal": plan.goal.tolist(),
                "reachable": True,
                "geodesic_distance_m": plan.geodesic_distance_m,
                "polyline_distance_m": plan.polyline_distance_m,
                "waypoints": plan.waypoints.tolist(),
                "duration_s": samples[-1][0],
                "collision_truth_validation": path_truth_validation,
            },
            "frame_count": len(samples),
            "vggt_estimates": {
                "status": "pending" if not args.skip_vggt else "skipped"
            },
        }
        write_json(partial_dir / "metadata.json", metadata)

        trajectory_rows: List[Dict[str, Any]] = []
        jsonl_path = partial_dir / "ground_truth_trajectory.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            for frame_id, (sim_time, path_distance, position, forward) in enumerate(samples):
                set_agent_pose(simulator, position, forward)
                observations = simulator.get_sensor_observations()
                camera_frame = np.asarray(observations["camera_sensor"])[..., :3]
                depth_frame = np.asarray(
                    observations["depth_sensor"],
                    dtype=np.float32,
                )
                extrinsic = capture_frame_extrinsic(simulator)
                frame_name = f"frame_{frame_id:06d}.png"
                depth_name = f"frame_{frame_id:06d}.npz"
                save_rgb(partial_dir / "camera" / frame_name, camera_frame)
                save_depth(partial_dir / "depth" / depth_name, depth_frame)
                bev_record: Dict[str, Any] = {}
                for key, extent in zip(extent_keys, args.bev_extents):
                    complete = render_ego_obstacle_map(
                        simulator,
                        full_scene_map=full_truth,
                        lower_bound=lower_bound,
                        source_meters_per_pixel=args.voxel_size,
                        size=args.bev_size,
                        extent=extent,
                    )
                    masked = render_visibility_masked_map(
                        complete,
                        horizontal_fov_degrees=parameters.horizontal_fov_degrees,
                    )
                    validate_bev_frame_quality(
                        complete,
                        masked,
                        extent_m=extent,
                    )
                    single_valid_coverage = render_ego_obstacle_map(
                        simulator,
                        full_scene_map=full_valid_coverage.astype(
                            np.uint8, copy=False
                        ),
                        lower_bound=lower_bound,
                        source_meters_per_pixel=args.voxel_size,
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
                    accumulator = accumulators[key]
                    accumulator.update(complete, masked, extrinsic)
                    save_grayscale(partial_dir / key / "masked" / frame_name, masked)
                    save_grayscale(partial_dir / key / "complete" / frame_name, complete)
                    merged_record: Dict[str, Any] = {}
                    for merged_extent in args.merged_extents:
                        extent_record: Dict[str, Any] = {}
                        merged_images: Dict[str, np.ndarray] = {}
                        for kind in ("masked", "complete"):
                            merged_image, merged_meta = accumulator.render_ego(
                                kind, extrinsic, merged_extent
                            )
                            merged_images[kind] = merged_image
                            modality = merged_modality(kind, merged_extent)
                            save_grayscale(
                                partial_dir / key / modality / frame_name,
                                merged_image,
                            )
                            extent_record[kind] = merged_meta
                        merged_known = merged_images["masked"] != UNKNOWN_VALUE
                        if not np.array_equal(
                            merged_images["masked"][merged_known],
                            merged_images["complete"][merged_known],
                        ):
                            raise RuntimeError(
                                f"{key} merged {merged_extent:g} m masked values "
                                "disagree with static complete truth"
                            )
                        validate_bev_frame_quality(
                            merged_images["complete"],
                            merged_images["masked"],
                            extent_m=merged_extent,
                            complete_may_contain_unknown=True,
                        )
                        merged_record[f"{merged_extent:g}m"] = extent_record
                    bev_record[key] = {
                        "current_shape": [args.bev_size, args.bev_size],
                        "current_extent_m": extent,
                        "current_meters_per_pixel": extent / args.bev_size,
                        "single_complete_gt_void_ratio": void_ratio,
                        "single_complete_gt_valid_ratio": 1.0 - void_ratio,
                        "void_coverage_algorithm": VOID_COVERAGE_ALGORITHM,
                        "merged": merged_record,
                    }
                row = {
                    "frame_id": frame_id,
                    "sim_time_s": sim_time,
                    "path_distance_m": path_distance,
                    "path_progress": path_distance / plan.polyline_distance_m,
                    "agent_position_world_m": extrinsic["agent_position_world_m"],
                    "bev_forward_xz": extrinsic["bev_forward_xz"],
                    "extrinsic": extrinsic,
                    "camera_file": f"camera/{frame_name}",
                    "depth_file": f"depth/{depth_name}",
                    "bev": bev_record,
                }
                trajectory_rows.append(row)
                jsonl.write(json.dumps(row, default=json_ready) + "\n")
                print(
                    f"  frame {frame_id + 1:03d}/{len(samples):03d} "
                    f"t={sim_time:5.2f}s d={path_distance:5.2f}m",
                    flush=True,
                )

        with (partial_dir / "camera_extrinsics.jsonl").open(
            "w",
            encoding="utf-8",
        ) as stream:
            for row in trajectory_rows:
                stream.write(
                    json.dumps(
                        {
                            "frame_id": row["frame_id"],
                            "sim_time_s": row["sim_time_s"],
                            "camera_file": row["camera_file"],
                            "depth_file": row["depth_file"],
                            "extrinsic": row["extrinsic"],
                        },
                        default=json_ready,
                    )
                    + "\n"
                )

        with (partial_dir / "ground_truth_trajectory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "frame_id",
                    "sim_time_s",
                    "path_distance_m",
                    "path_progress",
                    "agent_x",
                    "agent_y",
                    "agent_z",
                    "forward_x",
                    "forward_z",
                ),
            )
            writer.writeheader()
            for row in trajectory_rows:
                position = row["agent_position_world_m"]
                heading = row["bev_forward_xz"]
                writer.writerow(
                    {
                        "frame_id": row["frame_id"],
                        "sim_time_s": row["sim_time_s"],
                        "path_distance_m": row["path_distance_m"],
                        "path_progress": row["path_progress"],
                        "agent_x": position[0],
                        "agent_y": position[1],
                        "agent_z": position[2],
                        "forward_x": heading[0],
                        "forward_z": heading[1],
                    }
                )

        render_trajectory_image(
            simulator,
            plan,
            [row["agent_position_world_m"] for row in trajectory_rows],
            partial_dir / "ground_truth_trajectory.png",
        )
        final_position = np.asarray(trajectory_rows[-1]["agent_position_world_m"])
        final_error = float(np.linalg.norm(final_position - plan.goal))
        validation = validate_session(
            partial_dir,
            len(samples),
            extent_keys,
            args.bev_extents,
            args.merged_extents,
            (args.camera_width, args.camera_height),
            args.bev_size,
            args.max_single_bev_void_ratio,
        )
        validation["final_goal_error_m"] = final_error
        if final_error > 1e-4:
            raise RuntimeError(f"final pose misses goal by {final_error:.6f} m")
        measured_speeds = [
            (current["path_distance_m"] - previous["path_distance_m"])
            / (current["sim_time_s"] - previous["sim_time_s"])
            for previous, current in zip(trajectory_rows, trajectory_rows[1:])
        ]
        speed_error = max(
            (
                abs(value - parameters.robot_speed_mps)
                for value in measured_speeds
            ),
            default=0.0,
        )
        camera_height_error = max(
            abs(
                row["extrinsic"]["camera_position_world_m"][1]
                - row["extrinsic"]["agent_position_world_m"][1]
                - parameters.camera_height_m
            )
            for row in trajectory_rows
        )
        validation.update(
            {
                "maximum_speed_error_mps": speed_error,
                "maximum_camera_height_error_m": camera_height_error,
                "gt_quality_version": GT_QUALITY_VERSION,
                "all_frames_collision_truth_validated": True,
                "path_collision_truth_validation": path_truth_validation,
            }
        )
        if speed_error > 1e-6 or camera_height_error > 1e-5:
            raise RuntimeError(
                "randomized speed or camera height failed trajectory validation"
            )
        metadata.update(
            {
                "status": "complete",
                "completed_at_unix_s": time.time(),
                "wall_time_s": time.time() - started_at,
                "validation": validation,
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
            "coverage_algorithm": VOID_COVERAGE_ALGORITHM,
            "filter_contract": VOID_FILTER_CONTRACT,
            "passed": True,
        }
        write_json(partial_dir / "metadata.json", metadata)
        make_preview(
            partial_dir,
            extent_keys,
            args.merged_extents,
            f"frame_{len(samples) - 1:06d}.png",
            parameters,
        )
        (partial_dir / "COMPLETE").write_text("validated\n", encoding="utf-8")

    partial_dir.rename(final_dir)
    metadata["output_directory"] = str(final_dir)
    return metadata


def collect_session_group(
    *,
    asset: SceneAsset,
    first_session_index: int,
    scene_sampling_cycle: int,
    target_sessions: int,
    initialization_parameters: SessionParameters,
    rng: random.Random,
    output_root: Path,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect several independent paths from one loaded and voxelized scene."""

    group_started_at = time.time()
    sessions: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    with make_simulator(
        asset,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        horizontal_fov_degrees=(
            initialization_parameters.horizontal_fov_degrees
        ),
        sensor_height_m=initialization_parameters.camera_height_m,
        gpu_device_id=args.gpu_device_id,
    ) as simulator:
        navmesh, navmesh_source = validate_loaded_asset(
            simulator,
            asset,
            args.navmesh_cache,
            args.camera_width,
            args.camera_height,
        )
        first_plan = random_reachable_path(
            simulator,
            seed=initialization_parameters.seed,
            min_length=args.min_path_length,
            max_length=args.max_path_length,
            max_floor_variation=args.max_floor_variation,
            attempts=args.path_attempts,
        )
        first_samples = path_samples(
            first_plan,
            initialization_parameters.robot_speed_mps,
            args.capture_hz,
        )
        set_agent_pose(simulator, first_samples[0][2], first_samples[0][3])
        ready_image = np.asarray(
            simulator.get_sensor_observations()["camera_sensor"]
        )
        if ready_image.shape != (args.camera_height, args.camera_width, 4):
            raise AssetRejected("camera failed after verified path placement")

        lower_bound, navigable_map, estimated_voxel_cells = estimate_voxel_grid(
            simulator,
            first_plan.floor_height_m,
            args.voxel_size,
            args.obstacle_min_height,
            args.obstacle_max_height,
        )
        if (
            args.max_voxel_grid_cells > 0
            and estimated_voxel_cells > args.max_voxel_grid_cells
        ):
            raise AssetRejected(
                f"estimated voxel grid {estimated_voxel_cells:,} exceeds "
                f"limit {args.max_voxel_grid_cells:,}"
            )
        print(
            f"Initializing {asset.dataset}/{asset.scene_id} once for "
            f"{target_sessions} path sessions; voxelizing "
            f"{estimated_voxel_cells:,} cells",
            flush=True,
        )
        full_truth, full_valid_coverage, voxel_statistics = render_full_obstacle_map(
            simulator,
            floor_height=first_plan.floor_height_m,
            meters_per_pixel=args.voxel_size,
            rows=navigable_map.shape[0],
            columns=navigable_map.shape[1],
            lower_bound=lower_bound,
            obstacle_min_height=args.obstacle_min_height,
            obstacle_max_height=args.obstacle_max_height,
            navigable_map=navigable_map,
        )
        initialized_scene = InitializedScene(
            navmesh=navmesh,
            navmesh_source=navmesh_source,
            lower_bound=lower_bound,
            full_truth=full_truth,
            full_valid_coverage=full_valid_coverage,
            voxel_statistics=voxel_statistics,
            estimated_voxel_cells=estimated_voxel_cells,
            floor_height_m=first_plan.floor_height_m,
        )

        pass_attempt = 0
        # A successfully initialized scene is one collection run.  Rejected
        # path candidates are resampled without reloading it.  This finite
        # per-initialization budget counts every attempted session, successful
        # or failed.  Valid sessions remain committed and the caller samples a
        # new scene to fill the unchanged global quota after exhaustion.
        while len(sessions) < target_sessions:
            if pass_attempt >= args.max_initialization_attempts:
                failures.append(
                    {
                        "dataset": asset.dataset,
                        "scene": str(asset.scene_file),
                        "scene_id": asset.scene_id,
                        "scene_sampling_cycle": scene_sampling_cycle,
                        "initialization_attempt_guard_reached": True,
                        "initialization_attempts": pass_attempt,
                        "max_initialization_attempts": (
                            args.max_initialization_attempts
                        ),
                        "completed_sessions": len(sessions),
                        "target_sessions": target_sessions,
                        "reason": (
                            "scene initialization exhausted its total session "
                            "attempt budget"
                        ),
                    }
                )
                print(
                    "  initialization attempt guard reached "
                    f"{pass_attempt}/{args.max_initialization_attempts}; "
                    f"keeping {len(sessions)} valid sessions and yielding to "
                    "the next scene",
                    flush=True,
                )
                break
            pass_started_at = (
                group_started_at if pass_attempt == 0 else time.time()
            )
            if pass_attempt == 0:
                parameters = initialization_parameters
                plan = first_plan
            else:
                parameters = SessionParameters(
                    seed=rng.randrange(0, 2**31 - 1),
                    horizontal_fov_degrees=(
                        initialization_parameters.horizontal_fov_degrees
                    ),
                    camera_height_m=initialization_parameters.camera_height_m,
                    robot_speed_mps=rng.uniform(*args.robot_speed_range),
                )
                plan = None
            session_index = first_session_index + len(sessions)
            try:
                if plan is None:
                    plan = random_reachable_path(
                        simulator,
                        seed=parameters.seed,
                        min_length=args.min_path_length,
                        max_length=args.max_path_length,
                        max_floor_variation=args.max_floor_variation,
                        attempts=args.path_attempts,
                        target_floor_height=initialized_scene.floor_height_m,
                    )
                session = collect_session(
                    asset=asset,
                    session_index=session_index,
                    scene_sampling_cycle=scene_sampling_cycle,
                    parameters=parameters,
                    output_root=output_root,
                    args=args,
                    simulator=simulator,
                    initialized_scene=initialized_scene,
                    plan=plan,
                    started_at=pass_started_at,
                )
            except (
                AssetRejected,
                PointNavigationError,
                RuntimeError,
                ValueError,
            ) as error:
                partial_dir = output_root / (
                    f".session_{session_index:06d}_{asset.dataset}_"
                    f"{sanitize_identifier(asset.scene_id)}.partial"
                )
                if partial_dir.exists():
                    shutil.rmtree(partial_dir)
                failure = {
                        "dataset": asset.dataset,
                        "scene": str(asset.scene_file),
                        "scene_id": asset.scene_id,
                        "scene_sampling_cycle": scene_sampling_cycle,
                        "pass_attempt": pass_attempt + 1,
                        "initialization_attempt": pass_attempt + 1,
                        "max_initialization_attempts": (
                            args.max_initialization_attempts
                        ),
                        "reason": str(error),
                    }
                if isinstance(error, SingleBEVVoidRatioExceeded):
                    failure.update(error.as_dict())
                    failure["partial_session_cache_deleted"] = True
                failures.append(failure)
                print(
                    f"  pass attempt {pass_attempt + 1} "
                    f"({pass_attempt + 1}/"
                    f"{args.max_initialization_attempts}) rejected: {error}",
                    flush=True,
                )
            else:
                sessions.append(session)
                print(
                    f"  completed {session['session_id']} with "
                    f"{session['frame_count']} synchronized frames "
                    f"({len(sessions)}/{target_sessions} from this "
                    "initialization)",
                    flush=True,
                )
            pass_attempt += 1
    return sessions, failures


def write_inspection_html(output_root: Path, sessions: Sequence[Dict[str, Any]]) -> None:
    cards = []
    for session in sessions:
        directory = Path(session["output_directory"]).name
        parameters = session["random_parameters"]
        vggt = session.get("vggt_estimates", {})
        vggt_link = (
            f' · <a href="{html.escape(directory)}/vggt/metadata.json">'
            "VGGT estimates</a>"
            if vggt.get("status") == "complete"
            else f" · VGGT {html.escape(str(vggt.get('status', 'unknown')))}"
        )
        cards.append(
            f"""
            <article>
              <h2>{html.escape(session['session_id'])}</h2>
              <p>{html.escape(session.get('dataset', 'unknown'))}/{html.escape(session['scene_id'])} · FOV {parameters['horizontal_fov_degrees']:.1f}° · camera {parameters['camera_height_m']:.2f} m · speed {parameters['robot_speed_mps']:.2f} m/s · {session['frame_count']} frames</p>
              <a href="{html.escape(directory)}/metadata.json">metadata.json</a>{vggt_link}
              <img src="{html.escape(directory)}/preview.png" alt="{html.escape(session['session_id'])} preview">
            </article>
            """
        )
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>VGGNAV collection inspection</title>
<style>
body {{ margin: 0; padding: 24px; color: #eef3f8; background: #090c10; font: 14px system-ui; }}
article {{ max-width: 1100px; margin: 0 auto 28px; padding: 16px; background: #151b23; border: 1px solid #2b3542; border-radius: 12px; }}
h1, h2 {{ margin-top: 0; }} a {{ color: #40c9ff; }} img {{ display: block; width: 100%; margin-top: 12px; border-radius: 8px; }}
</style>
<h1>VGGNAV randomized collection</h1>
{''.join(cards)}
"""
    (output_root / "inspection.html").write_text(document, encoding="utf-8")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"output is not empty: {path}; pass --overwrite to replace it"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_vggt_root(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        candidates = [explicit.expanduser()]
    else:
        script_parent = Path(__file__).resolve().parent
        candidates = [
            script_parent.parent / "vggt",
            script_parent.parent,
            *DEFAULT_VGGT_ROOTS,
        ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "vggt_omega").is_dir():
            return resolved
    raise RuntimeError(
        "VGGT source tree was not found; pass --vggt-root or --skip-vggt"
    )


def resolve_vggt_python(explicit: Optional[Path], vggt_root: Path) -> Path:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    configured = os.environ.get("VGGNAV_VGGT_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            vggt_root / ".venv" / "bin" / "python",
            Path.home() / ".local/share/micromamba/envs/vggt/bin/python",
            Path(sys.executable),
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise RuntimeError(
        "VGGT Python was not found; pass --vggt-python or set "
        "VGGNAV_VGGT_PYTHON"
    )


def run_vggt_estimator(output_root: Path, args: argparse.Namespace) -> None:
    vggt_root = resolve_vggt_root(args.vggt_root)
    vggt_python = resolve_vggt_python(args.vggt_python, vggt_root)
    checkpoint = (
        args.vggt_checkpoint.expanduser().resolve()
        if args.vggt_checkpoint is not None
        else vggt_root / "checkpoints/VGGT-Omega-1B-512/model.pt"
    )
    if not checkpoint.is_file():
        raise RuntimeError(f"VGGT checkpoint not found: {checkpoint}")
    estimator = Path(__file__).resolve().with_name("run_vggt_estimation.py")
    if not estimator.is_file():
        raise RuntimeError(f"VGGT estimator script not found: {estimator}")
    command = [
        str(vggt_python),
        str(estimator),
        "--collection-root",
        str(output_root),
        "--vggt-root",
        str(vggt_root),
        "--checkpoint",
        str(checkpoint),
        "--image-resolution",
        str(args.vggt_image_resolution),
        "--device",
        args.vggt_device,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(vggt_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    print(
        f"Running VGGT once over {len(list(output_root.glob('session_*')))} "
        f"completed sessions with {vggt_python}",
        flush=True,
    )
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    args = parse_args()
    validate_args(args)
    raw_dataset_roots = (
        args.legacy_dataset_roots
        if args.legacy_dataset_roots
        else args.dataset_roots
    )
    dataset_roots = [path.expanduser().resolve() for path in raw_dataset_roots]
    output_root = args.output.expanduser().resolve()
    all_discovered_scenes = [
        scene
        for dataset_root in dataset_roots
        for scene in discover_scenes(dataset_root)
    ]
    enabled_datasets = tuple(args.enabled_datasets)
    enabled_scenes = [
        scene
        for scene in all_discovered_scenes
        if scene.dataset in enabled_datasets
    ]
    if not enabled_scenes:
        raise SystemExit(
            "No enabled HM3D, Replica, HSSD, or MP3D scenes found under: "
            + ", ".join(str(path) for path in dataset_roots)
        )
    partitions = partition_scene_assets(
        enabled_scenes,
        enabled_datasets=enabled_datasets,
        train_fraction=args.scene_train_fraction,
        split_seed=args.scene_split_seed,
    )
    split_payload = scene_split_manifest(
        partitions,
        train_fraction=args.scene_train_fraction,
        split_seed=args.scene_split_seed,
        selected_role=args.scene_split_role,
    )
    scenes = [
        scene
        for dataset in enabled_datasets
        for scene in partitions[dataset][args.scene_split_role]
    ]
    if args.scene_ids:
        requested_scene_ids = set(args.scene_ids)
        discovered_scene_ids = {scene.scene_id for scene in enabled_scenes}
        missing_scene_ids = sorted(requested_scene_ids - discovered_scene_ids)
        if missing_scene_ids:
            raise SystemExit(
                "Requested scene IDs were not discovered: "
                + ", ".join(missing_scene_ids)
            )
        selected_scene_ids = {scene.scene_id for scene in scenes}
        held_out_scene_ids = sorted(requested_scene_ids - selected_scene_ids)
        if held_out_scene_ids:
            opposite_role = (
                "validation"
                if args.scene_split_role == "train"
                else "train"
            )
            raise SystemExit(
                "Requested scene IDs belong to the "
                f"{opposite_role} scene split, not "
                f"{args.scene_split_role}: "
                + ", ".join(held_out_scene_ids)
            )
        scenes = [
            scene for scene in scenes if scene.scene_id in requested_scene_ids
        ]
    if not scenes:
        raise SystemExit(
            f"No scenes are available in the {args.scene_split_role} split"
        )
    requested_sessions_by_dataset: Optional[Dict[str, int]] = None
    quota_arguments = {
        "hm3d": args.hm3d_sessions,
        "replica": args.replica_sessions,
        "hssd": args.hssd_sessions,
        "mp3d": args.mp3d_sessions,
    }
    if any(value is not None for value in quota_arguments.values()):
        requested_sessions_by_dataset = {
            dataset: int(quota_arguments[dataset])
            for dataset in enabled_datasets
        }
        args.sessions = sum(requested_sessions_by_dataset.values())
    collect_all_scenes_once = (
        args.sessions is None and requested_sessions_by_dataset is None
    )
    if collect_all_scenes_once:
        args.sessions = len(scenes) * args.sessions_per_initialization
    scenes_by_dataset = {
        dataset: [scene for scene in scenes if scene.dataset == dataset]
        for dataset in enabled_datasets
    }
    if requested_sessions_by_dataset is not None:
        unavailable = [
            dataset
            for dataset, quota in requested_sessions_by_dataset.items()
            if quota > 0 and not scenes_by_dataset[dataset]
        ]
        if unavailable:
            raise SystemExit(
                "No scenes discovered for requested dataset quotas: "
                + ", ".join(unavailable)
            )
    prepare_output(output_root, args.overwrite)
    write_json(output_root / "scene_split_manifest.json", split_payload)
    args.navmesh_cache = (
        args.navmesh_cache.expanduser().resolve()
        if args.navmesh_cache is not None
        else output_root / "_navmesh_cache"
    )
    args.navmesh_cache.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    failures: List[Dict[str, Any]] = []
    completed: List[Dict[str, Any]] = []
    scene_queue: List[SceneAsset] = []
    scene_sampling_cycle = -1
    quota_scene_queues: Dict[str, List[SceneAsset]] = {
        dataset: [] for dataset in scenes_by_dataset
    }
    quota_sampling_cycles = {dataset: -1 for dataset in scenes_by_dataset}
    completed_sessions_by_dataset = {
        dataset: 0 for dataset in scenes_by_dataset
    }
    scene_attempts_by_dataset = {dataset: 0 for dataset in scenes_by_dataset}
    scene_attempts = 0
    requested_initializations = math.ceil(
        args.sessions / args.sessions_per_initialization
    )
    max_scene_attempts = (
        len(scenes)
        if collect_all_scenes_once
        else max(
            requested_initializations * 5,
            requested_initializations + len(scenes) * 4,
        )
    )
    while len(completed) < args.sessions and scene_attempts < max_scene_attempts:
        if requested_sessions_by_dataset is not None:
            remaining = {
                dataset: quota - completed_sessions_by_dataset[dataset]
                for dataset, quota in requested_sessions_by_dataset.items()
                if quota > completed_sessions_by_dataset[dataset]
            }
            if not remaining:
                break
            datasets = sorted(remaining)
            selected_dataset = rng.choices(
                datasets,
                weights=[remaining[dataset] for dataset in datasets],
                k=1,
            )[0]
            selected_queue = quota_scene_queues[selected_dataset]
            if not selected_queue:
                selected_queue.extend(scenes_by_dataset[selected_dataset])
                rng.shuffle(selected_queue)
                quota_sampling_cycles[selected_dataset] += 1
            asset = selected_queue.pop()
            scene_sampling_cycle = quota_sampling_cycles[selected_dataset]
            target_sessions = min(
                args.sessions_per_initialization,
                remaining[selected_dataset],
            )
        else:
            if not scene_queue:
                if collect_all_scenes_once and scene_sampling_cycle >= 0:
                    break
                scene_queue = list(scenes)
                rng.shuffle(scene_queue)
                scene_sampling_cycle += 1
            asset = scene_queue.pop()
            target_sessions = min(
                args.sessions_per_initialization,
                args.sessions - len(completed),
            )
        scene_attempts += 1
        scene_attempts_by_dataset[asset.dataset] += 1
        initialization_parameters = SessionParameters(
            seed=rng.randrange(0, 2**31 - 1),
            horizontal_fov_degrees=rng.uniform(*args.fov_range),
            camera_height_m=rng.uniform(*args.camera_height_range),
            robot_speed_mps=rng.uniform(*args.robot_speed_range),
        )
        print(
            f"Trying asset {asset.dataset}/{asset.scene_id} "
            f"(scene cycle {scene_sampling_cycle + 1}, "
            f"target passes={target_sessions}, "
            f"FOV={initialization_parameters.horizontal_fov_degrees:.1f}, "
            f"height={initialization_parameters.camera_height_m:.2f}m)",
            flush=True,
        )
        try:
            group_sessions, pass_failures = collect_session_group(
                asset=asset,
                first_session_index=len(completed),
                scene_sampling_cycle=scene_sampling_cycle,
                target_sessions=target_sessions,
                initialization_parameters=initialization_parameters,
                rng=rng,
                output_root=output_root,
                args=args,
            )
        except (AssetRejected, PointNavigationError, RuntimeError, ValueError) as error:
            partial_dir = output_root / (
                f".session_{len(completed):06d}_{asset.dataset}_"
                f"{sanitize_identifier(asset.scene_id)}.partial"
            )
            if partial_dir.exists():
                shutil.rmtree(partial_dir)
            failure = {
                "dataset": asset.dataset,
                "scene": str(asset.scene_file),
                "scene_id": asset.scene_id,
                "scene_sampling_cycle": scene_sampling_cycle,
                "reason": str(error),
            }
            failures.append(failure)
            print(f"  rejected: {error}", flush=True)
            continue
        failures.extend(pass_failures)
        completed.extend(group_sessions)
        completed_sessions_by_dataset[asset.dataset] += len(group_sessions)

    if len(completed) == args.sessions and not args.skip_vggt:
        run_vggt_estimator(output_root, args)
        completed = [
            json.loads(
                (Path(item["output_directory"]) / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            | {"output_directory": item["output_directory"]}
            for item in completed
        ]

    manifest = {
        "schema_version": GT_QUALITY_VERSION,
        "collection_seed": args.seed,
        "dataset_roots": [str(path) for path in dataset_roots],
        "enabled_datasets": list(enabled_datasets),
        "replica_enabled": "replica" in enabled_datasets,
        "mp3d_enabled": "mp3d" in enabled_datasets,
        "scene_split_manifest": "scene_split_manifest.json",
        "scene_split_role": args.scene_split_role,
        "scene_train_fraction": args.scene_train_fraction,
        "scene_split_seed": args.scene_split_seed,
        "source_scenes_by_dataset": {
            dataset: len(partitions[dataset]["train"])
            + len(partitions[dataset]["validation"])
            for dataset in enabled_datasets
        },
        "split_scenes_by_dataset": {
            dataset: {
                "train": len(partitions[dataset]["train"]),
                "validation": len(partitions[dataset]["validation"]),
            }
            for dataset in enabled_datasets
        },
        "discovered_scenes_by_dataset": {
            dataset: sum(scene.dataset == dataset for scene in scenes)
            for dataset in enabled_datasets
        },
        "requested_sessions": args.sessions,
        "completed_sessions": len(completed),
        "requested_sessions_by_dataset": requested_sessions_by_dataset,
        "completed_sessions_by_dataset": completed_sessions_by_dataset,
        "scene_sampling_strategy": (
            "exact per-dataset quotas with repeated independently shuffled "
            "scene pools; one scene initialization/voxelization produces up "
            "to the requested batch size; an initialization that exhausts its "
            "total attempt guard yields to a new scene, while the global quota "
            "remains unchanged"
            if requested_sessions_by_dataset is not None
            else (
                "single shuffled initialization pass over all scenes"
                if collect_all_scenes_once
                else "repeated shuffled scene initializations; each produces "
                "multiple independent same-floor paths"
            )
        ),
        "sessions_per_scene_initialization": args.sessions_per_initialization,
        "scene_attempts": scene_attempts,
        "scene_attempts_by_dataset": scene_attempts_by_dataset,
        "bev_extent_classes_m": list(args.bev_extents),
        "capture_hz": args.capture_hz,
        "collector_settings": {
            "enabled_datasets": list(enabled_datasets),
            "replica_enabled": "replica" in enabled_datasets,
            "mp3d_enabled": "mp3d" in enabled_datasets,
            "scene_split_role": args.scene_split_role,
            "scene_train_fraction": args.scene_train_fraction,
            "scene_split_seed": args.scene_split_seed,
            "camera_resolution": [args.camera_width, args.camera_height],
            "gpu_device_id": args.gpu_device_id,
            "camera_height_range_m": list(args.camera_height_range),
            "horizontal_fov_range_degrees": list(args.fov_range),
            "robot_speed_range_mps": list(args.robot_speed_range),
            "sessions_per_scene_initialization": (
                args.sessions_per_initialization
            ),
            "camera_parameters_shared_within_initialization": True,
            "path_and_speed_randomized_per_session": True,
            "max_initialization_attempts": args.max_initialization_attempts,
            "initialization_attempt_counting": (
                "every attempted session counts, successful or failed"
            ),
            "max_single_bev_void_ratio": args.max_single_bev_void_ratio,
            "void_check_version": VOID_CHECK_VERSION,
            "void_check_modality": (
                "single/current Complete GT geometry-valid coverage only"
            ),
            "void_coverage_algorithm": VOID_COVERAGE_ALGORITHM,
            "void_filter_contract": VOID_FILTER_CONTRACT,
            "bev_size": args.bev_size,
            "voxel_size_m": args.voxel_size,
            "voxel_grid_cell_limit": (
                args.max_voxel_grid_cells
                if args.max_voxel_grid_cells > 0
                else None
            ),
            "voxel_grid_limit_enabled": args.max_voxel_grid_cells > 0,
            "obstacle_height_band_m": [
                args.obstacle_min_height,
                args.obstacle_max_height,
            ],
            "path_length_range_m": [
                args.min_path_length,
                args.max_path_length,
            ],
            "merged_normalized_extents_m": list(args.merged_extents),
            "merged_normalized_size": [args.bev_size, args.bev_size],
            "navmesh_cache": str(args.navmesh_cache),
            "vggt_enabled": not args.skip_vggt,
            "vggt_image_resolution": args.vggt_image_resolution,
            "vggt_device": args.vggt_device,
        },
        "sessions": [
            {
                "session_id": item["session_id"],
                "dataset": item["dataset"],
                "scene_id": item["scene_id"],
                "scene_sampling_cycle": item["scene_sampling_cycle"],
                "output_directory": item["output_directory"],
                "random_parameters": item["random_parameters"],
                "frame_count": item["frame_count"],
                "validation": item["validation"],
                "vggt_estimates": item["vggt_estimates"],
            }
            for item in completed
        ],
        "rejected_assets": failures,
    }
    write_json(output_root / "collection_manifest.json", manifest)
    write_inspection_html(output_root, completed)
    if len(completed) != args.sessions:
        raise SystemExit(
            f"Only {len(completed)}/{args.sessions} sessions completed; "
            f"see {output_root / 'collection_manifest.json'}"
        )
    print(f"Collection complete: {output_root}")
    print(f"Inspection page: {output_root / 'inspection.html'}")


if __name__ == "__main__":
    main()
