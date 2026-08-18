#!/usr/bin/env python3
"""Interactive Habitat UI for the RGB-only P1B BEV head."""

import argparse
import base64
import io
import json
import math
import queue
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

import habitat_sim
import numpy as np
import quaternion
from habitat_sim.agent.controls.default_controls import (
    LookLeft,
    LookRight,
    MoveForward,
)
from habitat_sim.registry import registry
from habitat_sim.utils.common import quat_from_magnum, quat_to_magnum
from PIL import Image, ImageDraw

from bev_accumulator import BEVAccumulator
from collision_voxel import voxelize_stage_occupancy
from model_comparison import ComparisonFrame, ModelComparisonWorker
from point_navigation import PointNavigationError, PointNavigator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from vggt_bev_method1.interactive_verifier import (  # noqa: E402
    DEFAULT_INFLATION_RADIUS_M,
    VerifierPlanningError,
    compile_grid_path_to_open_loop_actions,
    plan_metric_target,
    relative_camera_motion_metric,
    transform_target_by_predicted_motion,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = Path(
    "/media/user/T9/scene_datasets/hm3d/val/"
    "00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
)
DEFAULT_START = (-1.7062, 0.1634, -2.3114)
HTML_FILE = ROOT / "web_ui/index.html"
MERGED_ROUTING_HTML_FILE = ROOT / "web_ui/merged_routing_geometry.html"
DEFAULT_CAMERA_SENSOR_HEIGHT_METERS = 0.35
DEFAULT_HORIZONTAL_FOV_DEGREES = 90.0
GLOBAL_MAP_METERS_PER_PIXEL = 0.05
DEFAULT_VOXEL_SIZE_METERS = 0.01
MANUAL_LINEAR_SPEED_METERS_PER_SECOND = 0.4
MANUAL_ANGULAR_SPEED_RADIANS_PER_SECOND = 0.3
MANUAL_MAX_INTEGRATION_STEP_SECONDS = 0.25
DEFAULT_MANUAL_NAVMESH_CLEARANCE_METERS = 0.15
MODEL_EXTENTS_METERS = {"p1b": 6.5}
MERGED_GT_EXTENT_METERS = 10.0
MERGED_GT_SIZE = 800
STRICT_OPEN_LOOP_ACTION_HZ = 30.0
CLOSED_LOOP_PREDICTED_SUCCESS_TOLERANCE_METERS = 0.05
MIN_INTERACTIVE_INFLATION_RADIUS_M = 0.0
MAX_INTERACTIVE_INFLATION_RADIUS_M = 0.50

# The strict open-loop executor needs different forward distances and fixed
# 45-degree turns.  Give those controls unique registry names: reusing the
# default ``move_forward``/``turn_*`` names makes GreedyGeodesicFollower find
# multiple ActionSpecs for one standard action and abort during construction.
registry.register_move_fn(
    MoveForward, name="strict_move_forward_cardinal", body_action=True
)
registry.register_move_fn(
    MoveForward, name="strict_move_forward_diagonal", body_action=True
)
registry.register_move_fn(
    LookLeft, name="strict_turn_left_45", body_action=True
)
registry.register_move_fn(
    LookRight, name="strict_turn_right_45", body_action=True
)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


@dataclass(frozen=True)
class SceneChoice:
    scene_id: str
    label: str
    scene: Path
    navmesh: Path


def load_scene_catalog(path: Path) -> Tuple[SceneChoice, ...]:
    catalog_path = path.expanduser().resolve()
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    records = payload.get("scenes")
    if not isinstance(records, list) or not records:
        raise ValueError("scene catalog must contain at least one scene")
    choices = []
    seen_ids = set()
    seen_scenes = set()
    for record in records:
        scene_id = str(record["id"])
        scene = Path(record["scene"]).expanduser()
        navmesh = Path(record["navmesh"]).expanduser()
        if not scene.is_absolute():
            scene = catalog_path.parent / scene
        if not navmesh.is_absolute():
            navmesh = catalog_path.parent / navmesh
        scene = scene.resolve()
        navmesh = navmesh.resolve()
        if scene_id in seen_ids or scene in seen_scenes:
            raise ValueError(f"duplicate scene catalog entry: {scene_id}")
        if not scene.is_file() or not navmesh.is_file():
            raise FileNotFoundError(
                f"local scene assets are incomplete for {scene_id}: "
                f"{scene}, {navmesh}"
            )
        seen_ids.add(scene_id)
        seen_scenes.add(scene)
        choices.append(
            SceneChoice(
                scene_id=scene_id,
                label=str(record.get("label", scene_id)),
                scene=scene,
                navmesh=navmesh,
            )
        )
    return tuple(choices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the P1B WASD comparison browser interface"
    )
    parser.add_argument("scene", type=Path, nargs="?", default=DEFAULT_SCENE)
    parser.add_argument("--navmesh", type=Path)
    parser.add_argument(
        "--scene-catalog",
        type=Path,
        help="local JSON catalog containing exactly ten selectable scenes",
    )
    parser.add_argument(
        "--scene-dataset-config",
        type=Path,
        help="Habitat scene-dataset config (required for furnished HSSD scenes)",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
        help="logical GPU index used by Habitat-Sim",
    )
    parser.add_argument(
        "--start",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_START,
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="ignore --start and sample a navigable start point",
    )
    parser.add_argument(
        "--start-yaw-degrees",
        type=float,
        default=0.0,
        help="initial agent yaw in degrees",
    )
    parser.add_argument(
        "--random-yaw",
        action="store_true",
        help="ignore --start-yaw-degrees and sample yaw uniformly",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed used for navigable start and yaw sampling",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--action-hz", type=float, default=4.0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--sensor-height-m",
        type=float,
        default=DEFAULT_CAMERA_SENSOR_HEIGHT_METERS,
        help="physical camera mounting height in metres",
    )
    parser.add_argument(
        "--horizontal-fov-degrees",
        type=float,
        default=DEFAULT_HORIZONTAL_FOV_DEGREES,
        help="horizontal pinhole camera field of view",
    )
    parser.add_argument("--bev-size", type=int, default=512)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_EXTENTS_METERS),
        default="p1b",
        help="trained single-frame extent to compare",
    )
    parser.add_argument(
        "--all-model-comparison",
        action="store_true",
        help="expect the shared runtime and display all three trained extents",
    )
    parser.add_argument(
        "--model-server-url",
        default="http://127.0.0.1:8765",
    )
    parser.add_argument("--model-hz", type=float, default=1.0)
    parser.add_argument("--model-max-history", type=int, default=34)
    parser.add_argument(
        "--bev-extent",
        type=float,
        help="deprecated; must equal the extent selected by --model",
    )
    parser.add_argument("--obstacle-min-height", type=float, default=0.2)
    parser.add_argument("--obstacle-max-height", type=float, default=1.4)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=DEFAULT_VOXEL_SIZE_METERS,
        help="collision occupancy voxel edge length in metres (default: 0.01)",
    )
    parser.add_argument(
        "--manual-clearance",
        type=float,
        default=DEFAULT_MANUAL_NAVMESH_CLEARANCE_METERS,
        help=(
            "extra navmesh-boundary clearance for WASD motion in metres "
            "(default: 0.15)"
        ),
    )
    parser.add_argument(
        "--occlusion-rays",
        type=int,
        default=720,
        help=(
            "deprecated compatibility option; exact grid shadowcasting "
            "does not use angular rays"
        ),
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--interactive-verifier",
        action="store_true",
        help=(
            "enable display-only GT click selection, ego-local target input, "
            "predicted-BEV A*, path overlays, and truth-free robot control"
        ),
    )
    parser.add_argument(
        "--merged-routing-visualizer",
        action="store_true",
        help="show only the trained 5090 Merged FOV Support and Observed Gate",
    )
    return parser.parse_args()


def make_web_simulator(
    scene: Path,
    scene_dataset_config: Optional[Path],
    gpu_device_id: int,
    camera_width: int,
    camera_height: int,
    sensor_height_m: float,
    horizontal_fov_degrees: float,
    strict_grid_cell_m: float,
) -> habitat_sim.Simulator:
    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(scene)
    if scene_dataset_config is not None:
        simulator_config.scene_dataset_config_file = str(scene_dataset_config)
    simulator_config.enable_physics = True
    simulator_config.gpu_device_id = gpu_device_id
    # A blocked action must stop, never slide along GT collision geometry.
    simulator_config.allow_sliding = False

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

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [camera]
    agent_config.action_space.update(
        {
            "strict_forward_cardinal": habitat_sim.agent.ActionSpec(
                "strict_move_forward_cardinal",
                habitat_sim.agent.ActuationSpec(amount=strict_grid_cell_m),
            ),
            "strict_forward_diagonal": habitat_sim.agent.ActionSpec(
                "strict_move_forward_diagonal",
                habitat_sim.agent.ActuationSpec(
                    amount=strict_grid_cell_m * math.sqrt(2.0)
                ),
            ),
            "strict_turn_left_45": habitat_sim.agent.ActionSpec(
                "strict_turn_left_45",
                habitat_sim.agent.ActuationSpec(amount=45.0),
            ),
            "strict_turn_right_45": habitat_sim.agent.ActionSpec(
                "strict_turn_right_45",
                habitat_sim.agent.ActuationSpec(amount=45.0),
            ),
        }
    )
    return habitat_sim.Simulator(
        habitat_sim.Configuration(simulator_config, [agent_config])
    )


def ensure_navmesh(simulator: habitat_sim.Simulator, navmesh: Path) -> None:
    if simulator.pathfinder.is_loaded:
        return
    if not navmesh.is_file():
        raise PointNavigationError(
            f"No navmesh was loaded and the file does not exist: {navmesh}"
        )
    if not simulator.pathfinder.load_nav_mesh(str(navmesh)):
        raise PointNavigationError(f"Could not load navmesh: {navmesh}")


def place_agent(
    simulator: habitat_sim.Simulator,
    requested_start: Optional[Tuple[float, float, float]],
    requested_yaw_degrees: Optional[float],
    seed: int,
    minimum_clearance_m: float = 0.0,
) -> Tuple[np.ndarray, float]:
    simulator.pathfinder.seed(seed)
    if requested_start is None:
        start = np.full(3, np.nan, dtype=np.float32)
        best_start = start
        best_clearance = -math.inf
        for _ in range(512):
            candidate = np.asarray(
                simulator.pathfinder.get_random_navigable_point(),
                dtype=np.float32,
            )
            if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
                continue
            clearance = float(
                simulator.pathfinder.distance_to_closest_obstacle(
                    candidate, 2.0
                )
            )
            if clearance > best_clearance:
                best_start = candidate
                best_clearance = clearance
            if clearance >= minimum_clearance_m:
                start = candidate
                break
        if not np.all(np.isfinite(start)):
            start = best_start
    else:
        requested = np.asarray(requested_start, dtype=np.float32)
        start = np.asarray(
            simulator.pathfinder.snap_point(requested), dtype=np.float32
        )
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise PointNavigationError("Could not place the agent on the navmesh")
    yaw_degrees = (
        float(np.random.default_rng(seed).uniform(-180.0, 180.0))
        if requested_yaw_degrees is None
        else float(requested_yaw_degrees)
    )
    agent = simulator.get_agent(0)
    state = agent.get_state()
    state.position = start
    state.rotation = quaternion.from_rotation_vector(
        np.asarray([0.0, math.radians(yaw_degrees), 0.0])
    )
    agent.set_state(state, reset_sensors=True)
    return start, yaw_degrees


def encode_jpeg(image: np.ndarray, quality: int = 88) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image)[..., :3]).save(
        buffer, format="JPEG", quality=quality, optimize=False
    )
    return buffer.getvalue()


def encode_png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def render_visibility_masked_map(
    ground_truth: np.ndarray,
    *,
    horizontal_fov_degrees: float,
) -> np.ndarray:
    """Mask an ego-centric ground-truth map with an ideal sensor view.

    Symmetric grid shadowcasting prevents angular-bin leaks through occupied
    cells. This function never creates occupancy from ray returns: it computes
    only visibility over the supplied simulator truth, then copies the exact
    truth values into visible cells. The first occupied cell is visible; cells
    behind it and outside the FOV are unknown.
    """

    if ground_truth.ndim != 2 or ground_truth.shape[0] != ground_truth.shape[1]:
        raise ValueError("ground_truth must be a square grayscale image")
    if not 0.0 < horizontal_fov_degrees <= 360.0:
        raise ValueError("horizontal_fov_degrees must be in (0, 360]")

    size = ground_truth.shape[0]
    obstacle = ground_truth == 0
    visible = np.zeros_like(obstacle, dtype=bool)
    origin_column = size // 2
    origin_row = size // 2
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

                in_bounds = (
                    0 <= column < size and 0 <= output_row < size
                )
                if in_bounds:
                    visible[output_row, column] = True
                cell_is_obstacle = (
                    not in_bounds or obstacle[output_row, column]
                )
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

    # The live camera uses a 90-degree forward FOV, so only the two octants
    # adjacent to image-up are needed. Wider compatibility FOVs cast all eight.
    if horizontal_fov_degrees <= 90.0 + 1e-12:
        octant_transforms = ((1, 0, 0, 1), (-1, 0, 0, 1))
    else:
        octant_transforms = (
            (1, 0, 0, 1),
            (0, 1, 1, 0),
            (0, -1, 1, 0),
            (-1, 0, 0, 1),
            (-1, 0, 0, -1),
            (0, -1, -1, 0),
            (0, 1, -1, 0),
            (1, 0, 0, -1),
        )
    for transform in octant_transforms:
        cast_octant(1, 1.0, 0.0, *transform)

    output_rows, output_columns = np.indices(
        ground_truth.shape, dtype=np.float64
    )
    local_right = output_columns - float(origin_column)
    local_forward = float(origin_row) - output_rows
    angle = np.arctan2(local_right, local_forward)
    half_fov = math.radians(horizontal_fov_degrees) / 2.0
    visible &= np.abs(angle) <= half_fov + 1e-12

    result = np.full_like(ground_truth, 112, dtype=np.uint8)
    result[visible] = ground_truth[visible]
    return result


def render_fov_complete_map(
    ground_truth: np.ndarray,
    *,
    horizontal_fov_degrees: float,
) -> np.ndarray:
    """Keep complete simulator occupancy inside the geometric camera FOV.

    Unlike an observed/visibility raster, this intentionally retains cells
    behind obstacles.  It matches P1B's FOV-complete supervision contract.
    """

    if ground_truth.ndim != 2 or ground_truth.shape[0] != ground_truth.shape[1]:
        raise ValueError("ground_truth must be a square grayscale image")
    size = ground_truth.shape[0]
    center = (size - 1) / 2.0
    half_grid = size / 2.0
    tangent = math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    if tangent <= 1.0:
        far_offset = half_grid * tangent
        vertices = (
            (center, center),
            (center - far_offset, -0.5),
            (center + far_offset, -0.5),
        )
    else:
        side_row = center - half_grid / tangent
        vertices = (
            (center, center),
            (-0.5, side_row),
            (-0.5, -0.5),
            (size - 0.5, -0.5),
            (size - 0.5, side_row),
        )
    # This is the same square-clipped polygon rasterization used by training's
    # local_fov_polygon + _metric_to_pixel + ImageDraw path.
    support_image = Image.new("L", (size, size), 0)
    ImageDraw.Draw(support_image).polygon(vertices, fill=1)
    inside = np.asarray(support_image, dtype=np.uint8).astype(bool)
    result = np.full_like(ground_truth, 112, dtype=np.uint8)
    result[inside] = ground_truth[inside]
    return result


def render_full_obstacle_map(
    simulator: habitat_sim.Simulator,
    *,
    floor_height: float,
    meters_per_pixel: float,
    rows: int,
    columns: int,
    lower_bound: np.ndarray,
    obstacle_min_height: float,
    obstacle_max_height: float,
    navigable_map: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return max-Y projection of the solid collision-geometry voxel truth."""

    return voxelize_stage_occupancy(
        simulator,
        floor_height=floor_height,
        lower_bound=lower_bound,
        rows=rows,
        columns=columns,
        voxel_size=meters_per_pixel,
        obstacle_min_height=obstacle_min_height,
        obstacle_max_height=obstacle_max_height,
        navigable_map=navigable_map,
    )


def render_ego_obstacle_map(
    simulator: habitat_sim.Simulator,
    *,
    full_scene_map: np.ndarray,
    lower_bound: np.ndarray,
    source_meters_per_pixel: float,
    size: int,
    extent: float,
) -> np.ndarray:
    """Crop and rotate the complete collision map around the robot pose."""

    agent_state = simulator.get_agent(0).get_state()
    position = np.asarray(agent_state.position, dtype=np.float64)
    forward_3d = habitat_sim.utils.common.quat_rotate_vector(
        agent_state.rotation, np.asarray([0.0, 0.0, -1.0])
    )
    forward = np.asarray([forward_3d[0], forward_3d[2]], dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    right = np.asarray([-forward[1], forward[0]], dtype=np.float64)

    center = (size - 1) / 2.0
    output_meters_per_pixel = extent / size
    output_rows, output_columns = np.indices((size, size), dtype=np.float64)
    local_right = (output_columns - center) * output_meters_per_pixel
    local_forward = (center - output_rows) * output_meters_per_pixel
    world_x = (
        position[0]
        + right[0] * local_right
        + forward[0] * local_forward
    )
    world_z = (
        position[2]
        + right[1] * local_right
        + forward[1] * local_forward
    )

    source_columns = np.floor(
        (world_x - float(lower_bound[0])) / source_meters_per_pixel
    ).astype(np.int32)
    unflipped_rows = np.floor(
        (world_z - float(lower_bound[2])) / source_meters_per_pixel
    ).astype(np.int32)
    source_rows = full_scene_map.shape[0] - 1 - unflipped_rows
    valid = (
        (source_columns >= 0)
        & (source_columns < full_scene_map.shape[1])
        & (source_rows >= 0)
        & (source_rows < full_scene_map.shape[0])
    )

    # Outside the scanned scene bounds is conservatively occupied.
    result = np.zeros((size, size), dtype=np.uint8)
    result[valid] = full_scene_map[source_rows[valid], source_columns[valid]]
    return result


def capture_frame_extrinsic(
    simulator: habitat_sim.Simulator,
) -> Dict[str, Any]:
    """Capture the exact agent, camera, and planar BEV pose for one frame."""

    agent_state = simulator.get_agent(0).get_state()
    agent_position = np.asarray(agent_state.position, dtype=np.float64)
    agent_rotation = agent_state.rotation
    forward_3d = habitat_sim.utils.common.quat_rotate_vector(
        agent_rotation, np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    )
    forward_xz = np.asarray(
        [forward_3d[0], forward_3d[2]], dtype=np.float64
    )
    forward_xz /= max(float(np.linalg.norm(forward_xz)), 1e-12)
    right_xz = np.asarray([-forward_xz[1], forward_xz[0]])

    sensor_state = agent_state.sensor_states["camera_sensor"]
    camera_position = np.asarray(sensor_state.position, dtype=np.float64)
    camera_rotation = sensor_state.rotation
    camera_rotation_matrix = quaternion.as_rotation_matrix(camera_rotation)
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = camera_rotation_matrix
    camera_to_world[:3, 3] = camera_position
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = camera_rotation_matrix.T
    world_to_camera[:3, 3] = -camera_rotation_matrix.T @ camera_position

    def xyzw(value: np.quaternion) -> list[float]:
        return [
            float(value.x),
            float(value.y),
            float(value.z),
            float(value.w),
        ]

    return {
        "agent_position_world_m": agent_position.tolist(),
        "agent_rotation_xyzw": xyzw(agent_rotation),
        "camera_position_world_m": camera_position.tolist(),
        "camera_rotation_xyzw": xyzw(camera_rotation),
        "camera_to_world_matrix": camera_to_world.tolist(),
        "world_to_camera_matrix": world_to_camera.tolist(),
        "bev_forward_xz": forward_xz.tolist(),
        "bev_right_xz": right_xz.tolist(),
        "world_from_bev_planar": [
            [float(right_xz[0]), float(forward_xz[0]), float(agent_position[0])],
            [float(right_xz[1]), float(forward_xz[1]), float(agent_position[2])],
            [0.0, 0.0, 1.0],
        ],
        "coordinate_convention": (
            "Habitat world x/y/z; BEV local axes are right/forward; "
            "camera matrices use the Habitat sensor local x/y/z axes"
        ),
    }


class LocalAStarActionFollower:
    """Replay a local A* action queue without simulator pose or navmesh input."""

    def __init__(self, actions: list[str]) -> None:
        self.actions = tuple(str(action) for action in actions)
        self.index = 0

    @property
    def executed_action_count(self) -> int:
        return self.index

    def next_action_along(self, _ignored_goal: np.ndarray) -> Optional[str]:
        if self.index >= len(self.actions):
            return None
        action = self.actions[self.index]
        self.index += 1
        return action


class NavigationEngine:
    """Own Habitat and its OpenGL context on one dedicated thread."""

    def __init__(
        self,
        *,
        scene: Path,
        navmesh: Path,
        scene_dataset_config: Optional[Path],
        gpu_device_id: int,
        model_keys: Tuple[str, ...],
        start: Optional[Tuple[float, float, float]],
        start_yaw_degrees: Optional[float],
        seed: int,
        fps: float,
        action_hz: float,
        camera_width: int,
        camera_height: int,
        sensor_height_m: float,
        horizontal_fov_degrees: float,
        bev_size: int,
        bev_extent: float,
        obstacle_min_height: float,
        obstacle_max_height: float,
        voxel_size: float,
        manual_clearance: float,
        occlusion_rays: int,
        model_server_url: str,
        model_hz: float,
        model_max_history: int,
        verifier_enabled: bool = False,
    ) -> None:
        if fps <= 0 or action_hz <= 0:
            raise ValueError("fps and action_hz must be positive")
        if camera_width <= 0 or camera_height <= 0 or bev_size <= 0:
            raise ValueError("camera and BEV dimensions must be positive")
        if not 0.05 <= sensor_height_m <= 3.0:
            raise ValueError("sensor-height-m must be between 0.05 and 3.0 metres")
        if not 1.0 <= horizontal_fov_degrees < 180.0:
            raise ValueError(
                "horizontal-fov-degrees must be between 1 and 180 degrees"
            )
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed must be between 0 and 4294967295")
        if bev_extent <= 0:
            raise ValueError("BEV extent must be positive")
        if obstacle_min_height < 0 or obstacle_max_height <= obstacle_min_height:
            raise ValueError(
                "obstacle heights must satisfy 0 <= minimum < maximum"
            )
        if not 0.001 <= voxel_size <= 0.2:
            raise ValueError("voxel-size must be between 0.001 and 0.2 metres")
        if not 0.0 <= manual_clearance <= 1.0:
            raise ValueError("manual-clearance must be between 0.0 and 1.0 metres")
        if occlusion_rays < 60:
            raise ValueError("occlusion-rays must be at least 60")
        self.scene = scene
        self.navmesh = navmesh
        self.scene_dataset_config = scene_dataset_config
        self.gpu_device_id = gpu_device_id
        self.model_keys = model_keys
        self.start = start
        self.start_yaw_degrees = start_yaw_degrees
        self.seed = seed
        self.fps = fps
        self.action_hz = action_hz
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.sensor_height_m = sensor_height_m
        self.horizontal_fov_degrees = horizontal_fov_degrees
        self.bev_size = bev_size
        self.bev_extent = bev_extent
        self.obstacle_min_height = obstacle_min_height
        self.obstacle_max_height = obstacle_max_height
        self.voxel_size = voxel_size
        self.manual_clearance = manual_clearance
        self.occlusion_rays = occlusion_rays
        self.comparison = ModelComparisonWorker(
            server_url=model_server_url,
            model_extents_m={
                key: MODEL_EXTENTS_METERS[key] for key in self.model_keys
            },
            bev_size=bev_size,
            sample_hz=model_hz,
            max_history=model_max_history,
        )
        self.verifier_enabled = bool(verifier_enabled)
        self.closed_loop_context: Optional[Dict[str, Any]] = None
        self.verifier_plan_revision = 0
        self.merged_gt_history: list[dict[str, Any]] = []
        self.manual_enabled = True
        self.manual_keys: Dict[str, bool] = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
        }

        self.commands: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.condition = threading.Condition()
        self.thread = threading.Thread(
            target=self._run, name="habitat-engine", daemon=True
        )
        self.state: Dict[str, Any] = {
            "ready": False,
            "status": "Starting Habitat…",
            "navigating": False,
            "paused": False,
            "step_count": 0,
            "frame_seq": 0,
            "manual_control": {
                "enabled": True,
                "keys": dict(self.manual_keys),
                "linear_speed_mps": MANUAL_LINEAR_SPEED_METERS_PER_SECOND,
                "angular_speed_radps": MANUAL_ANGULAR_SPEED_RADIANS_PER_SECOND,
                "extra_navmesh_clearance_m": self.manual_clearance,
                "navmesh_agent_radius_m": None,
                "effective_body_radius_m": None,
                "current_navmesh_clearance_m": None,
                "translation_blocked": False,
                "integration": "Habitat-Sim physics.VelocityControl",
                "collision_filter": "Habitat-Sim Simulator.step_filter",
            },
            "verifier": {
                "enabled": self.verifier_enabled,
                "planner": "8-connected A* on predicted Single BEV",
                "target_source": (
                    "normalized UI click converted directly to current ego-local "
                    "x/forward metres; GT pixels are display-only"
                ),
                "inflation_radius_m": DEFAULT_INFLATION_RADIUS_M,
                "inflation_radius_min_m": MIN_INTERACTIVE_INFLATION_RADIUS_M,
                "inflation_radius_max_m": MAX_INTERACTIVE_INFLATION_RADIUS_M,
                "alignment": "UI click -> ego-local metres -> predicted pixel",
                "runtime_truth_inputs": [],
                "forbidden_runtime_inputs": [
                    "GT occupancy values",
                    "Habitat pose/extrinsic",
                    "navmesh queries",
                    "GT depth",
                ],
                "available_modes": {
                    "open_loop": (
                        "one predicted-only A* plan; immutable grid-action "
                        "queue; no pose feedback, navmesh query, avoidance, "
                        "snap, or replanning"
                    ),
                    "closed_loop": (
                        "remember the local metric target with VGGT pose + Scale "
                        "Token; execute local grid actions; replan on each usable "
                        "predicted-BEV update and retain the old queue otherwise"
                    ),
                },
                "mode": "open_loop",
                "plan_revision": 0,
                "replan_count": 0,
                "deferred_replan_count": 0,
                "last_deferred_replan": None,
                "last_plan": None,
                "active_plan": None,
            },
        }
        self.camera_jpeg: Optional[bytes] = None
        self.bev_png: Optional[bytes] = None
        self.obstacle_png: Optional[bytes] = None
        self.bev_frame_extent: Optional[float] = None
        self.bev_frame_extrinsic: Optional[Dict[str, Any]] = None
        self.map_png: Optional[bytes] = None
        self.full_scene_obstacle_map: Optional[np.ndarray] = None
        self.full_scene_lower_bound: Optional[np.ndarray] = None
        self.full_scene_meters_per_pixel = self.voxel_size
    def start_engine(self) -> None:
        self.comparison.start()
        self.thread.start()

    def stop_engine(self) -> None:
        self.stop_event.set()
        self.commands.put({"type": "stop"})
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=5.0)
        self.comparison.stop()

    def wait_until_ready(self, timeout: float = 30.0) -> None:
        if not self.ready_event.wait(timeout):
            raise RuntimeError("Timed out while starting the Habitat engine")
        snapshot = self.snapshot()
        if not snapshot.get("ready"):
            raise RuntimeError(snapshot.get("status", "Habitat failed to start"))

    def snapshot(self) -> Dict[str, Any]:
        with self.condition:
            snapshot = json.loads(json.dumps(self.state))
        snapshot["model_comparison"] = self.comparison.status()
        return snapshot

    def latest_model_comparison(self) -> Dict[str, Any]:
        return self.comparison.snapshot()

    def plan_verifier_goal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Plan on one exact synchronized prediction and queue robot motion."""

        if not self.verifier_enabled:
            raise VerifierPlanningError(
                "verifier_disabled",
                "Interactive verifier mode was not enabled at server startup.",
            )
        model_key = str(payload.get("model_key", "p1b"))
        mode = str(payload.get("mode", "open_loop"))
        if mode not in {"open_loop", "closed_loop"}:
            raise VerifierPlanningError(
                "invalid_mode", "mode must be open_loop or closed_loop."
            )
        try:
            frame_seq = int(payload["frame_seq"])
            target_metric_m = [
                float(payload["target_metric_m"][0]),
                float(payload["target_metric_m"][1]),
            ]
            inflation_radius_m = float(
                payload.get("inflation_radius_m", DEFAULT_INFLATION_RADIUS_M)
            )
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise VerifierPlanningError(
                "invalid_request",
                "frame_seq, local target_metric_m=[x_right, z_forward], and a "
                "numeric inflation radius are required.",
            ) from error
        if not all(math.isfinite(value) for value in target_metric_m):
            raise VerifierPlanningError(
                "invalid_target", "The ego-local target must be finite."
            )
        if (
            not math.isfinite(inflation_radius_m)
            or inflation_radius_m < MIN_INTERACTIVE_INFLATION_RADIUS_M
            or inflation_radius_m > MAX_INTERACTIVE_INFLATION_RADIUS_M
        ):
            raise VerifierPlanningError(
                "invalid_inflation_radius",
                "One-sided obstacle inflation must be between "
                f"{MIN_INTERACTIVE_INFLATION_RADIUS_M * 100:.0f} and "
                f"{MAX_INTERACTIVE_INFLATION_RADIUS_M * 100:.0f} cm.",
                inflation_radius_m=inflation_radius_m,
            )
        try:
            frame = self.comparison.planning_snapshot(
                model_key, expected_frame_seq=frame_seq
            )
        except (KeyError, RuntimeError) as error:
            raise VerifierPlanningError(
                "stale_or_missing_frame", str(error)
            ) from error
        target_right_m, target_forward_m = target_metric_m
        half_fov_radians = math.radians(self.horizontal_fov_degrees / 2.0)
        target_bearing_radians = math.atan2(
            abs(target_right_m), target_forward_m
        )
        if (
            target_forward_m <= 0.0
            or target_bearing_radians > half_fov_radians
        ):
            raise VerifierPlanningError(
                "target_outside_initial_camera_fov",
                "The initial ego-local target must lie inside the calibrated "
                "camera FOV. Later closed-loop replans keep the target even "
                "when it leaves view.",
                target_metric_m=target_metric_m,
                horizontal_fov_degrees=self.horizontal_fov_degrees,
            )
        # The GT image is only the browser's click canvas.  The API boundary is
        # already an ego-local coordinate and carries no GT pixel or label.
        plan = plan_metric_target(
            predicted_semantic=frame["predicted_semantic"],
            predicted_extent_m=frame["predicted_extent_m"],
            target_metric_m=target_metric_m,
            frame_seq=frame["frame_seq"],
            model_key=model_key,
            inflation_radius_m=inflation_radius_m,
        )
        plan["metric_alignment"].update(
            {
                "lambda_m_per_vggt": float(frame["lambda_m_per_vggt"]),
                "scale_std_m_per_vggt": float(
                    frame["scale_std_m_per_vggt"]
                ),
                "scale_contract": (
                    "Scale Token is used only for VGGT relative translation "
                    "during closed-loop target memory; A* receives an ego-local "
                    "metric target and a fixed-metric predicted BEV"
                ),
            }
        )
        plan["mode"] = mode
        plan["replan_count"] = 0
        plan["deferred_replan_count"] = 0
        plan["pose_source"] = (
            "none: fixed actions compiled once from A* grid"
            if mode == "open_loop"
            else "VGGT predicted world-to-camera extrinsics + Scale Token"
        )
        actions = compile_grid_path_to_open_loop_actions(plan["path_pixels"])
        plan.update(
            {
                "execution_action_count": len(actions),
                "execution_waypoint_count": 0,
                "control_contract": {
                    "planner_input": (
                        "predicted semantic BEV + ego-local target coordinate"
                    ),
                    "target_click_semantics_read": False,
                    "controller": "45-degree/grid-step local action queue",
                    "simulator_pose_feedback": False,
                    "simulator_extrinsic": False,
                    "navmesh_queries": False,
                    "waypoint_snap": False,
                    "obstacle_avoidance": False,
                    "replanning": mode == "closed_loop",
                    "closed_loop_motion_estimate": (
                        "VGGT predicted extrinsics + Scale Token"
                        if mode == "closed_loop"
                        else None
                    ),
                    "gt_use": "display-only click canvas",
                },
            }
        )
        queued_plan = dict(plan)
        queued_plan["_local_actions"] = actions
        # Freeze the exact synchronized imagery in the browser so the path is
        # never overlaid on a later ego frame.  These images are presentation
        # payloads only and are not accepted by the planner function above.
        plan["images"] = frame["presentation_images"]
        queued_plan["images"] = frame["presentation_images"]
        self.submit({"type": "verifier_path", "plan": queued_plan})
        return plan

    def _publish_verifier_plan(
        self,
        plan: Dict[str, Any],
        *,
        execution_status: str,
    ) -> None:
        """Publish a compact browser-safe plan without repeating PNG payloads."""

        self.verifier_plan_revision += 1
        public_plan = {
            key: json.loads(json.dumps(plan[key]))
            for key in (
                "success",
                "frame_seq",
                "model_key",
                "mode",
                "replan_count",
                "deferred_replan_count",
                "target_metric_m",
                "path_metric_m",
                "path_length_m",
                "path_cell_count",
                "execution_waypoint_count",
                "execution_action_count",
                "inflation_radius_m",
                "inflation_radius_cells",
                "effective_inflation_radius_m",
                "inflation_semantics",
                "metric_alignment",
                "pose_source",
                "control_contract",
                "planner_runtime_inputs",
                "planner_forbidden_inputs",
                "executed_action_count",
                "predicted_target_error_m",
                "predicted_success_tolerance_m",
            )
            if key in plan
        }
        public_plan["plan_revision"] = self.verifier_plan_revision
        public_plan["execution_status"] = execution_status
        if "predicted_motion_current_from_previous_metric" in plan:
            public_plan["predicted_motion_current_from_previous_metric"] = (
                json.loads(
                    json.dumps(
                        plan[
                            "predicted_motion_current_from_previous_metric"
                        ]
                    )
                )
            )
        self.state["verifier"].update(
            {
                "mode": plan.get("mode", "open_loop"),
                "inflation_radius_m": float(plan["inflation_radius_m"]),
                "plan_revision": self.verifier_plan_revision,
                "replan_count": int(plan.get("replan_count", 0)),
                "deferred_replan_count": int(
                    plan.get("deferred_replan_count", 0)
                ),
                "last_deferred_replan": None,
                "last_plan": {
                    "success": True,
                    "frame_seq": int(plan["frame_seq"]),
                    "path_length_m": float(plan["path_length_m"]),
                    "inflation_radius_m": float(plan["inflation_radius_m"]),
                    "execution_status": execution_status,
                    "mode": plan.get("mode", "open_loop"),
                    "replan_count": int(plan.get("replan_count", 0)),
                    "deferred_replan_count": int(
                        plan.get("deferred_replan_count", 0)
                    ),
                },
                "active_plan": public_plan,
            }
        )

    def _record_local_action(self, follower: LocalAStarActionFollower) -> None:
        """Publish command progress without reading simulator state."""

        active_plan = self.state["verifier"].get("active_plan")
        if active_plan:
            self.verifier_plan_revision += 1
            active_plan.update(
                {
                    "plan_revision": self.verifier_plan_revision,
                    "executed_action_count": follower.executed_action_count,
                }
            )
            self.state["verifier"]["plan_revision"] = (
                self.verifier_plan_revision
            )

    def _finish_open_loop_without_truth(self) -> None:
        """End open-loop execution without GT terminal scoring."""

        self.state["status"] = (
            "Open-loop action queue complete; physical outcome is unverified "
            "because no simulator truth is read"
        )
        self._update_verifier_execution_status("completed_unverified")

    def _finish_closed_loop_from_prediction(
        self,
        *,
        frame_seq: int,
        target_metric_m: list[float],
    ) -> None:
        """Declare arrival only from the VGGT/Scale-updated target estimate."""

        predicted_error_m = float(np.hypot(*target_metric_m))
        self.closed_loop_context = None
        for key in ("last_plan", "active_plan"):
            record = self.state["verifier"].get(key)
            if record:
                record.update(
                    {
                        "frame_seq": int(frame_seq),
                        "target_metric_m": list(target_metric_m),
                        "predicted_target_error_m": predicted_error_m,
                        "predicted_success_tolerance_m": (
                            CLOSED_LOOP_PREDICTED_SUCCESS_TOLERANCE_METERS
                        ),
                    }
                )
        self.state.update(
            {
                "navigating": False,
                "last_action": None,
                "status": (
                    "Closed-loop target reached by VGGT/Scale estimate · "
                    f"{predicted_error_m:.3f} m"
                ),
            }
        )
        self._update_verifier_execution_status("reached_predicted")

    def _defer_closed_loop_replan(
        self,
        follower: Any,
        *,
        context: Dict[str, Any],
        frame_seq: int,
        code: str,
        reason: str,
        target_metric_m: Optional[list[float]] = None,
    ) -> Any:
        """Keep the last valid route when one new BEV cannot be replanned."""

        updated_context = dict(context)
        updated_context["last_attempted_frame_seq"] = int(frame_seq)
        if target_metric_m is not None:
            # The VGGT pose update succeeded, so retain the target in this
            # newer ego frame even though occupancy planning did not.
            updated_context["last_frame_seq"] = int(frame_seq)
            updated_context["target_metric_m"] = list(target_metric_m)
        deferred_count = int(context.get("deferred_replan_count", 0)) + 1
        updated_context["deferred_replan_count"] = deferred_count
        self.closed_loop_context = updated_context

        details = {
            "frame_seq": int(frame_seq),
            "code": str(code),
            "reason": str(reason),
            "continuing_previous_route": True,
        }
        self.state.update(
            {
                "navigating": True,
                "status": (
                    f"Closed-loop replan deferred [{code}] on frame "
                    f"{frame_seq}; continuing the previous valid route"
                ),
            }
        )
        verifier = self.state["verifier"]
        verifier["deferred_replan_count"] = deferred_count
        verifier["last_deferred_replan"] = details
        last_plan = verifier.get("last_plan")
        if last_plan:
            last_plan["deferred_replan_count"] = deferred_count
            last_plan["last_deferred_replan"] = details
            last_plan["execution_status"] = "running"
        active_plan = verifier.get("active_plan")
        if active_plan:
            self.verifier_plan_revision += 1
            active_plan["plan_revision"] = self.verifier_plan_revision
            active_plan["deferred_replan_count"] = deferred_count
            active_plan["last_deferred_replan"] = details
            active_plan["execution_status"] = "running"
            verifier["plan_revision"] = self.verifier_plan_revision
        return follower

    def _update_verifier_execution_status(
        self,
        execution_status: str,
        *,
        failure_code: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        """Publish execution-only changes without inventing a new A* plan."""

        last_plan = self.state["verifier"].get("last_plan")
        if last_plan:
            last_plan["execution_status"] = execution_status
            if failure_code is not None:
                last_plan["execution_failure_code"] = failure_code
            if failure_reason is not None:
                last_plan["execution_failure_reason"] = failure_reason
        active_plan = self.state["verifier"].get("active_plan")
        if active_plan:
            self.verifier_plan_revision += 1
            active_plan["plan_revision"] = self.verifier_plan_revision
            active_plan["execution_status"] = execution_status
            if failure_code is not None:
                active_plan["execution_failure_code"] = failure_code
            if failure_reason is not None:
                active_plan["execution_failure_reason"] = failure_reason
            self.state["verifier"]["plan_revision"] = (
                self.verifier_plan_revision
            )

    def _maybe_replan_closed_loop(
        self,
        follower: Optional[Any],
    ) -> Optional[Any]:
        """Consume each new predicted BEV and update the target via VGGT pose."""

        context = self.closed_loop_context
        if context is None or follower is None:
            return follower
        try:
            frame = self.comparison.planning_snapshot(context["model_key"])
        except (KeyError, RuntimeError):
            return follower
        frame_seq = int(frame["frame_seq"])
        previous_frame_seq = int(context["last_frame_seq"])
        last_attempted_frame_seq = int(
            context.get("last_attempted_frame_seq", previous_frame_seq)
        )
        if frame_seq <= last_attempted_frame_seq:
            return follower

        pose_frame_seqs = frame["vggt_pose_frame_seqs"]
        predicted_poses = frame["vggt_predicted_camera_from_world"]
        if len(pose_frame_seqs) < 2 or len(predicted_poses) < 2:
            return self._defer_closed_loop_replan(
                follower,
                context=context,
                frame_seq=frame_seq,
                code="vggt_pose_unavailable",
                reason="the updated prediction has fewer than two VGGT poses",
            )
        pose_frame_seqs = [int(value) for value in pose_frame_seqs]
        if pose_frame_seqs[-1] != frame_seq:
            return self._defer_closed_loop_replan(
                follower,
                context=context,
                frame_seq=frame_seq,
                code="vggt_pose_chain_discontinuity",
                reason="the latest VGGT pose does not match the updated BEV frame",
            )
        try:
            previous_pose_index = pose_frame_seqs.index(previous_frame_seq)
        except ValueError:
            return self._defer_closed_loop_replan(
                follower,
                context=context,
                frame_seq=frame_seq,
                code="vggt_pose_chain_discontinuity",
                reason="the last target-reference frame has left the VGGT pose window",
            )
        try:
            relative_motion = relative_camera_motion_metric(
                predicted_poses[previous_pose_index],
                predicted_poses[-1],
                frame["lambda_m_per_vggt"],
            )
            target_metric_m = transform_target_by_predicted_motion(
                context["target_metric_m"], relative_motion
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._defer_closed_loop_replan(
                follower,
                context=context,
                frame_seq=frame_seq,
                code="vggt_pose_transform_unavailable",
                reason=str(error) or type(error).__name__,
            )

        if (
            float(np.hypot(*target_metric_m))
            <= CLOSED_LOOP_PREDICTED_SUCCESS_TOLERANCE_METERS
        ):
            self._finish_closed_loop_from_prediction(
                frame_seq=frame_seq,
                target_metric_m=list(target_metric_m)
            )
            return None

        try:
            plan = plan_metric_target(
                predicted_semantic=frame["predicted_semantic"],
                predicted_extent_m=frame["predicted_extent_m"],
                target_metric_m=target_metric_m,
                frame_seq=frame_seq,
                model_key=context["model_key"],
                inflation_radius_m=float(context["inflation_radius_m"]),
            )
        except VerifierPlanningError as error:
            return self._defer_closed_loop_replan(
                follower,
                context=context,
                frame_seq=frame_seq,
                code=error.code,
                reason=error.message,
                target_metric_m=list(target_metric_m),
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._defer_closed_loop_replan(
                follower,
                context=context,
                frame_seq=frame_seq,
                code="replan_input_unavailable",
                reason=str(error) or type(error).__name__,
                target_metric_m=list(target_metric_m),
            )

        replan_count = int(context["replan_count"]) + 1
        deferred_replan_count = int(context.get("deferred_replan_count", 0))
        actions = compile_grid_path_to_open_loop_actions(plan["path_pixels"])
        plan.update(
            {
                "mode": "closed_loop",
                "replan_count": replan_count,
                "deferred_replan_count": deferred_replan_count,
                "execution_action_count": len(actions),
                "pose_source": (
                    "VGGT predicted adjacent extrinsics; translation "
                    "converted by Scale Token"
                ),
                "control_contract": {
                    "planner_input": (
                        "predicted semantic BEV + ego-local target coordinate"
                    ),
                    "target_click_semantics_read": False,
                    "controller": "receding local 45-degree/grid-step action queue",
                    "simulator_pose_feedback": False,
                    "simulator_extrinsic": False,
                    "navmesh_queries": False,
                    "waypoint_snap": False,
                    "obstacle_avoidance": False,
                    "replanning": True,
                    "closed_loop_motion_estimate": (
                        "VGGT predicted extrinsics + Scale Token"
                    ),
                    "gt_use": "display-only click canvas",
                },
                "predicted_motion_current_from_previous_metric": (
                    relative_motion.tolist()
                ),
            }
        )
        plan["metric_alignment"].update(
            {
                "lambda_m_per_vggt": float(frame["lambda_m_per_vggt"]),
                "scale_std_m_per_vggt": float(
                    frame["scale_std_m_per_vggt"]
                ),
                "scale_contract": (
                    "VGGT relative translation is multiplied by the predicted "
                    "Scale Token before the remembered target is transformed"
                ),
            }
        )
        replanned_follower = LocalAStarActionFollower(actions)

        self.closed_loop_context = {
            "model_key": context["model_key"],
            "last_frame_seq": frame_seq,
            "last_attempted_frame_seq": frame_seq,
            "target_metric_m": target_metric_m,
            "inflation_radius_m": float(context["inflation_radius_m"]),
            "replan_count": replan_count,
            "deferred_replan_count": deferred_replan_count,
        }
        self.state.update(
            {
                "goal": [
                    float(target_metric_m[0]),
                    0.0,
                    float(target_metric_m[1]),
                ],
                "path": [],
                "path_distance": float(plan["path_length_m"]),
                "navigating": True,
                "last_action": None,
                "status": (
                    f"Closed-loop A* replan {replan_count} on model frame "
                    f"{frame_seq} · {plan['path_length_m']:.2f} m predicted "
                    f"path · {len(actions)} local actions"
                ),
            }
        )
        self._publish_verifier_plan(plan, execution_status="running")
        return replanned_follower

    def submit(self, command: Dict[str, Any]) -> None:
        self.commands.put(command)

    def latest_image(self, kind: str) -> Optional[bytes]:
        with self.condition:
            return self._image_for_kind(kind)

    def latest_bev_pair(
        self,
    ) -> Tuple[
        int,
        float,
        Optional[Dict[str, Any]],
        Optional[bytes],
        Optional[bytes],
    ]:
        """Return one atomically published masked/complete BEV frame pair."""
        with self.condition:
            return (
                int(self.state.get("frame_seq", 0)),
                float(
                    self.bev_extent
                    if self.bev_frame_extent is None
                    else self.bev_frame_extent
                ),
                (
                    None
                    if self.bev_frame_extrinsic is None
                    else json.loads(json.dumps(self.bev_frame_extrinsic))
                ),
                self.bev_png,
                self.obstacle_png,
            )

    def _image_for_kind(self, kind: str) -> Optional[bytes]:
        if kind == "camera":
            return self.camera_jpeg
        if kind == "bev":
            return self.bev_png
        if kind == "obstacle":
            return self.obstacle_png
        raise ValueError(f"unknown image kind: {kind}")

    def wait_for_image(
        self, kind: str, last_sequence: int, timeout: float = 2.0
    ) -> Tuple[int, Optional[bytes]]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.state.get("frame_seq", 0) > last_sequence
                or self.stop_event.is_set(),
                timeout=timeout,
            )
            sequence = int(self.state.get("frame_seq", 0))
            return sequence, self._image_for_kind(kind)

    def _publish_map(
        self, simulator: habitat_sim.Simulator, agent_height: float
    ) -> None:
        meters_per_pixel = GLOBAL_MAP_METERS_PER_PIXEL
        topdown = simulator.pathfinder.get_topdown_view(
            meters_per_pixel, agent_height
        )
        image = np.zeros((*topdown.shape, 3), dtype=np.uint8)
        image[topdown] = [220, 226, 230]
        image[~topdown] = [27, 31, 36]
        self.map_png = encode_png(np.flipud(image))

        lower_bound, _ = simulator.pathfinder.get_bounds()
        lower_bound = np.asarray(lower_bound, dtype=np.float64)
        map_rows, map_columns = topdown.shape
        voxel_navigable_map = simulator.pathfinder.get_topdown_view(
            self.voxel_size, agent_height
        )
        voxel_rows, voxel_columns = voxel_navigable_map.shape
        full_obstacle_map, voxel_statistics = render_full_obstacle_map(
            simulator,
            floor_height=agent_height,
            meters_per_pixel=self.voxel_size,
            rows=voxel_rows,
            columns=voxel_columns,
            lower_bound=lower_bound,
            obstacle_min_height=self.obstacle_min_height,
            obstacle_max_height=self.obstacle_max_height,
            navigable_map=voxel_navigable_map,
        )
        self.full_scene_obstacle_map = full_obstacle_map
        self.full_scene_lower_bound = lower_bound
        self.full_scene_meters_per_pixel = self.voxel_size
        self.state["map"] = {
            "width": map_columns,
            "height": map_rows,
            "xmin": float(lower_bound[0]),
            "xmax": float(lower_bound[0] + map_columns * meters_per_pixel),
            "zmin": float(lower_bound[2]),
            "zmax": float(lower_bound[2] + map_rows * meters_per_pixel),
            "meters_per_pixel": meters_per_pixel,
        }
        self.state["obstacle_only_bev"] = {
            "width": self.bev_size,
            "height": self.bev_size,
            "extent_m": self.bev_extent,
            "meters_per_pixel": self.bev_extent / self.bev_size,
            "encoding": "8-bit grayscale PNG",
            "unoccupied_value": 255,
            "occupied_value": 0,
            "obstacle_height_min_m": self.obstacle_min_height,
            "obstacle_height_max_m": self.obstacle_max_height,
            "ground_truth_id": "simulator_solid_voxel_occupancy_v2",
            "source": (
                "ego crop of solid voxels from the active Habitat collision mesh"
            ),
            "construction": (
                "collision mesh voxelization, solidification, then max-Y projection"
            ),
            "voxel_statistics": voxel_statistics,
            "orientation": "ego-centric, robot forward is image up",
            "line_of_sight_limited": False,
            "range_limited": True,
        }

    def _set_manual_enabled(self, enabled: bool) -> None:
        self.manual_enabled = enabled
        if not enabled:
            for key in self.manual_keys:
                self.manual_keys[key] = False
        self.state["manual_control"]["enabled"] = enabled
        self.state["manual_control"]["keys"] = dict(self.manual_keys)
        if not enabled:
            self.state["manual_control"]["translation_blocked"] = False

    def _apply_manual_control(
        self,
        simulator: habitat_sim.Simulator,
        velocity_control: habitat_sim.physics.VelocityControl,
        delta_time: float,
    ) -> Optional[str]:
        linear_axis = int(self.manual_keys["forward"]) - int(
            self.manual_keys["backward"]
        )
        angular_axis = int(self.manual_keys["left"]) - int(
            self.manual_keys["right"]
        )
        if linear_axis == 0 and angular_axis == 0:
            return None

        velocity_control.controlling_lin_vel = True
        velocity_control.lin_vel_is_local = True
        velocity_control.linear_velocity = np.asarray(
            [
                0.0,
                0.0,
                -linear_axis * MANUAL_LINEAR_SPEED_METERS_PER_SECOND,
            ],
            dtype=np.float32,
        )
        velocity_control.controlling_ang_vel = True
        velocity_control.ang_vel_is_local = True
        velocity_control.angular_velocity = np.asarray(
            [
                0.0,
                angular_axis * MANUAL_ANGULAR_SPEED_RADIANS_PER_SECOND,
                0.0,
            ],
            dtype=np.float32,
        )

        agent = simulator.get_agent(0)
        agent_state = agent.get_state()
        current = habitat_sim.RigidState(
            quat_to_magnum(agent_state.rotation), agent_state.position
        )
        target = velocity_control.integrate_transform(delta_time, current)
        filtered_position = simulator.step_filter(
            current.translation, target.translation
        )
        current_clearance = float(
            simulator.pathfinder.distance_to_closest_obstacle(
                current.translation, 2.0
            )
        )
        target_clearance = float(
            simulator.pathfinder.distance_to_closest_obstacle(
                filtered_position, 2.0
            )
        )
        translation_blocked = (
            linear_axis != 0
            and target_clearance < self.manual_clearance
            and target_clearance < current_clearance - 1e-6
        )
        if translation_blocked:
            filtered_position = current.translation
            target_clearance = current_clearance
            self.state["status"] = (
                "Movement blocked by navmesh clearance — turn away "
                "from the obstacle"
            )
        else:
            self.state["status"] = "WASD active — W/S move, A/D rotate"
        self.state["manual_control"].update(
            {
                "current_navmesh_clearance_m": target_clearance,
                "translation_blocked": translation_blocked,
            }
        )
        agent_state.position = np.asarray(filtered_position, dtype=np.float32)
        agent_state.rotation = quat_from_magnum(target.rotation)
        agent.set_state(agent_state, reset_sensors=True)

        actions = []
        if linear_axis > 0:
            actions.append("forward")
        elif linear_axis < 0:
            actions.append("backward")
        if angular_axis > 0:
            actions.append("left")
        elif angular_axis < 0:
            actions.append("right")
        action = "manual_" + "+".join(actions)
        return action + ("_blocked" if translation_blocked else "")

    def _handle_commands(
        self,
        simulator: habitat_sim.Simulator,
        navigator: PointNavigator,
        follower: Optional[Any],
    ) -> Optional[Any]:
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                break
            command_type = command.get("type")
            if command_type == "stop":
                return follower
            if command_type == "verifier_path":
                self._set_manual_enabled(False)
                plan = command["plan"]
                mode = str(plan.get("mode", "open_loop"))
                try:
                    follower = LocalAStarActionFollower(
                        plan["_local_actions"]
                    )
                    goal = [
                        float(plan["target_metric_m"][0]),
                        0.0,
                        float(plan["target_metric_m"][1]),
                    ]
                    path = []
                    if mode == "open_loop":
                        self.closed_loop_context = None
                        status = (
                            "Strict open-loop A* succeeded; replaying "
                            f"{len(follower.actions)} immutable actions "
                            "without pose, GT, extrinsic, or navmesh input"
                        )
                    else:
                        status = (
                            "Closed-loop A* succeeded; executing "
                            f"{len(follower.actions)} local grid actions with "
                            "VGGT/Scale-only target memory"
                        )
                        self.closed_loop_context = {
                            "model_key": plan["model_key"],
                            "last_frame_seq": int(plan["frame_seq"]),
                            "last_attempted_frame_seq": int(plan["frame_seq"]),
                            "target_metric_m": list(plan["target_metric_m"]),
                            "inflation_radius_m": float(
                                plan["inflation_radius_m"]
                            ),
                            "replan_count": 0,
                            "deferred_replan_count": 0,
                        }
                    self.state.update(
                        {
                            "goal": goal,
                            "path": path,
                            "path_distance": float(plan["path_length_m"]),
                            "navigating": True,
                            "paused": False,
                            "step_count": 0,
                            "last_action": None,
                            "status": status,
                        }
                    )
                    self._publish_verifier_plan(
                        plan, execution_status="running"
                    )
                except (
                    AssertionError,
                    KeyError,
                    PointNavigationError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    follower = None
                    self.closed_loop_context = None
                    self.state.update(
                        {
                            "navigating": False,
                            "last_action": None,
                            "status": f"A* succeeded but execution failed: {error}",
                        }
                    )
                    self._publish_verifier_plan(
                        plan, execution_status="failed"
                    )
                    self.state["verifier"]["last_plan"][
                        "execution_failure_reason"
                    ] = str(error)
                    self.state["verifier"]["active_plan"][
                        "execution_failure_reason"
                    ] = str(error)
            elif command_type == "goal":
                self._set_manual_enabled(False)
                self.closed_loop_context = None
                position = simulator.get_agent(0).get_state().position
                requested_goal = [
                    float(command["x"]),
                    float(position[1]),
                    float(command["z"]),
                ]
                try:
                    plan = navigator.plan(requested_goal)
                    follower = habitat_sim.nav.GreedyGeodesicFollower(
                        simulator.pathfinder,
                        simulator.get_agent(0),
                        goal_radius=navigator.goal_radius,
                        stop_key=None,
                        forward_key="move_forward",
                        left_key="turn_left",
                        right_key="turn_right",
                    )
                    self.state.update(
                        {
                            "goal": plan.goal.tolist(),
                            "path": [point.tolist() for point in plan.waypoints],
                            "path_distance": plan.geodesic_distance,
                            "navigating": True,
                            "paused": False,
                            "step_count": 0,
                            "last_action": None,
                            "status": (
                                f"Navigating {plan.geodesic_distance:.2f} m"
                            ),
                        }
                    )
                except (
                    AssertionError,
                    KeyError,
                    PointNavigationError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    self.state.update(
                        {
                            "navigating": False,
                            "last_action": None,
                            "status": f"Cannot navigate there: {error}",
                        }
                    )
                    follower = None
            elif command_type == "pause" and follower is not None:
                self.state["paused"] = True
                self.state["status"] = "Paused"
            elif command_type == "resume" and follower is not None:
                self.state["paused"] = False
                self.state["status"] = "Navigation resumed"
            elif command_type == "cancel":
                follower = None
                self.closed_loop_context = None
                if self.state["verifier"].get("last_plan"):
                    self._update_verifier_execution_status("cancelled")
                for key in self.manual_keys:
                    self.manual_keys[key] = False
                self.state["manual_control"]["keys"] = dict(self.manual_keys)
                self.state.update(
                    {
                        "navigating": False,
                        "paused": False,
                        "goal": None,
                        "path": [],
                        "last_action": None,
                        "status": "Navigation cancelled",
                    }
                )
            elif command_type == "manual_control":
                if "enabled" in command:
                    self._set_manual_enabled(bool(command["enabled"]))
                if self.manual_enabled and "keys" in command:
                    self.manual_keys.update(command["keys"])
                    self.state["manual_control"]["keys"] = dict(
                        self.manual_keys
                    )
                if self.manual_enabled:
                    if follower is not None and self.state["verifier"].get(
                        "last_plan"
                    ):
                        self._update_verifier_execution_status("cancelled")
                    follower = None
                    self.closed_loop_context = None
                    self.state.update(
                        {
                            "navigating": False,
                            "paused": False,
                            "goal": None,
                            "path": [],
                            "last_action": None,
                            "status": (
                                "WASD active — W/S move, A/D rotate"
                            ),
                        }
                    )
                elif "enabled" in command:
                    self.state.update(
                        {
                            "last_action": None,
                            "status": "Click a navigable point to begin",
                        }
                    )
            elif command_type == "set_bev_extent":
                extent = float(command["extent_m"])
                self.bev_extent = extent
                self.state["bev"]["extent_m"] = extent
                self.state["bev"]["meters_per_pixel"] = (
                    extent / self.bev_size
                )
                self.state["obstacle_only_bev"]["extent_m"] = extent
                self.state["obstacle_only_bev"]["meters_per_pixel"] = (
                    extent / self.bev_size
                )
        return follower

    def _update_pose_state(
        self, simulator: habitat_sim.Simulator, last_action: Optional[Any]
    ) -> None:
        agent_state = simulator.get_agent(0).get_state()
        position = np.asarray(agent_state.position, dtype=np.float64)
        forward = habitat_sim.utils.common.quat_rotate_vector(
            agent_state.rotation, np.array([0.0, 0.0, -1.0])
        )
        pose = {
            "agent_position": position.tolist(),
            "heading_xz": [float(forward[0]), float(forward[2])],
        }
        # Keep the most recently executed action visible between action ticks.
        # Rendering normally runs faster than actuation, so clearing this on
        # every render-only frame would make the browser almost always say
        # "idle" even while the robot is turning or moving.
        if last_action is not None:
            pose["last_action"] = str(last_action)
        self.state.update(pose)

    def _run(self) -> None:
        simulator: Optional[habitat_sim.Simulator] = None
        try:
            simulator = make_web_simulator(
                self.scene,
                self.scene_dataset_config,
                self.gpu_device_id,
                self.camera_width,
                self.camera_height,
                self.sensor_height_m,
                self.horizontal_fov_degrees,
                self.bev_extent / self.bev_size,
            )
            ensure_navmesh(simulator, self.navmesh)
            navmesh_agent_radius = float(
                simulator.pathfinder.nav_mesh_settings.agent_radius
            )
            start, start_yaw_degrees = place_agent(
                simulator,
                self.start,
                self.start_yaw_degrees,
                self.seed,
                minimum_clearance_m=(
                    navmesh_agent_radius + self.manual_clearance + 0.1
                ),
            )
            initial_state = simulator.get_agent(0).get_state()
            initial_forward = habitat_sim.utils.common.quat_rotate_vector(
                initial_state.rotation, np.asarray([0.0, 0.0, -1.0])
            )
            navigator = PointNavigator(simulator, max_snap_distance=0.75)
            self.state["manual_control"].update(
                {
                    "navmesh_agent_radius_m": navmesh_agent_radius,
                    "effective_body_radius_m": (
                        navmesh_agent_radius + self.manual_clearance
                    ),
                }
            )
            self.state.update(
                {
                    "ready": True,
                    "status": "WASD active — W/S move, A/D rotate",
                    "goal": None,
                    "path": [],
                    "last_action": None,
                    "agent_position": start.tolist(),
                    "heading_xz": [
                        float(initial_forward[0]),
                        float(initial_forward[2]),
                    ],
                    "session": {
                        "scene": str(self.scene),
                        "scene_name": self.scene.stem,
                        "seed": self.seed,
                        "random_start": self.start is None,
                        "random_yaw": self.start_yaw_degrees is None,
                        "start_position_m": start.tolist(),
                        "start_yaw_degrees": start_yaw_degrees,
                    },
                    "camera": {
                        "width": self.camera_width,
                        "height": self.camera_height,
                        "hfov_degrees": self.horizontal_fov_degrees,
                        "sensor_height_m": self.sensor_height_m,
                    },
                    "bev": {
                        "size": self.bev_size,
                        "extent_m": self.bev_extent,
                        "meters_per_pixel": self.bev_extent / self.bev_size,
                        "obstacle_height_min_m": self.obstacle_min_height,
                        "obstacle_height_max_m": self.obstacle_max_height,
                        "visibility_algorithm": "symmetric_grid_shadowcasting",
                        "ray_count": None,
                        "encoding": "8-bit grayscale PNG",
                        "unoccupied_value": 255,
                        "occupied_value": 0,
                        "unknown_value": 112,
                        "ground_truth_id": "simulator_solid_voxel_occupancy_v2",
                        "source": (
                            "ideal visibility mask applied to the same solid "
                            "collision-mesh voxel truth"
                        ),
                        "representation": "free / obstacle / occluded",
                        "horizontal_fov_degrees": self.horizontal_fov_degrees,
                        "occupancy_from_sensor_observations": False,
                        "orientation": "ego-centric, robot forward is image up",
                    },
                    "render_fps": self.fps,
                    "action_hz": self.action_hz,
                }
            )
            self._publish_map(simulator, float(start[1]))
            observations = simulator.get_sensor_observations()
            velocity_control = habitat_sim.physics.VelocityControl()
            follower: Optional[Any] = None
            next_action_time = time.monotonic()
            frame_interval = 1.0 / self.fps
            previous_loop_time = time.monotonic()

            self.ready_event.set()
            while not self.stop_event.is_set():
                loop_start = time.monotonic()
                delta_time = min(
                    max(loop_start - previous_loop_time, 0.0),
                    MANUAL_MAX_INTEGRATION_STEP_SECONDS,
                )
                previous_loop_time = loop_start
                follower = self._handle_commands(simulator, navigator, follower)
                follower = self._maybe_replan_closed_loop(follower)
                last_action = None
                if self.manual_enabled:
                    last_action = self._apply_manual_control(
                        simulator, velocity_control, delta_time
                    )
                    if last_action is not None:
                        observations = simulator.get_sensor_observations()
                        self.state["step_count"] = (
                            int(self.state.get("step_count", 0)) + 1
                        )
                    else:
                        self.state["last_action"] = None
                        self.state["manual_control"]["translation_blocked"] = False
                elif (
                    follower is not None
                    and not self.state.get("paused", False)
                    and loop_start >= next_action_time
                ):
                    try:
                        last_action = follower.next_action_along(
                            np.asarray(self.state["goal"], dtype=np.float32)
                        )
                    except (
                        habitat_sim.errors.GreedyFollowerError,
                        AssertionError,
                        KeyError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as error:
                        follower = None
                        self.closed_loop_context = None
                        reason = str(error) or type(error).__name__
                        self.state.update(
                            {
                                "navigating": False,
                                "last_action": None,
                                "status": f"Navigation follower failed: {reason}",
                            }
                        )
                        if self.state["verifier"].get("last_plan"):
                            self._update_verifier_execution_status(
                                "failed",
                                failure_code="habitat_follower_failed",
                                failure_reason=reason,
                            )
                    if last_action is None and follower is not None:
                        local_queue = isinstance(
                            follower, LocalAStarActionFollower
                        )
                        if local_queue and self.closed_loop_context is not None:
                            # An empty local queue is not proof of arrival.  Wait
                            # for the next RGB/VGGT update, which will either
                            # declare predicted arrival or produce a new queue.
                            self.state.update(
                                {
                                    "navigating": True,
                                    "last_action": None,
                                    "status": (
                                        "Closed-loop local queue complete; "
                                        "waiting for VGGT/Scale pose update"
                                    ),
                                }
                            )
                            next_action_time = loop_start + 1.0 / self.action_hz
                        else:
                            follower = None
                            self.closed_loop_context = None
                            self.state.update(
                                {
                                    "navigating": False,
                                    "last_action": None,
                                    "status": "Action queue complete",
                                }
                            )
                            if local_queue:
                                self._finish_open_loop_without_truth()
                            elif self.state["verifier"].get("last_plan"):
                                self.state["status"] = "Navigation goal reached"
                                self._update_verifier_execution_status("reached")
                    elif last_action is not None:
                        try:
                            observations = simulator.step({0: last_action})[0]
                        except Exception as error:
                            failed_action = str(last_action)
                            local_failure = isinstance(
                                follower, LocalAStarActionFollower
                            )
                            follower = None
                            self.closed_loop_context = None
                            self.state.update(
                                {
                                    "navigating": False,
                                    "last_action": None,
                                    "status": (
                                        f"Action {failed_action} failed without "
                                        f"stopping the simulator: {error}"
                                    ),
                                }
                            )
                            if self.state["verifier"].get("last_plan"):
                                self._update_verifier_execution_status(
                                    "failed",
                                    failure_code=(
                                        "local_action_execution_failed"
                                        if local_failure
                                        else "habitat_action_execution_failed"
                                    ),
                                    failure_reason=str(error),
                                )
                            last_action = None
                            continue
                        self.state["step_count"] = (
                            int(self.state.get("step_count", 0)) + 1
                        )
                        if isinstance(follower, LocalAStarActionFollower):
                            self._record_local_action(follower)
                            action_hz = STRICT_OPEN_LOOP_ACTION_HZ
                        else:
                            action_hz = self.action_hz
                        next_action_time = loop_start + 1.0 / action_hz

                self._update_pose_state(simulator, last_action)
                camera_jpeg = encode_jpeg(observations["camera_sensor"])
                if (
                    self.full_scene_obstacle_map is None
                    or self.full_scene_lower_bound is None
                ):
                    raise RuntimeError("full-scene obstacle map is unavailable")
                gt_complete_by_model = {}
                gt_observed_by_model = {}
                gt_guessed_by_model = {}
                for model_key, extent_m in MODEL_EXTENTS_METERS.items():
                    complete_square = render_ego_obstacle_map(
                        simulator,
                        full_scene_map=self.full_scene_obstacle_map,
                        lower_bound=self.full_scene_lower_bound,
                        source_meters_per_pixel=(
                            self.full_scene_meters_per_pixel
                        ),
                        size=self.bev_size,
                        extent=extent_m,
                    )
                    complete = render_fov_complete_map(
                        complete_square,
                        horizontal_fov_degrees=self.horizontal_fov_degrees,
                    )
                    observed = render_visibility_masked_map(
                        complete_square,
                        horizontal_fov_degrees=self.horizontal_fov_degrees,
                    )
                    # Both role GTs must partition the exact same FOV support
                    # used by complete GT.  The shadowcaster and polygon
                    # rasterizer use slightly different boundary conventions,
                    # so clip the visibility result explicitly before taking
                    # the reverse mask.
                    observed[complete == 112] = 112
                    guessed = np.full_like(complete, 112, dtype=np.uint8)
                    guessed_domain = (complete != 112) & (observed == 112)
                    guessed[guessed_domain] = complete[guessed_domain]
                    gt_complete_by_model[model_key] = complete
                    gt_observed_by_model[model_key] = observed
                    gt_guessed_by_model[model_key] = guessed
                selected_model_key = min(
                    MODEL_EXTENTS_METERS,
                    key=lambda key: abs(
                        MODEL_EXTENTS_METERS[key] - self.bev_extent
                    ),
                )
                ego_obstacle_map = gt_complete_by_model[selected_model_key]
                occlusion_map = gt_observed_by_model[selected_model_key]
                frame_extrinsic = capture_frame_extrinsic(simulator)
                bev_png = encode_png(occlusion_map)
                obstacle_png = encode_png(ego_obstacle_map)
                with self.condition:
                    self.camera_jpeg = camera_jpeg
                    self.bev_png = bev_png
                    self.obstacle_png = obstacle_png
                    self.bev_frame_extent = self.bev_extent
                    self.bev_frame_extrinsic = frame_extrinsic
                    frame_sequence = int(self.state.get("frame_seq", 0)) + 1
                    self.state["frame_seq"] = frame_sequence
                    self.condition.notify_all()
                motion_step = int(self.state.get("step_count", 0))
                if self.comparison.should_sample(motion_step):
                    # Build merged masked GT from exactly the same sampled RGB
                    # history used by the runtime. Newer frames overwrite older
                    # labels in overlap, matching the historical collector.
                    merged_source_complete = {}
                    merged_source_fov_complete = {}
                    merged_source_observed = {}
                    for model_key, extent_m in MODEL_EXTENTS_METERS.items():
                        merged_size = self.comparison.merged_output_sizes.get(
                            model_key, MERGED_GT_SIZE
                        )
                        complete_merged_source = render_ego_obstacle_map(
                            simulator,
                            full_scene_map=self.full_scene_obstacle_map,
                            lower_bound=self.full_scene_lower_bound,
                            source_meters_per_pixel=(
                                self.full_scene_meters_per_pixel
                            ),
                            size=merged_size,
                            extent=extent_m,
                        )
                        observed_merged_source = render_visibility_masked_map(
                            complete_merged_source,
                            horizontal_fov_degrees=(
                                self.horizontal_fov_degrees
                            ),
                        )
                        fov_complete_merged_source = render_fov_complete_map(
                            complete_merged_source,
                            horizontal_fov_degrees=(
                                self.horizontal_fov_degrees
                            ),
                        )
                        observed_merged_source[
                            fov_complete_merged_source == 112
                        ] = 112
                        merged_source_complete[model_key] = (
                            complete_merged_source
                        )
                        merged_source_fov_complete[model_key] = (
                            fov_complete_merged_source
                        )
                        merged_source_observed[model_key] = (
                            observed_merged_source
                        )
                    self.merged_gt_history.append(
                        {
                            "complete": merged_source_complete,
                            "fov_complete": merged_source_fov_complete,
                            "observed": merged_source_observed,
                            "extrinsic": frame_extrinsic,
                        }
                    )
                    self.merged_gt_history = self.merged_gt_history[
                        -self.comparison.max_history :
                    ]
                    gt_merged_complete_by_model = {}
                    gt_merged_visible_by_model = {}
                    gt_merged_observed_by_model = {}
                    for model_key, extent_m in MODEL_EXTENTS_METERS.items():
                        merged_size = self.comparison.merged_output_sizes.get(
                            model_key, MERGED_GT_SIZE
                        )
                        merged_extent = self.comparison.merged_extents_m.get(
                            model_key, MERGED_GT_EXTENT_METERS
                        )
                        accumulator = BEVAccumulator(
                            extent=extent_m,
                            size=merged_size,
                        )
                        for record in self.merged_gt_history:
                            accumulator.update_routing_geometry(
                                record["complete"][model_key],
                                record["fov_complete"][model_key],
                                record["observed"][model_key],
                                record["extrinsic"],
                            )
                        complete_merged, visible_merged = (
                            accumulator.render_routing_geometry(
                                frame_extrinsic, merged_extent
                            )
                        )
                        gt_merged_complete_by_model[model_key] = complete_merged
                        gt_merged_visible_by_model[model_key] = visible_merged
                        # Retain the historical field with its literal masked
                        # meaning for legacy pages.
                        gt_merged_observed_by_model[model_key] = visible_merged
                    # Only RGB is sent to the model. Simulator occupancy is
                    # copied solely for synchronized visualization.
                    self.comparison.submit(
                        ComparisonFrame(
                            frame_seq=frame_sequence,
                            motion_step=motion_step,
                            camera_rgb=np.asarray(
                                observations["camera_sensor"]
                            )[..., :3].copy(),
                            gt_complete_by_model={
                                key: value.copy()
                                for key, value in gt_complete_by_model.items()
                            },
                            gt_observed_by_model={
                                key: value.copy()
                                for key, value in gt_observed_by_model.items()
                            },
                            gt_guessed_by_model={
                                key: value.copy()
                                for key, value in gt_guessed_by_model.items()
                            },
                            gt_merged_observed_by_model={
                                key: value.copy()
                                for key, value in (
                                    gt_merged_observed_by_model.items()
                                )
                            },
                            gt_merged_complete_by_model={
                                key: value.copy()
                                for key, value in (
                                    gt_merged_complete_by_model.items()
                                )
                            },
                            gt_merged_visible_by_model={
                                key: value.copy()
                                for key, value in (
                                    gt_merged_visible_by_model.items()
                                )
                            },
                            extrinsic=json.loads(json.dumps(frame_extrinsic)),
                        )
                    )

                elapsed = time.monotonic() - loop_start
                self.stop_event.wait(max(0.0, frame_interval - elapsed))
        except Exception as error:
            with self.condition:
                self.state.update(
                    {"ready": False, "status": f"Engine error: {error}"}
                )
                self.condition.notify_all()
            self.ready_event.set()
        finally:
            if simulator is not None:
                simulator.close()


def make_handler(
    engine: NavigationEngine,
    html: bytes,
    *,
    scene_choices: Tuple[SceneChoice, ...] = (),
    active_scene_id: Optional[str] = None,
    request_scene_switch: Optional[Callable[[str], None]] = None,
) -> type[BaseHTTPRequestHandler]:
    choices_by_id = {choice.scene_id: choice for choice in scene_choices}

    class NavigationRequestHandler(BaseHTTPRequestHandler):
        server_version = "VGGTBEVCompare/0.5"

        def _send_bytes(
            self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(
            self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            self._send_bytes(
                json.dumps(payload).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_bytes(html, "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send_json(engine.snapshot())
            elif path == "/api/scenes":
                self._send_json(
                    {
                        "active_scene_id": active_scene_id,
                        "scene_count": len(scene_choices),
                        "source_split": "validation",
                        "training_scene_overlap": 0,
                        "scenes": [
                            {"id": choice.scene_id, "label": choice.label}
                            for choice in scene_choices
                        ],
                    }
                )
            elif path == "/api/map.png":
                if engine.map_png is None:
                    self._send_json(
                        {"error": "map not ready"}, HTTPStatus.SERVICE_UNAVAILABLE
                    )
                else:
                    self._send_bytes(engine.map_png, "image/png")
            elif path == "/api/camera.jpg":
                image = engine.latest_image("camera")
                if image is None:
                    self._send_json(
                        {"error": "frame not ready"}, HTTPStatus.SERVICE_UNAVAILABLE
                    )
                else:
                    self._send_bytes(image, "image/jpeg")
            elif path == "/api/bev-pair":
                sequence, extent, extrinsic, masked, complete = (
                    engine.latest_bev_pair()
                )
                if masked is None or complete is None or extrinsic is None:
                    self._send_json(
                        {"error": "BEV frame pair not ready"},
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                else:
                    self._send_json(
                        {
                            "frame_seq": sequence,
                            "bev_extent_m": extent,
                            "extrinsic": extrinsic,
                            "masked_png_base64": base64.b64encode(masked).decode(
                                "ascii"
                            ),
                            "complete_png_base64": base64.b64encode(
                                complete
                            ).decode("ascii"),
                        }
                    )
            elif path == "/api/model-comparison":
                comparison = engine.latest_model_comparison()
                models = comparison.get("models")
                synchronized = isinstance(models, dict)
                if synchronized:
                    for model_key in engine.comparison.model_extents_m:
                        model = models.get(model_key)
                        if not isinstance(model, dict):
                            synchronized = False
                            break
                        mode = model.get("visualization_mode")
                        if mode == "merged_routing_geometry":
                            required = (
                                "gt_fov_support_png_base64",
                                "gt_observed_gate_png_base64",
                                "predicted_fov_support_probability_png_base64",
                                "predicted_fov_support_binary_png_base64",
                                "predicted_observed_gate_probability_png_base64",
                                "predicted_observed_gate_binary_png_base64",
                            )
                        else:
                            required = (
                                "gt_complete_png_base64",
                                "gt_observed_png_base64",
                                "gt_guessed_png_base64",
                                "predicted_png_base64",
                            )
                        if mode == "legacy_masked_dual":
                            required += (
                                "gt_merged_observed_png_base64",
                                "predicted_merged_png_base64",
                            )
                        elif mode != "merged_routing_geometry":
                            required += (
                                "observed_gate_confidence_png_base64",
                                "guessed_occupancy_confidence_png_base64",
                            )
                        if not all(model.get(field) for field in required):
                            synchronized = False
                            break
                if not synchronized:
                    self._send_json(
                        comparison,
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                else:
                    self._send_json(comparison)
            elif path == "/api/bev.png":
                image = engine.latest_image("bev")
                if image is None:
                    self._send_json(
                        {"error": "frame not ready"}, HTTPStatus.SERVICE_UNAVAILABLE
                    )
                else:
                    self._send_bytes(image, "image/png")
            elif path == "/api/obstacle.png":
                image = engine.latest_image("obstacle")
                if image is None:
                    self._send_json(
                        {"error": "frame not ready"}, HTTPStatus.SERVICE_UNAVAILABLE
                    )
                else:
                    self._send_bytes(image, "image/png")
            elif path == "/stream/camera.mjpg":
                self._stream_images("camera", "image/jpeg")
            elif path == "/stream/bev.mpng":
                self._stream_images("bev", "image/png")
            elif path == "/stream/obstacle.mpng":
                self._stream_images("obstacle", "image/png")
            elif path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 65536:
                    raise ValueError("request is too large")
                payload = json.loads(self.rfile.read(length) or b"{}")
                if path == "/api/verifier/plan":
                    result = engine.plan_verifier_goal(payload)
                    self._send_json(result, HTTPStatus.OK)
                    return
                elif path == "/api/goal":
                    raise ValueError("click-to-navigate is disabled; use WASD")
                elif path == "/api/control":
                    raise ValueError("automatic navigation controls are disabled")
                elif path == "/api/manual":
                    command: Dict[str, Any] = {"type": "manual_control"}
                    if "enabled" in payload:
                        if not isinstance(payload["enabled"], bool):
                            raise ValueError("enabled must be a boolean")
                        command["enabled"] = payload["enabled"]
                    if "keys" in payload:
                        keys = payload["keys"]
                        if not isinstance(keys, dict):
                            raise ValueError("keys must be an object")
                        allowed_keys = {"forward", "backward", "left", "right"}
                        if set(keys) != allowed_keys:
                            raise ValueError(
                                "keys must contain forward, backward, left, and right"
                            )
                        if not all(
                            isinstance(value, bool) for value in keys.values()
                        ):
                            raise ValueError("all key states must be boolean")
                        command["keys"] = keys
                    if len(command) == 1:
                        raise ValueError("enabled or keys is required")
                    engine.submit(command)
                elif path == "/api/scene":
                    if request_scene_switch is None or not scene_choices:
                        raise ValueError("scene switching is not configured")
                    scene_id = str(payload.get("scene_id", ""))
                    if scene_id not in choices_by_id:
                        raise ValueError("scene_id is not in the local 10-scene catalog")
                    self._send_json(
                        {
                            "accepted": True,
                            "scene_id": scene_id,
                            "restarting_simulator": scene_id != active_scene_id,
                        },
                        HTTPStatus.ACCEPTED,
                    )
                    if scene_id != active_scene_id:
                        request_scene_switch(scene_id)
                    return
                elif path == "/api/config":
                    extent = float(payload["bev_extent_m"])
                    if not np.isfinite(extent) or not 0.5 <= extent <= 30.0:
                        raise ValueError(
                            "bev_extent_m must be between 0.5 and 30.0 metres"
                        )
                    if abs(extent - engine.bev_extent) > 1e-6:
                        raise ValueError(
                            "BEV extent is fixed by the selected trained model"
                        )
                else:
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_json({"accepted": True}, HTTPStatus.ACCEPTED)
            except VerifierPlanningError as error:
                self._send_json(
                    error.payload(), HTTPStatus.UNPROCESSABLE_ENTITY
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(
                    {"error": str(error)}, HTTPStatus.BAD_REQUEST
                )

        def _stream_images(self, kind: str, image_content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            last_sequence = -1
            try:
                while not engine.stop_event.is_set():
                    sequence, image = engine.wait_for_image(kind, last_sequence)
                    if image is None or sequence == last_sequence:
                        continue
                    last_sequence = sequence
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(
                        f"Content-Type: {image_content_type}\r\n".encode("ascii")
                    )
                    self.wfile.write(
                        f"Content-Length: {len(image)}\r\n\r\n".encode("ascii")
                    )
                    self.wfile.write(image)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return

        def log_message(self, format_string: str, *args: Any) -> None:
            if "/api/state" not in str(args[0]):
                super().log_message(format_string, *args)

    return NavigationRequestHandler


def main() -> None:
    args = parse_args()
    if args.interactive_verifier and args.merged_routing_visualizer:
        raise SystemExit(
            "--interactive-verifier cannot be combined with "
            "--merged-routing-visualizer"
        )
    model_extent = MODEL_EXTENTS_METERS[args.model]
    if (
        args.bev_extent is not None
        and abs(args.bev_extent - model_extent) > 1e-6
    ):
        raise SystemExit(
            f"--model {args.model} requires --bev-extent {model_extent:g}"
        )
    initial_scene = args.scene.expanduser().resolve()
    if not initial_scene.is_file():
        raise SystemExit(f"Scene does not exist: {initial_scene}")
    initial_navmesh = (
        args.navmesh.expanduser().resolve()
        if args.navmesh is not None
        else initial_scene.with_suffix(".navmesh")
    )
    scene_dataset_config = (
        args.scene_dataset_config.expanduser().resolve()
        if args.scene_dataset_config is not None
        else None
    )
    if (
        scene_dataset_config is not None
        and not scene_dataset_config.is_file()
    ):
        raise SystemExit(
            f"Scene dataset config does not exist: {scene_dataset_config}"
        )
    model_keys = (
        tuple(MODEL_EXTENTS_METERS)
        if args.all_model_comparison
        else (args.model,)
    )
    html_file = (
        MERGED_ROUTING_HTML_FILE
        if args.merged_routing_visualizer
        else HTML_FILE
    )
    if not html_file.is_file():
        raise SystemExit(f"Web UI file is missing: {html_file}")
    html = html_file.read_bytes()
    if args.scene_catalog is None:
        scene_choices = (
            SceneChoice(
                scene_id=f"local:{initial_scene.parent.name}",
                label=initial_scene.parent.name,
                scene=initial_scene,
                navmesh=initial_navmesh,
            ),
        )
        switching_enabled = False
    else:
        scene_choices = load_scene_catalog(args.scene_catalog)
        switching_enabled = True
    choices_by_id = {choice.scene_id: choice for choice in scene_choices}
    current = next(
        (choice for choice in scene_choices if choice.scene == initial_scene),
        scene_choices[0],
    )
    url = f"http://{args.host}:{args.port}"
    browser_started = False

    while True:
        engine = NavigationEngine(
            scene=current.scene,
            navmesh=current.navmesh,
            scene_dataset_config=scene_dataset_config,
            gpu_device_id=args.gpu_device_id,
            model_keys=model_keys,
            start=None if args.random_start else tuple(args.start),
            start_yaw_degrees=(
                None if args.random_yaw else args.start_yaw_degrees
            ),
            seed=args.seed,
            fps=args.fps,
            action_hz=args.action_hz,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            sensor_height_m=args.sensor_height_m,
            horizontal_fov_degrees=args.horizontal_fov_degrees,
            bev_size=args.bev_size,
            bev_extent=model_extent,
            obstacle_min_height=args.obstacle_min_height,
            obstacle_max_height=args.obstacle_max_height,
            voxel_size=args.voxel_size,
            manual_clearance=args.manual_clearance,
            occlusion_rays=args.occlusion_rays,
            model_server_url=args.model_server_url,
            model_hz=args.model_hz,
            model_max_history=args.model_max_history,
            verifier_enabled=args.interactive_verifier,
        )
        requested_scene: list[str] = []
        server_holder: list[ReusableThreadingHTTPServer] = []

        def request_scene_switch(scene_id: str) -> None:
            if requested_scene:
                return
            requested_scene.append(scene_id)

            def stop_server() -> None:
                with engine.condition:
                    engine.state["status"] = (
                        f"Loading local scene {choices_by_id[scene_id].label}…"
                    )
                    engine.condition.notify_all()
                if server_holder:
                    server_holder[0].shutdown()

            threading.Thread(target=stop_server, daemon=True).start()

        engine.start_engine()
        server: Optional[ReusableThreadingHTTPServer] = None
        interrupted = False
        try:
            engine.wait_until_ready(timeout=300.0)
            with engine.condition:
                engine.state["session"].update(
                    {
                        "scene_catalog_id": current.scene_id,
                        "scene_label": current.label,
                        "source_split": "validation",
                        "training_scene_overlap": 0,
                    }
                )
            handler = make_handler(
                engine,
                html,
                scene_choices=scene_choices if switching_enabled else (),
                active_scene_id=current.scene_id,
                request_scene_switch=(
                    request_scene_switch if switching_enabled else None
                ),
            )
            server = ReusableThreadingHTTPServer((args.host, args.port), handler)
            server.daemon_threads = True
            server_holder.append(server)
            print(
                f"VGGTBEV comparison UI: {url} "
                + (
                    "(P1B merged routing geometry from runtime grid contract)"
                    if args.merged_routing_visualizer
                    else (
                        "(P1B verifier: GT display only -> ego-local target -> "
                        "predicted A*; no simulator truth; inflation=0-50cm)"
                        if args.interactive_verifier
                        else f"(P1B single={model_extent:g}m + metric scale; merged disabled)"
                    )
                )
            )
            snapshot = engine.snapshot()
            session = snapshot["session"]
            camera = snapshot["camera"]
            print(
                "Session: "
                f"scene={current.scene_id}, seed={session['seed']}, "
                f"height={camera['sensor_height_m']:.3f}m, "
                f"hfov={camera['hfov_degrees']:.2f}deg, "
                f"yaw={session['start_yaw_degrees']:.2f}deg"
            )
            if switching_enabled:
                print(
                    f"Local scene selector: {len(scene_choices)} validation-only scenes"
                )
            print("WASD control is active. Press Ctrl+C to stop.")
            if not args.no_browser and not browser_started:
                threading.Timer(0.4, lambda: webbrowser.open(url)).start()
                browser_started = True
            try:
                server.serve_forever(poll_interval=0.2)
            except KeyboardInterrupt:
                interrupted = True
                print("\nStopping VGGNAV…")
        finally:
            if server is not None:
                server.server_close()
            engine.stop_engine()

        if interrupted or not requested_scene:
            break
        current = choices_by_id[requested_scene[0]]
        print(f"Switching to local scene: {current.scene_id}", flush=True)


if __name__ == "__main__":
    main()
