#!/usr/bin/env python3
"""Interactive Habitat UI comparing simulator-truth and Method II BEVs."""

import argparse
import base64
import io
import json
import math
import queue
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import habitat_sim
import numpy as np
import quaternion
from habitat_sim.utils.common import quat_from_magnum, quat_to_magnum
from PIL import Image

from collision_voxel import voxelize_stage_occupancy
from model_comparison import ComparisonFrame, ModelComparisonWorker
from point_navigation import PointNavigationError, PointNavigator


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = Path(
    "/media/user/T9/scene_datasets/hm3d/val/"
    "00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
)
DEFAULT_START = (-1.7062, 0.1634, -2.3114)
HTML_FILE = ROOT / "web_ui/index.html"
DEFAULT_CAMERA_SENSOR_HEIGHT_METERS = 0.35
DEFAULT_HORIZONTAL_FOV_DEGREES = 90.0
GLOBAL_MAP_METERS_PER_PIXEL = 0.05
DEFAULT_VOXEL_SIZE_METERS = 0.01
MANUAL_LINEAR_SPEED_METERS_PER_SECOND = 0.4
MANUAL_ANGULAR_SPEED_RADIANS_PER_SECOND = 0.3
MANUAL_MAX_INTEGRATION_STEP_SECONDS = 0.25
DEFAULT_MANUAL_NAVMESH_CLEARANCE_METERS = 0.15
MODEL_EXTENTS_METERS = {"3p5m": 3.5, "5m": 5.0, "6p5m": 6.5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the VGGNAV click-to-navigate browser interface"
    )
    parser.add_argument("scene", type=Path, nargs="?", default=DEFAULT_SCENE)
    parser.add_argument("--navmesh", type=Path)
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
        default="5m",
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
    return parser.parse_args()


def make_web_simulator(
    scene: Path,
    scene_dataset_config: Optional[Path],
    gpu_device_id: int,
    camera_width: int,
    camera_height: int,
    sensor_height_m: float,
    horizontal_fov_degrees: float,
) -> habitat_sim.Simulator:
    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(scene)
    if scene_dataset_config is not None:
        simulator_config.scene_dataset_config_file = str(scene_dataset_config)
    simulator_config.enable_physics = True
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

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [camera]
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
        self.manual_enabled = False
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
                "enabled": False,
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
        follower: Optional[habitat_sim.nav.GreedyGeodesicFollower],
    ) -> Optional[habitat_sim.nav.GreedyGeodesicFollower]:
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                break
            command_type = command.get("type")
            if command_type == "stop":
                return follower
            if command_type == "goal":
                self._set_manual_enabled(False)
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
                except PointNavigationError as error:
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
                    follower = None
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
                    "status": "Click a navigable point to begin",
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
            follower: Optional[
                habitat_sim.nav.GreedyGeodesicFollower
            ] = None
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
                    except habitat_sim.errors.GreedyFollowerError:
                        follower = None
                        self.state.update(
                            {
                                "navigating": False,
                                "last_action": None,
                                "status": "Navigation failed: no valid action",
                            }
                        )
                    if last_action is None and follower is not None:
                        follower = None
                        self.state.update(
                            {
                                "navigating": False,
                                "last_action": None,
                                "status": "Goal reached — click another point",
                            }
                        )
                    elif last_action is not None:
                        observations = simulator.step({0: last_action})[0]
                        self.state["step_count"] = (
                            int(self.state.get("step_count", 0)) + 1
                        )
                        next_action_time = loop_start + 1.0 / self.action_hz

                self._update_pose_state(simulator, last_action)
                camera_jpeg = encode_jpeg(observations["camera_sensor"])
                if (
                    self.full_scene_obstacle_map is None
                    or self.full_scene_lower_bound is None
                ):
                    raise RuntimeError("full-scene obstacle map is unavailable")
                gt_complete_by_model = {}
                gt_masked_by_model = {}
                for model_key, extent_m in MODEL_EXTENTS_METERS.items():
                    complete = render_ego_obstacle_map(
                        simulator,
                        full_scene_map=self.full_scene_obstacle_map,
                        lower_bound=self.full_scene_lower_bound,
                        source_meters_per_pixel=(
                            self.full_scene_meters_per_pixel
                        ),
                        size=self.bev_size,
                        extent=extent_m,
                    )
                    gt_complete_by_model[model_key] = complete
                    gt_masked_by_model[model_key] = (
                        render_visibility_masked_map(
                            complete,
                            horizontal_fov_degrees=(
                                self.horizontal_fov_degrees
                            ),
                        )
                    )
                selected_model_key = min(
                    MODEL_EXTENTS_METERS,
                    key=lambda key: abs(
                        MODEL_EXTENTS_METERS[key] - self.bev_extent
                    ),
                )
                ego_obstacle_map = gt_complete_by_model[selected_model_key]
                occlusion_map = gt_masked_by_model[selected_model_key]
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
                self.comparison.submit(
                    ComparisonFrame(
                        frame_seq=frame_sequence,
                        motion_step=int(self.state.get("step_count", 0)),
                        camera_rgb=np.asarray(
                            observations["camera_sensor"]
                        )[..., :3].copy(),
                        gt_masked_by_model={
                            key: value.copy()
                            for key, value in gt_masked_by_model.items()
                        },
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
    engine: NavigationEngine, html: bytes
) -> type[BaseHTTPRequestHandler]:
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
                synchronized = isinstance(models, dict) and all(
                    isinstance(models.get(model_key), dict)
                    and "gt_png_base64" in models[model_key]
                    and "predicted_png_base64" in models[model_key]
                    for model_key in engine.comparison.model_extents_m
                )
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
                if path == "/api/goal":
                    x = float(payload["x"])
                    z = float(payload["z"])
                    if not np.isfinite(x) or not np.isfinite(z):
                        raise ValueError("goal coordinates must be finite")
                    engine.submit({"type": "goal", "x": x, "z": z})
                elif path == "/api/control":
                    action = str(payload.get("action", ""))
                    if action not in {"pause", "resume", "cancel"}:
                        raise ValueError("unknown control action")
                    engine.submit({"type": action})
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
    model_extent = MODEL_EXTENTS_METERS[args.model]
    if (
        args.bev_extent is not None
        and abs(args.bev_extent - model_extent) > 1e-6
    ):
        raise SystemExit(
            f"--model {args.model} requires --bev-extent {model_extent:g}"
        )
    scene = args.scene.expanduser().resolve()
    if not scene.is_file():
        raise SystemExit(f"Scene does not exist: {scene}")
    navmesh = (
        args.navmesh.expanduser().resolve()
        if args.navmesh is not None
        else scene.with_suffix(".navmesh")
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
    if not HTML_FILE.is_file():
        raise SystemExit(f"Web UI file is missing: {HTML_FILE}")
    html = HTML_FILE.read_bytes()

    engine = NavigationEngine(
        scene=scene,
        navmesh=navmesh,
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
    )
    engine.start_engine()
    try:
        engine.wait_until_ready(timeout=300.0)
        handler = make_handler(engine, html)
        server = ThreadingHTTPServer((args.host, args.port), handler)
        server.daemon_threads = True
        url = f"http://{args.host}:{args.port}"
        print(
            f"VGGTBEV comparison UI: {url} "
            f"(single={model_extent:g}m, merged=8m)"
        )
        snapshot = engine.snapshot()
        session = snapshot["session"]
        camera = snapshot["camera"]
        print(
            "Session: "
            f"scene={session['scene_name']}, seed={session['seed']}, "
            f"height={camera['sensor_height_m']:.3f}m, "
            f"hfov={camera['hfov_degrees']:.2f}deg, "
            f"yaw={session['start_yaw_degrees']:.2f}deg"
        )
        print("Click the global map to set a goal. Press Ctrl+C to stop.")
        if not args.no_browser:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            print("\nStopping VGGNAV…")
        finally:
            server.server_close()
    finally:
        engine.stop_engine()


if __name__ == "__main__":
    main()
