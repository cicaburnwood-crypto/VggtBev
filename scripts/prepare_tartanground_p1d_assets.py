#!/usr/bin/env python3
"""Convert TartanGround global semantic assets into P1D BEV sessions.

Unlike ``prepare_tartanground_p1d_compat.py``, this converter never uses depth
to reconstruct the complete or guessed BEV target.  Complete occupancy comes
from the official per-environment ``<env>_sem.pcd`` asset.  Metric depth is
copied only for the independent Scale-Token supervision contract.

The semantic point cloud is projected in the camera/robot frame.  Traversable
surface classes establish free-space support; non-traversable surfaces inside
the configured robot-height obstacle band establish occupied cells.  Exact
symmetric grid shadowcasting then derives the observed target from that same
complete raster, so observed labels are always a strict subset of complete GT.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from compact_storage import (
    DEPTH_INVALID_Q,
    DEPTH_SCALE_M,
    JPEG_QUALITY,
    load_depth_compact,
    save_bev_jpeg,
    save_depth_compact,
    save_rgb_jpeg,
    snap_bev_palette,
)

try:
    import torch
except ImportError:  # CPU-only conversion remains available.
    torch = None

from prepare_tartanground_p1d_compat import (
    CAMERA,
    CX,
    CY,
    FX,
    FY,
    HEIGHT,
    HFOV_DEG,
    OCCUPIED,
    UNKNOWN,
    WIDTH,
    FREE,
    boundary,
    choose_window,
    decode_depth,
    fov_union,
    member_map,
    metric_to_pixel,
    palette,
    quaternion_matrix,
    read_metadata,
    save_label,
    world_from_bev_planar,
)


DEFAULT_ROUTES = (
    "AbandonedFactory/Data_omni/P0000",
    "Hospital/Data_diff/P1000",
    "House/Data_omni/P0000",
    "Office/Data_omni/P0000",
    "Supermarket/Data_diff/P1000",
)

FLAT_TRAVERSABLE_CLASSES = frozenset(
    {
        "floor",
        "ground",
        "carpet",
        "doormat",
        "wooddeck",
        "sidewalk",
        "road",
        "roadway",
        "floorrug",
    }
)
TERRAIN_TRAVERSABLE_CLASSES = frozenset({"stairs", "stair", "ramp"})
# Several TartanGround environments label the entire static shell, including
# both floor tiles and walls, as ``building``.  Its near-ground points can seed
# free space, but the class as a whole must not be declared traversable.
AMBIGUOUS_GROUND_CLASSES = frozenset({"building"})
IGNORED_GEOMETRY_SUBSTRINGS = ("ceiling", "sky")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--seg-rgb-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--route", action="append", default=[])
    parser.add_argument(
        "--exclude-scene",
        action="append",
        default=[],
        help="exclude a scene from all-route/one-route-per-scene recovery",
    )
    parser.add_argument(
        "--all-routes",
        action="store_true",
        help="enumerate every structurally complete route whose scene asset exists",
    )
    parser.add_argument(
        "--one-route-per-scene",
        action="store_true",
        help=(
            "select one deterministic structurally complete route for every "
            "scene whose global semantic asset exists"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="rasterization device, for example cuda:0 (default: cpu)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="preserve the output root and skip already COMPLETE sessions",
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument(
        "--window-attempts-per-route",
        type=int,
        default=1,
        help=(
            "try this many ranked temporal windows in each route before "
            "rejecting it (useful when completing one valid sample per scene)"
        ),
    )
    parser.add_argument(
        "--window-start-rank",
        type=int,
        default=0,
        help="start candidate-window rank (rank 0 is the legacy best window)",
    )
    parser.add_argument("--obstacle-min-height-m", type=float, default=0.0)
    parser.add_argument("--obstacle-max-height-m", type=float, default=1.4)
    parser.add_argument("--surface-height-tolerance-m", type=float, default=2.0)
    parser.add_argument("--ground-surface-tolerance-m", type=float, default=0.45)
    parser.add_argument(
        "--minimum-observed-known-pixels",
        type=int,
        default=1024,
        help="minimum known pixels required in every Single and Merged Masked frame",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ranked_windows(
    frame_ids: list[int],
    poses: np.ndarray,
    count: int,
    step: int,
) -> list[list[int]]:
    """Return deterministic candidate windows in the converter's quality order.

    ``choose_window`` historically returned only the best kinematic window.
    Asset support varies spatially, so a valid scene can have one unsupported
    window and another fully trainable one. Ranking all windows lets the caller
    recover that sample without weakening any GT-quality gate.
    """

    available = set(frame_ids)
    stride = max(1, len(frame_ids) // 300)
    candidates: list[tuple[float, int, list[int]]] = []
    for start in frame_ids[::stride]:
        ids = [start + offset * step for offset in range(count)]
        if ids[-1] >= poses.shape[0] or any(frame not in available for frame in ids):
            continue
        xy = poses[ids, :2]
        path_length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        yaw = np.unwrap(
            [
                math.atan2(
                    quaternion_matrix(poses[frame, 3:])[1, 0],
                    quaternion_matrix(poses[frame, 3:])[0, 0],
                )
                for frame in ids
            ]
        )
        yaw_travel = float(np.abs(np.diff(yaw)).sum())
        if path_length > 8.0:
            continue
        score = -abs(path_length - 3.0) + min(yaw_travel, math.pi) * 0.35
        candidates.append((score, start, ids))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    if candidates:
        return [ids for _, _, ids in candidates]
    # Preserve the legacy deterministic fallback and its exact validation.
    return [choose_window(frame_ids, poses, count, step)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pcd_header(path: Path) -> tuple[int, int]:
    offset = 0
    points = None
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"missing DATA line in {path}")
            offset += len(line)
            text = line.decode("ascii").strip()
            if text.startswith("POINTS "):
                points = int(text.split()[1])
            if text == "DATA binary":
                break
            if text.startswith("DATA "):
                raise ValueError(f"only binary PCD is supported: {text}")
    if points is None:
        raise ValueError(f"missing POINTS count in {path}")
    return offset, points


def packed_palette(path: Path) -> np.ndarray:
    colors = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [int(value.strip()) for value in line.split(",")]
        if len(values) != 3:
            raise ValueError(f"invalid RGB palette row: {line!r}")
        # Official occupancy utility reverses each source RGB row before
        # matching Open3D's decoded PCD color.  PCD stores r<<16|g<<8|b.
        r, g, b = values[::-1]
        colors.append((r << 16) | (g << 8) | b)
    if len(colors) != 256:
        raise ValueError(f"expected 256 semantic colors, got {len(colors)}")
    return np.asarray(colors, dtype=np.uint32)


def load_scene_label_ids(asset_dir: Path) -> dict[str, int]:
    archive_path = asset_dir / "seg_labels.zip"
    with zipfile.ZipFile(archive_path) as archive:
        payload = json.loads(archive.read("seg_label_map.json"))
    return {str(name): int(index) for name, index in payload["name_map"].items()}


class SemanticAsset:
    """Memory-mapped official semantic PCD with route-local preselection."""

    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]
    )

    def __init__(
        self,
        asset_dir: Path,
        scene: str,
        palette: np.ndarray,
        route_poses: np.ndarray,
        *,
        margin_m: float = 8.0,
        device: str = "cpu",
    ) -> None:
        self.path = asset_dir / f"{scene}_sem.pcd"
        offset, count = parse_pcd_header(self.path)
        cloud = np.memmap(self.path, dtype=self.dtype, mode="r", offset=offset, shape=(count,))
        xmin = float(route_poses[:, 0].min() - margin_m)
        xmax = float(route_poses[:, 0].max() + margin_m)
        ymin = float(route_poses[:, 1].min() - margin_m)
        ymax = float(route_poses[:, 1].max() + margin_m)
        keep = (
            (cloud["x"] >= xmin)
            & (cloud["x"] <= xmax)
            & (cloud["y"] >= ymin)
            & (cloud["y"] <= ymax)
        )
        self.xyz = np.column_stack((cloud["x"][keep], cloud["y"][keep], cloud["z"][keep])).astype(
            np.float32, copy=False
        )
        packed = np.asarray(cloud["rgb"][keep], dtype=np.uint32)
        color_to_id = {int(color): index for index, color in enumerate(palette.tolist())}
        self.class_id = np.fromiter(
            (color_to_id.get(int(color), 0) for color in packed),
            dtype=np.int16,
            count=packed.size,
        )
        self.source_point_count = int(count)
        self.selected_point_count = int(self.xyz.shape[0])
        self.device = device
        self.gpu_xyz: Any | None = None
        self.gpu_class_id: Any | None = None
        if device != "cpu":
            if torch is None:
                raise RuntimeError("--device requires PyTorch")
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is unavailable for requested device {device}")
            self.gpu_xyz = torch.as_tensor(self.xyz, device=device)
            self.gpu_class_id = torch.as_tensor(
                self.class_id.astype(np.int64, copy=False), device=device
            )

    def _gpu_isin(self, values: Any, ids: np.ndarray) -> Any:
        if ids.size == 0:
            return torch.zeros_like(values, dtype=torch.bool)
        return torch.isin(
            values,
            torch.as_tensor(ids.astype(np.int64, copy=False), device=self.device),
        )

    def estimate_ground_down(
        self,
        pose: np.ndarray,
        *,
        flat_traversable_ids: np.ndarray,
        ambiguous_ground_ids: np.ndarray,
        radius_m: float = 2.5,
    ) -> float:
        """Find the current asset floor below the camera, including multilevel scenes.

        TartanGround's route ``robot_height`` is platform metadata and is not
        always the front optical-center height.  A dense horizontal semantic
        surface produces a sharp local-height mode, whereas walls contribute a
        broad distribution.  Selecting the strongest positive mode below the
        camera also rejects floors from levels above and below the route.
        """

        rotation = quaternion_matrix(pose[3:]).astype(np.float32)
        local = (self.xyz - pose[None, :3]) @ rotation
        radial = np.hypot(local[:, 0], local[:, 1]) <= radius_m
        ground_class = np.isin(
            self.class_id,
            np.concatenate((flat_traversable_ids, ambiguous_ground_ids)),
        )
        down = local[radial & ground_class, 2]
        down = down[np.isfinite(down) & (down >= 0.15) & (down <= 3.0)]
        if down.size < 100:
            raise ValueError("insufficient asset ground support below camera")
        edges = np.arange(0.15, 3.001, 0.02, dtype=np.float32)
        histogram, edges = np.histogram(down, edges)
        index = int(histogram.argmax())
        mode = 0.5 * float(edges[index] + edges[index + 1])
        support = down[np.abs(down - mode) <= 0.04]
        if support.size < 50:
            raise ValueError("asset ground-height mode has insufficient support")
        return float(np.median(support))

    def rasterize(
        self,
        pose: np.ndarray,
        *,
        ground_down_m: float,
        extent_m: float,
        flat_traversable_ids: np.ndarray,
        terrain_traversable_ids: np.ndarray,
        ambiguous_ground_ids: np.ndarray,
        ignored_ids: np.ndarray,
        obstacle_min_height_m: float,
        obstacle_max_height_m: float,
        surface_height_tolerance_m: float,
        ground_surface_tolerance_m: float,
        size: int = 512,
    ) -> np.ndarray:
        if self.gpu_xyz is not None:
            return self._rasterize_gpu(
                pose,
                ground_down_m=ground_down_m,
                extent_m=extent_m,
                flat_traversable_ids=flat_traversable_ids,
                terrain_traversable_ids=terrain_traversable_ids,
                ambiguous_ground_ids=ambiguous_ground_ids,
                ignored_ids=ignored_ids,
                obstacle_min_height_m=obstacle_min_height_m,
                obstacle_max_height_m=obstacle_max_height_m,
                surface_height_tolerance_m=surface_height_tolerance_m,
                ground_surface_tolerance_m=ground_surface_tolerance_m,
                size=size,
            )
        rotation = quaternion_matrix(pose[3:]).astype(np.float32)
        local = (self.xyz - pose[None, :3]) @ rotation
        # TartanGround local axes are forward/right/down.  BEV axes are
        # right/forward, with forward rendered upward.
        forward = local[:, 0]
        right = local[:, 1]
        down = local[:, 2]
        half = extent_m / 2.0
        spatial = (
            (right >= -half)
            & (right < half)
            & (forward >= -half)
            & (forward < half)
        )
        local = local[spatial]
        classes = self.class_id[spatial]
        forward = local[:, 0]
        right = local[:, 1]
        down = local[:, 2]
        near_ground = np.abs(down - ground_down_m) <= ground_surface_tolerance_m
        flat_semantic = np.isin(classes, flat_traversable_ids)
        terrain_semantic = np.isin(classes, terrain_traversable_ids)
        ambiguous_ground = np.isin(classes, ambiguous_ground_ids)
        semantic_traversable = flat_semantic | terrain_semantic
        traversable = (
            (flat_semantic & near_ground)
            | (ambiguous_ground & near_ground)
            | (
                terrain_semantic
                & (np.abs(down - ground_down_m) <= surface_height_tolerance_m)
            )
        )
        ignored = np.isin(classes, ignored_ids)
        height_above_ground = ground_down_m - down
        obstacle = (
            ~ignored
            & ~semantic_traversable
            & ~(ambiguous_ground & near_ground)
            & (height_above_ground >= obstacle_min_height_m)
            & (height_above_ground <= obstacle_max_height_m)
        )

        free_seed = np.zeros((size, size), dtype=np.uint8)
        occupied_seed = np.zeros_like(free_seed)
        metric = np.column_stack((right, forward))
        rows, columns = metric_to_pixel(metric, extent_m, size)
        in_bounds = (rows >= 0) & (rows < size) & (columns >= 0) & (columns < size)
        free_index = in_bounds & traversable
        occupied_index = in_bounds & obstacle
        free_seed[rows[free_index], columns[free_index]] = 1
        occupied_seed[rows[occupied_index], columns[occupied_index]] = 1

        # The official global cloud is a surface sampling, not a solid mesh.
        # Close only sub-sampling holes at the native PCD spacing.  No semantic
        # completion, depth fusion, or large-area topology filling is applied.
        kernel = np.ones((3, 3), dtype=np.uint8)
        free = cv2.morphologyEx(free_seed, cv2.MORPH_CLOSE, kernel, iterations=1)
        free = cv2.dilate(free, kernel, iterations=1)
        occupied = cv2.morphologyEx(occupied_seed, cv2.MORPH_CLOSE, kernel, iterations=1)
        occupied = cv2.dilate(occupied, kernel, iterations=1)
        labels = np.full((size, size), UNKNOWN, dtype=np.uint8)
        labels[free.astype(bool)] = FREE
        labels[occupied.astype(bool)] = OCCUPIED

        # A recorded robot pose is guaranteed collision-free.  This tiny
        # anchor only protects visibility connectivity against point sampling
        # holes; it cannot erase a real obstacle away from the pose center.
        center = size // 2
        labels[center - 1 : center + 2, center - 1 : center + 2] = FREE
        return labels

    def _rasterize_gpu(
        self,
        pose: np.ndarray,
        *,
        ground_down_m: float,
        extent_m: float,
        flat_traversable_ids: np.ndarray,
        terrain_traversable_ids: np.ndarray,
        ambiguous_ground_ids: np.ndarray,
        ignored_ids: np.ndarray,
        obstacle_min_height_m: float,
        obstacle_max_height_m: float,
        surface_height_tolerance_m: float,
        ground_surface_tolerance_m: float,
        size: int,
    ) -> np.ndarray:
        """CUDA implementation of the expensive transform/filter/scatter path."""

        assert torch is not None and self.gpu_xyz is not None and self.gpu_class_id is not None
        with torch.inference_mode():
            rotation = torch.as_tensor(
                quaternion_matrix(pose[3:]).astype(np.float32), device=self.device
            )
            origin = torch.as_tensor(pose[:3].astype(np.float32), device=self.device)
            local = (self.gpu_xyz - origin[None, :]) @ rotation
            forward = local[:, 0]
            right = local[:, 1]
            down = local[:, 2]
            half = extent_m / 2.0
            spatial = (
                (right >= -half)
                & (right < half)
                & (forward >= -half)
                & (forward < half)
            )
            forward = forward[spatial]
            right = right[spatial]
            down = down[spatial]
            classes = self.gpu_class_id[spatial]
            near_ground = torch.abs(down - ground_down_m) <= ground_surface_tolerance_m
            flat_semantic = self._gpu_isin(classes, flat_traversable_ids)
            terrain_semantic = self._gpu_isin(classes, terrain_traversable_ids)
            ambiguous_ground = self._gpu_isin(classes, ambiguous_ground_ids)
            semantic_traversable = flat_semantic | terrain_semantic
            traversable = (
                (flat_semantic & near_ground)
                | (ambiguous_ground & near_ground)
                | (
                    terrain_semantic
                    & (torch.abs(down - ground_down_m) <= surface_height_tolerance_m)
                )
            )
            ignored = self._gpu_isin(classes, ignored_ids)
            height_above_ground = ground_down_m - down
            obstacle = (
                ~ignored
                & ~semantic_traversable
                & ~(ambiguous_ground & near_ground)
                & (height_above_ground >= obstacle_min_height_m)
                & (height_above_ground <= obstacle_max_height_m)
            )
            columns = torch.floor((right + half) / extent_m * size).to(torch.int64)
            rows = torch.floor((half - forward) / extent_m * size).to(torch.int64)
            in_bounds = (rows >= 0) & (rows < size) & (columns >= 0) & (columns < size)
            flat_index = rows * size + columns
            free_seed = torch.zeros(size * size, dtype=torch.uint8, device=self.device)
            occupied_seed = torch.zeros_like(free_seed)
            free_seed[flat_index[in_bounds & traversable]] = 1
            occupied_seed[flat_index[in_bounds & obstacle]] = 1
            free_seed_np = free_seed.reshape(size, size).cpu().numpy()
            occupied_seed_np = occupied_seed.reshape(size, size).cpu().numpy()

        kernel = np.ones((3, 3), dtype=np.uint8)
        free = cv2.morphologyEx(free_seed_np, cv2.MORPH_CLOSE, kernel, iterations=1)
        free = cv2.dilate(free, kernel, iterations=1)
        occupied = cv2.morphologyEx(occupied_seed_np, cv2.MORPH_CLOSE, kernel, iterations=1)
        occupied = cv2.dilate(occupied, kernel, iterations=1)
        labels = np.full((size, size), UNKNOWN, dtype=np.uint8)
        labels[free.astype(bool)] = FREE
        labels[occupied.astype(bool)] = OCCUPIED
        center = size // 2
        labels[center - 1 : center + 2, center - 1 : center + 2] = FREE
        return labels


def shadowcast_visible(obstacle: np.ndarray, horizontal_fov_degrees: float) -> np.ndarray:
    """Exact symmetric grid shadowcasting; occupied first-hit cells stay visible."""

    if obstacle.ndim != 2 or obstacle.shape[0] != obstacle.shape[1]:
        raise ValueError("obstacle must be a square boolean raster")
    size = obstacle.shape[0]
    visible = np.zeros_like(obstacle, dtype=bool)
    origin_column = origin_row = size // 2
    visible[origin_row, origin_column] = True

    def cast_octant(row: int, start: float, end: float, xx: int, xy: int, yx: int, yy: int) -> None:
        if start < end:
            return
        next_start = start
        for distance in range(row, size + 1):
            dx = -distance - 1
            dy = -distance
            blocked = False
            while dx <= 0:
                dx += 1
                column = origin_column + dx * xx + dy * xy
                output_row = origin_row + dx * yx + dy * yy
                left = (dx - 0.5) / (dy + 0.5)
                right = (dx + 0.5) / (dy - 0.5)
                if start < right:
                    continue
                if end > left:
                    break
                inside = 0 <= column < size and 0 <= output_row < size
                if inside:
                    visible[output_row, column] = True
                cell_obstacle = not inside or obstacle[output_row, column]
                if blocked:
                    if cell_obstacle:
                        next_start = right
                        continue
                    blocked = False
                    start = next_start
                elif cell_obstacle and distance < size:
                    blocked = True
                    cast_octant(distance + 1, start, left, xx, xy, yx, yy)
                    next_start = right
            if blocked:
                break

    for transform in (
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0, -1, 1, 0),
        (-1, 0, 0, 1),
        (-1, 0, 0, -1),
        (0, -1, -1, 0),
        (0, 1, -1, 0),
        (1, 0, 0, -1),
    ):
        cast_octant(1, 1.0, 0.0, *transform)
    rows, columns = np.indices(obstacle.shape, dtype=np.float64)
    angle = np.arctan2(columns - origin_column, origin_row - rows)
    visible &= np.abs(angle) <= math.radians(horizontal_fov_degrees) / 2.0 + 1e-12
    return visible


def observed_from_complete(complete: np.ndarray) -> np.ndarray:
    visible = shadowcast_visible(complete == OCCUPIED, HFOV_DEG)
    observed = np.full_like(complete, UNKNOWN)
    known = visible & (complete != UNKNOWN)
    observed[known] = complete[known]
    # Keep only observed free space connected to the robot.  Occupied surface
    # cells remain visible even though they are not part of the free component.
    free = observed == FREE
    labels, _ = ndimage.label(free, structure=ndimage.generate_binary_structure(2, 1))
    center = observed.shape[0] // 2
    origin_label = int(labels[center, center])
    if origin_label == 0:
        raise ValueError("asset BEV robot origin is not free")
    observed[free & (labels != origin_label)] = UNKNOWN
    return observed


def warp_known_to_target(
    source_known: np.ndarray,
    source_world_from_bev: np.ndarray,
    target_world_from_bev: np.ndarray,
    *,
    source_extent_m: float = 6.5,
    target_extent_m: float = 10.0,
    target_size: int = 512,
) -> np.ndarray:
    rows, columns = np.nonzero(source_known)
    source_size = source_known.shape[0]
    right = (columns.astype(np.float64) + 0.5) / source_size * source_extent_m - source_extent_m / 2.0
    forward = source_extent_m / 2.0 - (rows.astype(np.float64) + 0.5) / source_size * source_extent_m
    local = np.column_stack((right, forward, np.ones_like(right)))
    world = (source_world_from_bev @ local.T).T
    target = (np.linalg.inv(target_world_from_bev) @ world.T).T[:, :2]
    target_rows, target_columns = metric_to_pixel(target, target_extent_m, target_size)
    valid = (
        (target_rows >= 0)
        & (target_rows < target_size)
        & (target_columns >= 0)
        & (target_columns < target_size)
    )
    result = np.zeros((target_size, target_size), dtype=np.uint8)
    result[target_rows[valid], target_columns[valid]] = 1
    return result.astype(bool)


def merged_observed(
    single_observed: list[np.ndarray],
    transforms: list[np.ndarray],
    target_index: int,
    complete: np.ndarray,
) -> np.ndarray:
    known = np.zeros_like(complete, dtype=bool)
    target = transforms[target_index]
    for index in range(target_index + 1):
        known |= warp_known_to_target(
            single_observed[index] != UNKNOWN,
            transforms[index],
            target,
        )
    known &= complete != UNKNOWN
    result = np.full_like(complete, UNKNOWN)
    result[known] = complete[known]
    return result


def add_overlays(value: np.ndarray, observed: np.ndarray, fov: np.ndarray) -> np.ndarray:
    rgb = palette(value)
    rgb[boundary(fov)] = (35, 120, 255)
    rgb[boundary(observed)] = (255, 40, 40)
    return rgb


def make_visualization(
    output: Path,
    session_id: str,
    rgb: Image.Image,
    single_masked: np.ndarray,
    single_complete: np.ndarray,
    merged_masked: np.ndarray,
    merged_complete: np.ndarray,
    merged_fov: np.ndarray,
) -> Path:
    guessed = np.full_like(merged_complete, UNKNOWN)
    guessed_known = (merged_masked == UNKNOWN) & (merged_complete != UNKNOWN)
    guessed[guessed_known] = merged_complete[guessed_known]
    panels = (
        ("latest RGB", rgb),
        ("single asset-complete", Image.fromarray(palette(single_complete))),
        ("single observed", Image.fromarray(palette(single_masked))),
        ("merged asset-complete", Image.fromarray(palette(merged_complete))),
        ("merged observed", Image.fromarray(palette(merged_masked))),
        (
            "merged guessed | red=Gate blue=FOV",
            Image.fromarray(add_overlays(guessed, merged_masked != UNKNOWN, merged_fov)),
        ),
    )
    tile, title_h = 360, 34
    canvas = Image.new("RGB", (3 * tile, 2 * (tile + title_h) + 44), (18, 24, 31))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), session_id + "  (BEV truth comes from the global semantic PCD asset)", fill=(230, 235, 242))
    for index, (title, panel) in enumerate(panels):
        row, column = divmod(index, 3)
        x = column * tile
        y = 44 + row * (tile + title_h)
        resample = Image.Resampling.BILINEAR if index == 0 else Image.Resampling.NEAREST
        canvas.paste(panel.convert("RGB").resize((tile, tile), resample), (x, y + title_h))
        draw.text((x + 10, y + 9), title, fill=(220, 228, 238))
    path = output / "visualization" / f"{session_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def build_route(
    args: argparse.Namespace,
    relative: str,
    palette_values: np.ndarray,
    *,
    window_rank: int = 0,
    selected_frame_ids: list[int] | None = None,
    session_id_suffix: str = "",
    selection_metadata: dict[str, Any] | None = None,
    create_visualization: bool = True,
    semantic_asset: SemanticAsset | None = None,
    semantic_pcd_sha256: str | None = None,
    seg_rgb_sha256: str | None = None,
    ground_estimate_cache: dict[int, tuple[float | None, str]] | None = None,
    single_bev_cache: dict[
        tuple[int, float], tuple[np.ndarray, np.ndarray]
    ] | None = None,
    merged_complete_cache: dict[tuple[int, float], np.ndarray] | None = None,
) -> dict:
    source_root = args.source_root.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    compact_storage = bool(getattr(args, "compact_storage", False))
    raster_suffix = ".jpeg" if compact_storage else ".png"
    route = source_root / relative
    scene = relative.split("/")[0]
    asset_dir = asset_root / scene
    for required in ("metadata.zip", "image_lcam_front.zip", "depth_lcam_front.zip"):
        if not (route / required).is_file():
            raise FileNotFoundError(route / required)
    for required in (f"{scene}_sem.pcd", "seg_labels.zip"):
        if not (asset_dir / required).is_file():
            raise FileNotFoundError(asset_dir / required)

    poses, route_metadata = read_metadata(route)
    robot_height = float(route_metadata["robot_height"])
    label_ids = load_scene_label_ids(asset_dir)
    flat_traversable_ids = np.asarray(
        [
            label_ids[name]
            for name in sorted(FLAT_TRAVERSABLE_CLASSES)
            if name in label_ids
        ],
        dtype=np.int16,
    )
    terrain_traversable_ids = np.asarray(
        [
            label_ids[name]
            for name in sorted(TERRAIN_TRAVERSABLE_CLASSES)
            if name in label_ids
        ],
        dtype=np.int16,
    )
    ambiguous_ground_ids = np.asarray(
        [
            label_ids[name]
            for name in sorted(AMBIGUOUS_GROUND_CLASSES)
            if name in label_ids
        ],
        dtype=np.int16,
    )
    ignored_ids = np.asarray(
        [
            index
            for name, index in label_ids.items()
            if any(token in name for token in IGNORED_GEOMETRY_SUBSTRINGS)
        ],
        dtype=np.int16,
    )

    with zipfile.ZipFile(route / "image_lcam_front.zip") as rgb_zip, zipfile.ZipFile(
        route / "depth_lcam_front.zip"
    ) as depth_zip:
        rgb_members = member_map(rgb_zip, rf"/(\d{{6}})_{CAMERA}\.png$")
        depth_members = member_map(depth_zip, rf"/(\d{{6}})_{CAMERA}_depth\.png$")
        available = sorted(set(rgb_members) & set(depth_members) & set(range(len(poses))))
        if selected_frame_ids is None:
            windows = ranked_windows(available, poses, args.frames, args.frame_step)
            if window_rank >= len(windows):
                raise IndexError(
                    f"route has only {len(windows)} candidate windows; requested rank {window_rank}"
                )
            selected = windows[window_rank]
        else:
            selected = [int(value) for value in selected_frame_ids]
            if len(selected) != args.frames:
                raise ValueError(
                    f"external window has {len(selected)} frames; expected {args.frames}"
                )
            if any(left >= right for left, right in zip(selected, selected[1:])):
                raise ValueError("external window frame IDs must be strictly increasing")
            missing = [frame for frame in selected if frame not in available]
            if missing:
                raise ValueError(f"external window references unavailable frame {missing[0]}")
        selected_poses = poses[selected]
        if semantic_asset is None:
            asset = SemanticAsset(
                asset_dir,
                scene,
                palette_values,
                selected_poses,
                device=args.device,
            )
        else:
            asset = semantic_asset
            expected_asset_path = (asset_dir / f"{scene}_sem.pcd").resolve()
            if asset.path.resolve() != expected_asset_path:
                raise ValueError(
                    "shared semantic asset does not match source scene: "
                    f"{asset.path} != {expected_asset_path}"
                )
        transforms = [world_from_bev_planar(pose) for pose in selected_poses]
        ground_down_by_frame: list[float | None] = []
        ground_source_by_frame: list[str] = []
        for source_index, pose in zip(selected, selected_poses):
            cached_ground = (
                None
                if ground_estimate_cache is None
                else ground_estimate_cache.get(source_index)
            )
            if cached_ground is not None:
                ground_down_by_frame.append(cached_ground[0])
                ground_source_by_frame.append(cached_ground[1])
                continue
            estimate = None
            source = ""
            for radius in (2.5, 4.0, 6.0):
                try:
                    estimate = asset.estimate_ground_down(
                        pose,
                        flat_traversable_ids=flat_traversable_ids,
                        ambiguous_ground_ids=ambiguous_ground_ids,
                        radius_m=radius,
                    )
                except ValueError:
                    continue
                source = f"asset_semantic_height_mode_radius_{radius:g}m"
                break
            ground_down_by_frame.append(estimate)
            ground_source_by_frame.append(source)
            if ground_estimate_cache is not None:
                ground_estimate_cache[source_index] = (estimate, source)

        valid_ground_indices = [
            index for index, value in enumerate(ground_down_by_frame) if value is not None
        ]
        if valid_ground_indices:
            valid_values = [float(ground_down_by_frame[index]) for index in valid_ground_indices]
            interpolated = np.interp(
                np.arange(len(ground_down_by_frame), dtype=np.float64),
                np.asarray(valid_ground_indices, dtype=np.float64),
                np.asarray(valid_values, dtype=np.float64),
            )
            for index, value in enumerate(ground_down_by_frame):
                if value is None:
                    ground_down_by_frame[index] = float(interpolated[index])
                    ground_source_by_frame[index] = "asset_temporal_neighbor_interpolation"
        else:
            ground_down_by_frame = [robot_height for _ in selected_poses]
            ground_source_by_frame = ["route_robot_height_metadata_fallback" for _ in selected_poses]
        ground_down_by_frame = [float(value) for value in ground_down_by_frame]
        single_complete: list[np.ndarray] = []
        single_masked: list[np.ndarray] = []
        for output_index, (source_index, pose) in enumerate(
            zip(selected, selected_poses)
        ):
            cache_key = (source_index, ground_down_by_frame[output_index])
            cached_bev = (
                None if single_bev_cache is None else single_bev_cache.get(cache_key)
            )
            if cached_bev is None:
                complete = asset.rasterize(
                    pose,
                    ground_down_m=ground_down_by_frame[output_index],
                    extent_m=6.5,
                    flat_traversable_ids=flat_traversable_ids,
                    terrain_traversable_ids=terrain_traversable_ids,
                    ambiguous_ground_ids=ambiguous_ground_ids,
                    ignored_ids=ignored_ids,
                    obstacle_min_height_m=args.obstacle_min_height_m,
                    obstacle_max_height_m=args.obstacle_max_height_m,
                    surface_height_tolerance_m=args.surface_height_tolerance_m,
                    ground_surface_tolerance_m=args.ground_surface_tolerance_m,
                )
                observed = observed_from_complete(complete)
                if single_bev_cache is not None:
                    single_bev_cache[cache_key] = (complete, observed)
            else:
                complete, observed = cached_bev
            known_pixels = int(np.count_nonzero(observed != UNKNOWN))
            if known_pixels < args.minimum_observed_known_pixels:
                raise ValueError(
                    "observed BEV has insufficient known support (preflight): "
                    f"known={known_pixels}, "
                    f"required={args.minimum_observed_known_pixels}, "
                    f"frame={output_index}, directory=masked"
                )
            single_complete.append(complete)
            single_masked.append(observed)

        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id_suffix).strip("_")
        session_id = "session_tartanground_asset_" + relative.replace("/", "_")
        if suffix:
            session_id += "_" + suffix
        final_session = output_root / "sessions" / session_id
        session = output_root / "sessions" / f".{session_id}.partial"
        if session.exists():
            shutil.rmtree(session)
        if final_session.exists():
            shutil.rmtree(final_session)
        extrinsics = []
        final_visual = None
        for output_index, source_index in enumerate(selected):
            pose = poses[source_index]
            rgb = Image.open(io.BytesIO(rgb_zip.read(rgb_members[source_index]))).convert("RGB")
            depth = decode_depth(depth_zip.read(depth_members[source_index]))
            camera_path = (
                session / "camera" / f"frame_{output_index:06d}{raster_suffix}"
            )
            camera_path.parent.mkdir(parents=True, exist_ok=True)
            if compact_storage:
                save_rgb_jpeg(camera_path, np.asarray(rgb, dtype=np.uint8))
            else:
                rgb.save(camera_path)
            depth_path = session / "depth" / f"frame_{output_index:06d}.npz"
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            if compact_storage:
                save_depth_compact(depth_path, depth)
            else:
                np.savez_compressed(depth_path, depth=depth.astype(np.float32))

            cache_key = (source_index, ground_down_by_frame[output_index])
            merged_complete = (
                None
                if merged_complete_cache is None
                else merged_complete_cache.get(cache_key)
            )
            if merged_complete is None:
                merged_complete = asset.rasterize(
                    pose,
                    ground_down_m=ground_down_by_frame[output_index],
                    extent_m=10.0,
                    flat_traversable_ids=flat_traversable_ids,
                    terrain_traversable_ids=terrain_traversable_ids,
                    ambiguous_ground_ids=ambiguous_ground_ids,
                    ignored_ids=ignored_ids,
                    obstacle_min_height_m=args.obstacle_min_height_m,
                    obstacle_max_height_m=args.obstacle_max_height_m,
                    surface_height_tolerance_m=args.surface_height_tolerance_m,
                    ground_surface_tolerance_m=args.ground_surface_tolerance_m,
                )
                if merged_complete_cache is not None:
                    merged_complete_cache[cache_key] = merged_complete
            merged_masked = merged_observed(
                single_masked, transforms, output_index, merged_complete
            )
            frame_name = f"frame_{output_index:06d}{raster_suffix}"
            bev_outputs = (
                ("bev_6p5m/masked", single_masked[output_index]),
                ("bev_6p5m/complete", single_complete[output_index]),
                ("bev_6p5m/merged_masked_10m", merged_masked),
                ("bev_6p5m/merged_complete_10m", merged_complete),
            )
            for directory, value in bev_outputs:
                path = session / directory / frame_name
                path.parent.mkdir(parents=True, exist_ok=True)
                if compact_storage:
                    save_bev_jpeg(path, value)
                else:
                    save_label(path, value)

            rotation = quaternion_matrix(pose[3:])
            camera_to_world = np.eye(4, dtype=np.float64)
            camera_to_world[:3, :3] = rotation
            camera_to_world[:3, 3] = pose[:3]
            extrinsics.append(
                {
                    "frame_id": output_index,
                    "source_frame_id": source_index,
                    "sim_time_s": (
                        float(source_index - selected[0])
                        * float(route_metadata.get("time_step", 0.1))
                    ),
                    "camera_file": (
                        f"camera/frame_{output_index:06d}{raster_suffix}"
                    ),
                    "depth_file": f"depth/frame_{output_index:06d}.npz",
                    "extrinsic": {
                        "camera_position_world_m": pose[:3].tolist(),
                        "camera_rotation_xyzw": pose[3:].tolist(),
                        "camera_to_world_matrix": camera_to_world.tolist(),
                        "world_to_camera_matrix": np.linalg.inv(camera_to_world).tolist(),
                        "bev_forward_xy": rotation[:2, 0].tolist(),
                        "bev_right_xy": rotation[:2, 1].tolist(),
                        "world_from_bev_planar": transforms[output_index].tolist(),
                        "coordinate_convention": "TartanGround NED world; local forward/right/down; BEV right/forward",
                    },
                }
            )
            if output_index == len(selected) - 1:
                support = fov_union(transforms, transforms[-1], 10.0, 512)
                final_visual = (
                    rgb,
                    single_masked[output_index],
                    single_complete[output_index],
                    merged_masked,
                    merged_complete,
                    support,
                )

        (session / "camera_extrinsics.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in extrinsics), encoding="utf-8"
        )
        camera_intrinsics = {
            "model": "pinhole",
            "width": WIDTH,
            "height": HEIGHT,
            "fx": FX,
            "fy": FY,
            "cx": CX,
            "cy": CY,
            "K": [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]],
            "horizontal_fov_degrees": HFOV_DEG,
            "vertical_fov_degrees": HFOV_DEG,
            "pixel_coordinate_convention": (
                "integer coordinates address pixel centers; origin is top-left"
            ),
            "source": "TartanGround official camera contract",
        }
        (session / "camera_intrinsics.json").write_text(
            json.dumps(camera_intrinsics, indent=2) + "\n", encoding="utf-8"
        )
        pcd_path = asset_dir / f"{scene}_sem.pcd"
        metadata = {
            "schema_version": 6,
            "storage_format": (
                {
                    "profile": "pro6000-compact-v1",
                    "rgb": (
                        f"JPEG quality {JPEG_QUALITY}, 4:4:4, optimized"
                    ),
                    "bev": (
                        f"grayscale JPEG quality {JPEG_QUALITY}; "
                        "decode then snap to {0,112,255}"
                    ),
                    "depth": "uint16 NPZ v1 at 0.002 metres per unit",
                    "synchronized_dense_saved_frame_ids": True,
                }
                if compact_storage
                else {
                    "profile": "lossless-png-float32-depth-v1",
                    "rgb": "PNG RGB",
                    "bev": "grayscale PNG categorical labels",
                    "depth": "float32 compressed NPZ",
                    "synchronized_dense_saved_frame_ids": True,
                }
            ),
            "status": "complete",
            "session_id": session_id,
            "dataset": "tartanground-global-semantic-asset",
            "scene_id": scene,
            "source_route": relative,
            "source_window_rank": window_rank,
            "source_frame_ids": selected,
            "frame_count": len(selected),
            "capture_hz": (
                None
                if selection_metadata is not None
                else 1.0
                / (
                    float(args.frame_step)
                    * float(route_metadata.get("time_step", 0.1))
                )
            ),
            "frame_selection": selection_metadata,
            "camera_intrinsics": camera_intrinsics,
            "camera_extrinsics_file": "camera_extrinsics.jsonl",
            "depth": {
                "directory": "depth",
                "filename_pattern": "frame_INDEX.npz",
                "format": (
                    "compressed NumPy NPZ, compact depth format v1"
                    if compact_storage
                    else "compressed NumPy NPZ"
                ),
                "dtype": "uint16" if compact_storage else "float32",
                "units": "metres",
                **(
                    {
                        "scale_m": float(DEPTH_SCALE_M),
                        "invalid_q": int(DEPTH_INVALID_Q),
                        "maximum_quantization_error_m": float(DEPTH_SCALE_M) / 2.0,
                    }
                    if compact_storage
                    else {}
                ),
                "source": "TartanGround DepthPlanar packed float32 ground truth",
                "convention": "camera_axis_z_depth_m",
                "usage": "Scale-Token supervision only; never used to generate BEV truth",
            },
            "random_parameters": {
                "horizontal_fov_degrees": HFOV_DEG,
                "camera_height_m": float(np.median(ground_down_by_frame)),
                "camera_height_by_frame_m": ground_down_by_frame,
                "camera_height_source_by_frame": ground_source_by_frame,
                "source_robot_height_metadata_m": robot_height,
            },
            "bev": {
                "size": 512,
                "extent_classes_m": [6.5],
                "merged_normalized_extents_m": [10.0],
                "merged_normalized_size": [512, 512],
                "merged_orientation": "ego-centric; latest robot centered and forward up",
                "masked_values": {"occupied": 0, "unknown": 112, "free": 255},
                "truth_source": "official TartanGround global semantic point-cloud asset",
                "flat_traversable_classes": sorted(FLAT_TRAVERSABLE_CLASSES),
                "terrain_traversable_classes": sorted(TERRAIN_TRAVERSABLE_CLASSES),
                "ambiguous_near_ground_classes": sorted(AMBIGUOUS_GROUND_CLASSES),
                "ignored_geometry_substrings": list(IGNORED_GEOMETRY_SUBSTRINGS),
                "ground_surface_tolerance_m": args.ground_surface_tolerance_m,
                "obstacle_height_band_m": [
                    args.obstacle_min_height_m,
                    args.obstacle_max_height_m,
                ],
                "visibility_algorithm": "symmetric_grid_shadowcasting_from_asset_complete_v1",
                "fusion_rule": "history visibility union in latest ego; labels copied from latest asset-complete target",
                "complete_gt_contract": "asset complete within semantic surface coverage; unsampled cells remain unknown/ignored",
                "training_eligibility": "M05/P1D asset-grounded complete/observed BEV",
            },
            "path": {
                "start": [float(selected_poses[0, 0]), float(-selected_poses[0, 2] - ground_down_by_frame[0]), float(selected_poses[0, 1])],
                "goal": [float(selected_poses[-1, 0]), float(-selected_poses[-1, 2] - ground_down_by_frame[-1]), float(selected_poses[-1, 1])],
            },
            "asset": {
                "semantic_pcd": str(pcd_path),
                "semantic_pcd_sha256": (
                    semantic_pcd_sha256
                    if semantic_pcd_sha256 is not None
                    else sha256(pcd_path)
                ),
                "source_point_count": asset.source_point_count,
                "route_selected_point_count": asset.selected_point_count,
                "seg_rgb_sha256": (
                    seg_rgb_sha256
                    if seg_rgb_sha256 is not None
                    else sha256(args.seg_rgb_file.expanduser().resolve())
                ),
            },
            "conversion": {
                "version": "tartanground-m05-p1d-global-asset-v2",
                "original_data_mutated": False,
                "depth_generated_complete": False,
                "ground_level_source": "per-frame local horizontal semantic-surface mode from global asset",
                "ground_level_fallback_contract": (
                    "expand asset radius, then interpolate only failed frames from neighboring "
                    "asset-derived route frames; route robot_height is used only if the entire "
                    "window lacks semantic ground support"
                ),
                "unknown_policy": "retain unsupported asset cells as unknown and exclude them from BEV loss",
                "minimum_single_observed_known_pixels": args.minimum_observed_known_pixels,
                "minimum_merged_observed_known_pixels": max(
                    1,
                    round(args.minimum_observed_known_pixels * (6.5 / 10.0) ** 2),
                ),
                "observed_support_gate": "equal minimum physical area across BEV extents",
            },
        }
        (session / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        validate_session(
            session,
            expected_frames=len(selected),
            require_complete=False,
            minimum_observed_known_pixels=args.minimum_observed_known_pixels,
        )
        session.rename(final_session)
        (final_session / "COMPLETE").touch()
        visual = None
        if create_visualization:
            assert final_visual is not None
            visual = make_visualization(
                output_root,
                session_id,
                final_visual[0],
                final_visual[1],
                final_visual[2],
                final_visual[3],
                final_visual[4],
                final_visual[5],
            )
        return {
            "session_id": session_id,
            "relative_path": f"sessions/{session_id}",
            "scene_id": scene,
            "source_route": relative,
            "source_window_rank": window_rank,
            "source_frame_ids": selected,
            "frame_count": len(selected),
            "visualization": (
                None if visual is None else str(visual.relative_to(output_root))
            ),
            "training_eligibility": metadata["bev"]["training_eligibility"],
        }


def write_index(output_root: Path, records: list[dict]) -> None:
    (output_root / "manifest.json").write_text(
        json.dumps({"schema": "m05-p1d-tartanground-global-asset-v2", "sessions": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    cards = "\n".join(
        (
            '<button class="card" type="button" '
            f'data-search="{html.escape((record.get("scene_id", "") + " " + record["source_route"]).lower())}" '
            f'data-src="{html.escape(record["visualization"])}">'
            f'<img loading="lazy" src="{html.escape(record["visualization"])}" '
            f'alt="{html.escape(record["source_route"])}">'
            '<span class="card-copy">'
            f'<strong>{html.escape(record.get("scene_id", record["source_route"].split("/")[0]))}</strong>'
            f'<small>{html.escape(record["source_route"])}</small>'
            f'<small>{len(record["source_frame_ids"])} frames · 2 Hz · click to inspect</small>'
            '</span></button>'
        )
        for record in records
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TartanGround trainable session gallery</title>
<style>
:root{{--bg:#0b1017;--panel:#141c27;--line:#2a394b;--text:#e9f0f8;--muted:#9fb0c3;--accent:#56b7ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:4;padding:18px 24px;background:rgba(11,16,23,.94);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}}
h1{{font-size:24px;margin:0 0 5px}} p{{margin:4px 0;color:var(--muted)}}
.toolbar{{display:flex;gap:12px;align-items:center;margin-top:14px}} input{{width:min(560px,80vw);padding:10px 13px;border:1px solid var(--line);border-radius:9px;background:#0f1620;color:var(--text)}}
#count{{color:var(--accent);font-weight:700;white-space:nowrap}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:16px;padding:20px}}
.card{{appearance:none;text-align:left;padding:0;overflow:hidden;border:1px solid var(--line);border-radius:12px;background:var(--panel);color:var(--text);cursor:zoom-in;transition:.16s transform,.16s border-color}}
.card:hover{{transform:translateY(-2px);border-color:var(--accent)}} .card[hidden]{{display:none}}
.card img{{display:block;width:100%;aspect-ratio:3/2;object-fit:cover;background:#080b10}}
.card-copy{{display:grid;gap:3px;padding:11px 13px}} .card-copy strong{{font-size:17px}} .card-copy small{{color:var(--muted);overflow-wrap:anywhere}}
dialog{{width:min(96vw,1500px);padding:0;border:1px solid var(--line);border-radius:12px;background:#080d13;color:var(--text)}}
dialog::backdrop{{background:rgba(0,0,0,.86)}} dialog img{{display:block;width:100%;height:auto}}
.close{{position:sticky;top:0;float:right;margin:10px;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:#172231;color:white;cursor:pointer}}
</style></head><body>
<header><h1>TartanGround · actual trainable sessions</h1>
<p>One deterministic 10-frame session per scene. Complete BEV is projected from the official global semantic PCD; Masked BEV is visibility over that same truth. Depth is Scale-only.</p>
<div class="toolbar"><input id="filter" type="search" placeholder="Search scene or route…" autofocus><span id="count"></span></div></header>
<main id="grid">{cards}</main>
<dialog id="viewer"><button class="close" type="button">Close</button><img alt="Full-resolution session audit"></dialog>
<script>
const cards=[...document.querySelectorAll('.card')], q=document.querySelector('#filter'), count=document.querySelector('#count');
function apply(){{const value=q.value.trim().toLowerCase();let shown=0;for(const card of cards){{const visible=!value||card.dataset.search.includes(value);card.hidden=!visible;shown+=visible?1:0}}count.textContent=`${{shown}} / ${{cards.length}} scenes`}}
q.addEventListener('input',apply);apply();
const viewer=document.querySelector('#viewer'), full=viewer.querySelector('img');
for(const card of cards)card.addEventListener('click',()=>{{full.src=card.dataset.src;full.alt=card.dataset.search;viewer.showModal()}});
viewer.querySelector('.close').addEventListener('click',()=>viewer.close());viewer.addEventListener('click',event=>{{if(event.target===viewer)viewer.close()}});
document.addEventListener('keydown',event=>{{if(event.key==='Escape'&&viewer.open)viewer.close()}});
</script></body></html>"""
    (output_root / "index.html").write_text(
        page,
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        "# TartanGround global-asset M05/P1D conversion\n\n"
        "Complete BEV is rasterized from the official per-environment semantic point cloud. "
        "Observed BEV is an exact visibility mask over that same complete raster. Metric depth "
        "is retained only for Scale-Token supervision and does not generate occupancy labels.\n",
        encoding="utf-8",
    )


def validate_session(
    session: Path,
    *,
    expected_frames: int,
    require_complete: bool,
    minimum_observed_known_pixels: int = 1024,
) -> dict[str, float | int]:
    if minimum_observed_known_pixels <= 0:
        raise ValueError("minimum observed known pixels must be positive")
    if require_complete and not (session / "COMPLETE").is_file():
        raise ValueError(f"session lacks COMPLETE marker: {session}")
    metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    compact_storage = (
        metadata.get("storage_format", {}).get("profile")
        == "pro6000-compact-v1"
    )
    raster_suffix = ".jpeg" if compact_storage else ".png"
    intrinsics = json.loads(
        (session / "camera_intrinsics.json").read_text(encoding="utf-8")
    )
    if int(metadata["frame_count"]) != expected_frames:
        raise ValueError(f"frame count metadata mismatch: {session}")
    if metadata["camera_intrinsics"]["K"] != intrinsics["K"]:
        raise ValueError(f"embedded/external intrinsics disagree: {session}")
    extrinsics = [
        json.loads(line)
        for line in (session / "camera_extrinsics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(extrinsics) != expected_frames:
        raise ValueError(f"extrinsic count mismatch: {session}")

    allowed = {int(OCCUPIED), int(UNKNOWN), int(FREE)}
    pair_disagreements = 0
    observed_outside_complete = 0
    minimum_complete_known_fraction = 1.0
    minimum_observed_known_pixels_seen = 512 * 512
    pairs = (
        ("complete", "masked"),
        ("merged_complete_10m", "merged_masked_10m"),
    )
    for index in range(expected_frames):
        name = f"frame_{index:06d}{raster_suffix}"
        with Image.open(session / "camera" / name) as rgb:
            if rgb.convert("RGB").size != (WIDTH, HEIGHT):
                raise ValueError(f"unexpected RGB shape: {session / 'camera' / name}")
        depth_path = session / "depth" / f"frame_{index:06d}.npz"
        if compact_storage:
            depth = load_depth_compact(depth_path)
            if depth.shape != (HEIGHT, WIDTH):
                raise ValueError(f"unexpected compact depth payload: {session}")
        else:
            with np.load(depth_path, allow_pickle=False) as payload:
                if "depth" not in payload or payload["depth"].shape != (HEIGHT, WIDTH):
                    raise ValueError(f"unexpected depth payload: {session}")
        for complete_directory, observed_directory in pairs:
            with Image.open(
                session / "bev_6p5m" / complete_directory / name
            ) as encoded_complete, Image.open(
                session / "bev_6p5m" / observed_directory / name
            ) as encoded_observed:
                if encoded_complete.mode != "L" or encoded_complete.size != (512, 512):
                    raise ValueError(f"invalid complete BEV encoding: {session}")
                if encoded_observed.mode != "L" or encoded_observed.size != (512, 512):
                    raise ValueError(f"invalid observed BEV encoding: {session}")
                complete = np.asarray(encoded_complete, dtype=np.uint8)
                observed = np.asarray(encoded_observed, dtype=np.uint8)
            if compact_storage:
                complete = snap_bev_palette(complete)
                observed = snap_bev_palette(observed)
            if not set(np.unique(complete).tolist()) <= allowed:
                raise ValueError(f"complete BEV contains invalid labels: {session}")
            if not set(np.unique(observed).tolist()) <= allowed:
                raise ValueError(f"observed BEV contains invalid labels: {session}")
            known = observed != UNKNOWN
            complete_known = complete != UNKNOWN
            pair_disagreements += int(
                np.count_nonzero(known & (observed != complete))
            )
            observed_outside_complete += int(
                np.count_nonzero(known & ~complete_known)
            )
            minimum_complete_known_fraction = min(
                minimum_complete_known_fraction,
                float(complete_known.mean()),
            )
            minimum_observed_known_pixels_seen = min(
                minimum_observed_known_pixels_seen,
                int(known.sum()),
            )
            center = complete.shape[0] // 2
            if int(complete[center, center]) != int(FREE):
                raise ValueError(f"robot origin is not free in complete BEV: {session}")
            # Both rasters are 512x512, but the Merged grid covers 10 m while
            # Single covers 6.5 m.  Compare equal physical support area rather
            # than applying an accidentally 2.37x stricter Merged pixel gate.
            required_known_pixels = minimum_observed_known_pixels
            if observed_directory == "merged_masked_10m":
                required_known_pixels = max(
                    1,
                    round(minimum_observed_known_pixels * (6.5 / 10.0) ** 2),
                )
            if int(known.sum()) < required_known_pixels:
                raise ValueError(
                    "observed BEV has insufficient known support: "
                    f"known={int(known.sum())}, "
                    f"required={required_known_pixels}, "
                    f"frame={index}, directory={observed_directory}, "
                    f"session={session}"
                )
    if pair_disagreements or observed_outside_complete:
        raise ValueError(
            "observed/complete BEV contract failed: "
            f"disagreements={pair_disagreements}, "
            f"outside={observed_outside_complete}, session={session}"
        )
    return {
        "frames": expected_frames,
        "pair_disagreements": pair_disagreements,
        "observed_outside_complete": observed_outside_complete,
        "minimum_complete_known_fraction": minimum_complete_known_fraction,
        "minimum_observed_known_pixels": minimum_observed_known_pixels_seen,
    }


def validate_output(
    output_root: Path,
    records: list[dict],
    *,
    minimum_observed_known_pixels: int = 1024,
) -> dict[str, float | int]:
    checks: dict[str, float | int] = {
        "sessions": 0,
        "frames": 0,
        "pair_disagreements": 0,
        "observed_outside_complete": 0,
        "minimum_complete_known_fraction": 1.0,
        "minimum_observed_known_pixels": 512 * 512,
    }
    for record in records:
        session = output_root / record["relative_path"]
        result = validate_session(
            session,
            expected_frames=int(record.get("frame_count", len(record["source_frame_ids"]))),
            require_complete=True,
            minimum_observed_known_pixels=minimum_observed_known_pixels,
        )
        checks["sessions"] = int(checks["sessions"]) + 1
        checks["frames"] = int(checks["frames"]) + int(result["frames"])
        checks["pair_disagreements"] = int(checks["pair_disagreements"]) + int(
            result["pair_disagreements"]
        )
        checks["observed_outside_complete"] = int(
            checks["observed_outside_complete"]
        ) + int(result["observed_outside_complete"])
        checks["minimum_complete_known_fraction"] = min(
            float(checks["minimum_complete_known_fraction"]),
            float(result["minimum_complete_known_fraction"]),
        )
        checks["minimum_observed_known_pixels"] = min(
            int(checks["minimum_observed_known_pixels"]),
            int(result["minimum_observed_known_pixels"]),
        )
    return checks


def session_id_for_route(relative: str) -> str:
    return "session_tartanground_asset_" + relative.replace("/", "_")


def enumerate_routes(source_root: Path, asset_root: Path) -> tuple[list[str], dict[str, int]]:
    """Return complete source routes with currently available scene assets."""

    all_complete = 0
    missing_assets = 0
    routes: list[str] = []
    for metadata in sorted(source_root.glob("*/Data_*/*/metadata.zip")):
        route = metadata.parent
        if not all(
            (route / name).is_file()
            for name in ("metadata.zip", "image_lcam_front.zip", "depth_lcam_front.zip")
        ):
            continue
        all_complete += 1
        scene = route.relative_to(source_root).parts[0]
        asset_dir = asset_root / scene
        if not (asset_dir / f"{scene}_sem.pcd").is_file() or not (
            asset_dir / "seg_labels.zip"
        ).is_file():
            missing_assets += 1
            continue
        routes.append(route.relative_to(source_root).as_posix())
    return routes, {
        "source_routes_complete": all_complete,
        "routes_with_assets": len(routes),
        "routes_waiting_for_assets": missing_assets,
    }


def main() -> None:
    args = parse_args()
    if args.window_attempts_per_route <= 0:
        raise ValueError("--window-attempts-per-route must be positive")
    if args.window_start_rank < 0:
        raise ValueError("--window-start-rank cannot be negative")
    source_root = args.source_root.expanduser().resolve()
    asset_root = args.asset_root.expanduser().resolve()
    selection_modes = int(bool(args.route)) + int(args.all_routes) + int(
        args.one_route_per_scene
    )
    if selection_modes > 1:
        raise ValueError(
            "--route, --all-routes, and --one-route-per-scene are mutually exclusive"
        )
    target_scenes: set[str] | None = None
    if args.all_routes or args.one_route_per_scene:
        routes, route_counts = enumerate_routes(source_root, asset_root)
        excluded_scenes = set(args.exclude_scene)
        if excluded_scenes:
            routes = [
                route
                for route in routes
                if route.split("/", 1)[0] not in excluded_scenes
            ]
            route_counts = {
                **route_counts,
                "excluded_scenes": sorted(excluded_scenes),
                "routes_after_scene_exclusion": len(routes),
            }
        if args.one_route_per_scene:
            target_scenes = {route.split("/", 1)[0] for route in routes}
            route_counts = {
                **route_counts,
                "target_scenes": len(target_scenes),
                "selection_policy": "try sorted complete routes until one passes per scene",
            }
        print(json.dumps({"route_enumeration": route_counts}), flush=True)
    else:
        routes = list(args.route) if args.route else list(DEFAULT_ROUTES)
    output = args.output_root.expanduser().resolve()
    if output.exists():
        if not args.overwrite and not args.resume:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")
        if args.overwrite:
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    palette_values = packed_palette(args.seg_rgb_file.expanduser().resolve())
    manifest_path = output / "manifest.json"
    records: list[dict] = []
    if args.resume and manifest_path.is_file():
        records = list(json.loads(manifest_path.read_text(encoding="utf-8")).get("sessions", []))
    record_by_id = {record["session_id"]: record for record in records}
    completed_scenes = {
        str(record.get("scene_id") or record["source_route"].split("/", 1)[0])
        for record in records
    }
    failures_path = output / "failures.jsonl"
    completed = 0
    skipped = 0
    failed = 0
    for index, route in enumerate(routes, start=1):
        scene = route.split("/", 1)[0]
        if args.one_route_per_scene and scene in completed_scenes:
            continue
        session_id = session_id_for_route(route)
        complete_marker = output / "sessions" / session_id / "COMPLETE"
        if args.resume and complete_marker.is_file() and session_id in record_by_id:
            skipped += 1
            print(f"SKIP {index}/{len(routes)} {route}", flush=True)
            continue
        record = None
        for window_rank in range(
            args.window_start_rank,
            args.window_start_rank + args.window_attempts_per_route,
        ):
            print(
                f"BUILD {index}/{len(routes)} {route} window={window_rank} "
                f"device={args.device}",
                flush=True,
            )
            try:
                record = build_route(
                    args,
                    route,
                    palette_values,
                    window_rank=window_rank,
                )
            except Exception as error:
                failed += 1
                partial = output / "sessions" / f".{session_id}.partial"
                if partial.exists():
                    shutil.rmtree(partial)
                with failures_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "source_route": route,
                                "source_window_rank": window_rank,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                        + "\n"
                    )
                print(
                    f"FAILED {route} window={window_rank}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                if isinstance(error, IndexError):
                    break
                continue
            break
        if record is None:
            continue
        record_by_id[record["session_id"]] = record
        completed_scenes.add(scene)
        records = sorted(record_by_id.values(), key=lambda value: value["source_route"])
        write_index(output, records)
        completed += 1
        print(
            f"DONE {record['session_id']} progress={index}/{len(routes)} "
            f"new={completed} skipped={skipped} failed={failed}",
            flush=True,
        )
    write_index(output, records)
    checks = validate_output(
        output,
        records,
        minimum_observed_known_pixels=args.minimum_observed_known_pixels,
    )
    missing_scenes = sorted((target_scenes or set()) - completed_scenes)
    print(
        json.dumps(
            {
                "output": str(output),
                "validation": checks,
                "new": completed,
                "skipped": skipped,
                "failed": failed,
                "missing_scenes": missing_scenes,
            },
            indent=2,
        )
    )
    if missing_scenes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
