#!/usr/bin/env python3
"""Simulator-ground-truth BEV construction for procedural AI2-THOR houses."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import trimesh

from bev_visibility import (
    VOID_FILTER_CONTRACT,
    VOID_REPAIR_ALGORITHM,
    VISIBILITY_ALGORITHM,
    repair_valid_coverage,
    render_visibility_masked_map,
)

UNKNOWN_VALUE = np.uint8(112)


def extent_key(extent: float) -> str:
    return f"bev_{extent:g}m".replace(".", "p")


def merged_modality(kind: str, extent: float) -> str:
    if kind not in {"masked", "complete"}:
        raise ValueError(f"unsupported merged BEV kind: {kind}")
    return f"merged_{kind}_{extent:g}m".replace(".", "p")


def _iter_objects(objects: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for obj in objects:
        yield obj
        yield from _iter_objects(obj.get("children", []))


def entity_ids(house: dict[str, Any]) -> list[str]:
    """Return each independently modeled static entity exactly once."""
    values = [obj["id"] for obj in _iter_objects(house.get("objects", []))]
    for key in ("doors", "windows", "walls"):
        values.extend(
            item["id"] for item in house.get(key, []) if not item.get("empty", False)
        )
    return list(dict.fromkeys(values))


def _floor_polygons(
    house: dict[str, Any], floor_y: float, tolerance: float = 0.15
) -> list[list[dict[str, float]]]:
    polygons: list[list[dict[str, float]]] = []
    for room in house.get("rooms", []):
        polygon = room.get("floorPolygon") or []
        if polygon and abs(float(np.median([p["y"] for p in polygon])) - floor_y) <= tolerance:
            polygons.append(polygon)
    if not polygons:
        raise RuntimeError(f"house has no room floor polygons near y={floor_y:.3f}")
    return polygons


def _geometry_bounds(
    polygons: Sequence[Sequence[dict[str, float]]], margin_m: float
) -> tuple[float, float, float, float]:
    xs = [float(point["x"]) for polygon in polygons for point in polygon]
    zs = [float(point["z"]) for polygon in polygons for point in polygon]
    return (
        min(xs) - margin_m,
        max(xs) + margin_m,
        min(zs) - margin_m,
        max(zs) + margin_m,
    )


def build_complete_truth(
    controller: Any,
    house: dict[str, Any],
    *,
    floor_y: float,
    voxel_size: float,
    obstacle_min_height: float,
    obstacle_max_height: float,
    bounds_margin_m: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Voxelize every entity mesh in the height band and project to world XZ.

    Room floor polygons define free scene support. Everything outside those
    polygons is conservatively occupied. Walls, doors, windows, furniture and
    recursively generated small objects come directly from the simulator's
    ``GetInSceneAssetGeometry`` action in their final world transforms.
    """

    if not 0.001 <= voxel_size <= 0.2:
        raise ValueError("voxel_size must be within [0.001, 0.2] metres")
    if not 0 <= obstacle_min_height < obstacle_max_height:
        raise ValueError("invalid obstacle height band")
    polygons = _floor_polygons(house, floor_y)
    min_x, max_x, min_z, max_z = _geometry_bounds(polygons, bounds_margin_m)
    # Align the global raster origin to the same integer lattice used by
    # trimesh voxelization. Mixing rounded voxel centers with a floating local
    # origin and ``floor`` can duplicate one index and skip the next, creating
    # the regular horizontal/vertical stripe artifact on solid tabletops.
    minimum_voxel_x = math.floor(min_x / voxel_size)
    maximum_voxel_x = math.ceil(max_x / voxel_size)
    minimum_voxel_z = math.floor(min_z / voxel_size)
    maximum_voxel_z = math.ceil(max_z / voxel_size)
    min_x = minimum_voxel_x * voxel_size
    max_x = maximum_voxel_x * voxel_size
    min_z = minimum_voxel_z * voxel_size
    max_z = maximum_voxel_z * voxel_size
    columns = maximum_voxel_x - minimum_voxel_x + 1
    rows = maximum_voxel_z - minimum_voxel_z + 1
    truth_image = Image.new("L", (columns, rows), color=0)
    draw = ImageDraw.Draw(truth_image)

    def pixel(point: dict[str, float]) -> tuple[int, int]:
        column = int(round(float(point["x"]) / voxel_size)) - minimum_voxel_x
        unflipped_row = (
            int(round(float(point["z"]) / voxel_size)) - minimum_voxel_z
        )
        return column, rows - 1 - unflipped_row

    for polygon in polygons:
        draw.polygon([pixel(point) for point in polygon], fill=255)
    truth = np.asarray(truth_image, dtype=np.uint8).copy()
    valid_coverage = truth == 255

    minimum_y = floor_y + obstacle_min_height
    maximum_y = floor_y + obstacle_max_height
    identifiers = entity_ids(house)
    mesh_components = 0
    voxel_points_total = 0
    projected_points_total = 0
    skipped_above_or_below = 0
    for entity_index, object_id in enumerate(identifiers):
        # Solidify each entity independently in the projected plane.  This is
        # important for imperfect/non-watertight asset meshes: a tabletop or
        # chair seat must not become a hollow contour merely because its source
        # mesh has a seam.
        entity_projection = np.zeros((rows, columns), dtype=bool)
        event = controller.step(
            action="GetInSceneAssetGeometry",
            objectId=object_id,
            triangles=True,
            renderImage=False,
        )
        if not event:
            raise RuntimeError(
                f"GetInSceneAssetGeometry failed for {object_id}: "
                f"{event.metadata.get('errorMessage')}"
            )
        geometry = event.metadata.get("actionReturn") or []
        if not geometry:
            raise RuntimeError(f"simulator returned no geometry for entity {object_id}")
        for component in geometry:
            vertices = np.asarray(
                [
                    [float(point["x"]), float(point["y"]), float(point["z"])]
                    for point in component.get("vertices", [])
                ],
                dtype=np.float64,
            )
            triangle_indices = np.asarray(component.get("triangles", []), dtype=np.int64)
            if vertices.size == 0 or triangle_indices.size == 0:
                continue
            if triangle_indices.size % 3:
                raise RuntimeError(f"invalid triangle index count for {object_id}")
            if float(vertices[:, 1].max()) < minimum_y - voxel_size or float(
                vertices[:, 1].min()
            ) > maximum_y + voxel_size:
                skipped_above_or_below += 1
                continue
            faces = triangle_indices.reshape(-1, 3)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            # ``max_iter=10`` is trimesh's default subdivision ceiling.  Large
            # ProcTHOR wall triangles legitimately need more than ten bisection
            # levels at 1 cm pitch.  Trimesh 5.0's automatic estimate can be one
            # iteration short because of floating-point/edge propagation, so
            # use a generous explicit ceiling; subdivision still stops as soon
            # as every edge satisfies the requested pitch.
            try:
                voxels = mesh.voxelized(pitch=voxel_size, max_iter=32)
            except Exception as exc:
                raise RuntimeError(
                    f"voxelization failed for entity {object_id} at "
                    f"{voxel_size:g} m: {exc}"
                ) from exc
            try:
                voxels = voxels.fill()
            except Exception as exc:
                raise RuntimeError(
                    f"solid voxel fill failed for entity {object_id}: {exc}"
                ) from exc
            points = np.asarray(voxels.points, dtype=np.float64)
            mesh_components += 1
            voxel_points_total += len(points)
            if not len(points):
                continue
            in_band = (
                (points[:, 1] >= minimum_y - voxel_size * 0.5)
                & (points[:, 1] <= maximum_y + voxel_size * 0.5)
            )
            points = points[in_band]
            if not len(points):
                continue
            columns_index = (
                np.rint(points[:, 0] / voxel_size).astype(np.int64)
                - minimum_voxel_x
            )
            unflipped_rows = (
                np.rint(points[:, 2] / voxel_size).astype(np.int64)
                - minimum_voxel_z
            )
            rows_index = rows - 1 - unflipped_rows
            valid = (
                (columns_index >= 0)
                & (columns_index < columns)
                & (rows_index >= 0)
                & (rows_index < rows)
            )
            entity_projection[rows_index[valid], columns_index[valid]] = True
            projected_points_total += int(valid.sum())
        if entity_projection.any():
            entity_projection = ndimage.binary_fill_holes(entity_projection)
            valid_coverage |= entity_projection
            truth[entity_projection] = 0
        if (entity_index + 1) % 25 == 0 or entity_index + 1 == len(identifiers):
            print(
                f"  BEV geometry {entity_index + 1:03d}/{len(identifiers):03d} entities",
                flush=True,
            )

    lower_bound = np.asarray([min_x, floor_y, min_z], dtype=np.float64)
    raw_valid_count = int(np.count_nonzero(valid_coverage))
    valid_coverage = repair_valid_coverage(valid_coverage)
    repaired_valid_count = int(np.count_nonzero(valid_coverage))
    void_coverage_algorithm = "procthor-solid-entity-or-room-floor-coverage-v1"
    statistics = {
        "truth_source": (
            "AI2-THOR GetInSceneAssetGeometry complete world meshes; solid voxel "
            "occupancy projected through configured height band on one aligned "
            "integer voxel lattice"
        ),
        "floor_source": "ProcTHOR room floorPolygon geometry",
        "voxel_size_m": voxel_size,
        "height_band_relative_to_floor_m": [
            obstacle_min_height,
            obstacle_max_height,
        ],
        "floor_y_world_m": floor_y,
        "shape_rows_columns": [rows, columns],
        "lower_bound_world_xyz_m": lower_bound.tolist(),
        "entity_count": len(identifiers),
        "mesh_component_count": mesh_components,
        "mesh_components_outside_height_band": skipped_above_or_below,
        "solid_voxel_point_count": voxel_points_total,
        "projected_voxel_point_count": projected_points_total,
        "voxel_index_policy": "global integer lattice; nearest integer index",
        "free_pixel_count": int(np.count_nonzero(truth == 255)),
        "occupied_pixel_count": int(np.count_nonzero(truth == 0)),
        "strict_void_coverage": {
            "algorithm": void_coverage_algorithm,
            "filter_contract": VOID_FILTER_CONTRACT,
            "repair": VOID_REPAIR_ALGORITHM,
            "valid_sources": [
                "complete room floorPolygon support",
                "solid projected in-band simulator entity voxels",
            ],
            "raw_valid_column_count": raw_valid_count,
            "repaired_valid_column_count": repaired_valid_count,
            "scene_domain_repaired_cell_count": (
                repaired_valid_count - raw_valid_count
            ),
            "repaired_valid_fraction": float(np.mean(valid_coverage)),
            "void_definition": (
                "outside all same-floor room polygons and without any solid "
                "in-band simulator entity voxel"
            ),
            "independent_of_fov_masked_rgb_depth": True,
        },
    }
    return truth, valid_coverage, lower_bound, statistics


def render_ego_obstacle_map(
    *,
    full_scene_map: np.ndarray,
    lower_bound: np.ndarray,
    source_meters_per_pixel: float,
    position: Sequence[float],
    forward: Sequence[float],
    right: Sequence[float],
    size: int,
    extent: float,
) -> np.ndarray:
    """Crop and rotate complete static truth around an explicit robot pose."""
    center = float(size // 2)
    output_meters_per_pixel = extent / size
    output_rows, output_columns = np.indices((size, size), dtype=np.float64)
    local_right = (output_columns - center) * output_meters_per_pixel
    local_forward = (center - output_rows) * output_meters_per_pixel
    world_x = position[0] + right[0] * local_right + forward[0] * local_forward
    world_z = position[2] + right[1] * local_right + forward[1] * local_forward
    source_columns = np.rint(
        (world_x - float(lower_bound[0])) / source_meters_per_pixel
    ).astype(np.int32)
    unflipped_rows = np.rint(
        (world_z - float(lower_bound[2])) / source_meters_per_pixel
    ).astype(np.int32)
    source_rows = full_scene_map.shape[0] - 1 - unflipped_rows
    valid = (
        (source_columns >= 0)
        & (source_columns < full_scene_map.shape[1])
        & (source_rows >= 0)
        & (source_rows < full_scene_map.shape[0])
    )
    result = np.zeros((size, size), dtype=np.uint8)
    result[valid] = full_scene_map[source_rows[valid], source_columns[valid]]
    return result


def _legacy_render_visibility_masked_map(
    ground_truth: np.ndarray, *, horizontal_fov_degrees: float
) -> np.ndarray:
    """Copy visible truth cells through exact symmetric grid shadowcasting."""
    if ground_truth.ndim != 2 or ground_truth.shape[0] != ground_truth.shape[1]:
        raise ValueError("ground_truth must be a square grayscale image")
    if not 0.0 < horizontal_fov_degrees <= 360.0:
        raise ValueError("horizontal_fov_degrees must be in (0, 360]")
    size = ground_truth.shape[0]
    obstacle = ground_truth == 0
    visible = np.zeros_like(obstacle, dtype=bool)
    origin_column = origin_row = size // 2
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
                in_bounds = 0 <= column < size and 0 <= output_row < size
                if in_bounds:
                    visible[output_row, column] = True
                cell_is_obstacle = not in_bounds or obstacle[output_row, column]
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

    transforms = (
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0, -1, 1, 0),
        (-1, 0, 0, 1),
        (-1, 0, 0, -1),
        (0, -1, -1, 0),
        (0, 1, -1, 0),
        (1, 0, 0, -1),
    )
    for transform in transforms:
        cast_octant(1, 1.0, 0.0, *transform)
    output_rows, output_columns = np.indices(ground_truth.shape, dtype=np.float64)
    local_right = output_columns - float(origin_column)
    local_forward = float(origin_row) - output_rows
    angle = np.arctan2(local_right, local_forward)
    half_fov = math.radians(horizontal_fov_degrees) / 2.0
    visible &= np.abs(angle) <= half_fov + 1e-12
    result = np.full_like(ground_truth, UNKNOWN_VALUE, dtype=np.uint8)
    result[visible] = ground_truth[visible]
    return result


def crop_corners(
    position: Sequence[float],
    forward: Sequence[float],
    right: Sequence[float],
    extent: float,
) -> np.ndarray:
    half = extent / 2.0
    return np.asarray(
        [
            [
                position[0] + right[0] * local_right + forward[0] * local_forward,
                position[2] + right[1] * local_right + forward[1] * local_forward,
            ]
            for local_right in (-half, half)
            for local_forward in (-half, half)
        ],
        dtype=np.float64,
    )


class BEVAccumulator:
    """Accumulate observed coverage while always sampling one static truth."""

    def __init__(self, extent: float, size: int) -> None:
        self.extent = float(extent)
        self.size = int(size)
        self.meters_per_pixel = self.extent / self.size
        self.minimum_x: float | None = None
        self.maximum_z: float | None = None
        self.complete = np.full((1, 1), UNKNOWN_VALUE, dtype=np.uint8)
        self.complete_known = np.zeros((1, 1), dtype=bool)
        self.observed = np.zeros((1, 1), dtype=bool)
        self.all_crop_corners: list[np.ndarray] = []

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
        old_min_x, old_max_z = self.minimum_x, self.maximum_z
        old_max_x = old_min_x + (self.complete.shape[1] - 1) * mpp
        old_min_z = old_max_z - (self.complete.shape[0] - 1) * mpp
        new_min_x, new_max_x = min(old_min_x, requested_min_x), max(
            old_max_x, requested_max_x
        )
        new_min_z, new_max_z = min(old_min_z, requested_min_z), max(
            old_max_z, requested_max_z
        )
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
        self.minimum_x, self.maximum_z = new_min_x, new_max_z

    def _ego_to_world_transform(
        self,
        position: Sequence[float],
        forward: Sequence[float],
        right: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.minimum_x is None or self.maximum_z is None:
            raise RuntimeError("accumulator bounds are not initialized")
        center = float(self.size // 2)
        mpp = self.meters_per_pixel
        matrix = np.asarray(
            [[forward[1], -forward[0]], [-right[1], right[0]]], dtype=np.float64
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
        self, complete: np.ndarray, masked: np.ndarray, extrinsic: dict[str, Any]
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
        observed_warped = ndimage.affine_transform(
            (masked != UNKNOWN_VALUE).astype(np.uint8),
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
        self, kind: str, extrinsic: dict[str, Any], merged_extent: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.all_crop_corners or self.minimum_x is None or self.maximum_z is None:
            raise RuntimeError("cannot render an empty accumulator")
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
        rendered = ndimage.affine_transform(
            self.complete,
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
            self.observed & self.complete_known if kind == "masked" else self.complete_known
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


def validate_bev_frame(
    complete: np.ndarray,
    masked: np.ndarray,
    *,
    allow_complete_unknown: bool = False,
) -> None:
    if complete.shape != masked.shape or complete.ndim != 2:
        raise RuntimeError("BEV pair has invalid or mismatched shapes")
    complete_allowed = {0, 255, 112} if allow_complete_unknown else {0, 255}
    if not set(int(v) for v in np.unique(complete)).issubset(complete_allowed):
        raise RuntimeError("complete BEV has invalid label values")
    if not set(int(v) for v in np.unique(masked)).issubset({0, 112, 255}):
        raise RuntimeError("masked BEV has invalid label values")
    if not np.any(complete == 255):
        raise RuntimeError("complete BEV contains no free cells")
    if np.all(masked == UNKNOWN_VALUE):
        raise RuntimeError("masked BEV is entirely unknown")
    known = masked != UNKNOWN_VALUE
    if not np.array_equal(masked[known], complete[known]):
        raise RuntimeError("masked BEV known cells disagree with complete truth")
    center = complete.shape[0] // 2
    if complete[center, center] != 255:
        raise RuntimeError("robot origin is occupied in complete BEV")
