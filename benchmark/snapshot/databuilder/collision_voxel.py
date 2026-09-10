"""Deterministic solid occupancy voxelization for all Habitat scene geometry."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from bev_visibility import (
    VOID_COVERAGE_ALGORITHM,
    VOID_FILTER_CONTRACT,
    VOID_REPAIR_ALGORITHM,
    repair_valid_coverage,
)
from build_void_coverage import ALGORITHM as STRICT_VOID_ALGORITHM
from build_void_coverage import _rasterize_coverage


_GLB_JSON_CHUNK = 0x4E4F534A
_GLB_BINARY_CHUNK = 0x004E4942
_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}
_SUPPORT_SURFACE_TOLERANCE_M = 0.05
_SUPPORT_SURFACE_MIN_ABS_NORMAL_Y = 0.90
_VOID_FLOOR_MARGIN_BELOW_M = 0.30
_VOID_OBSTACLE_MARGIN_ABOVE_M = 0.05
_VOID_TILE_SIZE = 256
_VOID_SURFACE_SEAL_VOXELS = 1


class CollisionVoxelError(RuntimeError):
    """Raised when collision geometry cannot be converted to occupancy."""


def _read_glb(path: Path) -> Tuple[Dict[str, Any], bytes]:
    with path.open("rb") as stream:
        magic, version, total_length = struct.unpack("<4sII", stream.read(12))
        if magic != b"glTF" or version != 2:
            raise CollisionVoxelError(f"Unsupported GLB header: {path}")
        chunks: Dict[int, bytes] = {}
        while stream.tell() < total_length:
            chunk_length, chunk_type = struct.unpack("<II", stream.read(8))
            chunks[chunk_type] = stream.read(chunk_length)

    if _GLB_JSON_CHUNK not in chunks or _GLB_BINARY_CHUNK not in chunks:
        raise CollisionVoxelError(f"GLB has no embedded JSON/BIN chunks: {path}")
    document = json.loads(chunks[_GLB_JSON_CHUNK])
    if len(document.get("buffers", [])) != 1:
        raise CollisionVoxelError(
            "Only single-buffer GLB collision assets are supported"
        )
    return document, chunks[_GLB_BINARY_CHUNK]


def _read_accessor(
    document: Dict[str, Any], binary: bytes, accessor_index: int
) -> np.ndarray:
    accessor = document["accessors"][accessor_index]
    if "sparse" in accessor or "bufferView" not in accessor:
        raise CollisionVoxelError("Sparse GLTF accessors are not supported")
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(_COMPONENT_DTYPES[accessor["componentType"]])
    components = _TYPE_COMPONENTS[accessor["type"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    item_size = dtype.itemsize * components
    stride = int(view.get("byteStride", item_size))
    result = np.ndarray(
        (int(accessor["count"]), components),
        dtype=dtype,
        buffer=binary,
        offset=offset,
        strides=(stride, dtype.itemsize),
    ).copy()
    if components == 1:
        return result[:, 0]
    return result


def _node_transform(node: Dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        # GLTF matrices are serialized in column-major order.
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T

    x, y, z, w = np.asarray(
        node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64
    )
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    scale = np.eye(4, dtype=np.float64)
    scale[:3, :3] = np.diag(node.get("scale", [1.0, 1.0, 1.0]))
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    return translation @ rotation @ scale


def _asset_to_world_matrix(
    orient_up: Sequence[float], orient_front: Sequence[float]
) -> np.ndarray:
    up = np.asarray(orient_up, dtype=np.float64)
    front = np.asarray(orient_front, dtype=np.float64)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    front /= max(float(np.linalg.norm(front)), 1e-12)
    right = np.cross(front, up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    # Habitat world convention is +Y up and -Z forward.
    return np.stack([right, up, -front], axis=0)


def _load_glb_triangles(asset_path: Path) -> np.ndarray:
    document, binary = _read_glb(asset_path)
    triangle_groups: List[np.ndarray] = []

    def visit(node_index: int, parent_transform: np.ndarray) -> None:
        node = document["nodes"][node_index]
        transform = parent_transform @ _node_transform(node)
        if "mesh" in node:
            mesh = document["meshes"][node["mesh"]]
            for primitive in mesh.get("primitives", []):
                if int(primitive.get("mode", 4)) != 4:
                    raise CollisionVoxelError(
                        "Only GLTF triangle primitives are supported"
                    )
                positions = np.asarray(
                    _read_accessor(
                        document, binary, primitive["attributes"]["POSITION"]
                    ),
                    dtype=np.float64,
                )
                homogeneous = np.concatenate(
                    [positions, np.ones((len(positions), 1), dtype=np.float64)],
                    axis=1,
                )
                asset_positions = (homogeneous @ transform.T)[:, :3]
                if "indices" in primitive:
                    indices = np.asarray(
                        _read_accessor(document, binary, primitive["indices"]),
                        dtype=np.int64,
                    )
                else:
                    indices = np.arange(len(world_positions), dtype=np.int64)
                if len(indices) % 3:
                    raise CollisionVoxelError(
                        "Triangle index count is not divisible by 3"
                    )
                triangle_groups.append(asset_positions[indices.reshape(-1, 3)])

        for child_index in node.get("children", []):
            visit(int(child_index), transform)

    scene_index = int(document.get("scene", 0))
    for root_node in document["scenes"][scene_index].get("nodes", []):
        visit(int(root_node), np.eye(4, dtype=np.float64))

    if not triangle_groups:
        raise CollisionVoxelError(
            f"No triangles found in collision asset: {asset_path}"
        )
    return np.concatenate(triangle_groups, axis=0)


def _load_binary_ply_triangles(asset_path: Path) -> np.ndarray:
    """Read Replica's binary little-endian triangle PLY without extra packages."""
    header_lines: List[str] = []
    with asset_path.open("rb") as stream:
        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise CollisionVoxelError(f"PLY header is incomplete: {asset_path}")
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise CollisionVoxelError(
                    f"PLY header is not ASCII: {asset_path}"
                ) from error
            header_lines.append(line)
            if line == "end_header":
                data_offset = stream.tell()
                break
    if not header_lines or header_lines[0] != "ply":
        raise CollisionVoxelError(f"Not a PLY file: {asset_path}")
    if "format binary_little_endian 1.0" not in header_lines:
        raise CollisionVoxelError(
            f"Only binary little-endian PLY is supported: {asset_path}"
        )

    vertex_count = 0
    face_count = 0
    active_element: Optional[str] = None
    vertex_fields: List[Tuple[str, str]] = []
    face_list: Optional[Tuple[str, str, str]] = None
    for line in header_lines:
        tokens = line.split()
        if len(tokens) == 3 and tokens[0] == "element":
            active_element = tokens[1]
            if active_element == "vertex":
                vertex_count = int(tokens[2])
            elif active_element == "face":
                face_count = int(tokens[2])
        elif tokens[:1] == ["property"] and active_element == "vertex":
            if len(tokens) != 3 or tokens[1] not in _PLY_DTYPES:
                raise CollisionVoxelError(
                    f"Unsupported PLY vertex property: {line}"
                )
            vertex_fields.append((tokens[2], _PLY_DTYPES[tokens[1]]))
        elif tokens[:2] == ["property", "list"] and active_element == "face":
            if len(tokens) != 5:
                raise CollisionVoxelError(f"Unsupported PLY face property: {line}")
            face_list = (tokens[2], tokens[3], tokens[4])
    if not vertex_count or not face_count or face_list is None:
        raise CollisionVoxelError(f"PLY has no vertices/faces: {asset_path}")
    if not {"x", "y", "z"}.issubset(name for name, _ in vertex_fields):
        raise CollisionVoxelError(f"PLY has no xyz vertex fields: {asset_path}")
    count_type, index_type, _ = face_list
    if _PLY_DTYPES.get(count_type) != "u1" or _PLY_DTYPES.get(index_type) != "<i4":
        raise CollisionVoxelError(
            "Only uint8-count/int32-index PLY triangle faces are supported"
        )

    vertex_dtype = np.dtype(vertex_fields)
    vertices = np.memmap(
        asset_path,
        mode="r",
        dtype=vertex_dtype,
        offset=data_offset,
        shape=(vertex_count,),
    )
    vertex_data_size = vertex_count * vertex_dtype.itemsize
    remaining_bytes = asset_path.stat().st_size - (
        data_offset + vertex_data_size
    )
    bytes_per_face = remaining_bytes / face_count
    if bytes_per_face == 13:
        vertices_per_face = 3
    elif bytes_per_face == 17:
        vertices_per_face = 4
    else:
        raise CollisionVoxelError(
            f"Unsupported variable/extended PLY face records: {asset_path}"
        )
    face_dtype = np.dtype(
        [("count", "u1"), ("indices", "<i4", (vertices_per_face,))]
    )
    faces = np.memmap(
        asset_path,
        mode="r",
        dtype=face_dtype,
        offset=data_offset + vertex_data_size,
        shape=(face_count,),
    )
    if np.any(faces["count"] != vertices_per_face):
        raise CollisionVoxelError(
            f"PLY contains variable-size faces: {asset_path}"
        )
    positions = np.column_stack(
        [vertices["x"], vertices["y"], vertices["z"]]
    ).astype(np.float64, copy=False)
    indices = np.asarray(faces["indices"], dtype=np.int64)
    if vertices_per_face == 4:
        indices = np.concatenate(
            [indices[:, (0, 1, 2)], indices[:, (0, 2, 3)]],
            axis=0,
        )
    return positions[indices]


def _load_asset_triangles(asset_path: Path) -> np.ndarray:
    suffix = asset_path.suffix.lower()
    if suffix == ".glb":
        return _load_glb_triangles(asset_path)
    if suffix == ".ply":
        return _load_binary_ply_triangles(asset_path)
    raise CollisionVoxelError(
        f"Unsupported collision mesh format {suffix!r}: {asset_path}"
    )


def _template_triangles(
    attributes: Any,
    cache: Optional[Dict[Tuple[Any, ...], np.ndarray]] = None,
) -> np.ndarray:
    if not attributes.use_mesh_for_collision:
        raise CollisionVoxelError(
            f"Collision template does not use a mesh: {attributes.handle}"
        )
    asset_path = Path(attributes.collision_asset_fullpath)
    if not asset_path.is_file():
        raise CollisionVoxelError(f"Collision asset is unavailable: {asset_path}")
    scale = tuple(float(value) for value in attributes.scale)
    orient_up = tuple(float(value) for value in attributes.orient_up)
    orient_front = tuple(float(value) for value in attributes.orient_front)
    cache_key = (str(asset_path), scale, orient_up, orient_front)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    triangles = _load_asset_triangles(asset_path)
    asset_to_canonical = _asset_to_world_matrix(
        orient_up, orient_front
    )
    transformed = (triangles * np.asarray(scale)) @ asset_to_canonical.T
    if cache is not None:
        cache[cache_key] = transformed
    return transformed


def load_scene_collision_triangles(
    simulator: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load stage plus every instantiated rigid object's collision geometry."""
    stage = simulator.get_stage_initialization_template()
    if stage is None:
        raise CollisionVoxelError("The active stage template is unavailable")
    template_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
    triangle_groups = [_template_triangles(stage, template_cache)]
    sources = [str(stage.collision_asset_fullpath)]
    object_count = 0

    object_manager = simulator.get_rigid_object_manager()
    objects = object_manager.get_objects_by_handle_substring("")
    object_values = objects.values() if hasattr(objects, "values") else objects
    for rigid_object in object_values:
        attributes = rigid_object.creation_attributes
        if not attributes.use_mesh_for_collision:
            continue
        object_triangles = _template_triangles(attributes, template_cache)
        transform = np.asarray(rigid_object.transformation, dtype=np.float64)
        homogeneous = np.concatenate(
            [
                object_triangles.reshape(-1, 3),
                np.ones((object_triangles.size // 3, 1), dtype=np.float64),
            ],
            axis=1,
        )
        transformed = (homogeneous @ transform.T)[:, :3]
        triangle_groups.append(transformed.reshape(-1, 3, 3))
        sources.append(str(attributes.collision_asset_fullpath))
        object_count += 1

    return np.concatenate(triangle_groups, axis=0), {
        "stage_source": str(stage.collision_asset_fullpath),
        "rigid_object_count": object_count,
        "collision_asset_count": len(sources),
        "collision_assets": sources,
    }


def _clip_against_y(
    polygon: List[np.ndarray], boundary: float, keep_above: bool
) -> List[np.ndarray]:
    if not polygon:
        return []
    result: List[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = (
        previous[1] >= boundary if keep_above else previous[1] <= boundary
    )
    for current in polygon:
        current_inside = (
            current[1] >= boundary if keep_above else current[1] <= boundary
        )
        if current_inside != previous_inside:
            denominator = current[1] - previous[1]
            if abs(float(denominator)) > 1e-12:
                fraction = (boundary - previous[1]) / denominator
                result.append(previous + fraction * (current - previous))
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return result


def _clip_triangle_to_slab(
    triangle: np.ndarray, lower_y: float, upper_y: float
) -> List[np.ndarray]:
    polygon = [triangle[0], triangle[1], triangle[2]]
    polygon = _clip_against_y(polygon, lower_y, keep_above=True)
    return _clip_against_y(polygon, upper_y, keep_above=False)


def voxelize_stage_occupancy(
    simulator: Any,
    *,
    floor_height: float,
    lower_bound: np.ndarray,
    rows: int,
    columns: int,
    voxel_size: float,
    obstacle_min_height: float,
    obstacle_max_height: float,
    navigable_map: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Voxelize collision geometry, solidify it, and max-project over Y.

    The requested height band is relative to the navigable support height.
    Habitat collision meshes often place the support floor a few centimetres
    above that height.  Those near-horizontal support triangles are removed
    explicitly instead of discarding every real obstacle below an arbitrary
    positive minimum height.
    """

    if navigable_map.shape != (rows, columns):
        raise ValueError("navigable_map shape must match the voxel grid")
    sampled_min_height = float(obstacle_min_height)
    lower_y = floor_height + sampled_min_height
    upper_y = floor_height + float(obstacle_max_height)
    layer_count = int(math.ceil((upper_y - lower_y) / voxel_size))
    if layer_count <= 0:
        raise ValueError("The obstacle height band contains no voxels")

    triangles, geometry_statistics = load_scene_collision_triangles(simulator)
    coverage_triangles = triangles
    loaded_triangle_count = int(len(triangles))
    triangle_edges_a = triangles[:, 1] - triangles[:, 0]
    triangle_edges_b = triangles[:, 2] - triangles[:, 0]
    triangle_normals = np.cross(triangle_edges_a, triangle_edges_b)
    triangle_normal_lengths = np.linalg.norm(triangle_normals, axis=1)
    abs_normal_y = np.divide(
        np.abs(triangle_normals[:, 1]),
        triangle_normal_lengths,
        out=np.zeros_like(triangle_normal_lengths),
        where=triangle_normal_lengths > 1e-12,
    )
    near_support_height = np.max(
        np.abs(triangles[:, :, 1] - float(floor_height)),
        axis=1,
    ) <= _SUPPORT_SURFACE_TOLERANCE_M
    support_surface = near_support_height & (
        abs_normal_y >= _SUPPORT_SURFACE_MIN_ABS_NORMAL_Y
    )
    filtered_support_triangle_count = int(np.count_nonzero(support_surface))
    triangles = triangles[~support_surface]

    triangle_min_y = triangles[:, :, 1].min(axis=1)
    triangle_max_y = triangles[:, :, 1].max(axis=1)
    intersects_band = (triangle_min_y <= upper_y) & (triangle_max_y >= lower_y)
    triangles = triangles[intersects_band]
    triangle_min_y = triangle_min_y[intersects_band]
    triangle_max_y = triangle_max_y[intersects_band]

    first_layer = np.floor((triangle_min_y - lower_y) / voxel_size).astype(np.int32)
    last_layer = np.floor((triangle_max_y - lower_y) / voxel_size).astype(np.int32)
    first_layer = np.clip(first_layer, 0, layer_count - 1)
    last_layer = np.clip(last_layer, 0, layer_count - 1)
    layer_triangles: List[List[int]] = [[] for _ in range(layer_count)]
    for triangle_index, (first, last) in enumerate(zip(first_layer, last_layer)):
        for layer in range(int(first), int(last) + 1):
            layer_triangles[layer].append(triangle_index)

    xmin = float(lower_bound[0])
    zmin = float(lower_bound[2])
    surface_voxels = np.zeros((layer_count, rows, columns), dtype=bool)
    for layer, triangle_indices in enumerate(layer_triangles):
        image = Image.new("1", (columns, rows), 0)
        draw = ImageDraw.Draw(image)
        slab_lower = lower_y + layer * voxel_size
        slab_upper = min(upper_y, slab_lower + voxel_size)
        for triangle_index in triangle_indices:
            polygon = _clip_triangle_to_slab(
                triangles[triangle_index], slab_lower, slab_upper
            )
            if len(polygon) < 2:
                continue
            points = [
                (
                    (float(vertex[0]) - xmin) / voxel_size,
                    (float(vertex[2]) - zmin) / voxel_size,
                )
                for vertex in polygon
            ]
            if len(points) >= 3:
                draw.polygon(points, fill=1)
            draw.line(points + [points[0]], fill=1, width=1)
        surface_voxels[layer] = np.asarray(image, dtype=bool)

    # The stage mesh is frequently a non-watertight boundary representation.
    # Navmesh cells are used only as known-air seeds. Propagation through the
    # complement of the collision surface labels air without copying the
    # navmesh's agent-radius inflation into occupancy. Unreached voxels are the
    # solid interiors bounded by collision geometry.
    known_air_seeds = np.broadcast_to(
        np.asarray(navigable_map, dtype=bool), surface_voxels.shape
    ) & ~surface_voxels
    free_voxels = ndimage.binary_propagation(
        known_air_seeds,
        structure=ndimage.generate_binary_structure(3, 1),
        mask=~surface_voxels,
    )
    occupied_voxels = ~free_voxels
    fill_only_voxels = occupied_voxels & ~surface_voxels
    projected_occupied = np.any(occupied_voxels, axis=0)
    projected_surface = np.any(surface_voxels, axis=0)
    projected_fill_only = np.any(fill_only_voxels, axis=0)
    result = np.where(projected_occupied, 0, 255).astype(np.uint8)

    # Reuse the independently verified Strict-VOID v7 geometry scan.  This is
    # intentionally separate from occupancy solidification: exterior air is
    # VOID, while collision surfaces, genuinely enclosed solids, and direct
    # same-floor NavMesh support are GT-valid.  FOV, RGB, depth, and masked BEV
    # labels do not participate.
    if STRICT_VOID_ALGORITHM != VOID_COVERAGE_ALGORITHM:
        raise RuntimeError("Strict-VOID geometry algorithm identifier mismatch")
    strict_valid_coverage, strict_statistics = _rasterize_coverage(
        coverage_triangles,
        navigable_map=np.asarray(navigable_map, dtype=bool),
        lower_y=float(floor_height) - _VOID_FLOOR_MARGIN_BELOW_M,
        upper_y=(
            float(floor_height)
            + float(obstacle_max_height)
            + _VOID_OBSTACLE_MARGIN_ABOVE_M
        ),
        lower_bound=np.asarray(lower_bound, dtype=np.float64),
        rows=rows,
        columns=columns,
        voxel_size_m=float(voxel_size),
        tile_size=_VOID_TILE_SIZE,
        surface_seal_voxels=_VOID_SURFACE_SEAL_VOXELS,
    )
    raw_strict_valid_count = int(np.count_nonzero(strict_valid_coverage))
    strict_valid_coverage = repair_valid_coverage(strict_valid_coverage)
    repaired_strict_valid_count = int(np.count_nonzero(strict_valid_coverage))
    strict_statistics.update(
        {
            "algorithm": VOID_COVERAGE_ALGORITHM,
            "filter_contract": VOID_FILTER_CONTRACT,
            "repair": VOID_REPAIR_ALGORITHM,
            "floor_margin_below_m": _VOID_FLOOR_MARGIN_BELOW_M,
            "obstacle_margin_above_m": _VOID_OBSTACLE_MARGIN_ABOVE_M,
            "raw_valid_column_count": raw_strict_valid_count,
            "repaired_valid_column_count": repaired_strict_valid_count,
            "scene_domain_repaired_cell_count": (
                repaired_strict_valid_count - raw_strict_valid_count
            ),
            "repaired_valid_fraction": float(np.mean(strict_valid_coverage)),
            "void_definition": (
                "no collision surface voxel, no enclosed solid voxel, and no "
                "same-floor NavMesh proof in the complete full-resolution 3-D "
                "scene scan"
            ),
            "independent_of_fov_masked_rgb_depth": True,
        }
    )
    statistics = {
        "loaded_collision_triangle_count": loaded_triangle_count,
        "collision_triangle_count": int(len(triangles)),
        "filtered_support_triangle_count": filtered_support_triangle_count,
        "support_surface_tolerance_m": _SUPPORT_SURFACE_TOLERANCE_M,
        "support_surface_min_abs_normal_y": (
            _SUPPORT_SURFACE_MIN_ABS_NORMAL_Y
        ),
        "voxel_layers": layer_count,
        "voxel_size_m": float(voxel_size),
        "surface_voxel_count": int(np.count_nonzero(surface_voxels)),
        "fill_only_voxel_count": int(np.count_nonzero(fill_only_voxels)),
        "solid_voxel_count": int(np.count_nonzero(occupied_voxels)),
        "occupied_column_count": int(np.count_nonzero(projected_occupied)),
        "surface_column_count": int(np.count_nonzero(projected_surface)),
        "fill_only_column_count": int(np.count_nonzero(projected_fill_only)),
        "navigable_surface_conflict_column_count": int(
            np.count_nonzero(projected_surface & navigable_map)
        ),
        "navigable_fill_only_conflict_column_count": int(
            np.count_nonzero(projected_fill_only & navigable_map)
        ),
        "source": "active Habitat stage plus all instantiated rigid objects",
        "ground_handling": (
            "0-maximum height truth with only near-horizontal collision "
            "triangles within 0.05 m of the local support height excluded"
        ),
        "strict_void_coverage": strict_statistics,
        **geometry_statistics,
    }
    return np.flipud(result), strict_valid_coverage, statistics
