#!/usr/bin/env python3
"""Build immutable strict 3-D geometry coverage sidecars for legacy BEV data.

Collision triangles are clipped into every one-voxel-thick Y slab at the source
BEV resolution.  Empty-volume components are labeled in bounded-memory XZ tiles
and joined across tile borders.  Same-floor NavMesh cells and true full-scene
XZ boundaries seed known air; unseeded enclosed volume is solid.  A BEV column
is valid only when it contains a collision surface/solid voxel or direct
same-floor NavMesh proof.  No 2-D topology repair, FOV, or masked GT participates
in this build.  The raw sessions and semantic labels are never modified.

The command has three deliberately separate phases so a 200K-session manifest
is scanned once and scene builds can be distributed across independent GPUs:

  plan     scan metadata and group scene floor heights
  build    build one deterministic worker shard
  finalize verify every artifact and write the training index
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

FORMAT_VERSION = 2
ALGORITHM = "strict-solid-voxel-or-navmesh-coverage-v4"


@dataclass(frozen=True)
class FloorBand:
    index: int
    minimum_floor_m: float
    maximum_floor_m: float
    session_count: int

    @property
    def center_floor_m(self) -> float:
        return 0.5 * (self.minimum_floor_m + self.maximum_floor_m)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(_json_bytes(payload))
    os.replace(temporary, path)


def _manifest_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(payload.get("train", ())) + list(payload.get("validation", ()))
    if not rows:
        raise ValueError("split manifest contains no train/validation sessions")
    if len({str(row["key"]) for row in rows}) != len(rows):
        raise ValueError("split manifest contains duplicate session keys")
    return rows


def _greedy_floor_bands(
    floor_counts: dict[float, int], maximum_span_m: float
) -> list[FloorBand]:
    if maximum_span_m <= 0.0:
        raise ValueError("maximum floor-band span must be positive")
    values = sorted((float(floor), int(count)) for floor, count in floor_counts.items())
    bands: list[FloorBand] = []
    active: list[tuple[float, int]] = []
    for floor, count in values:
        if active and floor - active[0][0] > maximum_span_m:
            bands.append(
                FloorBand(
                    index=len(bands),
                    minimum_floor_m=active[0][0],
                    maximum_floor_m=active[-1][0],
                    session_count=sum(item[1] for item in active),
                )
            )
            active = []
        active.append((floor, count))
    if active:
        bands.append(
            FloorBand(
                index=len(bands),
                minimum_floor_m=active[0][0],
                maximum_floor_m=active[-1][0],
                session_count=sum(item[1] for item in active),
            )
        )
    return bands


def _artifact_relative_path(scene_key: str, band_index: int) -> str:
    digest = hashlib.sha256(scene_key.encode("utf-8")).hexdigest()
    return f"coverage/{digest[:2]}/{digest}/floor_{band_index:03d}.npz"


def create_plan(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _manifest_rows(payload)
    if Path(payload["dataset_root"]).resolve() != dataset_root:
        raise ValueError("manifest dataset_root does not match --dataset-root")

    floor_counts: dict[str, dict[float, int]] = defaultdict(lambda: defaultdict(int))
    scene_metadata: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        session = dataset_root / str(row["key"])
        metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
        scene_key = f"{metadata['dataset']}:{metadata['scene_id']}"
        if scene_key != str(row["scene_key"]):
            raise ValueError(f"scene mismatch for {row['key']}")
        floor_m = float(metadata["path"]["start"][1])
        # Habitat values vary at float noise scale.  One millimetre precision
        # preserves real ramps while avoiding thousands of duplicate labels.
        floor_counts[scene_key][round(floor_m, 3)] += 1
        fields = {
            "dataset": str(metadata["dataset"]),
            "scene_id": str(metadata["scene_id"]),
            "scene_file": str(metadata["scene_file"]),
            "dataset_root": str(metadata["dataset_root"]),
            "scene_dataset_config": metadata.get("scene_dataset_config"),
            "navmesh_file": metadata.get("navmesh_file"),
            "voxel_size_m": float(metadata["bev"]["voxel_size_m"]),
            "obstacle_max_height_m": float(
                metadata["bev"]["obstacle_height_band_m"][1]
            ),
        }
        previous = scene_metadata.setdefault(scene_key, fields)
        # Collector workers keep identical generated NavMeshes in separate
        # GPU-specific cache directories.  Their paths may differ while all
        # geometry-defining fields remain identical.
        geometry_fields = tuple(key for key in fields if key != "navmesh_file")
        if any(previous[key] != fields[key] for key in geometry_fields):
            raise ValueError(f"inconsistent source metadata for {scene_key}")
        if not previous.get("navmesh_file") and fields.get("navmesh_file"):
            previous["navmesh_file"] = fields["navmesh_file"]
        if number % args.log_every_sessions == 0:
            print(f"planned {number:,}/{len(rows):,} sessions", flush=True)

    scenes: list[dict[str, Any]] = []
    total_bands = 0
    for scene_key in sorted(scene_metadata):
        bands = _greedy_floor_bands(
            floor_counts[scene_key], args.maximum_floor_band_span_m
        )
        total_bands += len(bands)
        scenes.append(
            {
                "scene_key": scene_key,
                **scene_metadata[scene_key],
                "bands": [
                    {
                        **asdict(band),
                        "center_floor_m": band.center_floor_m,
                        "coverage_lower_y_m": (
                            band.minimum_floor_m - args.floor_margin_below_m
                        ),
                        "coverage_upper_y_m": (
                            band.maximum_floor_m
                            + scene_metadata[scene_key]["obstacle_max_height_m"]
                            + args.obstacle_margin_above_m
                        ),
                        "artifact": _artifact_relative_path(scene_key, band.index),
                    }
                    for band in bands
                ],
            }
        )

    plan = {
        "format_version": FORMAT_VERSION,
        "algorithm": ALGORITHM,
        "source_manifest": str(manifest_path),
        "source_manifest_content_sha256": str(payload["content_sha256"]),
        "dataset_root": str(dataset_root),
        "session_count": len(rows),
        "scene_count": len(scenes),
        "floor_band_count": total_bands,
        "maximum_floor_band_span_m": args.maximum_floor_band_span_m,
        "floor_margin_below_m": args.floor_margin_below_m,
        "obstacle_margin_above_m": args.obstacle_margin_above_m,
        "strict_voxel_tile_size": args.strict_voxel_tile_size,
        "surface_seal_voxels": args.surface_seal_voxels,
        "scenes": scenes,
    }
    plan["content_sha256"] = _sha256_bytes(_json_bytes(plan))
    _atomic_json(args.plan.expanduser().resolve(), plan)
    print(
        f"wrote plan: {len(rows):,} sessions, {len(scenes):,} scenes, "
        f"{total_bands:,} floor bands",
        flush=True,
    )


def _load_collector_modules(collector_root: Path) -> tuple[Any, Any]:
    root = collector_root.expanduser().resolve()
    if not (root / "collect_random_sessions.py").is_file():
        raise FileNotFoundError(f"collector source is unavailable: {root}")
    sys.path.insert(0, str(root))
    collector = importlib.import_module("collect_random_sessions")
    collision = importlib.import_module("collision_voxel")
    return collector, collision


def _clip_against_y(
    polygon: list[np.ndarray], boundary: float, *, keep_above: bool
) -> list[np.ndarray]:
    """Clip a 3-D polygon against one horizontal half-space."""

    if not polygon:
        return []
    result: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = (
        previous[1] >= boundary if keep_above else previous[1] <= boundary
    )
    for current in polygon:
        current_inside = (
            current[1] >= boundary if keep_above else current[1] <= boundary
        )
        if current_inside != previous_inside:
            denominator = float(current[1] - previous[1])
            if abs(denominator) > 1e-12:
                fraction = (boundary - float(previous[1])) / denominator
                result.append(previous + fraction * (current - previous))
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return result


def _clip_triangle_to_slab(
    triangle: np.ndarray, lower_y: float, upper_y: float
) -> list[np.ndarray]:
    polygon = _clip_against_y(
        [triangle[0], triangle[1], triangle[2]], lower_y, keep_above=True
    )
    return _clip_against_y(polygon, upper_y, keep_above=False)


class _DisjointSet:
    """Small union-find used to join 3-D air components across XZ tiles."""

    def __init__(self) -> None:
        self.parent = [0]
        self.rank = [0]
        self.seeded_air = [False]

    def add(self, count: int) -> int:
        base = len(self.parent) - 1
        for item in range(count):
            identifier = base + item + 1
            self.parent.append(identifier)
            self.rank.append(0)
            self.seeded_air.append(False)
        return base

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: int, second: int) -> None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        self.seeded_air[left] = self.seeded_air[left] or self.seeded_air[right]
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1

    def mark_seeded_air(self, item: int) -> None:
        if item:
            self.seeded_air[self.find(item)] = True

    def is_seeded_air(self, item: int) -> bool:
        return bool(item and self.seeded_air[self.find(item)])


class _StrictSurfaceVoxelizer:
    """Conservatively intersect collision triangles with every 3-D Y slab.

    Unlike the v3 builder, triangles are never projected wholesale into XZ.
    Each triangle is first clipped to each one-voxel-thick Y slab, then its
    clipped polygon is rasterized into that slab.  Processing XZ tiles keeps
    memory bounded without changing the source voxel size.
    """

    def __init__(
        self,
        triangles: np.ndarray,
        *,
        lower_y: float,
        upper_y: float,
        lower_bound: np.ndarray,
        rows: int,
        columns: int,
        voxel_size_m: float,
    ) -> None:
        self.lower_y = float(lower_y)
        self.upper_y = float(upper_y)
        self.lower_x = float(lower_bound[0])
        self.lower_z = float(lower_bound[2])
        self.rows = int(rows)
        self.columns = int(columns)
        self.voxel_size_m = float(voxel_size_m)
        self.layer_count = int(
            math.ceil((self.upper_y - self.lower_y) / self.voxel_size_m)
        )
        if self.layer_count <= 0:
            raise ValueError("strict voxel height band contains no layers")

        triangles = np.asarray(triangles, dtype=np.float64)
        minimum = triangles.min(axis=1)
        maximum = triangles.max(axis=1)
        in_height_band = (minimum[:, 1] <= self.upper_y) & (
            maximum[:, 1] >= self.lower_y
        )
        self.triangles = triangles[in_height_band]
        self.minimum = minimum[in_height_band]
        self.maximum = maximum[in_height_band]

    def tile(
        self, row_start: int, row_stop: int, column_start: int, column_stop: int
    ) -> tuple[np.ndarray, int]:
        height = row_stop - row_start
        width = column_stop - column_start
        surface = np.zeros((self.layer_count, height, width), dtype=np.uint8)
        tile_min_x = self.lower_x + column_start * self.voxel_size_m
        tile_max_x = self.lower_x + column_stop * self.voxel_size_m
        tile_min_z = self.lower_z + row_start * self.voxel_size_m
        tile_max_z = self.lower_z + row_stop * self.voxel_size_m
        overlaps = (
            (self.minimum[:, 0] <= tile_max_x)
            & (self.maximum[:, 0] >= tile_min_x)
            & (self.minimum[:, 2] <= tile_max_z)
            & (self.maximum[:, 2] >= tile_min_z)
        )
        indices = np.flatnonzero(overlaps)
        layer_contours: list[list[np.ndarray]] = [
            [] for _ in range(self.layer_count)
        ]
        for triangle_index in indices:
            triangle = self.triangles[triangle_index]
            first = max(
                0,
                int(
                    math.floor(
                        (float(self.minimum[triangle_index, 1]) - self.lower_y)
                        / self.voxel_size_m
                    )
                ),
            )
            last = min(
                self.layer_count - 1,
                int(
                    math.floor(
                        (float(self.maximum[triangle_index, 1]) - self.lower_y)
                        / self.voxel_size_m
                    )
                ),
            )
            for layer in range(first, last + 1):
                slab_lower = self.lower_y + layer * self.voxel_size_m
                slab_upper = min(
                    self.upper_y, slab_lower + self.voxel_size_m
                )
                polygon = _clip_triangle_to_slab(
                    triangle, slab_lower, slab_upper
                )
                if len(polygon) < 2:
                    continue
                points = np.asarray(
                    [
                        [
                            math.floor(
                                (float(vertex[0]) - self.lower_x)
                                / self.voxel_size_m
                            )
                            - column_start,
                            math.floor(
                                (float(vertex[2]) - self.lower_z)
                                / self.voxel_size_m
                            )
                            - row_start,
                        ]
                        for vertex in polygon
                    ],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                layer_contours[layer].append(points)

        try:
            import cv2

            for layer, contours in enumerate(layer_contours):
                if not contours:
                    continue
                polygons = [item for item in contours if len(item) >= 3]
                if polygons:
                    cv2.fillPoly(surface[layer], polygons, 1)
                cv2.polylines(surface[layer], contours, True, 1, 1)
        except ImportError:
            from PIL import Image, ImageDraw

            for layer, contours in enumerate(layer_contours):
                if not contours:
                    continue
                image = Image.new("1", (width, height), 0)
                draw = ImageDraw.Draw(image)
                for contour in contours:
                    points = [
                        tuple(int(value) for value in point)
                        for point in contour[:, 0]
                    ]
                    if len(points) >= 3:
                        draw.polygon(points, fill=1)
                    draw.line(points + [points[0]], fill=1, width=1)
                surface[layer] = np.asarray(image, dtype=np.uint8)
        return surface.astype(bool, copy=False), len(indices)


def _union_border_components(
    components: _DisjointSet,
    first_labels: np.ndarray,
    first_base: int,
    second_labels: np.ndarray,
    second_base: int,
) -> None:
    connected = (first_labels != 0) & (second_labels != 0)
    if not np.any(connected):
        return
    pairs = np.column_stack(
        [
            first_labels[connected].astype(np.int64) + first_base,
            second_labels[connected].astype(np.int64) + second_base,
        ]
    )
    for first, second in np.unique(pairs, axis=0):
        components.union(int(first), int(second))


def _sealed_surface_tile(
    voxelizer: _StrictSurfaceVoxelizer,
    *,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
    seal_voxels: int,
) -> tuple[np.ndarray, int]:
    """Rasterize with an XZ halo and close only sub-voxel mesh seams in 3-D."""

    if seal_voxels < 0:
        raise ValueError("surface seal radius cannot be negative")
    if not seal_voxels:
        return voxelizer.tile(row_start, row_stop, column_start, column_stop)
    halo_row_start = max(0, row_start - seal_voxels)
    halo_row_stop = min(voxelizer.rows, row_stop + seal_voxels)
    halo_column_start = max(0, column_start - seal_voxels)
    halo_column_stop = min(voxelizer.columns, column_stop + seal_voxels)
    surface, selected_count = voxelizer.tile(
        halo_row_start,
        halo_row_stop,
        halo_column_start,
        halo_column_stop,
    )
    surface = ndimage.binary_dilation(
        surface,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=seal_voxels,
    )
    row_slice = slice(
        row_start - halo_row_start,
        row_stop - halo_row_start,
    )
    column_slice = slice(
        column_start - halo_column_start,
        column_stop - halo_column_start,
    )
    return surface[:, row_slice, column_slice], selected_count


def _rasterize_coverage(
    triangles: np.ndarray,
    *,
    navigable_map: np.ndarray,
    lower_y: float,
    upper_y: float,
    lower_bound: np.ndarray,
    rows: int,
    columns: int,
    voxel_size_m: float,
    tile_size: int,
    surface_seal_voxels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if tile_size <= 0:
        raise ValueError("strict voxel tile size must be positive")
    if navigable_map.shape != (rows, columns):
        raise ValueError("NavMesh and collision coverage shapes do not match")
    navmesh = np.asarray(navigable_map, dtype=bool)
    voxelizer = _StrictSurfaceVoxelizer(
        triangles,
        lower_y=lower_y,
        upper_y=upper_y,
        lower_bound=lower_bound,
        rows=rows,
        columns=columns,
        voxel_size_m=voxel_size_m,
    )
    structure = ndimage.generate_binary_structure(3, 1)
    tile_rows = int(math.ceil(rows / tile_size))
    tile_columns = int(math.ceil(columns / tile_size))
    components = _DisjointSet()
    records: dict[tuple[int, int], tuple[int, int]] = {}
    above_borders: dict[int, tuple[np.ndarray, int]] = {}
    surface_count = 0
    selected_triangle_visits = 0

    # Pass one: label empty 3-D components per tile and join components across
    # all XZ tile borders.  Only same-floor NavMesh air and true full-scene XZ
    # boundaries seed air; Y slab boundaries are not treated as exterior because
    # they may cut through a wall or object.
    for tile_row in range(tile_rows):
        current_bottoms: dict[int, tuple[np.ndarray, int]] = {}
        left_border: tuple[np.ndarray, int] | None = None
        for tile_column in range(tile_columns):
            row_start = tile_row * tile_size
            row_stop = min(rows, row_start + tile_size)
            column_start = tile_column * tile_size
            column_stop = min(columns, column_start + tile_size)
            surface, selected_count = _sealed_surface_tile(
                voxelizer,
                row_start=row_start,
                row_stop=row_stop,
                column_start=column_start,
                column_stop=column_stop,
                seal_voxels=surface_seal_voxels,
            )
            selected_triangle_visits += selected_count
            surface_count += int(np.count_nonzero(surface))
            labels, label_count = ndimage.label(~surface, structure=structure)
            base = components.add(int(label_count))
            records[(tile_row, tile_column)] = (base, int(label_count))

            local_navmesh = navmesh[
                row_start:row_stop, column_start:column_stop
            ]
            if np.any(local_navmesh):
                for label in np.unique(labels[:, local_navmesh]):
                    if label:
                        components.mark_seeded_air(base + int(label))
            if row_start == 0:
                for label in np.unique(labels[:, 0, :]):
                    if label:
                        components.mark_seeded_air(base + int(label))
            if row_stop == rows:
                for label in np.unique(labels[:, -1, :]):
                    if label:
                        components.mark_seeded_air(base + int(label))
            if column_start == 0:
                for label in np.unique(labels[:, :, 0]):
                    if label:
                        components.mark_seeded_air(base + int(label))
            if column_stop == columns:
                for label in np.unique(labels[:, :, -1]):
                    if label:
                        components.mark_seeded_air(base + int(label))

            if left_border is not None:
                _union_border_components(
                    components,
                    left_border[0],
                    left_border[1],
                    labels[:, :, 0],
                    base,
                )
            if tile_column in above_borders:
                above, above_base = above_borders[tile_column]
                _union_border_components(
                    components,
                    above,
                    above_base,
                    labels[:, 0, :],
                    base,
                )
            left_border = (labels[:, :, -1].copy(), base)
            current_bottoms[tile_column] = (labels[:, -1, :].copy(), base)
        above_borders = current_bottoms

    # Pass two: deterministically recreate each tile and retain collision
    # surfaces, genuinely enclosed solid voxels, and direct NavMesh floor proof.
    coverage = np.zeros((rows, columns), dtype=bool)
    solid_count = 0
    air_count = 0
    for tile_row in range(tile_rows):
        for tile_column in range(tile_columns):
            row_start = tile_row * tile_size
            row_stop = min(rows, row_start + tile_size)
            column_start = tile_column * tile_size
            column_stop = min(columns, column_start + tile_size)
            surface, _ = _sealed_surface_tile(
                voxelizer,
                row_start=row_start,
                row_stop=row_stop,
                column_start=column_start,
                column_stop=column_stop,
                seal_voxels=surface_seal_voxels,
            )
            labels, label_count = ndimage.label(~surface, structure=structure)
            base, expected_count = records[(tile_row, tile_column)]
            if int(label_count) != expected_count:
                raise RuntimeError("strict voxel component pass is not deterministic")
            air_lookup = np.zeros(int(label_count) + 1, dtype=bool)
            for label in range(1, int(label_count) + 1):
                air_lookup[label] = components.is_seeded_air(base + label)
            air = (~surface) & air_lookup[labels]
            solid = (~surface) & ~air
            modeled = surface | solid
            local_navmesh = navmesh[
                row_start:row_stop, column_start:column_stop
            ]
            coverage[row_start:row_stop, column_start:column_stop] = (
                np.any(modeled, axis=0) | local_navmesh
            )
            solid_count += int(np.count_nonzero(solid))
            air_count += int(np.count_nonzero(air))

    # Match the source complete-BEV image convention used by the collector.
    flipped_coverage = np.flipud(coverage)
    flipped_navmesh = np.flipud(navmesh)
    return flipped_coverage, {
        "collision_triangle_count": int(len(triangles)),
        "slab_triangle_count": int(len(voxelizer.triangles)),
        "voxel_layer_count": int(voxelizer.layer_count),
        "surface_voxel_count": surface_count,
        "enclosed_solid_voxel_count": solid_count,
        "seeded_air_voxel_count": air_count,
        "navmesh_valid_column_count": int(np.count_nonzero(flipped_navmesh)),
        "valid_column_count": int(np.count_nonzero(flipped_coverage)),
        "valid_fraction": float(np.mean(flipped_coverage)),
        "strict_voxel_tile_size": int(tile_size),
        "surface_seal_voxels": int(surface_seal_voxels),
        "surface_seal_m": float(surface_seal_voxels * voxel_size_m),
        "tile_count": int(tile_rows * tile_columns),
        "selected_triangle_tile_visits": int(selected_triangle_visits),
        "connectivity": "3d-6-neighbour-across-all-xz-tiles",
        "air_seeds": "same-floor-navmesh-plus-full-scene-xz-boundary",
        "surface_backend": "triangle-clipped-per-y-voxel-slab",
        "two_dimensional_hole_fill": False,
    }


def _save_coverage(path: Path, coverage: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    packed = np.packbits(coverage.reshape(-1), bitorder="little")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            valid_bits=packed,
            shape=np.asarray(coverage.shape, dtype=np.int32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    os.replace(temporary, path)


def _worker_owns(scene_key: str, worker_index: int, worker_count: int) -> bool:
    digest = hashlib.sha256(scene_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % worker_count == worker_index


def build_worker(args: argparse.Namespace) -> None:
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker index must be in [0, worker-count)")
    plan_path = args.plan.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("algorithm") != ALGORITHM:
        raise ValueError("unsupported coverage plan")
    output_root = args.output_root.expanduser().resolve()
    collector, collision = _load_collector_modules(args.collector_root)
    scenes = [
        scene
        for scene in plan["scenes"]
        if _worker_owns(scene["scene_key"], args.worker_index, args.worker_count)
    ]
    if args.maximum_scenes is not None:
        scenes = scenes[: args.maximum_scenes]
    completed = 0
    for scene_number, scene in enumerate(scenes, start=1):
        expected = [output_root / band["artifact"] for band in scene["bands"]]
        if args.resume and all(path.is_file() for path in expected):
            completed += 1
            continue
        asset = collector.SceneAsset(
            dataset=scene["dataset"],
            scene_id=scene["scene_id"],
            scene_file=Path(scene["scene_file"]),
            dataset_root=Path(scene["dataset_root"]),
            scene_dataset_config=(
                Path(scene["scene_dataset_config"])
                if scene.get("scene_dataset_config")
                else None
            ),
            navmesh_file=(
                Path(scene["navmesh_file"]) if scene.get("navmesh_file") else None
            ),
        )
        first_band = scene["bands"][0]
        with collector.make_simulator(
            asset,
            camera_width=64,
            camera_height=64,
            horizontal_fov_degrees=90.0,
            sensor_height_m=0.5,
            gpu_device_id=args.habitat_gpu_device,
        ) as simulator:
            collector.validate_loaded_asset(
                simulator,
                asset,
                args.navmesh_cache.expanduser().resolve(),
                64,
                64,
            )
            lower_bound, _ = simulator.pathfinder.get_bounds()
            lower_bound = np.asarray(lower_bound, dtype=np.float64)
            representative = float(first_band["center_floor_m"])
            representative_navigable = simulator.pathfinder.get_topdown_view(
                float(scene["voxel_size_m"]), representative
            )
            rows, columns = (
                int(value) for value in representative_navigable.shape
            )
            triangles, geometry = collision.load_scene_collision_triangles(simulator)
            for band in scene["bands"]:
                path = output_root / band["artifact"]
                if args.resume and path.is_file():
                    continue
                navigable = simulator.pathfinder.get_topdown_view(
                    float(scene["voxel_size_m"]),
                    float(band["center_floor_m"]),
                )
                coverage, statistics = _rasterize_coverage(
                    triangles,
                    navigable_map=navigable,
                    lower_y=float(band["coverage_lower_y_m"]),
                    upper_y=float(band["coverage_upper_y_m"]),
                    lower_bound=lower_bound,
                    rows=rows,
                    columns=columns,
                    voxel_size_m=float(scene["voxel_size_m"]),
                    tile_size=int(plan["strict_voxel_tile_size"]),
                    surface_seal_voxels=int(plan.get("surface_seal_voxels", 0)),
                )
                metadata = {
                    "format_version": FORMAT_VERSION,
                    "algorithm": ALGORITHM,
                    "plan_content_sha256": plan["content_sha256"],
                    "scene_key": scene["scene_key"],
                    "band_index": int(band["index"]),
                    "minimum_floor_m": float(band["minimum_floor_m"]),
                    "maximum_floor_m": float(band["maximum_floor_m"]),
                    "coverage_lower_y_m": float(band["coverage_lower_y_m"]),
                    "coverage_upper_y_m": float(band["coverage_upper_y_m"]),
                    "lower_bound_xz_m": [
                        float(lower_bound[0]),
                        float(lower_bound[2]),
                    ],
                    "voxel_size_m": float(scene["voxel_size_m"]),
                    "shape": [rows, columns],
                    "strict_voxel_tile_size": int(
                        plan["strict_voxel_tile_size"]
                    ),
                    "surface_seal_voxels": int(
                        plan.get("surface_seal_voxels", 0)
                    ),
                    "geometry": geometry,
                    "statistics": statistics,
                }
                _save_coverage(path, coverage, metadata)
        completed += 1
        print(
            f"worker {args.worker_index}: {scene_number}/{len(scenes)} "
            f"{scene['scene_key']} ({len(scene['bands'])} bands)",
            flush=True,
        )
    print(
        f"worker {args.worker_index} complete: {completed}/{len(scenes)} scenes",
        flush=True,
    )


def _load_artifact_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        shape = tuple(int(value) for value in archive["shape"])
        expected_bytes = (shape[0] * shape[1] + 7) // 8
        if archive["valid_bits"].dtype != np.uint8:
            raise ValueError(f"coverage bits are not uint8: {path}")
        if int(archive["valid_bits"].size) != expected_bytes:
            raise ValueError(f"coverage bit count does not match shape: {path}")
        return json.loads(str(archive["metadata_json"].item()))


def finalize(args: argparse.Namespace) -> None:
    plan_path = args.plan.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scenes: dict[str, Any] = {}
    missing: list[str] = []
    for scene in plan["scenes"]:
        bands = []
        for band in scene["bands"]:
            path = output_root / band["artifact"]
            if not path.is_file():
                missing.append(str(path))
                continue
            metadata = _load_artifact_metadata(path)
            if metadata["plan_content_sha256"] != plan["content_sha256"]:
                raise ValueError(f"coverage artifact belongs to another plan: {path}")
            bands.append(
                {
                    "index": int(band["index"]),
                    "minimum_floor_m": float(band["minimum_floor_m"]),
                    "maximum_floor_m": float(band["maximum_floor_m"]),
                    "center_floor_m": float(band["center_floor_m"]),
                    "session_count": int(band["session_count"]),
                    "artifact": str(Path(band["artifact"])),
                    "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "lower_bound_xz_m": metadata["lower_bound_xz_m"],
                    "voxel_size_m": metadata["voxel_size_m"],
                    "shape": metadata["shape"],
                    "valid_fraction": metadata["statistics"]["valid_fraction"],
                }
            )
        if bands:
            scenes[scene["scene_key"]] = {"bands": bands}
    if missing and not args.allow_partial:
        raise FileNotFoundError(
            f"{len(missing)} coverage artifacts are missing; first: {missing[0]}"
        )
    indexed_band_count = sum(
        len(scene["bands"]) for scene in scenes.values()
    )
    indexed_session_count = sum(
        int(band["session_count"])
        for scene in scenes.values()
        for band in scene["bands"]
    )
    index = {
        "format_version": FORMAT_VERSION,
        "algorithm": ALGORITHM,
        "plan_content_sha256": plan["content_sha256"],
        "source_manifest_content_sha256": plan["source_manifest_content_sha256"],
        "dataset_root": plan["dataset_root"],
        "session_count": indexed_session_count,
        "scene_count": len(scenes),
        "floor_band_count": indexed_band_count,
        "planned_session_count": plan["session_count"],
        "planned_scene_count": plan["scene_count"],
        "planned_floor_band_count": plan["floor_band_count"],
        "partial": bool(missing),
        "missing_floor_band_count": len(missing),
        "artifact_root": str(output_root),
        "void_definition": (
            "no collision surface voxel, no enclosed solid voxel, and no "
            "same-floor NavMesh proof in the full-resolution 3-D scan; raw "
            "semantic GT remains immutable"
        ),
        "scenes": scenes,
    }
    index["content_sha256"] = _sha256_bytes(_json_bytes(index))
    _atomic_json(args.index.expanduser().resolve(), index)
    print(
        f"finalized {index['scene_count']:,} scenes / "
        f"{index['floor_band_count']:,} floor bands "
        f"(partial={index['partial']})",
        flush=True,
    )


def bind_manifest(args: argparse.Namespace) -> None:
    """Bind a complete parent coverage index to a verified manifest subset.

    Coverage is scene/floor based, so a frozen session subset can reuse the
    exact same immutable sidecars.  The new index records both parent hashes
    and refuses any session or scene that is not present in the parent
    manifest/index.
    """

    parent_index_path = args.parent_index.expanduser().resolve()
    parent_manifest_path = args.parent_manifest.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    parent_index = json.loads(parent_index_path.read_text(encoding="utf-8"))
    parent_manifest = json.loads(
        parent_manifest_path.read_text(encoding="utf-8")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if bool(parent_index.get("partial", True)):
        raise ValueError("cannot bind a partial Void coverage index")
    if int(parent_index.get("missing_floor_band_count", -1)) != 0:
        raise ValueError("parent Void coverage index has missing floor bands")
    if (
        str(parent_index["source_manifest_content_sha256"])
        != str(parent_manifest["content_sha256"])
    ):
        raise ValueError("parent index and parent manifest hashes do not match")
    if Path(parent_index["dataset_root"]).resolve() != Path(
        manifest["dataset_root"]
    ).resolve():
        raise ValueError("target manifest belongs to another dataset root")

    parent_rows = _manifest_rows(parent_manifest)
    target_rows = _manifest_rows(manifest)
    parent_keys = {str(row["key"]) for row in parent_rows}
    target_keys = {str(row["key"]) for row in target_rows}
    missing_keys = sorted(target_keys - parent_keys)
    if missing_keys:
        raise ValueError(
            f"target manifest is not a parent subset; first: {missing_keys[0]}"
        )
    target_scene_keys = {str(row["scene_key"]) for row in target_rows}
    missing_scenes = sorted(target_scene_keys - set(parent_index["scenes"]))
    if missing_scenes:
        raise ValueError(
            f"parent coverage lacks target scene: {missing_scenes[0]}"
        )

    scenes = {
        scene_key: parent_index["scenes"][scene_key]
        for scene_key in sorted(target_scene_keys)
    }
    artifact_root = Path(parent_index["artifact_root"]).resolve()
    for scene in scenes.values():
        for band in scene["bands"]:
            artifact = (artifact_root / str(band["artifact"])).resolve()
            if not artifact.is_relative_to(artifact_root):
                raise ValueError("coverage artifact escapes its root")
            if not artifact.is_file():
                raise FileNotFoundError(f"coverage artifact is missing: {artifact}")
            actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if actual_hash != str(band["artifact_sha256"]):
                raise ValueError(f"coverage artifact hash mismatch: {artifact}")

    floor_band_count = sum(
        len(scene["bands"]) for scene in scenes.values()
    )
    index = {
        "format_version": int(parent_index["format_version"]),
        "algorithm": str(parent_index["algorithm"]),
        "plan_content_sha256": str(parent_index["plan_content_sha256"]),
        "source_manifest_content_sha256": str(manifest["content_sha256"]),
        "source_parent_manifest_content_sha256": str(
            parent_manifest["content_sha256"]
        ),
        "source_parent_index_content_sha256": str(
            parent_index["content_sha256"]
        ),
        "dataset_root": str(Path(parent_index["dataset_root"]).resolve()),
        "session_count": len(target_rows),
        "scene_count": len(scenes),
        "floor_band_count": floor_band_count,
        "planned_session_count": len(target_rows),
        "planned_scene_count": len(scenes),
        "planned_floor_band_count": floor_band_count,
        "partial": False,
        "missing_floor_band_count": 0,
        "artifact_root": str(artifact_root),
        "void_definition": str(parent_index["void_definition"]),
        "subset_binding": "verified-session-subset-reuses-parent-sidecars-v1",
        "scenes": scenes,
    }
    index["content_sha256"] = _sha256_bytes(_json_bytes(index))
    _atomic_json(args.index.expanduser().resolve(), index)
    print(
        f"bound {index['session_count']:,} sessions / "
        f"{index['scene_count']:,} scenes / "
        f"{index['floor_band_count']:,} floor bands",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--dataset-root", type=Path, required=True)
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--maximum-floor-band-span-m", type=float, default=0.50)
    # Habitat NavMesh points can sit about 0.20 m above the underlying HSSD
    # floor triangles. A 0.30 m margin includes the complete support polygon,
    # which defines a continuous modeled interior instead of sparse columns.
    plan.add_argument("--floor-margin-below-m", type=float, default=0.30)
    plan.add_argument("--obstacle-margin-above-m", type=float, default=0.05)
    plan.add_argument("--strict-voxel-tile-size", type=int, default=256)
    plan.add_argument("--surface-seal-voxels", type=int, default=1)
    plan.add_argument("--log-every-sessions", type=int, default=10_000)

    build = subparsers.add_parser("build")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--collector-root", type=Path, required=True)
    build.add_argument("--navmesh-cache", type=Path, required=True)
    build.add_argument("--worker-index", type=int, required=True)
    build.add_argument("--worker-count", type=int, required=True)
    build.add_argument("--habitat-gpu-device", type=int, default=0)
    build.add_argument("--maximum-scenes", type=int)
    build.add_argument("--resume", action="store_true")

    finish = subparsers.add_parser("finalize")
    finish.add_argument("--plan", type=Path, required=True)
    finish.add_argument("--output-root", type=Path, required=True)
    finish.add_argument("--index", type=Path, required=True)
    finish.add_argument("--allow-partial", action="store_true")

    bind = subparsers.add_parser("bind-manifest")
    bind.add_argument("--parent-index", type=Path, required=True)
    bind.add_argument("--parent-manifest", type=Path, required=True)
    bind.add_argument("--manifest", type=Path, required=True)
    bind.add_argument("--index", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        create_plan(args)
    elif args.command == "build":
        build_worker(args)
    elif args.command == "finalize":
        finalize(args)
    elif args.command == "bind-manifest":
        bind_manifest(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
