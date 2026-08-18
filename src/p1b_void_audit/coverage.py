from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage

SUPPORTED_ALGORITHMS = {
    "collision-surface-column-coverage-v1",
    "collision-surface-or-navmesh-column-coverage-v2",
    "floor-domain-surface-or-navmesh-coverage-v3",
    "strict-solid-voxel-or-navmesh-coverage-v4",
}
SCENE_DOMAIN_REPAIR = "enclosed-scene-and-output-grid-hole-fill-v2"
STRICT_OUTER_VOID_REPAIR = "outer-connected-void-only-v3"
FINAL_GT_VOID_FILTER = "strict-voxel-outer-void-only-gt-validity-v7"


def fill_enclosed_scene_domain(valid_map: np.ndarray) -> np.ndarray:
    """Restore invalid islands enclosed by the complete scene floor domain.

    A per-FOV boundary test alone is insufficient: cropping can make a globally
    enclosed mesh hole touch the local FOV boundary.  Hole filling must happen
    on the complete scene/floor sidecar before it is sampled into an ego BEV.
    Eight-connected exterior propagation is deliberately conservative around
    diagonal openings.
    """

    valid = np.asarray(valid_map, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("scene-domain coverage must be a 2-D raster")
    return ndimage.binary_fill_holes(
        valid,
        structure=np.ones((3, 3), dtype=bool),
    ).astype(bool, copy=False)


class VoidCoverageIndex:
    """Render read-only validity masks from immutable scene sidecars."""

    def __init__(
        self,
        path: str | Path,
        *,
        dataset_root: str | Path,
        expected_manifest_sha256: str | None = None,
        require_complete: bool = False,
        verify_artifacts: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("algorithm") not in SUPPORTED_ALGORITHMS:
            raise ValueError("unsupported Void coverage algorithm")
        resolved_dataset = Path(dataset_root).expanduser().resolve()
        if Path(payload["dataset_root"]).resolve() != resolved_dataset:
            raise ValueError("Void coverage dataset root does not match the dataset")
        source_hash = str(payload["source_manifest_content_sha256"])
        if expected_manifest_sha256 is not None and source_hash != str(
            expected_manifest_sha256
        ):
            raise ValueError("Void coverage index belongs to another split snapshot")
        if require_complete:
            if bool(payload.get("partial", True)):
                raise ValueError("Void audit requires a complete coverage index")
            if int(payload.get("missing_floor_band_count", -1)) != 0:
                raise ValueError("Void audit coverage has missing floor bands")
        self.payload = payload
        self.algorithm = str(payload["algorithm"])
        self.artifact_root = Path(payload["artifact_root"]).expanduser().resolve()
        self.scenes = dict(payload["scenes"])
        self.content_sha256 = str(payload["content_sha256"])
        self.source_manifest_sha256 = source_hash
        self._valid_cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        if verify_artifacts:
            for scene in self.scenes.values():
                for band in scene["bands"]:
                    artifact = (self.artifact_root / str(band["artifact"])).resolve()
                    if not artifact.is_relative_to(self.artifact_root):
                        raise ValueError("Void coverage artifact escapes its root")
                    if not artifact.is_file():
                        raise FileNotFoundError(
                            f"Void coverage artifact is missing: {artifact}"
                        )
                    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
                    if actual != str(band["artifact_sha256"]):
                        raise ValueError(
                            f"Void coverage artifact hash mismatch: {artifact}"
                        )

    def _band(self, scene_key: str, floor_height_m: float) -> dict[str, Any]:
        try:
            bands = self.scenes[scene_key]["bands"]
        except KeyError as error:
            raise KeyError(f"Void coverage has no scene {scene_key!r}") from error
        if not bands:
            raise ValueError(f"Void coverage scene has no floor bands: {scene_key}")
        floor = float(floor_height_m)

        # A band may span many closely spaced floor heights while its neighbour
        # contains only one.  Selecting solely by distance to the band centre can
        # therefore choose the neighbour even when ``floor`` belongs to this band.
        # Prefer exact interval membership; the distance fallback only serves
        # legacy/external samples whose floor lies outside every indexed band.
        containing = [
            band
            for band in bands
            if float(band.get("minimum_floor_m", band["center_floor_m"])) - 1e-6
            <= floor
            <= float(band.get("maximum_floor_m", band["center_floor_m"])) + 1e-6
        ]
        if containing:
            return min(
                containing,
                key=lambda band: abs(float(band["center_floor_m"]) - floor),
            )

        def interval_distance(band: dict[str, Any]) -> tuple[float, float]:
            lower = float(band.get("minimum_floor_m", band["center_floor_m"]))
            upper = float(band.get("maximum_floor_m", band["center_floor_m"]))
            distance = max(lower - floor, floor - upper, 0.0)
            return distance, abs(float(band["center_floor_m"]) - floor)

        return min(bands, key=interval_distance)

    def _load_valid_map(
        self,
        artifact: str,
        expected_sha256: str,
        *,
        repair_scene_domain: bool,
    ) -> np.ndarray:
        key = (artifact, f"{expected_sha256}:{int(repair_scene_domain)}")
        cached = self._valid_cache.get(key)
        if cached is not None:
            self._valid_cache.move_to_end(key)
            return cached
        path = (self.artifact_root / artifact).resolve()
        if not path.is_relative_to(self.artifact_root):
            raise ValueError("Void coverage artifact escapes its root")
        with np.load(path, allow_pickle=False) as archive:
            shape = tuple(int(value) for value in archive["shape"])
            bit_count = shape[0] * shape[1]
            valid = np.unpackbits(
                archive["valid_bits"], bitorder="little"
            )[:bit_count].reshape(shape)
        valid = valid.astype(bool, copy=False)
        if (
            repair_scene_domain
            and self.algorithm
            in {
                "floor-domain-surface-or-navmesh-coverage-v3",
                "strict-solid-voxel-or-navmesh-coverage-v4",
            }
        ):
            valid = fill_enclosed_scene_domain(valid)
        valid.setflags(write=False)
        self._valid_cache[key] = valid
        self._valid_cache.move_to_end(key)
        while len(self._valid_cache) > 2:
            self._valid_cache.popitem(last=False)
        return valid

    def render_valid_mask(
        self,
        *,
        scene_key: str,
        floor_height_m: float,
        world_from_bev_planar: np.ndarray,
        output_size: int,
        output_extent_m: float,
        repair_scene_domain: bool = True,
    ) -> torch.Tensor:
        band = self._band(scene_key, floor_height_m)
        valid_map = self._load_valid_map(
            str(band["artifact"]),
            str(band["artifact_sha256"]),
            repair_scene_domain=repair_scene_domain,
        )
        transform = np.asarray(world_from_bev_planar, dtype=np.float64)
        if transform.shape != (3, 3) or abs(np.linalg.det(transform)) < 1e-8:
            raise ValueError("world_from_bev_planar must be an invertible 3x3 matrix")
        if output_size <= 0 or output_extent_m <= 0.0:
            raise ValueError("BEV output size and extent must be positive")

        center = float(output_size // 2)
        meters_per_pixel = float(output_extent_m) / float(output_size)
        rows, columns = np.indices((output_size, output_size), dtype=np.float64)
        local_right = (columns - center) * meters_per_pixel
        local_forward = (center - rows) * meters_per_pixel
        world_x = (
            transform[0, 0] * local_right
            + transform[0, 1] * local_forward
            + transform[0, 2]
        )
        world_z = (
            transform[1, 0] * local_right
            + transform[1, 1] * local_forward
            + transform[1, 2]
        )
        lower_x, lower_z = (float(value) for value in band["lower_bound_xz_m"])
        source_mpp = float(band["voxel_size_m"])
        source_columns = np.floor((world_x - lower_x) / source_mpp).astype(np.int64)
        unflipped_rows = np.floor((world_z - lower_z) / source_mpp).astype(np.int64)
        source_rows = valid_map.shape[0] - 1 - unflipped_rows
        inside = (
            (source_columns >= 0)
            & (source_columns < valid_map.shape[1])
            & (source_rows >= 0)
            & (source_rows < valid_map.shape[0])
        )
        result = np.zeros((output_size, output_size), dtype=bool)
        result[inside] = valid_map[source_rows[inside], source_columns[inside]]
        if (
            repair_scene_domain
            and self.algorithm
            in {
                "floor-domain-surface-or-navmesh-coverage-v3",
                "strict-solid-voxel-or-navmesh-coverage-v4",
            }
        ):
            # Cropping a globally exterior-connected invalid component can hide
            # its connection outside this ego grid and leave a small enclosed
            # island after rasterization. Repair that output-grid topology as a
            # second, still geometry-only step. No FOV or masked GT is involved.
            result = fill_enclosed_scene_domain(result)
        return torch.from_numpy(result)
