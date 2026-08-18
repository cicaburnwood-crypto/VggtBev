from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _builder_module():
    path = Path(__file__).parents[1] / "tools" / "build_void_coverage.py"
    spec = importlib.util.spec_from_file_location("strict_void_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _visualizer_module():
    path = Path(__file__).parents[1] / "tools" / "visualize_void_samples.py"
    spec = importlib.util.spec_from_file_location("strict_void_visualizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _vertical_wall(
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    lower_y: float = 0.0,
    upper_y: float = 4.0,
) -> np.ndarray:
    x0, z0 = first
    x1, z1 = second
    vertices = np.asarray(
        [
            [x0, lower_y, z0],
            [x1, lower_y, z1],
            [x1, upper_y, z1],
            [x0, upper_y, z0],
        ],
        dtype=np.float64,
    )
    return vertices[np.asarray([[0, 1, 2], [0, 2, 3]])]


def test_strict_voxel_preserves_enclosed_wall_volume_across_tiles() -> None:
    builder = _builder_module()
    triangles = np.concatenate(
        [
            _vertical_wall((2.0, 2.0), (7.0, 2.0)),
            _vertical_wall((7.0, 2.0), (7.0, 7.0)),
            _vertical_wall((7.0, 7.0), (2.0, 7.0)),
            _vertical_wall((2.0, 7.0), (2.0, 2.0)),
        ]
    )
    navmesh = np.zeros((10, 10), dtype=bool)
    coverage, statistics = builder._rasterize_coverage(
        triangles,
        navigable_map=navmesh,
        lower_y=0.0,
        upper_y=4.0,
        lower_bound=np.asarray([0.0, 0.0, 0.0]),
        rows=10,
        columns=10,
        voxel_size_m=1.0,
        tile_size=3,
        surface_seal_voxels=1,
    )

    # z=4 is row 5 after the source top-down flip.  It is not a surface cell;
    # preserving it proves that 3-D solid connectivity crossed tile borders.
    assert bool(coverage[5, 4])
    assert not bool(coverage[9, 0])
    assert statistics["enclosed_solid_voxel_count"] > 0
    assert statistics["two_dimensional_hole_fill"] is False


def test_strict_voxel_keeps_open_air_void_and_navmesh_as_direct_proof() -> None:
    builder = _builder_module()
    triangles = _vertical_wall((3.0, 1.0), (3.0, 6.0))
    navmesh = np.zeros((8, 8), dtype=bool)
    navmesh[1, 6] = True
    coverage, _ = builder._rasterize_coverage(
        triangles,
        navigable_map=navmesh,
        lower_y=0.0,
        upper_y=4.0,
        lower_bound=np.asarray([0.0, 0.0, 0.0]),
        rows=8,
        columns=8,
        voxel_size_m=1.0,
        tile_size=2,
        surface_seal_voxels=1,
    )

    assert bool(coverage[6, 6])  # raw NavMesh row 1 after vertical flip
    assert bool(coverage[4, 3])  # collision surface at raw z=3
    # One conservative surface voxel is retained on both sides; farther open
    # exterior air remains Void.
    assert not bool(coverage[4, 5])


def test_outer_void_only_audit_rejects_internal_void() -> None:
    visualizer = _visualizer_module()
    grid = np.ones((5, 5), dtype=bool)
    void = np.zeros_like(grid)
    void[2, 2] = True
    free = grid & ~void
    sample = visualizer.SampleMasks(
        key="session",
        scene_key="test:scene",
        dataset="test",
        branch="single",
        target_frame=0,
        grid=grid,
        original_free=grid,
        original_occupied=np.zeros_like(grid),
        original_unknown=np.zeros_like(grid),
        previous_void=void,
        void=void,
        free=free,
        occupied=np.zeros_like(grid),
    )

    audit = visualizer._audit_samples([sample])
    assert audit["passed"] is False
    assert audit["internal_void_policy"].startswith("filled")


def test_manifest_binding_reuses_only_verified_parent_sidecars(
    tmp_path: Path,
) -> None:
    builder = _builder_module()
    artifact_root = tmp_path / "sidecars"
    artifact = artifact_root / "coverage/scene/floor_000.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"strict-coverage")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    parent_manifest = {
        "dataset_root": str(tmp_path),
        "content_sha256": "parent-manifest",
        "train": [
            {"key": "a/session", "scene_key": "test:scene"},
            {"key": "b/session", "scene_key": "test:other"},
        ],
        "validation": [],
    }
    target_manifest = {
        "dataset_root": str(tmp_path),
        "content_sha256": "target-manifest",
        "train": [{"key": "a/session", "scene_key": "test:scene"}],
        "validation": [],
    }
    parent_index = {
        "format_version": 4,
        "algorithm": builder.ALGORITHM,
        "plan_content_sha256": "plan",
        "source_manifest_content_sha256": "parent-manifest",
        "content_sha256": "parent-index",
        "dataset_root": str(tmp_path),
        "artifact_root": str(artifact_root),
        "partial": False,
        "missing_floor_band_count": 0,
        "void_definition": "strict",
        "scenes": {
            "test:scene": {
                "bands": [
                    {
                        "artifact": "coverage/scene/floor_000.npz",
                        "artifact_sha256": artifact_hash,
                    }
                ]
            },
            "test:other": {"bands": []},
        },
    }
    parent_manifest_path = tmp_path / "parent_manifest.json"
    target_manifest_path = tmp_path / "target_manifest.json"
    parent_index_path = tmp_path / "parent_index.json"
    output_path = tmp_path / "target_index.json"
    parent_manifest_path.write_text(json.dumps(parent_manifest), encoding="utf-8")
    target_manifest_path.write_text(json.dumps(target_manifest), encoding="utf-8")
    parent_index_path.write_text(json.dumps(parent_index), encoding="utf-8")

    builder.bind_manifest(
        argparse.Namespace(
            parent_index=parent_index_path,
            parent_manifest=parent_manifest_path,
            manifest=target_manifest_path,
            index=output_path,
        )
    )
    rebound = json.loads(output_path.read_text(encoding="utf-8"))
    assert rebound["source_manifest_content_sha256"] == "target-manifest"
    assert rebound["source_parent_index_content_sha256"] == "parent-index"
    assert rebound["session_count"] == 1
    assert set(rebound["scenes"]) == {"test:scene"}
    assert rebound["partial"] is False
