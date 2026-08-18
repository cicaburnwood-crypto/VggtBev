from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vggt_bev_method1.data.void_coverage import (
    VoidCoverageIndex,
    fill_enclosed_scene_domain,
)


def test_void_coverage_renders_in_latest_ego_coordinates(tmp_path: Path) -> None:
    valid = np.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=bool,
    )
    artifact = tmp_path / "coverage.npz"
    with artifact.open("wb") as stream:
        np.savez_compressed(
            stream,
            valid_bits=np.packbits(valid.reshape(-1), bitorder="little"),
            shape=np.asarray(valid.shape, dtype=np.int32),
            metadata_json=np.asarray("{}"),
        )
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "algorithm": "floor-domain-surface-or-navmesh-coverage-v3",
                "dataset_root": str(tmp_path),
                "artifact_root": str(tmp_path),
                "source_manifest_content_sha256": "manifest",
                "content_sha256": "index",
                "scenes": {
                    "test:scene": {
                        "bands": [
                            {
                                "center_floor_m": 0.0,
                                "artifact": artifact.name,
                                "artifact_sha256": artifact_hash,
                                "lower_bound_xz_m": [0.0, 0.0],
                                "voxel_size_m": 1.0,
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    index = VoidCoverageIndex(
        index_path,
        dataset_root=tmp_path,
        expected_manifest_sha256="manifest",
    )
    rendered = index.render_valid_mask(
        scene_key="test:scene",
        floor_height_m=0.0,
        world_from_bev_planar=np.asarray(
            [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        output_size=4,
        output_extent_m=4.0,
    )
    assert rendered.dtype == torch.bool
    assert torch.equal(rendered, torch.from_numpy(valid))


def test_floor_band_selection_prefers_containing_interval(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "algorithm": "floor-domain-surface-or-navmesh-coverage-v3",
                "dataset_root": str(tmp_path),
                "artifact_root": str(tmp_path),
                "source_manifest_content_sha256": "manifest",
                "content_sha256": "index",
                "scenes": {
                    "test:scene": {
                        "bands": [
                            {
                                "minimum_floor_m": 0.0,
                                "maximum_floor_m": 0.5,
                                "center_floor_m": 0.25,
                                "artifact": "wide.npz",
                                "artifact_sha256": "wide",
                            },
                            {
                                "minimum_floor_m": 0.61,
                                "maximum_floor_m": 0.61,
                                "center_floor_m": 0.61,
                                "artifact": "narrow.npz",
                                "artifact_sha256": "narrow",
                            },
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    index = VoidCoverageIndex(
        index_path,
        dataset_root=tmp_path,
        expected_manifest_sha256="manifest",
    )

    # Centre-only selection would incorrectly choose the 0.61 m band.
    assert index._band("test:scene", 0.5)["artifact"] == "wide.npz"
    # A floor outside every interval falls back to the nearest interval.
    assert index._band("test:scene", 0.57)["artifact"] == "narrow.npz"


def test_complete_scene_topology_fills_only_enclosed_invalid_islands() -> None:
    valid = np.zeros((9, 9), dtype=bool)
    valid[1:8, 1:8] = True
    valid[3:6, 3:6] = False
    valid[0:3, 7] = False

    repaired = fill_enclosed_scene_domain(valid)

    assert bool(repaired[4, 4])
    assert not bool(repaired[0, 7])
    assert not bool(repaired[1, 7])


def test_scene_gt_validity_api_has_no_fov_or_mask_input() -> None:
    parameters = set(
        inspect.signature(VoidCoverageIndex.render_valid_mask).parameters
    )
    assert parameters == {
        "self",
        "scene_key",
        "floor_height_m",
        "world_from_bev_planar",
        "output_size",
        "output_extent_m",
        "repair_scene_domain",
    }


def test_output_grid_repair_fills_crop_enclosed_geometry_hole(
    monkeypatch,
) -> None:
    valid = np.ones((5, 5), dtype=bool)
    valid[2, 2] = False
    index = object.__new__(VoidCoverageIndex)
    index.algorithm = "floor-domain-surface-or-navmesh-coverage-v3"
    monkeypatch.setattr(
        index,
        "_band",
        lambda scene_key, floor_height_m: {
            "artifact": "unused.npz",
            "artifact_sha256": "unused",
            "lower_bound_xz_m": [0.0, 0.0],
            "voxel_size_m": 1.0,
        },
    )
    monkeypatch.setattr(
        index,
        "_load_valid_map",
        lambda artifact, expected_sha256, repair_scene_domain: valid,
    )
    transform = np.asarray(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
    )
    raw = index.render_valid_mask(
        scene_key="test:scene",
        floor_height_m=0.0,
        world_from_bev_planar=transform,
        output_size=5,
        output_extent_m=5.0,
        repair_scene_domain=False,
    )
    repaired = index.render_valid_mask(
        scene_key="test:scene",
        floor_height_m=0.0,
        world_from_bev_planar=transform,
        output_size=5,
        output_extent_m=5.0,
        repair_scene_domain=True,
    )
    assert not bool(raw[2, 2])
    assert bool(repaired[2, 2])


def test_strict_v4_restores_an_enclosed_output_void(monkeypatch) -> None:
    valid = np.ones((5, 5), dtype=bool)
    valid[2, 2] = False
    index = object.__new__(VoidCoverageIndex)
    index.algorithm = "strict-solid-voxel-or-navmesh-coverage-v4"
    monkeypatch.setattr(
        index,
        "_band",
        lambda scene_key, floor_height_m: {
            "artifact": "unused.npz",
            "artifact_sha256": "unused",
            "lower_bound_xz_m": [0.0, 0.0],
            "voxel_size_m": 1.0,
        },
    )
    monkeypatch.setattr(
        index,
        "_load_valid_map",
        lambda artifact, expected_sha256, repair_scene_domain: valid,
    )
    rendered = index.render_valid_mask(
        scene_key="test:scene",
        floor_height_m=0.0,
        world_from_bev_planar=np.asarray(
            [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
        ),
        output_size=5,
        output_extent_m=5.0,
        repair_scene_domain=True,
    )
    assert bool(rendered[2, 2])


def test_strict_v4_preserves_void_connected_to_output_boundary(monkeypatch) -> None:
    valid = np.ones((7, 7), dtype=bool)
    valid[0:4, 3] = False
    index = object.__new__(VoidCoverageIndex)
    index.algorithm = "strict-solid-voxel-or-navmesh-coverage-v4"
    monkeypatch.setattr(
        index,
        "_band",
        lambda scene_key, floor_height_m: {
            "artifact": "unused.npz",
            "artifact_sha256": "unused",
            "lower_bound_xz_m": [0.0, 0.0],
            "voxel_size_m": 1.0,
        },
    )
    monkeypatch.setattr(
        index,
        "_load_valid_map",
        lambda artifact, expected_sha256, repair_scene_domain: valid,
    )
    rendered = index.render_valid_mask(
        scene_key="test:scene",
        floor_height_m=0.0,
        world_from_bev_planar=np.asarray(
            [[1.0, 0.0, 3.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]
        ),
        output_size=7,
        output_extent_m=7.0,
        repair_scene_domain=True,
    )
    assert not bool(rendered[0, 3])
    assert not bool(rendered[3, 3])


def test_training_rejects_partial_void_index(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "algorithm": "strict-solid-voxel-or-navmesh-coverage-v4",
                "dataset_root": str(tmp_path),
                "artifact_root": str(tmp_path),
                "source_manifest_content_sha256": "manifest",
                "content_sha256": "partial",
                "partial": True,
                "missing_floor_band_count": 1,
                "scenes": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete Void coverage"):
        VoidCoverageIndex(
            index_path,
            dataset_root=tmp_path,
            expected_manifest_sha256="manifest",
            require_complete=True,
        )


def test_training_verifies_void_artifact_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "coverage.npz"
    artifact.write_bytes(b"coverage")
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "algorithm": "strict-solid-voxel-or-navmesh-coverage-v4",
                "dataset_root": str(tmp_path),
                "artifact_root": str(tmp_path),
                "source_manifest_content_sha256": "manifest",
                "content_sha256": "complete",
                "partial": False,
                "missing_floor_band_count": 0,
                "scenes": {
                    "test:scene": {
                        "bands": [
                            {
                                "artifact": artifact.name,
                                "artifact_sha256": "not-the-real-hash",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        VoidCoverageIndex(
            index_path,
            dataset_root=tmp_path,
            expected_manifest_sha256="manifest",
            require_complete=True,
            verify_artifacts=True,
        )
