from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from p1b_void_audit.audit import audit_manifest, measure_gt_bev


def test_void_fraction_is_reported_without_changing_gt() -> None:
    target = np.asarray(
        [[255, 255, 0, 112], [255, 0, 112, 112]],
        dtype=np.uint8,
    )
    original = target.copy()
    void = np.asarray(
        [[False, True, False, True], [False, False, True, True]],
        dtype=bool,
    )
    result = measure_gt_bev(void, target)
    assert result.grid_pixels == 8
    assert result.void_pixels == 4
    assert result.void_fraction_of_grid == 0.5
    assert result.gt_known_pixels == 5
    assert result.void_inside_gt_known_pixels == 1
    assert result.void_fraction_of_gt_known == 0.2
    assert np.array_equal(target, original)


def test_audit_writes_four_gt_bev_rows_without_training_dependency(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    session_key = "GPU0/session_000001_test"
    session = dataset_root / session_key
    for directory in (
        "bev_6p5m/complete",
        "bev_6p5m/masked",
        "bev_6p5m/merged_complete_10m",
        "bev_6p5m/merged_masked_10m",
    ):
        (session / directory).mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "test",
        "scene_id": "scene",
        "frame_count": 1,
        "camera_extrinsics_file": "camera_extrinsics.jsonl",
        "path": {"start": [0.0, 0.0, 0.0]},
    }
    (session / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (session / "camera_extrinsics.jsonl").write_text(
        json.dumps(
            {
                "frame_id": 0,
                "extrinsic": {
                    "world_from_bev_planar": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    complete = np.full((512, 512), 255, dtype=np.uint8)
    complete[:, 250:255] = 0
    observed = complete.copy()
    observed[:256] = 112
    for directory, raster in (
        ("complete", complete),
        ("masked", observed),
        ("merged_complete_10m", complete),
        ("merged_masked_10m", observed),
    ):
        Image.fromarray(raster).save(
            session / "bev_6p5m" / directory / "frame_000000.png"
        )

    manifest = {
        "dataset_root": str(dataset_root.resolve()),
        "content_sha256": "manifest-hash",
        "train": [{"key": session_key, "scene_key": "test:scene"}],
        "validation": [],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    artifact_root = tmp_path / "coverage"
    artifact_root.mkdir()
    valid = np.ones((1024, 1024), dtype=bool)
    valid[:, :350] = False
    artifact_path = artifact_root / "scene.npz"
    with artifact_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            valid_bits=np.packbits(valid.reshape(-1), bitorder="little"),
            shape=np.asarray(valid.shape, dtype=np.int32),
            metadata_json=np.asarray("{}"),
        )
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    index = {
        "algorithm": "strict-solid-voxel-or-navmesh-coverage-v4",
        "dataset_root": str(dataset_root.resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "source_manifest_content_sha256": "manifest-hash",
        "content_sha256": "coverage-hash",
        "partial": False,
        "missing_floor_band_count": 0,
        "scenes": {
            "test:scene": {
                "bands": [
                    {
                        "minimum_floor_m": 0.0,
                        "maximum_floor_m": 0.0,
                        "center_floor_m": 0.0,
                        "artifact": artifact_path.name,
                        "artifact_sha256": artifact_hash,
                        "lower_bound_xz_m": [-6.4, -6.4],
                        "voxel_size_m": 0.0125,
                    }
                ]
            }
        },
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    result = audit_manifest(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        void_index_path=index_path,
        output_dir=tmp_path / "report",
        log_every_sessions=0,
    )
    assert result["training_integration"] is False
    assert result["record_count"] == 4
    assert set(result["by_gt_bev"]) == {
        "single_complete",
        "single_observed",
        "merged_complete",
        "merged_observed",
    }
    with Path(result["csv"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(float(row["void_fraction_of_grid"]) > 0.0 for row in rows)
    assert (
        float(
            next(
                row["void_fraction_of_gt_known"]
                for row in rows
                if row["gt_bev"] == "single_observed"
            )
        )
        <= float(
            next(
                row["void_fraction_of_gt_known"]
                for row in rows
                if row["gt_bev"] == "single_complete"
            )
        )
    )
