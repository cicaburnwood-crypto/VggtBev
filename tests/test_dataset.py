from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vggt_bev_method1.data import (
    RGBResizePad,
    VGGNAVMethod1Dataset,
    create_split_manifest,
    load_split_manifest,
    manifest_session_keys,
    method1_collate,
    split_sessions_by_scene,
)


def make_session(
    root: Path,
    *,
    shard: str,
    index: int,
    scene_id: str,
    frame_count: int = 2,
) -> str:
    path = root / shard / f"session_{index:06d}_hm3d_{scene_id}"
    for relative in (
        "camera",
        "bev_6p5m/masked",
        "bev_6p5m/complete",
        "bev_6p5m/merged_masked_10m",
        "bev_6p5m/merged_complete_10m",
    ):
        (path / relative).mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 4,
        "status": "complete",
        "dataset": "hm3d",
        "scene_id": scene_id,
        "frame_count": frame_count,
        "random_parameters": {
            "camera_height_m": 1.25,
            "horizontal_fov_degrees": 90.0,
        },
        "camera_intrinsics": {
            "horizontal_fov_degrees": 90.0,
            "width": 64,
            "height": 48,
            "fx": 32.0,
            "fy": 32.0,
            "cx": 32.0,
            "cy": 24.0,
        },
        "camera_intrinsics_file": "camera_intrinsics.json",
        "camera_extrinsics_file": "camera_extrinsics.jsonl",
        "bev": {
            "size": 512,
            "extent_classes_m": [6.5],
            "merged_normalized_extents_m": [10.0],
            "merged_normalized_size": [512, 512],
            "masked_values": {
                "occupied": 0,
                "unknown": 112,
                "free": 255,
            },
            "merged_orientation": (
                "ego-centric in every output; latest robot centered and forward up"
            ),
            "truth_source": "solid Habitat collision-mesh voxel occupancy",
        },
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (path / "camera_intrinsics.json").write_text(
        json.dumps(metadata["camera_intrinsics"]),
        encoding="utf-8",
    )
    trajectory_lines = []
    camera_extrinsic_lines = []
    for frame in range(frame_count):
        name = f"frame_{frame:06d}.png"
        Image.fromarray(np.full((48, 64, 3), 80 + frame, dtype=np.uint8)).save(
            path / "camera" / name
        )
        masked = np.full((512, 512), 112, dtype=np.uint8)
        masked[100:300, 120:340] = 255
        masked[180:220, 240:260] = 0
        complete = np.full((512, 512), 255, dtype=np.uint8)
        complete[180:220, 240:260] = 0
        Image.fromarray(masked).save(path / "bev_6p5m/masked" / name)
        Image.fromarray(complete).save(path / "bev_6p5m/complete" / name)
        Image.fromarray(masked).save(
            path / "bev_6p5m/merged_masked_10m" / name
        )
        Image.fromarray(complete).save(
            path / "bev_6p5m/merged_complete_10m" / name
        )
        trajectory_lines.append(
            json.dumps(
                {
                    "frame_id": frame,
                    "extrinsic": {
                        "camera_position_world_m": [0.0, 0.0, frame * 0.1]
                    },
                }
            )
        )
        camera_extrinsic_lines.append(
            json.dumps(
                {
                    "frame_id": frame,
                    "extrinsic": {
                        "world_from_bev_planar": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, frame * 0.1],
                            [0.0, 0.0, 1.0],
                        ]
                    },
                }
            )
        )
    (path / "ground_truth_trajectory.jsonl").write_text(
        "\n".join(trajectory_lines) + "\n", encoding="utf-8"
    )
    (path / "camera_extrinsics.jsonl").write_text(
        "\n".join(camera_extrinsic_lines) + "\n",
        encoding="utf-8",
    )
    (path / "COMPLETE").touch()
    return path.relative_to(root).as_posix()


def test_dataset_exposes_runtime_height_but_excludes_gt_pose_and_depth(
    tmp_path: Path,
) -> None:
    key = make_session(
        tmp_path, shard="GPU0-1", index=0, scene_id="00001-test", frame_count=2
    )
    dataset = VGGNAVMethod1Dataset(
        tmp_path,
        supervision="observed",
        preprocess=RGBResizePad(32, 48),
        session_keys=[key],
    )
    sample = dataset[-1]
    assert sample["images"].shape == (2, 3, 32, 48)
    assert sample["single_target"].shape == (512, 512)
    assert sample["merged_target"].shape == (800, 800)
    assert float(sample["single_target_extent_m"]) == 6.5
    assert float(sample["merged_target_extent_m"]) == 10.0
    assert float(sample["camera_height_m"]) == pytest.approx(1.25)
    assert sample["alignment_camera_centers_m"].shape == (2, 3)
    forbidden = {
        "depth",
        "intrinsics",
        "extrinsics",
        "trajectory",
        "camera_to_world",
        "world_to_camera",
    }
    assert not forbidden.intersection(sample)
    batch = method1_collate([sample])
    assert batch["images"].shape == (1, 2, 3, 32, 48)
    assert batch["single_target_extent_m"].shape == (1,)
    assert batch["merged_target_extent_m"].shape == (1,)
    assert batch["camera_height_m"].shape == (1,)
    assert float(batch["camera_height_m"][0]) == pytest.approx(1.25)
    assert batch["alignment_camera_centers_m"].shape == (1, 2, 3)


def test_joint_dataset_returns_and_collates_fov_partitioned_targets(
    tmp_path: Path,
) -> None:
    key = make_session(
        tmp_path,
        shard="GPU0-1",
        index=0,
        scene_id="joint-test",
        frame_count=1,
    )
    dataset = VGGNAVMethod1Dataset(
        tmp_path,
        supervision="joint",
        preprocess=RGBResizePad(32, 48),
        session_keys=[key],
    )
    sample = dataset[0]
    target_keys = {
        "single_observed_target",
        "single_complete_target",
        "merged_observed_target",
        "merged_complete_target",
        "single_fov_complete_target",
        "single_fov_visible_target",
        "single_fov_support_target",
        "merged_fov_complete_target",
        "merged_fov_visible_target",
        "merged_fov_support_target",
    }
    assert target_keys <= sample.keys()
    batch = method1_collate([sample])
    for key in target_keys:
        expected = (1, 800, 800) if key.startswith("merged_") else (1, 512, 512)
        assert batch[key].shape == expected
    assert not {"single_target", "merged_target"}.intersection(batch)


def test_scene_grouped_split_prevents_scene_leakage(tmp_path: Path) -> None:
    make_session(tmp_path, shard="GPU0-1", index=0, scene_id="scene-a")
    make_session(tmp_path, shard="GPU5-1", index=1, scene_id="scene-a")
    make_session(tmp_path, shard="GPU5-1", index=2, scene_id="scene-b")
    make_session(tmp_path, shard="GPU6-1", index=3, scene_id="scene-c")
    train_keys, validation_keys = split_sessions_by_scene(
        tmp_path, validation_fraction=0.34, seed=3
    )
    train = VGGNAVMethod1Dataset(
        tmp_path, supervision="complete", session_keys=train_keys
    )
    validation = VGGNAVMethod1Dataset(
        tmp_path, supervision="complete", session_keys=validation_keys
    )
    assert train.scene_keys.isdisjoint(validation.scene_keys)


def test_frozen_manifest_ignores_later_dataset_growth(tmp_path: Path) -> None:
    make_session(tmp_path, shard="GPU0-1", index=0, scene_id="scene-a")
    make_session(tmp_path, shard="GPU5-1", index=1, scene_id="scene-b")
    make_session(tmp_path, shard="GPU6-1", index=2, scene_id="scene-c")
    manifest_path = tmp_path / "splits" / "method1.json"
    first = create_split_manifest(
        tmp_path,
        manifest_path,
        validation_fraction=0.34,
        seed=3,
    )
    make_session(tmp_path, shard="GPU6-1", index=3, scene_id="scene-d")
    second = load_split_manifest(
        manifest_path,
        dataset_root=tmp_path,
        validation_fraction=0.34,
        seed=3,
    )
    train_keys, validation_keys = manifest_session_keys(second)
    assert first["content_sha256"] == second["content_sha256"]
    assert len(train_keys) + len(validation_keys) == 3
    frozen = VGGNAVMethod1Dataset(
        tmp_path,
        supervision="observed",
        session_keys=train_keys + validation_keys,
    )
    assert len(frozen.sessions) == 3


def test_manifest_backed_dataset_does_not_scan_unlisted_sessions(
    tmp_path: Path,
) -> None:
    key = make_session(
        tmp_path,
        shard="GPU0-1",
        index=0,
        scene_id="scene-a",
        frame_count=1,
    )
    broken = tmp_path / "GPU9-1/session_999999_hm3d_broken"
    broken.mkdir(parents=True)
    (broken / "COMPLETE").touch()
    dataset = VGGNAVMethod1Dataset(
        tmp_path,
        supervision="complete",
        session_keys=[key],
    )
    assert len(dataset.sessions) == 1


def test_manifest_detects_training_artifact_mutation(tmp_path: Path) -> None:
    make_session(tmp_path, shard="GPU0-1", index=0, scene_id="scene-a")
    make_session(tmp_path, shard="GPU0-1", index=1, scene_id="scene-b")
    manifest_path = tmp_path / "splits" / "method1.json"
    create_split_manifest(
        tmp_path,
        manifest_path,
        validation_fraction=0.5,
        seed=3,
    )
    image = (
        tmp_path
        / "GPU0-1/session_000000_hm3d_scene-a"
        / "camera/frame_000000.png"
    )
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="training artifacts changed"):
        load_split_manifest(
            manifest_path,
            dataset_root=tmp_path,
            validation_fraction=0.5,
            seed=3,
        )


def test_long_session_is_capped_by_configured_history(tmp_path: Path) -> None:
    key = make_session(
        tmp_path,
        shard="GPU0-1",
        index=0,
        scene_id="scene-long",
        frame_count=35,
    )
    dataset = VGGNAVMethod1Dataset(
        tmp_path,
        supervision="observed",
        session_keys=[key],
        maximum_history=10,
    )
    assert len(dataset) == 10
    assert dataset.samples[-1].target_frame == 9
