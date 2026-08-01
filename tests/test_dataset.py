import json
import shutil
from pathlib import Path

import pytest
import torch

from vggt_bev.data import (
    CalibrationAwareResize,
    VGGNAVMethod2Dataset,
    method2_collate,
    split_sessions,
)
from vggt_bev.data.vggnav_dataset import discover_sessions

DATA_ROOT = Path("/home/user/Project/VGGNAV/output/random_sessions_test_3")


def test_single_frame_dataset_alignment() -> None:
    dataset = VGGNAVMethod2Dataset(DATA_ROOT, extent_key="bev_5m", target_mode="single")
    assert len(dataset) == 27
    sample = dataset[0]
    assert sample["images"].shape == (1, 3, 384, 512)
    assert sample["intrinsics"].shape == (1, 3, 3)
    assert sample["target_labels"].shape == (512, 512)
    assert float(sample["target_extent_m"]) == 5.0
    assert sample["metadata"]["source_frame_ids"] == [0]
    assert "revealed_labels_audit_only" not in sample
    assert torch.allclose(sample["intrinsics"][0, :2, 2], torch.tensor([255.5, 191.5]))
    assert set(torch.unique(sample["target_labels"]).tolist()) <= {0, 112, 255}


def test_merged_dataset_preserves_cumulative_history() -> None:
    dataset = VGGNAVMethod2Dataset(DATA_ROOT, extent_key="bev_5m", target_mode="merged")
    last_first_session = dataset[7]
    assert last_first_session["images"].shape[0] == 8
    assert last_first_session["metadata"]["source_frame_ids"] == list(range(8))
    assert last_first_session["metadata"]["declared_history_frame_count"] == 8
    assert float(last_first_session["target_extent_m"]) == 8.0


def test_both_dataset_returns_aligned_single_and_merged_targets() -> None:
    dataset = VGGNAVMethod2Dataset(DATA_ROOT, extent_key="bev_5m", target_mode="both")
    sample = dataset[1]
    assert sample["images"].shape[0] == 2
    assert sample["single_target_labels"].shape == (512, 512)
    assert sample["merged_target_labels"].shape == (512, 512)
    assert float(sample["single_target_extent_m"]) == 5.0
    assert float(sample["merged_target_extent_m"]) == 8.0
    assert sample["metadata"]["source_frame_ids"] == [0, 1]
    assert set(sample["metadata"]["target_paths"]) == {"single", "merged"}


def test_both_collate_preserves_dual_targets() -> None:
    dataset = VGGNAVMethod2Dataset(DATA_ROOT, target_mode="both")
    batch = method2_collate([dataset[0], dataset[1]])
    assert batch["single_target_labels"].shape == (2, 512, 512)
    assert batch["merged_target_labels"].shape == (2, 512, 512)
    assert batch["single_target_extent_m"].tolist() == [5.0, 5.0]
    assert batch["merged_target_extent_m"].tolist() == [8.0, 8.0]


def test_vggt_geometry_training_batch_excludes_simulator_camera_geometry() -> None:
    dataset = VGGNAVMethod2Dataset(
        DATA_ROOT,
        extent_key="bev_5m",
        target_mode="both",
        geometry_source="vggt",
    )
    sample = dataset[1]
    forbidden = {
        "intrinsics",
        "camera_to_world",
        "reference_world_from_bev",
        "floor_y",
    }

    assert forbidden.isdisjoint(sample)
    assert sample["metadata"]["geometry_source"] == "vggt"
    assert torch.isclose(
        sample["camera_height_m"],
        torch.tensor(sample["metadata"]["camera_height_m"]),
    )
    assert 0.3 <= float(sample["camera_height_m"]) <= 0.8

    batch = method2_collate([dataset[0], sample])
    assert forbidden.isdisjoint(batch)
    assert batch["use_predicted_geometry"] is True
    assert batch["camera_height_m"].shape == (2,)
    assert batch["single_target_extent_m"].tolist() == [5.0, 5.0]
    assert batch["merged_target_extent_m"].tolist() == [8.0, 8.0]


def test_raw_vggt_batch_excludes_camera_height() -> None:
    dataset = VGGNAVMethod2Dataset(
        DATA_ROOT,
        extent_key="bev_5m",
        target_mode="both",
        geometry_source="vggt",
        coordinate_mode="vggt_raw",
    )
    first = dataset[0]
    second = dataset[1]

    assert "camera_height_m" not in first
    assert "camera_height_m" not in first["metadata"]
    assert first["metadata"]["coordinate_mode"] == "vggt_raw"

    batch = method2_collate([first, second])
    assert "camera_height_m" not in batch
    assert batch["coordinate_mode"] == "vggt_raw"
    assert batch["use_predicted_geometry"] is True


def test_normalized_vggt_targets_are_pixel_only_and_dual_resolution() -> None:
    dataset = VGGNAVMethod2Dataset(
        DATA_ROOT,
        extent_key="bev_5m",
        target_mode="both",
        geometry_source="vggt",
        coordinate_mode="vggt_normalized",
        single_target_size=512,
        merged_target_size=800,
    )
    first = dataset[0]
    second = dataset[1]

    assert first["single_target_labels"].shape == (512, 512)
    assert first["merged_target_labels"].shape == (800, 800)
    assert "single_target_extent_m" not in first
    assert "merged_target_extent_m" not in first
    assert "camera_height_m" not in first
    assert first["metadata"]["target_source_extents_m"] == {
        "single": 5.0,
        "merged": 8.0,
    }

    batch = method2_collate([first, second])
    assert batch["single_target_labels"].shape == (2, 512, 512)
    assert batch["merged_target_labels"].shape == (2, 800, 800)
    assert "single_target_extent_m" not in batch
    assert "merged_target_extent_m" not in batch
    assert batch["coordinate_mode"] == "vggt_normalized"


def test_expected_merged_extent_rejects_wrong_p2_labels() -> None:
    with pytest.raises(ValueError, match="merged extent is 8 m, expected 10 m"):
        VGGNAVMethod2Dataset(
            DATA_ROOT,
            extent_key="bev_5m",
            target_mode="both",
            geometry_source="vggt",
            expected_merged_extent_m=10.0,
        )


def test_variable_history_collate_has_explicit_frame_mask() -> None:
    dataset = VGGNAVMethod2Dataset(DATA_ROOT, target_mode="merged")
    batch = method2_collate([dataset[0], dataset[1]])
    assert batch["images"].shape[:2] == (2, 2)
    assert batch["frame_valid"].tolist() == [[True, False], [True, True]]
    assert not batch["image_valid"][0, 1].any()


def test_session_split_has_no_overlap() -> None:
    train, validation = split_sessions(DATA_ROOT, validation_fraction=0.34, seed=4)
    assert train
    assert validation
    assert set(train).isdisjoint(validation)


def test_sharded_collection_uses_root_relative_session_keys(tmp_path: Path) -> None:
    source_sessions = sorted(path for path in DATA_ROOT.glob("session_*") if path.is_dir())[:2]
    for shard, source in zip(("gpu0", "gpu8"), source_sessions, strict=True):
        destination = tmp_path / shard / source.name
        destination.parent.mkdir(parents=True)
        destination.symlink_to(source, target_is_directory=True)

    sessions = VGGNAVMethod2Dataset(tmp_path, extent_key="bev_5m", target_mode="single")
    assert sessions.session_names == (
        f"gpu0/{source_sessions[0].name}",
        f"gpu8/{source_sessions[1].name}",
    )
    train, validation = split_sessions(tmp_path, validation_fraction=0.5, seed=4)
    assert len(train) == len(validation) == 1
    assert "/" in train[0] and "/" in validation[0]


def test_database_manifest_is_authoritative_session_subset(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "gpu0" / "session_selected"
    excluded = tmp_path / "gpu1" / "session_excluded"
    for session in (selected, excluded):
        session.mkdir(parents=True)
        (session / "COMPLETE").touch()
    (tmp_path / "database_manifest.json").write_text(
        json.dumps(
            {
                "counts": {"sessions": 1},
                "sessions": [{"key": "gpu0/session_selected"}],
            }
        ),
        encoding="utf-8",
    )

    assert discover_sessions(tmp_path) == [selected]


def test_database_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    (tmp_path / "database_manifest.json").write_text(
        json.dumps(
            {
                "counts": {"sessions": 1},
                "sessions": [{"key": "../session_unsafe"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe or duplicate key"):
        discover_sessions(tmp_path)


def test_scene_hash_split_keeps_repeated_scene_sessions_together(
    tmp_path: Path,
) -> None:
    for index, scene_id in enumerate(
        ("scene_a", "scene_a", "scene_b", "scene_c", "scene_d", "scene_e")
    ):
        session = tmp_path / f"gpu{index % 2}" / f"session_{index:03d}"
        session.mkdir(parents=True)
        (session / "COMPLETE").touch()
        (session / "metadata.json").write_text(
            json.dumps(
                {
                    "dataset": "hm3d",
                    "scene_id": scene_id,
                }
            ),
            encoding="utf-8",
        )

    train, validation = split_sessions(
        tmp_path,
        validation_fraction=0.4,
        seed=7,
        group_by="scene",
        strategy="stable_hash",
    )
    split_by_session = {
        session: "train" for session in train
    } | {
        session: "validation" for session in validation
    }
    assert split_by_session["gpu0/session_000"] == split_by_session["gpu1/session_001"]
    assert train
    assert validation


def test_schema_v4_normalized_merged_layout_is_loaded(tmp_path: Path) -> None:
    source = next(path for path in DATA_ROOT.glob("session_*") if path.is_dir())
    copied = tmp_path / source.name
    shutil.copytree(source, copied, symlinks=True)
    bev_root = copied / "bev_5m"
    (bev_root / "merged_masked").rename(bev_root / "merged_masked_10m")
    (bev_root / "merged_complete").rename(bev_root / "merged_complete_10m")

    trajectory_path = copied / "ground_truth_trajectory.jsonl"
    frames = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for frame in frames:
        metadata = frame["bev"]["bev_5m"]
        masked = metadata.pop("merged_masked")
        complete = metadata.pop("merged_complete")
        for target in (masked, complete):
            target["extent_m"] = 10.0
            target["meters_per_pixel"] = 10.0 / 512.0
        metadata["merged"] = {
            "10m": {
                "masked": masked,
                "complete": complete,
            }
        }
    trajectory_path.write_text(
        "".join(json.dumps(frame) + "\n" for frame in frames),
        encoding="utf-8",
    )

    dataset = VGGNAVMethod2Dataset(
        tmp_path,
        extent_key="bev_5m",
        target_mode="both",
        geometry_source="vggt",
        expected_merged_extent_m=10.0,
    )
    sample = dataset[1]
    assert float(sample["single_target_extent_m"]) == 5.0
    assert float(sample["merged_target_extent_m"]) == 10.0
    assert "merged_masked_10m" in sample["metadata"]["target_paths"]["merged"]


def test_required_merged_fusion_version_rejects_legacy_sessions() -> None:
    with pytest.raises(ValueError, match="do not provide merged fusion version 2"):
        VGGNAVMethod2Dataset(
            DATA_ROOT,
            extent_key="bev_5m",
            target_mode="merged",
            required_merged_fusion_version=2,
        )


def test_visibility_v2_modality_mapping_is_loaded(tmp_path: Path) -> None:
    source = next(path for path in DATA_ROOT.glob("session_*") if path.is_dir())
    copied = tmp_path / source.name
    shutil.copytree(source, copied, symlinks=True)
    bev_root = copied / "bev_5m"
    shutil.copytree(
        bev_root / "merged_masked",
        bev_root / "merged_masked_8m_visibility_v2",
    )
    shutil.copytree(
        bev_root / "merged_complete",
        bev_root / "merged_complete_8m_visibility_v2",
    )
    metadata_path = copied / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["bev"]["merged_fusion_version"] = 2
    metadata["bev"]["merged_fusion_modalities"] = {
        "8m": {
            "masked": "merged_masked_8m_visibility_v2",
            "complete": "merged_complete_8m_visibility_v2",
        }
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    dataset = VGGNAVMethod2Dataset(
        tmp_path,
        extent_key="bev_5m",
        target_mode="both",
        expected_merged_extent_m=8.0,
        required_merged_fusion_version=2,
    )
    sample = dataset[1]

    assert "merged_masked_8m_visibility_v2" in (
        sample["metadata"]["target_paths"]["merged"]
    )


def test_new_extent_name_is_discovered_from_data(tmp_path: Path) -> None:
    source = next(path for path in DATA_ROOT.glob("session_*") if path.is_dir())
    copied = tmp_path / source.name
    shutil.copytree(source, copied, symlinks=True)
    (copied / "bev_6p5m").symlink_to(copied / "bev_5m", target_is_directory=True)
    dataset = VGGNAVMethod2Dataset(tmp_path, extent_key="bev_6p5m", target_mode="single")
    assert len(dataset) > 0


def test_resize_updates_pixel_center_intrinsics() -> None:
    from PIL import Image

    resize = CalibrationAwareResize(height=256, width=512)
    image = Image.new("RGB", (640, 480))
    intrinsics = torch.tensor([[100.0, 0.0, 319.5], [0.0, 100.0, 239.5], [0.0, 0.0, 1.0]])
    _, calibrated, valid = resize(image, intrinsics)
    # Scale is 256/480 with horizontal white padding.
    expected_width = round(640 * 256 / 480)
    pad_left = (512 - expected_width) // 2
    assert torch.isclose(calibrated[0, 0], torch.tensor(100.0 * expected_width / 640))
    assert torch.isclose(calibrated[1, 1], torch.tensor(100.0 * 256 / 480))
    assert valid[:, :pad_left].sum() == 0
