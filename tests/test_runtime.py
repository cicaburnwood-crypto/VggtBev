import json
from pathlib import Path

import torch
from PIL import Image

from vggt_bev.runtime import (
    RuntimePrediction,
    RuntimeSequence,
    checkpoint_sha256,
    default_model_path,
    default_vggt_paths,
    extent_key_to_meters,
    load_vggnav_runtime_sequence,
    save_runtime_prediction,
)

DATA_ROOT = Path("/home/user/Project/VGGNAV/output/random_sessions_test_3")
PROJECT_ROOT = Path("/home/user/Project/VGGTBEV")


def test_extent_key_and_model_catalog() -> None:
    assert extent_key_to_meters("bev_3p5m") == 3.5
    assert extent_key_to_meters("bev_5m") == 5.0
    assert extent_key_to_meters("bev_6p5m") == 6.5
    assert default_model_path("5m", PROJECT_ROOT).name.endswith("_bev_5m.pt")


def test_vggt_path_discovery_supports_parent_checkout(tmp_path: Path) -> None:
    project = tmp_path / "vggt" / "VGGTBEV"
    source = project.parent
    checkpoint = source / "checkpoints" / "VGGT-Omega-1B-512" / "model.pt"
    (source / "vggt_omega").mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    discovered_source, discovered_checkpoint = default_vggt_paths(project)

    assert discovered_source == source
    assert discovered_checkpoint == checkpoint


def test_runtime_sequence_requires_only_camera_frames(tmp_path: Path) -> None:
    source = next(path for path in DATA_ROOT.glob("session_*") if path.is_dir())
    session = tmp_path / source.name
    session.mkdir()
    (session / "camera").symlink_to(source / "camera", target_is_directory=True)

    sequence = load_vggnav_runtime_sequence(
        session,
        target_frame=1,
        image_height=64,
        image_width=64,
    )
    assert sequence.images.shape == (2, 3, 64, 64)
    assert sequence.frame_ids == (0, 1)
    assert sequence.target_frame_id == 1
    assert [path.name for path in session.iterdir()] == ["camera"]
    assert sequence.camera_height_m is None


def test_save_runtime_prediction_contract(tmp_path: Path) -> None:
    source = next(path for path in DATA_ROOT.glob("session_*") if path.is_dir())
    sequence = load_vggnav_runtime_sequence(
        source,
        target_frame=0,
        image_height=64,
        image_width=64,
    )
    labels = torch.tensor([[0, 112], [255, 112]], dtype=torch.uint8)
    prediction = RuntimePrediction(
        single_labels=labels,
        merged_labels=labels.clone(),
        geometry_single_labels=labels.flip(0),
        geometry_merged_labels=labels.flip(1),
        single_occupancy_probability=torch.full((2, 2), 0.25),
        single_observed_probability=torch.full((2, 2), 0.75),
        merged_occupancy_probability=torch.full((2, 2), 0.5),
        merged_observed_probability=torch.full((2, 2), 0.6),
        depth_scale=2.5,
        single_extent_m=5.0,
        merged_extent_m=8.0,
        output_size=2,
        checkpoint_epoch=10,
        checkpoint_global_step=172820,
        geometry_estimated_intrinsic=torch.eye(3),
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    paths = save_runtime_prediction(
        prediction,
        tmp_path / "output",
        sequence=sequence,
        checkpoint_path=checkpoint,
        threshold=0.5,
    )
    with Image.open(paths["single"]) as image:
        assert image.mode == "L"
        assert list(image.getdata()) == [0, 112, 255, 112]
    metadata = json.loads(Path(paths["metadata"]).read_text())
    assert metadata["checkpoint_sha256"] == checkpoint_sha256(checkpoint)
    assert metadata["single"]["extent_m"] == 5.0
    assert metadata["merged"]["extent_m"] == 8.0
    assert Path(paths["geometry_single"]).is_file()
    assert Path(paths["geometry_merged"]).is_file()
    assert metadata["geometry_projection"]["uses_bev_head"] is False
    assert metadata["label_values"] == {"occupied": 0, "unknown": 112, "free": 255}


def test_save_normalized_prediction_has_learned_ranges_and_dual_sizes(
    tmp_path: Path,
) -> None:
    sequence = RuntimeSequence(
        images=torch.zeros((1, 3, 4, 4)),
        image_valid=torch.ones((1, 4, 4), dtype=torch.bool),
        camera_height_m=None,
        frame_ids=(0,),
        session_path=Path("synthetic"),
        target_frame_id=0,
    )
    single = torch.full((2, 2), 112, dtype=torch.uint8)
    merged = torch.full((3, 3), 112, dtype=torch.uint8)
    prediction = RuntimePrediction(
        single_labels=single,
        merged_labels=merged,
        geometry_single_labels=single.clone(),
        geometry_merged_labels=merged.clone(),
        single_occupancy_probability=torch.zeros((2, 2)),
        single_observed_probability=torch.zeros((2, 2)),
        merged_occupancy_probability=torch.zeros((3, 3)),
        merged_observed_probability=torch.zeros((3, 3)),
        depth_scale=1.0,
        single_extent_m=2.2,
        merged_extent_m=3.4375,
        output_size=2,
        checkpoint_epoch=1,
        checkpoint_global_step=3,
        geometry_estimated_intrinsic=torch.eye(3),
        metric_scale_mode="vggt_normalized",
        coordinate_mode="vggt_normalized",
        merged_output_size=3,
        reference_scale_vggt=4.5,
        normalized_units_per_output_pixel=1.1,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"normalized checkpoint")

    paths = save_runtime_prediction(
        prediction,
        tmp_path / "normalized",
        sequence=sequence,
        checkpoint_path=checkpoint,
        threshold=0.5,
    )
    metadata = json.loads(Path(paths["metadata"]).read_text())

    assert metadata["single"]["shape"] == [2, 2]
    assert metadata["merged"]["shape"] == [3, 3]
    assert metadata["single"]["span_normalized_vggt"] == 2.2
    assert metadata["merged"]["span_normalized_vggt"] == 3.4375
    assert metadata["reference_scale_vggt"] == 4.5
    assert "extent_m" not in metadata["single"]
