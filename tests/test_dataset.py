from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggt_bev_method1.data import VGGNAVMethod1Dataset, method1_collate


def _session(root: Path, *, procthor: bool = False) -> Path:
    path = root / "GPU0" / "session_000001_test_scene"
    for directory in (
        "camera",
        "depth",
        "bev_6p5m/masked",
        "bev_6p5m/complete",
        "bev_6p5m/merged_masked_10m",
        "bev_6p5m/merged_complete_10m",
    ):
        (path / directory).mkdir(parents=True, exist_ok=True)
    (path / "COMPLETE").write_text("", encoding="utf-8")
    metadata = {
        "schema_version": 4,
        "status": "complete",
        "dataset": "procthor-10k" if procthor else "test",
        "scene_id": "scene",
        "frame_count": 1,
        "camera_intrinsics": {
            "width": 8,
            "height": 6,
            "K": [[4.0, 0.0, 3.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]],
            "horizontal_fov_degrees": 90.0,
        },
        "camera_extrinsics_file": "camera_extrinsics.jsonl",
        "depth": {
            "filename_pattern": "frame_INDEX.npy",
            "units": "metres",
            "source": (
                "AI2-THOR synchronized third-party metric depth ground truth"
                if procthor
                else "Habitat-Sim pinhole depth sensor ground truth"
            ),
        },
        "bev": {
            "size": 512,
            "extent_classes_m": [6.5],
            "merged_normalized_extents_m": [10.0],
            "merged_normalized_size": [512, 512],
            "masked_values": {"occupied": 0, "unknown": 112, "free": 255},
            "merged_orientation": (
                "ego-centric; latest robot centered and forward up"
                if procthor
                else "ego-centric in every output; latest robot centered and forward up"
            ),
        },
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (path / "camera_intrinsics.json").write_text(
        json.dumps(metadata["camera_intrinsics"]),
        encoding="utf-8",
    )
    (path / "camera_extrinsics.jsonl").write_text(
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
    Image.new("RGB", (8, 6), (20, 30, 40)).save(
        path / "camera/frame_000000.png"
    )
    if procthor:
        x = (np.arange(8, dtype=np.float32) - 3.5) / 4.0
        y = (np.arange(6, dtype=np.float32) - 2.5) / 4.0
        depth = 2.0 * np.sqrt(1.0 + y[:, None] ** 2 + x[None, :] ** 2)
    else:
        depth = np.full((6, 8), 2.0, np.float32)
    np.save(path / "depth/frame_000000.npy", depth)
    complete = np.full((512, 512), 255, np.uint8)
    complete[200:320, 250:300] = 0
    observed = np.full((512, 512), 112, np.uint8)
    observed[200:220, 250:260] = complete[200:220, 250:260]
    observed[300:350, 200:300] = complete[300:350, 200:300]
    Image.fromarray(observed).save(path / "bev_6p5m/masked/frame_000000.png")
    Image.fromarray(complete).save(path / "bev_6p5m/complete/frame_000000.png")
    Image.fromarray(observed).save(
        path / "bev_6p5m/merged_masked_10m/frame_000000.png"
    )
    Image.fromarray(complete).save(
        path / "bev_6p5m/merged_complete_10m/frame_000000.png"
    )
    return path


def test_dataset_separates_runtime_rgb_from_training_only_labels(tmp_path: Path) -> None:
    _session(tmp_path)
    dataset = VGGNAVMethod1Dataset(
        tmp_path,
        supervision="metric_fov_complete_evidential",
        maximum_history=1,
    )
    sample = dataset[0]
    assert sample["images"].shape == (1, 3, 384, 512)
    assert sample["scale_gt_depth_m"].shape == (1, 384, 512)
    assert sample["scale_gt_intrinsics"].shape == (1, 3, 3)
    assert sample["single_fov_complete_target"].shape == (512, 512)
    assert sample["single_visible_target"].shape == (512, 512)
    assert sample["single_fov_support_target"].shape == (512, 512)
    assert sample["merged_fov_complete_target"].shape == (800, 800)
    assert sample["merged_visible_target"].shape == (800, 800)
    assert sample["merged_fov_support_target"].shape == (800, 800)
    assert torch.any(sample["single_fov_complete_target"] == 112)
    assert torch.any(
        (sample["single_fov_complete_target"] != 112)
        & (sample["single_visible_target"] == 112)
    )
    assert torch.equal(
        sample["single_fov_support_target"],
        sample["single_fov_complete_target"] != 112,
    )
    assert sample["metadata"]["runtime_model_inputs"] == ["rgb_window"]
    assert sample["metadata"]["single_bev_cell_size_m"] == 6.5 / 512
    assert sample["metadata"]["merged_bev_cell_size_m"] == 10.0 / 800

    batch = method1_collate([sample])
    assert batch["images"].shape == (1, 1, 3, 384, 512)
    assert batch["scale_gt_depth_m"].shape == (1, 1, 384, 512)
    assert batch["single_fov_complete_target"].shape == (1, 512, 512)
    assert batch["single_visible_target"].shape == (1, 512, 512)
    assert batch["single_fov_support_target"].shape == (1, 512, 512)
    assert batch["merged_fov_complete_target"].shape == (1, 800, 800)
    assert batch["merged_visible_target"].shape == (1, 800, 800)
    assert batch["merged_fov_support_target"].shape == (1, 800, 800)
    assert torch.all(batch["scale_gt_depth_m"] == 2.0)


def test_procthor_ray_distance_is_converted_to_camera_axis_z(tmp_path: Path) -> None:
    _session(tmp_path, procthor=True)
    dataset = VGGNAVMethod1Dataset(
        tmp_path,
        supervision="metric_fov_complete_evidential",
        maximum_history=1,
    )
    sample = dataset[0]
    assert sample["metadata"]["source_gt_depth_convention"] == (
        "euclidean_camera_ray_distance_m"
    )
    assert sample["metadata"]["gt_depth_convention"] == "camera_axis_z_depth_m"
    assert torch.allclose(
        sample["scale_gt_depth_m"],
        torch.full_like(sample["scale_gt_depth_m"], 2.0),
        atol=1e-5,
    )
