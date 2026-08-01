from __future__ import annotations

from pathlib import Path

import torch

from vggt_bev_method1.teacher_cache import TeacherCache


def test_teacher_cache_round_trip_is_bound_to_window_contract(tmp_path: Path) -> None:
    cache = TeacherCache(
        tmp_path,
        checkpoint_sha256="abc",
        preprocessing_version="rgb-depth-resize-pad-v2",
    )
    metadata = [
        {
            "sample_id": "scene:frame_000001",
            "source_frame_ids": [0, 1],
            "reference_frame_id": 1,
        }
    ]
    extraction = {
        "tokens": {4: torch.randn(1, 2, 6, 8)},
        "patch_grid": (2, 3),
    }
    geometry = {
        "estimated_depth_vggt": torch.rand(1, 2, 4, 6),
        "estimated_depth_confidence": torch.rand(1, 2, 4, 6) + 1,
        "estimated_intrinsics": torch.eye(3)[None, None].expand(1, 2, 3, 3),
        "estimated_camera_from_world_vggt": torch.zeros(1, 2, 3, 4),
    }
    cache.save(metadata, extraction, geometry)
    loaded = cache.load(metadata, device=torch.device("cpu"))
    assert loaded is not None
    loaded_extraction, loaded_geometry = loaded
    assert loaded_extraction["tokens"][4].shape == (1, 2, 6, 8)
    assert loaded_extraction["patch_grid"] == (2, 3)
    assert loaded_geometry["estimated_depth_vggt"].shape == (1, 2, 4, 6)
