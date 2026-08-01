from __future__ import annotations

import torch

from vggt_bev_method1.models.decoder import DirectBEVDecoder
from vggt_bev_method1.models.geometry_conditioning import (
    GeometryBuilderConfig,
    build_p1a_geometry,
    fit_metric_per_native_scale,
)


def _synthetic_ground_geometry(
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Create a horizontal camera-y=scale plane in VGGT native units."""

    frames = 2
    height = width = 24
    focal = 16.0
    principal = 8.0
    rows = torch.arange(height, dtype=torch.float32)
    valid_rows = rows > principal + 1
    row_depth = torch.where(
        valid_rows,
        scale * focal / (rows - principal).clamp_min(1.0),
        torch.zeros_like(rows),
    )
    depth = row_depth.view(1, 1, height, 1).expand(
        1,
        frames,
        height,
        width,
    ).clone()
    confidence = torch.full_like(depth, 10.0)
    intrinsics = (
        torch.eye(3).view(1, 1, 3, 3).expand(1, frames, -1, -1).clone()
    )
    intrinsics[..., 0, 0] = focal
    intrinsics[..., 1, 1] = focal
    intrinsics[..., 0, 2] = width / 2
    intrinsics[..., 1, 2] = principal
    camera = torch.zeros(1, frames, 3, 4)
    camera[..., :3, :3] = torch.eye(3)
    camera[:, 1, 0, 3] = -0.5 * scale
    return {
        "depth": depth,
        "confidence": confidence,
        "intrinsics": intrinsics,
        "camera": camera,
    }


def test_p1a_camera_height_anchor_cancels_one_global_vggt_scalar() -> None:
    config = GeometryBuilderConfig(
        sample_stride=1,
        confidence_threshold=0.1,
        minimum_points=16,
        minimum_ground_quality=0.01,
    )
    first = _synthetic_ground_geometry()
    scaled = _synthetic_ground_geometry(3.0)
    output = build_p1a_geometry(
        depth=first["depth"],
        confidence=first["confidence"],
        intrinsics=first["intrinsics"],
        camera_from_world=first["camera"],
        camera_height_m=torch.tensor([1.5]),
        config=config,
    )
    scaled_output = build_p1a_geometry(
        depth=scaled["depth"],
        confidence=scaled["confidence"],
        intrinsics=scaled["intrinsics"],
        camera_from_world=scaled["camera"],
        camera_height_m=torch.tensor([1.5]),
        config=config,
    )
    assert output["geometry_valid"].all()
    assert scaled_output["geometry_valid"].all()
    assert torch.allclose(
        output["metric_per_vggt_native_unit"],
        scaled_output["metric_per_vggt_native_unit"] * 3.0,
        atol=1e-4,
    )
    assert torch.allclose(
        output["ground_origin_m"],
        scaled_output["ground_origin_m"],
        atol=1e-4,
    )
    for key, expected in (
        ("single_extent_normalized_scale", 6.5),
        ("merged_extent_normalized_scale", 10.0),
    ):
        assert torch.equal(output[key], torch.tensor([expected]))
        assert torch.equal(scaled_output[key], torch.tensor([expected]))


def test_p1a_decoder_has_no_free_extent_predictor() -> None:
    decoder = DirectBEVDecoder(
        supervision="complete",
        output_size=4,
        hidden_dim=8,
        geometry_cue_dim=24,
        heads=2,
        layers=1,
        self_attention_mode="exact",
        cross_attention_mode="exact",
        feature_levels=1,
        deformable_samples=2,
        cross_query_chunk_size=8,
        gradient_checkpointing=False,
    )
    assert not hasattr(decoder, "extent_predictor")
    output = decoder(
        torch.randn(1, 5, 8),
        torch.zeros(1, 24),
        extent_normalized_scale=torch.tensor([6.5]),
    )
    assert torch.equal(
        output["extent_normalized_scale"],
        torch.tensor([6.5]),
    )
    assert torch.equal(
        output["cell_size_normalized_scale"],
        torch.tensor([1.625]),
    )


def test_training_scale_fit_uses_one_isotropic_scalar() -> None:
    predicted = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 2.0]]]
    )
    ground_truth = predicted * 0.25 + torch.tensor([[[4.0, 2.0, -1.0]]])
    scale, quality = fit_metric_per_native_scale(predicted, ground_truth)
    assert torch.allclose(scale, torch.tensor([0.25]))
    assert quality[0] > 0.99


def test_invalid_ground_is_flagged_instead_of_predicting_an_extent() -> None:
    geometry = _synthetic_ground_geometry()
    geometry["confidence"].zero_()
    output = build_p1a_geometry(
        depth=geometry["depth"],
        confidence=geometry["confidence"],
        intrinsics=geometry["intrinsics"],
        camera_from_world=geometry["camera"],
        camera_height_m=torch.tensor([1.5]),
        config=GeometryBuilderConfig(minimum_points=16),
    )
    assert not output["geometry_valid"].any()
    assert output["geometry_quality"].item() == 0.0
    assert output["single_extent_normalized_scale"].item() == 6.5
    assert output["merged_extent_normalized_scale"].item() == 10.0
