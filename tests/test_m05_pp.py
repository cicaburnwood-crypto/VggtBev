from __future__ import annotations

from pathlib import Path

import torch

from vggt_bev_method1.m05_pp_config import load_m05_pp_config
from vggt_bev_method1.m05_pp_contract import PIPELINE_ID, RESOLUTION_HIERARCHY
from vggt_bev_method1.models.m05_pp import M05PPHead


def _head() -> M05PPHead:
    return M05PPHead(
        cached_layers=(1, 2),
        spatial_scales=(2.0, 1.0),
        vggt_token_dim=16,
        hidden_dim=8,
        query_content_dim=4,
        heads=2,
        latest_decoder_layers=1,
        temporal_proposal_layers=1,
        fine_correction_layers=1,
        scale_hidden_dim=8,
        scale_decoder_layers=1,
        self_attention_mode="linear",
        deformable_samples=2,
        cross_query_chunk_size=64,
        coarse_bev_size=4,
        merged_bev_size=8,
        merged_extent_m=10.0,
        query_fourier_bands=2,
        shared_refinement_layers=1,
        routing_refinement_layers=1,
        evidence_refinement_layers=1,
        predict_scale_uncertainty=True,
        patch_stream_dim=4,
        prefix_context_hidden_dim=8,
        prefix_context_heads=2,
        prefix_context_layers=1,
        maximum_history=3,
        maximum_prefix_tokens=3,
        frame_reliability_hidden_dim=8,
        frame_reliability_minimum=0.25,
        frame_reliability_maximum=1.75,
        temporal_null_initial_probability=0.9,
        history_proposal_batch_size=2,
    )


def _extraction(frames: int = 3) -> dict:
    return {
        "tokens": {
            layer: torch.randn(1, frames, 6, 16)
            for layer in (1, 2)
        },
        "camera_register_tokens": torch.randn(1, frames, 3, 16),
        "patch_grid": (2, 3),
    }


def test_m05_pp_production_resolution_and_capacity_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_m05_pp_config(
        root / "configs/m05_pp_a100_8gpu_10e_frozen.toml"
    )
    model = config["model"]
    assert model["pipeline_variant"] == PIPELINE_ID
    assert model["resolution_hierarchy"] == RESOLUTION_HIERARCHY
    assert model["coarse_bev_size"] == 256
    assert model["merged_bev_output_size"] == 512
    assert model["prefix_context_layers"] == 4
    assert model["prefix_context_hidden_dim"] == 512
    assert model["patch_stream_dim"] == 96
    assert model["fine_correction_layers"] == 1
    assert model["latest_high_resolution_skip"]
    assert not model["explicit_geometry_module"]
    assert not model["geometry_auxiliary_loss"]


def test_m05_pp_keeps_temporal_reasoning_coarse_and_outputs_fine() -> None:
    torch.manual_seed(41)
    head = _head().eval()
    output = head(
        _extraction(),
        include_latest_auxiliary=True,
        assemble_runtime_outputs=False,
    )
    assert head.coarse_query.size == 4
    assert head.high_resolution_correction.fine_query.size == 8
    assert output["merged_bev"]["guessed"]["occupancy_probability"].shape == (
        1,
        8,
        8,
    )
    assert output["latest_auxiliary_bev"]["guessed"][
        "occupancy_probability"
    ].shape == (1, 8, 8)
    assert output["temporal_frame_attention_mean"].shape == (1, 2)


def test_high_resolution_correction_reads_latest_patch_pyramid() -> None:
    torch.manual_seed(43)
    head = _head().eval()
    correction = head.high_resolution_correction
    with torch.no_grad():
        correction.delta_projection.weight.fill_(0.02)
    coarse = torch.randn(1, 16, 8)
    first = [torch.randn(1, 1, 8, 4, 6), torch.randn(1, 1, 8, 2, 3)]
    second = [value + 0.5 for value in first]
    output_a = correction(coarse, first)
    output_b = correction(coarse, second)
    assert output_a.shape == (1, 64, 8)
    assert not torch.equal(output_a, output_b)
