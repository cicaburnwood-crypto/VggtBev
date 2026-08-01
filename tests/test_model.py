from __future__ import annotations

import torch
from torch import nn

from vggt_bev_method1.models import (
    Method1Head,
    Method1System,
    compose_fov_complete_semantic,
)


def _head() -> Method1Head:
    return Method1Head(
        cached_layers=(1, 2),
        spatial_scales=(1.0, 0.5),
        vggt_token_dim=8,
        hidden_dim=16,
        heads=4,
        decoder_layers=1,
        scale_decoder_layers=1,
        single_latent_bev_size=4,
        merged_latent_bev_size=4,
        single_output_size=16,
        merged_output_size=20,
        single_bev_extent_m=6.5,
        merged_bev_extent_m=10.0,
        cross_query_chunk_size=16,
    )


def _extraction() -> dict:
    return {
        "tokens": {
            1: torch.randn(2, 3, 16, 8),
            2: torch.randn(2, 3, 16, 8),
        },
        "patch_grid": (4, 4),
    }


def test_parallel_head_outputs_metric_bev_and_positive_scale() -> None:
    output = _head()(_extraction())
    assert output["single_bev"]["occupancy_probability"].shape == (2, 16, 16)
    assert output["merged_bev"]["occupancy_probability"].shape == (2, 20, 20)
    assert output["single_bev"]["raw_evidence"].shape == (2, 2, 16, 16)
    assert output["single_bev"]["raw_output"].shape == (2, 3, 16, 16)
    assert output["single_bev"]["fov_support_probability"].shape == (
        2,
        16,
        16,
    )
    assert torch.all(output["single_bev"]["fov_support_probability"] >= 0)
    assert torch.all(output["single_bev"]["fov_support_probability"] <= 1)
    assert torch.all(output["single_bev"]["evidence_confidence"] >= 0)
    assert torch.all(output["single_bev"]["evidence_confidence"] <= 1)
    assert torch.allclose(
        output["single_bev"]["evidence_confidence"]
        + output["single_bev"]["epistemic_uncertainty"],
        torch.ones_like(output["single_bev"]["evidence_confidence"]),
    )
    assert output["scale"]["lambda_m_per_vggt"].shape == (2,)
    assert torch.all(output["scale"]["lambda_m_per_vggt"] > 0)


def test_fov_complete_postprocess_keeps_outside_unknown() -> None:
    prediction = {
        "fov_support_probability": torch.tensor([[0.2, 0.8, 0.9]]),
        "occupancy_probability": torch.tensor([[0.9, 0.2, 0.7]]),
    }
    semantic = compose_fov_complete_semantic(prediction)
    assert semantic.tolist() == [[112, 255, 0]]


def test_single_uses_latest_tokens_while_merged_uses_full_window() -> None:
    torch.manual_seed(5)
    head = _head().eval()
    first = _extraction()
    changed = {
        "tokens": {
            layer: value.clone()
            for layer, value in first["tokens"].items()
        },
        "patch_grid": first["patch_grid"],
    }
    for value in changed["tokens"].values():
        value[:, 0] += 10.0
    with torch.no_grad():
        first_output = head(first)
        changed_output = head(changed)
    torch.testing.assert_close(
        first_output["single_bev"]["raw_evidence"],
        changed_output["single_bev"]["raw_evidence"],
    )
    assert not torch.allclose(
        first_output["merged_bev"]["raw_evidence"],
        changed_output["merged_bev"]["raw_evidence"],
    )


def test_bev_and_scale_use_gradient_isolated_projectors() -> None:
    head = _head()
    output = head(_extraction())
    output["single_bev"]["raw_evidence"].mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in head.token_projector.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.scale_token_projector.parameters()
    )

    head.zero_grad(set_to_none=True)
    output = head(_extraction())
    output["scale"]["log_lambda_m_per_vggt"].mean().backward()
    assert all(
        parameter.grad is None
        for parameter in head.token_projector.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in head.scale_token_projector.parameters()
    )


class FakeAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.geometry_calls = 0

    def aggregate(self, images: torch.Tensor) -> dict:
        batch, frames = images.shape[:2]
        return {
            "tokens": {
                1: torch.randn(batch, frames, 16, 8),
                2: torch.randn(batch, frames, 16, 8),
            },
            "patch_grid": (4, 4),
        }

    def decode_geometry(self, extraction: dict) -> dict:
        self.geometry_calls += 1
        return {"estimated_depth_vggt": torch.ones(1, 1, 4, 4)}


def test_runtime_forward_never_calls_geometry_heads() -> None:
    adapter = FakeAdapter()
    system = Method1System(
        adapter,
        cached_layers=(1, 2),
        spatial_scales=(1.0, 0.5),
        vggt_token_dim=8,
        hidden_dim=16,
        heads=4,
        decoder_layers=1,
        scale_decoder_layers=1,
        single_latent_bev_size=4,
        merged_latent_bev_size=4,
        single_output_size=16,
        merged_output_size=20,
        single_bev_extent_m=6.5,
        merged_bev_extent_m=10.0,
        cross_query_chunk_size=16,
    )
    output = system(torch.randn(1, 2, 3, 8, 8))
    assert adapter.geometry_calls == 0
    assert output["coordinate_mode"] == "p1b_fixed_metric"
    assert output["runtime_inputs"] == ("rgb_window",)
    assert output["bev_waits_for_geometry_heads"] is False
    assert output["single_bev_output_size"] == 16
    assert output["merged_bev_output_size"] == 20
    assert "unknown outside" in output["bev_content"]
    assert "FOV-support" in output["bev_support"]
    assert "Beta evidence" in output["bev_confidence"]
