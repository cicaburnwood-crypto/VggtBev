from __future__ import annotations

import torch

from vggt_bev_method1.cli_audit_wtbd_scale import summarize_scale_audit
from vggt_bev_method1.data.vggt_unit_targets import (
    regrid_merged_metric_targets_to_vggt_units,
    restore_metric_grid_contract,
)
from vggt_bev_method1.models.wtbd_merge_scale import (
    ImplicitGeometryContextTrunk,
    WTBDMergeScaleHead,
    WTBDMergeScaleSystem,
)


def _tiny_inputs() -> dict:
    return {
        "tokens": {1: torch.randn(1, 2, 4, 8)},
        "camera_register_tokens": torch.randn(1, 2, 3, 8),
        "patch_grid": (2, 2),
    }


def _tiny_head() -> WTBDMergeScaleHead:
    return WTBDMergeScaleHead(
        cached_layers=(1,),
        spatial_scales=(1.0,),
        vggt_token_dim=8,
        hidden_dim=4,
        heads=1,
        decoder_layers=1,
        scale_decoder_layers=1,
        self_attention_mode="linear",
        cross_attention_mode="linear",
        deformable_samples=1,
        cross_query_chunk_size=4,
        merged_latent_bev_size=2,
        merged_output_size=2,
        merged_extent_vggt=2.0,
        implicit_geometry_hidden_dim=4,
        implicit_geometry_heads=1,
        implicit_geometry_layers=2,
        maximum_history=2,
        maximum_prefix_tokens=3,
    )


def test_implicit_geometry_trunk_is_frame_order_and_reference_aware() -> None:
    torch.manual_seed(2)
    trunk = ImplicitGeometryContextTrunk(
        input_dim=8,
        hidden_dim=4,
        output_dim=4,
        heads=1,
        layers=1,
        maximum_history=2,
        maximum_prefix_tokens=3,
    )
    tokens = _tiny_inputs()["camera_register_tokens"]
    first = trunk(tokens)
    reversed_output = trunk(tokens.flip(1)).flip(1)
    assert first.shape == (1, 2, 4)
    assert not torch.allclose(first, reversed_output)


def test_no_single_and_scale_is_not_a_merged_input() -> None:
    torch.manual_seed(3)
    head = _tiny_head().eval()
    assert not any(name.startswith("single_") for name, _ in head.named_parameters())
    extraction = _tiny_inputs()
    with torch.no_grad():
        first = head(extraction, assemble_runtime_outputs=False)
        for parameter in head.scale_token_projector.parameters():
            parameter.add_(10.0 * torch.randn_like(parameter))
        for parameter in head.scale_decoder.parameters():
            parameter.add_(10.0 * torch.randn_like(parameter))
        second = head(extraction, assemble_runtime_outputs=False)
    assert torch.equal(
        first["merged_bev"]["observed_gate_logit"],
        second["merged_bev"]["observed_gate_logit"],
    )
    assert not torch.equal(
        first["scale"]["log_lambda_m_per_vggt"],
        second["scale"]["log_lambda_m_per_vggt"],
    )


def test_merged_and_scale_trainable_gradients_are_disjoint() -> None:
    torch.manual_seed(4)
    head = _tiny_head()
    extraction = _tiny_inputs()
    output = head(extraction, assemble_runtime_outputs=False)
    merged_parameters = [
        parameter
        for name, parameter in head.named_parameters()
        if name.startswith(("merged_", "implicit_geometry_trunk."))
    ]
    scale_parameters = [
        parameter
        for name, parameter in head.named_parameters()
        if name.startswith("scale_")
    ]
    merged_loss = output["merged_bev"]["observed_gate_logit"].sum()
    scale_loss = output["scale"]["log_lambda_m_per_vggt"].sum()
    merged_to_scale = torch.autograd.grad(
        merged_loss, scale_parameters, allow_unused=True, retain_graph=True
    )
    scale_to_merged = torch.autograd.grad(
        scale_loss, merged_parameters, allow_unused=True
    )
    assert all(gradient is None for gradient in merged_to_scale)
    assert all(gradient is None for gradient in scale_to_merged)


def test_runtime_forward_never_calls_geometry_heads() -> None:
    class RuntimeAdapter(torch.nn.Module):
        def aggregate(self, images: torch.Tensor) -> dict:
            return _tiny_inputs()

        def decode_geometry(self, extraction: dict) -> dict:
            raise AssertionError("runtime must not invoke VGGT geometry heads")

    system = WTBDMergeScaleSystem(
        RuntimeAdapter(),
        cached_layers=(1,),
        spatial_scales=(1.0,),
        vggt_token_dim=8,
        hidden_dim=4,
        heads=1,
        decoder_layers=1,
        scale_decoder_layers=1,
        self_attention_mode="linear",
        cross_attention_mode="linear",
        deformable_samples=1,
        cross_query_chunk_size=4,
        merged_latent_bev_size=2,
        merged_output_size=2,
        merged_extent_vggt=2.0,
        implicit_geometry_hidden_dim=4,
        implicit_geometry_heads=1,
        implicit_geometry_layers=1,
        maximum_history=2,
        maximum_prefix_tokens=3,
    ).eval()
    with torch.no_grad():
        output = system(torch.empty(1, 2, 3, 4, 4))
    assert output["extrinsic_input_present"] is False
    assert output["bev_waits_for_geometry_heads"] is False
    assert output["geometry_conditioning"] == (
        "implicit_multiview_token_cross_attention"
    )


def test_metric_target_is_regridded_by_lambda_not_resized() -> None:
    complete = torch.tensor(
        [[[0, 0, 0, 0, 0], [0, 255, 255, 255, 0], [0, 255, 255, 255, 0],
          [0, 255, 255, 255, 0], [0, 0, 0, 0, 0]]],
        dtype=torch.uint8,
    )
    support = torch.ones_like(complete, dtype=torch.bool)
    visible = complete.clone()
    valid = torch.ones_like(support)
    identity = regrid_merged_metric_targets_to_vggt_units(
        complete,
        visible,
        support,
        valid,
        torch.ones(1),
        torch.ones(1, dtype=torch.bool),
        source_extent_m=4.0,
        target_extent_vggt=4.0,
        target_size=5,
    )
    assert torch.equal(identity["complete_target"], complete)
    cropped = regrid_merged_metric_targets_to_vggt_units(
        complete,
        visible,
        support,
        valid,
        torch.full((1,), 2.0),
        torch.ones(1, dtype=torch.bool),
        source_extent_m=4.0,
        target_extent_vggt=4.0,
        target_size=5,
    )
    assert cropped["source_coverage_fraction"].item() < 1.0
    assert not bool(cropped["gt_valid_mask"][0, 0, 0])


def test_runtime_metric_restoration_changes_coordinates_only() -> None:
    restored = restore_metric_grid_contract(
        torch.tensor([2.0]), extent_vggt=6.0, output_size=600
    )
    assert restored["extent_m"].item() == 12.0
    assert torch.allclose(restored["cell_size_m"], torch.tensor([0.02]))
    assert torch.allclose(
        restored["bounds_m"], torch.tensor([[-6.0, 6.0, -6.0, 6.0]])
    )


def test_scale_audit_recommends_low_available_extent_quantile() -> None:
    report = summarize_scale_audit(
        torch.tensor([1.0, 2.0, 4.0, 5.0]),
        torch.ones(4),
        torch.full((4,), 0.05),
        source_extent_m=10.0,
        configured_extent_vggt=6.5,
    )
    recommendation = report["recommended_max_extent_vggt"]
    assert (
        recommendation["for_99_percent_full_source_coverage"]
        <= recommendation["for_95_percent_full_source_coverage"]
        <= recommendation["for_90_percent_full_source_coverage"]
    )
