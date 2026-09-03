from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from vggt_bev_method1.cli_train_m05 import (
    _contract,
    _load_checkpoint,
    _save_checkpoint,
)
from vggt_bev_method1.m05_config import load_m05_config
from vggt_bev_method1.m05_losses import m05_bev_loss, m05_loss_weights
from vggt_bev_method1.models import DenseNativeQuery, M05System


class _Adapter(nn.Module):
    def aggregate(self, images: torch.Tensor) -> dict:
        raise NotImplementedError


def _model(
    adapter: nn.Module | None = None,
    *,
    cross_query_chunk_size: int = 16,
) -> M05System:
    return M05System(
        adapter or _Adapter(),
        cached_layers=(1,),
        spatial_scales=(1.0,),
        vggt_token_dim=16,
        hidden_dim=8,
        heads=2,
        latest_decoder_layers=1,
        history_update_layers=1,
        scale_decoder_layers=1,
        self_attention_mode="linear",
        deformable_samples=2,
        cross_query_chunk_size=cross_query_chunk_size,
        merged_bev_size=8,
        merged_extent_vggt=6.5,
        query_fourier_bands=2,
        refinement_layers=1,
        predict_scale_uncertainty=True,
        implicit_geometry_hidden_dim=8,
        implicit_geometry_heads=2,
        implicit_geometry_layers=1,
        maximum_history=4,
        maximum_prefix_tokens=3,
        frame_reliability_hidden_dim=8,
        history_gate_initial_bias=-1.0,
    )


def _extraction(frames: int) -> dict:
    return {
        "tokens": {1: torch.randn(2, frames, 4, 16)},
        "camera_register_tokens": torch.randn(2, frames, 3, 16),
        "patch_grid": (2, 2),
    }


def test_m05_template_preserves_native_existing_gt_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_m05_config(root / "configs/m05_reverse_gated_10m_template.toml")
    assert config["data"]["merged_source_extent_m"] == 10.0
    assert config["data"]["merged_source_image_size"] == 512
    assert config["data"]["merged_source_output_size"] == 512
    assert config["model"]["merged_bev_output_size"] == 512
    assert config["model"]["full_per_pixel_query"]
    assert config["model"]["reverse_history_weight_sharing"]
    assert config["model"]["cross_query_chunk_size"] == 65536
    assert config["training"]["bev_objective"] == (
        "single_baseline_dual_supervision"
    )


def test_dense_native_query_restores_one_content_vector_per_cell() -> None:
    query = DenseNativeQuery(
        size=16,
        extent_vggt=6.5,
        hidden_dim=8,
        fourier_bands=2,
    )
    assert query.query_content.shape == (16 * 16, 8)
    assert query(3).shape == (3, 16 * 16, 8)


def test_m05_is_latest_anchored_and_updates_history_newest_to_oldest() -> None:
    torch.manual_seed(5)
    model = _model().eval()
    calls: list[bool] = []
    handle = model.head.history_updates[0].register_forward_hook(
        lambda *arguments: calls.append(True)
    )
    one = model.forward_head(_extraction(1))
    assert calls == []
    assert one["history_frame_indices_newest_to_oldest"] == ()
    assert one["history_update_gate_mean"].shape == (2, 0)
    four = model.forward_head(_extraction(4))
    handle.remove()
    assert calls == [True, True, True]
    assert four["history_frame_indices_newest_to_oldest"] == (2, 1, 0)
    assert four["history_update_gate_mean"].shape == (2, 3)
    assert four["merged_bev"]["occupancy_probability"].shape == (2, 8, 8)
    assert four["runtime_inputs"] == ("rgb_window",)
    assert four["runtime_passes"] == 1
    assert not four["single_bev_present"]
    assert not four["latest_auxiliary_runtime_output"]
    assert not four["extrinsic_input_present"]
    assert not four["camera_height_input_present"]


def test_latest_auxiliary_is_training_only_and_shares_native_head() -> None:
    model = _model().eval()
    runtime = model.forward_head(_extraction(3))
    assert "latest_auxiliary_bev" not in runtime
    training = model.forward_head(
        _extraction(3),
        include_latest_auxiliary=True,
        assemble_runtime_outputs=False,
    )
    assert training["latest_auxiliary_bev"]["observed_gate_logit"].shape == (
        2,
        8,
        8,
    )
    assert training["merged_bev"]["observed_gate_logit"].shape == (2, 8, 8)
    assert len([name for name, _ in model.head.named_modules() if name == "evidence_head"]) == 1


def test_scale_parameters_cannot_change_m05_bev() -> None:
    torch.manual_seed(8)
    model = _model().eval()
    extraction = _extraction(3)
    before = model.forward_head(extraction)["merged_bev"]["occupancy_probability"]
    with torch.no_grad():
        for module in (
            model.head.scale_token_projector,
            model.head.scale_decoder,
            model.head.scale_frame_reliability,
        ):
            for parameter in module.parameters():
                parameter.add_(torch.randn_like(parameter) * 10.0)
    after = model.forward_head(extraction)["merged_bev"]["occupancy_probability"]
    torch.testing.assert_close(after, before)


def test_query_chunk_optimization_preserves_predictions() -> None:
    torch.manual_seed(9)
    small_chunks = _model(cross_query_chunk_size=4).eval()
    large_chunks = _model(cross_query_chunk_size=64).eval()
    large_chunks.load_state_dict(small_chunks.state_dict(), strict=True)
    extraction = _extraction(3)
    small = small_chunks.forward_head(extraction)
    large = large_chunks.forward_head(extraction)
    torch.testing.assert_close(
        large["merged_bev"]["occupancy_probability"],
        small["merged_bev"]["occupancy_probability"],
    )
    torch.testing.assert_close(
        large["scale"]["lambda_m_per_vggt"],
        small["scale"]["lambda_m_per_vggt"],
    )


def _target() -> dict[str, torch.Tensor]:
    complete = torch.full((2, 8, 8), 112, dtype=torch.uint8)
    complete[:, 1:7, 1:7] = 255
    complete[:, 2:4, 4:6] = 0
    visible = torch.full_like(complete, 112)
    visible[:, 4:7, 2:6] = complete[:, 4:7, 2:6]
    support = complete != 112
    return {
        "complete_target": complete,
        "visible_target": visible,
        "support_target": support,
        "gt_valid_mask": torch.ones_like(support),
    }


def test_m05_dual_single_loss_backpropagates_latest_and_history() -> None:
    torch.manual_seed(12)
    model = _model().train()
    prediction = model.forward_head(
        _extraction(3),
        include_scale=False,
        include_latest_auxiliary=True,
        assemble_runtime_outputs=False,
    )
    result = m05_bev_loss(
        prediction["merged_bev"],
        prediction["latest_auxiliary_bev"],
        _target(),
        _target(),
        weights=m05_loss_weights({}),
        global_step=10,
        total_steps=100,
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert model.head.query.query_content.grad is not None
    assert model.head.history_updates[0].gate_bias.grad is not None
    assert model.head.evidence_head.weight.grad is not None
    torch.testing.assert_close(
        result["loss"],
        0.5 * (result["merged_loss"] + result["latest_auxiliary_loss"]),
    )


def test_m05_dual_loss_supports_a_convex_diagnostic_mix() -> None:
    model = _model().train()
    prediction = model.forward_head(
        _extraction(3),
        include_scale=False,
        include_latest_auxiliary=True,
        assemble_runtime_outputs=False,
    )
    result = m05_bev_loss(
        prediction["merged_bev"],
        prediction["latest_auxiliary_bev"],
        _target(),
        _target(),
        weights=m05_loss_weights({}),
        global_step=10,
        total_steps=100,
        latest_auxiliary_weight=0.25,
    )
    torch.testing.assert_close(
        result["loss"],
        0.75 * result["merged_loss"] + 0.25 * result["latest_auxiliary_loss"],
    )
    with pytest.raises(ValueError, match="latest auxiliary weight"):
        m05_bev_loss(
            prediction["merged_bev"],
            prediction["latest_auxiliary_bev"],
            _target(),
            _target(),
            weights=m05_loss_weights({}),
            global_step=0,
            total_steps=1,
            latest_auxiliary_weight=1.1,
        )


def test_m05_checkpoint_roundtrip_is_strict(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    contract = {
        "pipeline_id": model.head.pipeline_id,
        "checkpoint_schema": "unit-m05",
        "manifest_sha256": "manifest",
        "vggt_checkpoint_sha256": "vggt",
    }
    path = tmp_path / "checkpoint.pt"
    expected = {
        name: value.detach().clone() for name, value in model.head.state_dict().items()
    }
    _save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config={"test": True},
        contract=contract,
        epoch=3,
        global_step=17,
        batch_in_epoch=5,
    )
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
    assert _load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        contract=contract,
    ) == (3, 17, 5)
    for name, value in model.head.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    with pytest.raises(ValueError, match="checkpoint_schema"):
        _load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            contract={**contract, "checkpoint_schema": "wrong"},
        )


def test_m05_contract_has_no_runtime_gt_or_navigation() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_m05_config(root / "configs/m05_reverse_gated_10m_template.toml")
    contract = _contract(config, "manifest", "vggt")
    assert contract["runtime_external_inputs"] == ["rgb_window"]
    assert contract["history_order"] == "newest_to_oldest_after_latest_anchor"
    assert contract["latest_auxiliary_training_only"]
    assert not contract["extrinsic_input_present"]
    assert not contract["camera_height_input_present"]
    assert not contract["navigation_objective_present"]
