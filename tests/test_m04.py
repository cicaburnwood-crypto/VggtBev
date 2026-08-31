from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from vggt_bev_method1.cli_train_m04 import (
    _contract,
    _load_checkpoint,
    _save_checkpoint,
)
from vggt_bev_method1.data.vggt_unit_targets import (
    regrid_merged_metric_targets_to_vggt_units,
)
from vggt_bev_method1.m04_config import load_m04_config
from vggt_bev_method1.m04_losses import m04_bev_loss, m04_scale_loss
from vggt_bev_method1.models import FactorizedNativeQuery, M04System


class _Adapter(nn.Module):
    def aggregate(self, images: torch.Tensor) -> dict:
        raise NotImplementedError


def _model(adapter: nn.Module | None = None) -> M04System:
    return M04System(
        adapter or _Adapter(),
        cached_layers=(1,),
        spatial_scales=(1.0,),
        vggt_token_dim=16,
        hidden_dim=8,
        heads=2,
        latest_decoder_layers=1,
        history_decoder_layers=1,
        scale_decoder_layers=1,
        self_attention_mode="linear",
        deformable_samples=2,
        cross_query_chunk_size=16,
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
    )


def _extraction(frames: int) -> dict:
    return {
        "tokens": {1: torch.randn(2, frames, 4, 16)},
        "camera_register_tokens": torch.randn(2, frames, 3, 16),
        "patch_grid": (2, 2),
    }


def test_m04_template_keeps_existing_10m_supervision() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_m04_config(
        root / "configs/m04_parallel_anchor_history_10m_template.toml"
    )
    assert config["data"]["merged_source_extent_m"] == 10.0
    assert config["data"]["merged_complete_directory"] == "merged_complete_10m"
    assert config["data"]["merged_source_output_size"] == 512
    assert config["model"]["merged_bev_output_size"] == 512
    assert config["model"]["merged_bev_extent_vggt"] == 6.5
    assert not config["training"]["ddp_static_graph"]
    assert config["training"]["ddp_find_unused_parameters"]


def test_m04_merged_grid_is_native_512_without_upsampling() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_m04_config(
        root / "configs/m04_parallel_anchor_history_10m_template.toml"
    )
    assert (
        config["data"]["merged_source_image_size"]
        == config["data"]["merged_source_output_size"]
        == config["model"]["merged_bev_output_size"]
        == 512
    )


def test_m04_checkpoint_contract_binds_existing_gt_without_runtime_gt() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_m04_config(
        root / "configs/m04_parallel_anchor_history_10m_template.toml"
    )
    contract = _contract(config, "manifest", "vggt")
    assert contract["merged_source_extent_m"] == 10.0
    assert contract["source_outside_loss_policy"] == "hard_ignore"
    assert contract["runtime_external_inputs"] == ["rgb_window"]
    assert not contract["camera_height_input_present"]
    assert not contract["extrinsic_input_present"]
    assert not contract["navigation_objective_present"]


def test_m04_checkpoint_roundtrip_is_strict(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _: 1.0
    )
    contract = {
        "pipeline_id": model.head.pipeline_id,
        "checkpoint_schema": "unit-m04",
        "manifest_sha256": "manifest",
        "vggt_checkpoint_sha256": "vggt",
    }
    path = tmp_path / "checkpoint.pt"
    expected = {
        name: value.detach().clone()
        for name, value in model.head.state_dict().items()
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


def test_factorized_query_has_no_per_pixel_content_table() -> None:
    query = FactorizedNativeQuery(
        size=16,
        extent_vggt=6.5,
        hidden_dim=8,
        fourier_bands=2,
    )
    names = dict(query.named_parameters())
    assert set(names) == {
        "row_embedding",
        "column_embedding",
        "coordinate_projection.0.weight",
        "coordinate_projection.0.bias",
        "coordinate_projection.2.weight",
        "coordinate_projection.2.bias",
    }
    assert query(3).shape == (3, 16 * 16, 8)


def test_m04_is_one_pass_rgb_only_and_history_is_structurally_disabled_at_n1() -> None:
    torch.manual_seed(4)
    model = _model().eval()
    history_calls = []
    handle = model.head.history_branch.register_forward_hook(
        lambda *arguments: history_calls.append(True)
    )
    one = model.forward_head(_extraction(1))
    assert history_calls == []
    three = model.forward_head(_extraction(3))
    assert history_calls == [True]
    handle.remove()
    assert one["merged_bev"]["occupancy_probability"].shape == (2, 8, 8)
    assert three["merged_bev"]["occupancy_probability"].shape == (2, 8, 8)
    assert one["runtime_inputs"] == ("rgb_window",)
    assert one["runtime_passes"] == 1
    assert not one["single_bev_present"]
    assert not one["extrinsic_input_present"]
    assert not one["camera_height_input_present"]
    assert not one["scale_is_merged_input"]
    assert one["frame_reliability"].shape == (2, 1)
    assert one["scale_frame_reliability"].shape == (2, 1)


def test_scale_parameters_cannot_change_m04_bev() -> None:
    torch.manual_seed(8)
    model = _model().eval()
    extraction = _extraction(3)
    before = model.forward_head(extraction)["merged_bev"]["occupancy_probability"]
    with torch.no_grad():
        for parameter in model.head.scale_token_projector.parameters():
            parameter.add_(torch.randn_like(parameter) * 10.0)
        for parameter in model.head.scale_decoder.parameters():
            parameter.add_(torch.randn_like(parameter) * 10.0)
        for parameter in model.head.scale_frame_reliability.parameters():
            parameter.add_(torch.randn_like(parameter) * 10.0)
    after = model.forward_head(extraction)["merged_bev"]["occupancy_probability"]
    torch.testing.assert_close(after, before)


def _module_gradient_sum(module: nn.Module) -> float:
    return sum(
        0.0 if parameter.grad is None else float(parameter.grad.abs().sum())
        for parameter in module.parameters()
    )


def test_scale_and_bev_trainable_paths_are_gradient_isolated() -> None:
    torch.manual_seed(12)
    scale_model = _model().train()
    scale_prediction = scale_model.forward_head(
        _extraction(3),
        include_merged=False,
        include_scale=True,
        assemble_runtime_outputs=False,
    )
    scale_prediction["scale"]["log_lambda_m_per_vggt"].sum().backward()
    assert _module_gradient_sum(scale_model.head.scale_frame_reliability) > 0.0
    assert _module_gradient_sum(scale_model.head.scale_token_projector) > 0.0
    assert _module_gradient_sum(scale_model.head.scale_decoder) > 0.0
    assert _module_gradient_sum(scale_model.head.frame_reliability) == 0.0
    assert _module_gradient_sum(scale_model.head.spatial_token_projector) == 0.0

    bev_model = _model().train()
    bev_prediction = bev_model.forward_head(
        _extraction(3),
        include_merged=True,
        include_scale=False,
        assemble_runtime_outputs=False,
    )
    complete = torch.full((2, 8, 8), 255, dtype=torch.uint8)
    complete[:, 2:4, 3:6] = 0
    visible = torch.full_like(complete, 112)
    visible[:, 4:7, 1:7] = complete[:, 4:7, 1:7]
    support = torch.ones_like(complete, dtype=torch.bool)
    m04_bev_loss(
        bev_prediction["merged_bev"],
        complete,
        visible,
        support,
        gt_valid_mask=support,
    )["loss"].backward()
    assert _module_gradient_sum(bev_model.head.frame_reliability) > 0.0
    assert _module_gradient_sum(bev_model.head.spatial_token_projector) > 0.0
    assert _module_gradient_sum(bev_model.head.scale_frame_reliability) == 0.0
    assert _module_gradient_sum(bev_model.head.scale_token_projector) == 0.0
    assert _module_gradient_sum(bev_model.head.scale_decoder) == 0.0


def test_runtime_releases_teacher_only_aggregation_before_head() -> None:
    class RuntimeAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.aggregate_calls = 0

        def aggregate(self, images: torch.Tensor) -> dict:
            self.aggregate_calls += 1
            batch, frames = images.shape[:2]
            return {
                "tokens": {1: torch.randn(batch, frames, 4, 16)},
                "camera_register_tokens": torch.randn(batch, frames, 3, 16),
                "patch_grid": (2, 2),
                "_aggregated": object(),
                "_patch_start": 3,
                "_images": images,
            }

    adapter = RuntimeAdapter()
    model = _model(adapter).eval()
    head_keys: list[set[str]] = []
    handle = model.head.register_forward_pre_hook(
        lambda _module, arguments: head_keys.append(set(arguments[0]))
    )
    output = model(torch.randn(1, 3, 3, 32, 32))
    handle.remove()
    assert adapter.aggregate_calls == 1
    assert len(head_keys) == 1
    assert not {"_aggregated", "_patch_start", "_images"} & head_keys[0]
    assert output["runtime_inputs"] == ("rgb_window",)


def test_m04_hierarchical_loss_backpropagates_all_map_outputs() -> None:
    complete = torch.full((2, 8, 8), 112, dtype=torch.uint8)
    complete[:, 1:7, 1:7] = 255
    complete[:, 2:4, 4:6] = 0
    visible = torch.full_like(complete, 112)
    visible[:, 4:7, 2:6] = complete[:, 4:7, 2:6]
    support = complete != 112
    raw = torch.randn(2, 2, 8, 8, requires_grad=True)
    alpha = torch.nn.functional.softplus(raw[:, 0]) + 1.0
    beta = torch.nn.functional.softplus(raw[:, 1]) + 1.0
    gate = torch.randn(2, 8, 8, requires_grad=True)
    fov = torch.randn(2, 8, 8, requires_grad=True)
    result = m04_bev_loss(
        {
            "guessed": {
                "alpha_occupied": alpha,
                "beta_free": beta,
            },
            "observed_gate_logit": gate,
            "fov_support_logit": fov,
        },
        complete,
        visible,
        support,
        gt_valid_mask=torch.ones_like(support),
        evidence_kl_weight=1e-3,
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert gate.grad is not None and torch.isfinite(gate.grad).all()
    assert fov.grad is not None and torch.isfinite(fov.grad).all()


def test_m04_student_t_scale_loss_has_zero_scale_gradient_at_gt() -> None:
    vggt = torch.rand(1, 2, 8, 8) + 0.5
    gt = 2.5 * vggt
    predicted_log = torch.tensor([2.5]).log().requires_grad_()
    log_variance = torch.zeros(1, requires_grad=True)
    target = {
        "log_lambda_gt": torch.tensor([2.5]).log(),
        "quality_weight": torch.ones(1),
        "target_valid": torch.ones(1, dtype=torch.bool),
        "dense_inlier_mask": torch.ones_like(vggt, dtype=torch.bool),
        "dense_weight": torch.ones_like(vggt),
        "vggt_depth": vggt,
        "gt_depth_m": gt,
    }
    result = m04_scale_loss(
        {
            "log_lambda_m_per_vggt": predicted_log,
            "lambda_m_per_vggt": predicted_log.exp(),
            "log_variance": log_variance,
        },
        target,
    )
    result["loss"].backward()
    torch.testing.assert_close(predicted_log.grad, torch.zeros_like(predicted_log))
    assert log_variance.grad is not None


def test_existing_10m_source_is_hard_ignored_outside_coverage() -> None:
    complete = torch.full((1, 512, 512), 255, dtype=torch.uint8)
    visible = complete.clone()
    support = torch.ones_like(complete, dtype=torch.bool)
    target = regrid_merged_metric_targets_to_vggt_units(
        complete,
        visible,
        support,
        torch.ones_like(support),
        torch.tensor([2.5]),
        torch.ones(1, dtype=torch.bool),
        source_extent_m=10.0,
        target_extent_vggt=6.5,
        target_size=512,
    )
    assert target["complete_target"].shape == (1, 512, 512)
    assert float(target["source_coverage_fraction"][0]) < 0.5
    assert torch.equal(
        target["gt_valid_mask"], target["source_coverage_mask"]
    )


def test_10m_source_is_pixel_identical_when_vggt_grid_spans_10m() -> None:
    row = torch.arange(512, dtype=torch.int64)[:, None]
    column = torch.arange(512, dtype=torch.int64)[None, :]
    complete = torch.where(
        (row * 17 + column * 29) % 7 < 3,
        torch.tensor(0, dtype=torch.uint8),
        torch.tensor(255, dtype=torch.uint8),
    )[None]
    support = torch.ones_like(complete, dtype=torch.bool)
    target = regrid_merged_metric_targets_to_vggt_units(
        complete,
        complete,
        support,
        support,
        torch.tensor([10.0 / 6.5]),
        torch.ones(1, dtype=torch.bool),
        source_extent_m=10.0,
        target_extent_vggt=6.5,
        target_size=512,
    )
    assert torch.equal(target["complete_target"], complete)
    assert torch.equal(target["visible_target"], complete)
    assert torch.equal(target["support_target"], support)
    assert torch.equal(target["gt_valid_mask"], support)


def test_effective_supervision_is_distinct_from_geometric_coverage() -> None:
    complete = torch.full((1, 512, 512), 255, dtype=torch.uint8)
    support = torch.ones_like(complete, dtype=torch.bool)
    target = regrid_merged_metric_targets_to_vggt_units(
        complete,
        complete,
        support,
        support,
        torch.ones(1),
        torch.zeros(1, dtype=torch.bool),
        source_extent_m=10.0,
        target_extent_vggt=6.5,
        target_size=512,
    )
    assert float(target["source_coverage_fraction"][0]) == 1.0
    assert float(target["gt_valid_mask"].float().mean()) == 0.0
