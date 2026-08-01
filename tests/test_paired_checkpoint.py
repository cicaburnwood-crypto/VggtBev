from __future__ import annotations

import torch
from torch import nn

from vggt_bev_method1.cli_train_paired import save_path_checkpoint
from vggt_bev_method1.models import PairedMethod1System


class FakeAdapter(nn.Module):
    def forward(
        self,
        images: torch.Tensor,
        camera_height_m: torch.Tensor,
    ) -> dict:
        batch, frames = images.shape[:2]
        return {
            "tokens": {
                layer: torch.randn(batch, frames, 6, 8)
                for layer in (4, 11, 17, 23)
            },
            "patch_grid": (2, 3),
            "geometry_cue": torch.randn(batch, 24),
            "scene_radius_vggt": torch.ones(batch),
            "p1a_geometry": {
                "single_extent_normalized_scale": torch.full((batch,), 6.5),
                "merged_extent_normalized_scale": torch.full((batch,), 10.0),
                "camera_height_m": camera_height_m,
                "geometry_quality": torch.ones(batch),
                "geometry_valid": torch.ones(batch, dtype=torch.bool),
                "single_fov_support": torch.ones(
                    batch, 2, 2, dtype=torch.bool
                ),
                "merged_fov_support": torch.ones(
                    batch, 3, 3, dtype=torch.bool
                ),
            },
            "geometry_source": "fake live VGGT",
        }


def paired_config() -> dict:
    return {
        "data": {
            "supervision": "joint",
            "coordinate_mode": (
                "camera_height_anchored_fixed_normalized_scale"
            ),
            "single_target_extent_m": 6.5,
            "merged_target_extent_m": 10.0,
            "image_height": 32,
            "image_width": 48,
            "sample_stride": 1,
            "minimum_history": 1,
            "maximum_history": 2,
        },
        "model": {
            "hidden_dim": 8,
            "single_output_size": 2,
            "merged_output_size": 3,
            "single_output_extent_normalized_scale": 6.5,
            "merged_output_extent_normalized_scale": 10.0,
            "runtime_metric_anchor": "camera_height_m",
        },
        "training": {
            "pipeline": "paired_evidential",
            "batch_size": 1,
            "seed": 17,
            "geometry_warmup_epochs": 1,
            "geometry_ramp_epochs": 2,
            "observed_learning_rate": 1e-4,
            "complete_learning_rate": 1e-4,
            "observed_class_weights": [0.25, 1.0, 2.0],
            "observed_categorical_weight": 1.0,
            "observed_dice_weight": 0.25,
            "evidential_occupancy_weight": 1.0,
            "evidential_dice_weight": 0.25,
            "incorrect_evidence_weight": 0.01,
            "observation_relation_weight": 0.01,
            "evidence_calibration_weight": 0.05,
            "evidence_relation_margin": 0.0,
            "confidence_regularizer_warmup_epochs": 1,
            "confidence_regularizer_ramp_epochs": 2,
            "single_task_weight": 0.5,
            "merged_task_weight": 0.5,
        },
    }


def test_paired_checkpoints_are_separate_but_share_pair_identity(tmp_path) -> None:
    model = PairedMethod1System(
        FakeAdapter(),
        spatial_scales=(1.0, 1.0, 1.0, 1.0),
        vggt_token_dim=8,
        hidden_dim=8,
        geometry_cue_dim=24,
        heads=2,
        decoder_layers=1,
        self_attention_mode="exact",
        cross_attention_mode="exact",
        deformable_samples=2,
        cross_query_chunk_size=16,
        single_output_size=2,
        merged_output_size=3,
        gradient_checkpointing=False,
    )
    observed_optimizer = torch.optim.AdamW(
        model.unwrapped_observed_model().parameters()
    )
    complete_optimizer = torch.optim.AdamW(
        model.unwrapped_complete_model().parameters()
    )
    observed_scaler = torch.amp.GradScaler("cuda", enabled=False)
    complete_scaler = torch.amp.GradScaler("cuda", enabled=False)
    pair_id = "manifest-step-00000003"
    paths = {
        "observed": tmp_path / "observed.pt",
        "complete_evidential": tmp_path / "complete.pt",
    }
    for kind, optimizer, scaler in (
        ("observed", observed_optimizer, observed_scaler),
        ("complete_evidential", complete_optimizer, complete_scaler),
    ):
        save_path_checkpoint(
            paths[kind],
            model=model,
            model_kind=kind,
            optimizer=optimizer,
            scaler=scaler,
            config=paired_config(),
            epoch=0,
            global_step=3,
            next_epoch=0,
            next_batch_in_epoch=3,
            split_manifest_sha256="manifest",
            checkpoint_pair_id=pair_id,
        )
    observed = torch.load(paths["observed"], weights_only=False)
    complete = torch.load(paths["complete_evidential"], weights_only=False)
    assert observed["checkpoint_pair_id"] == complete["checkpoint_pair_id"] == pair_id
    assert observed["model_kind"] == "observed"
    assert complete["model_kind"] == "complete_evidential"
    assert observed["output_contract"]["confidence"] is False
    assert complete["output_contract"]["confidence_activation"] == "none"
    assert (
        observed["output_contract"]["coordinate_mode"]
        == "camera_height_anchored_fixed_normalized_scale"
    )
    assert observed["checkpoint_schema"] == "p1a-fov-complete-confidence-v3"
    assert observed["format_version"] == 14
    assert observed["output_contract"]["single_extent_normalized_scale"] == 6.5
    assert observed["output_contract"]["merged_extent_normalized_scale"] == 10.0
    assert observed["output_contract"]["single_output_size"] == 2
    assert observed["output_contract"]["merged_output_size"] == 3
    assert complete["runtime_input_contract"]["camera_height"] is True
    assert complete["runtime_input_contract"]["metric_calibration"] is True
    assert set(observed["model_state_dict"]).isdisjoint(
        set(complete["model_state_dict"])
    ) is False
    assert observed["model_state_dict"] is not complete["model_state_dict"]


def test_complete_only_system_does_not_instantiate_model_a() -> None:
    model = PairedMethod1System(
        FakeAdapter(),
        spatial_scales=(1.0, 1.0, 1.0, 1.0),
        vggt_token_dim=8,
        hidden_dim=8,
        geometry_cue_dim=24,
        heads=2,
        decoder_layers=1,
        self_attention_mode="exact",
        cross_attention_mode="exact",
        deformable_samples=2,
        cross_query_chunk_size=16,
        single_output_size=2,
        merged_output_size=3,
        gradient_checkpointing=False,
        enable_observed=False,
    )
    assert model.observed_model is None
    assert all(
        not name.startswith("observed_model.")
        for name, _ in model.named_parameters()
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == sum(
        parameter.numel()
        for parameter in model.unwrapped_complete_model().parameters()
    )
