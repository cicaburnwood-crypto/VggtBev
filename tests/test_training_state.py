from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from vggt_bev_method1.cli_train_metric import (
    _configure_head_compilation,
    _direct_priority_gradient_correction,
    _fov_complete_loss_weights,
    _normalized_resume_contract,
    _resume_contract,
    _set_training_stage,
    _step_losses,
    _surface_tolerance_pixels,
    _validation_batches_per_rank,
    guessed_supervision_scale,
    learning_rate_factor,
    validation_checkpoint_scores,
)
from vggt_bev_method1.models import Method1Head
from vggt_bev_method1.nccl import (
    parse_gpu_inventory,
    parse_topology_numa,
    resolve_visible_devices,
)
from vggt_bev_method1.training_state import (
    DistributedEpochShuffleSampler,
    EpochOffsetSampler,
    EpochShuffleSampler,
    SessionPrefixSampler,
    StratifiedValidationSampler,
    capture_rng_state,
    restore_rng_state,
    stratified_validation_indices,
)


def test_epoch_offset_sampler_resumes_without_reloading_completed_indices() -> None:
    base = EpochShuffleSampler(list(range(12)), seed=31)
    resumed = EpochOffsetSampler(base)
    resumed.set_epoch(4)
    full_order = list(base)
    resumed.set_start_index(7)
    assert list(resumed) == full_order[7:]
    assert len(resumed) == 5

    resumed.set_epoch(5)
    resumed.set_start_index(0)
    assert list(resumed) == list(base)


def test_validation_budget_is_split_across_all_ranks() -> None:
    assert _validation_batches_per_rank(200, 4) == 50
    assert _validation_batches_per_rank(3, 4) == 1
    assert _validation_batches_per_rank(0, 4) == 0


def test_learning_rate_keeps_a_nonzero_floor_across_sampler_epochs() -> None:
    training = {
        "learning_rate": 1.0e-4,
        "minimum_learning_rate": 1.0e-5,
        "warmup_fraction": 0.02,
    }
    assert abs(learning_rate_factor(0, 1000, training) - 0.145) < 1.0e-12
    assert learning_rate_factor(19, 1000, training) == 1.0
    assert learning_rate_factor(20, 1000, training) == 1.0
    assert learning_rate_factor(1000, 1000, training) == 0.1


def test_best_checkpoint_scores_require_support_and_surface_quality() -> None:
    metrics = {
        "single_bev_support_iou": 0.96,
        "single_bev_observed_surface_recall": 0.94,
        "single_bev_observed_free_false_occupied_rate": 0.14,
        "single_bev_guessed_occupied_iou_gain_over_all_occupied": 0.03,
        "single_bev_observed_confidence_ece": 0.08,
        "single_bev_guessed_confidence_ece": 0.12,
    }
    assert validation_checkpoint_scores(metrics) == {
        "best_observed": 0.14,
        "best_completion": -0.03,
        "best_calibrated": 0.12,
    }
    metrics["single_bev_support_iou"] = 0.949
    assert validation_checkpoint_scores(metrics) == {}


def test_nccl_preflight_resolves_uuid_and_parses_numa_topology() -> None:
    inventory = parse_gpu_inventory(
        "\n".join(
            (
                "0, GPU-aaaa",
                "1, GPU-bbbb",
                "2, GPU-cccc",
                "3, GPU-dddd",
            )
        )
    )
    assert resolve_visible_devices("GPU-bbbb,2", inventory) == [1, 2]
    topology = """\
        GPU0 GPU1 GPU2 GPU3 CPU Affinity NUMA Affinity GPU NUMA ID
GPU0    X    NODE SYS  SYS  0-31         0             N/A
GPU1    NODE X    SYS  SYS  0-31         0             N/A
GPU2    SYS  SYS  X    NODE 32-63        1             N/A
GPU3    SYS  SYS  NODE X    32-63        1             N/A
"""
    assert parse_topology_numa(topology, gpu_count=4) == {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
    }


def test_epoch_sampler_reconstructs_order_without_storing_permutation() -> None:
    source = list(range(17))
    first = EpochShuffleSampler(source, seed=23)
    first.set_epoch(4)
    order = list(first)

    resumed = EpochShuffleSampler(source, seed=23)
    resumed.set_epoch(4)
    assert list(resumed) == order

    resumed.set_epoch(5)
    assert list(resumed) != order
    assert sorted(order) == list(range(len(source)))


def test_rng_state_restores_python_numpy_and_torch() -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))

    random.random()
    np.random.rand()
    torch.rand(3)
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_resume_contract_rejects_order_or_model_changes() -> None:
    config = {
        "data": {
            "supervision": "metric_fov_complete_evidential",
            "coordinate_mode": "p1b_fixed_metric",
            "single_bev_extent_m": 6.5,
            "merged_bev_extent_m": 10.0,
            "image_height": 384,
            "image_width": 512,
            "sample_stride": 1,
            "minimum_history": 1,
            "maximum_history": 34,
        },
        "model": {
            "hidden_dim": 64,
            "single_bev_output_size": 512,
            "merged_bev_output_size": 800,
            "single_latent_bev_size": 64,
            "merged_latent_bev_size": 80,
        },
        "scale_fit": {"minimum_valid_pixels": 100},
        "training": {
            "stage": "joint",
            "guessed_completion_class_weights": [1.0, 1.0],
            "bev_loss_weight": 1.0,
            "single_task_weight": 0.5,
            "merged_task_weight": 0.5,
            "observed_free_nll_weight": 1.0,
            "observed_surface_nll_weight": 0.35,
            "guessed_completion_nll_weight": 1.0,
            "guessed_completion_dice_weight": 0.2,
            "observed_region_weight": 1.0,
            "guessed_region_weight": 0.25,
            "surface_tolerance_latent_cell_fraction": 0.5,
            "guessed_supervision_warmup_fraction": 0.05,
            "guessed_supervision_ramp_fraction": 0.10,
            "direct_priority_pcgrad": True,
            "train_guessed_completion": True,
            "maximum_guessed_to_direct_gradient_ratio": 0.25,
            "fov_support_bce_weight": 0.5,
            "fov_support_dice_weight": 0.5,
            "incorrect_evidence_weight": 0.05,
            "confidence_calibration_weight": 0.25,
            "guessed_incorrect_evidence_multiplier": 4.0,
            "guessed_confidence_calibration_multiplier": 4.0,
            "observation_relation_weight": 0.10,
            "evidence_relation_margin": 0.10,
            "confidence_regularizer_warmup_epochs": 0,
            "confidence_regularizer_ramp_epochs": 2,
            "scale_loss_weight": 1.0,
            "depth_scale_loss_weight": 0.5,
            "uncertainty_loss_weight": 0.05,
        },
    }
    original = _resume_contract(config, "manifest-a")
    assert original == _resume_contract(config, "manifest-a")
    assert (
        original["loss_schema"]
        == "observed-pointwise-confidence-pcgrad-v4"
    )
    assert "single_observed_surface_nll_weight" not in original["loss"]
    assert "merged_observed_surface_nll_weight" not in original["loss"]
    single = _fov_complete_loss_weights(config["training"], "single")
    merged = _fov_complete_loss_weights(config["training"], "merged")
    assert single.observed_surface == 0.35
    assert merged.observed_surface == 0.35

    config["training"]["single_observed_surface_nll_weight"] = 0.40
    config["training"]["merged_observed_surface_nll_weight"] = 0.30
    split_contract = _resume_contract(config, "manifest-a")
    assert (
        split_contract["loss_schema"]
        == "observed-pointwise-confidence-pcgrad-v5"
    )
    assert split_contract["loss"]["single_observed_surface_nll_weight"] == 0.40
    assert split_contract["loss"]["merged_observed_surface_nll_weight"] == 0.30
    single = _fov_complete_loss_weights(config["training"], "single")
    merged = _fov_complete_loss_weights(config["training"], "merged")
    assert single.observed_surface == 0.40
    assert merged.observed_surface == 0.30
    assert original != split_contract

    config["training"]["observed_surface_continuity_weight"] = 0.25
    continuity_contract = _resume_contract(config, "manifest-a")
    assert (
        continuity_contract["loss_schema"]
        == "observed-surface-continuity-confidence-pcgrad-v6"
    )
    assert (
        continuity_contract["loss"]["observed_surface_continuity_weight"]
        == 0.25
    )
    single = _fov_complete_loss_weights(config["training"], "single")
    merged = _fov_complete_loss_weights(config["training"], "merged")
    assert single.observed_surface_continuity == 0.25
    assert merged.observed_surface_continuity == 0.25
    assert split_contract != continuity_contract

    config["training"]["stage"] = "scale_only"
    assert original != _resume_contract(config, "manifest-a")


def test_query_chunk_is_execution_only_for_checkpoint_resume() -> None:
    config = {
        "data": {"maximum_history": 10},
        "model": {
            "hidden_dim": 64,
            "cross_query_chunk_size": 4096,
        },
        "scale_fit": {"minimum_valid_pixels": 100},
        "training": {
            "stage": "joint",
            "guessed_completion_class_weights": [1.0, 1.0],
            "bev_loss_weight": 1.0,
            "single_task_weight": 0.5,
            "merged_task_weight": 0.5,
            "observed_free_nll_weight": 1.0,
            "observed_surface_nll_weight": 1.5,
            "observed_surface_continuity_weight": 0.25,
            "guessed_completion_nll_weight": 1.0,
            "guessed_completion_dice_weight": 0.2,
            "observed_region_weight": 1.0,
            "guessed_region_weight": 0.25,
            "surface_tolerance_latent_cell_fraction": 0.5,
            "guessed_supervision_warmup_fraction": 0.0,
            "guessed_supervision_ramp_fraction": 0.0,
            "direct_priority_pcgrad": True,
            "train_guessed_completion": True,
            "maximum_guessed_to_direct_gradient_ratio": 0.25,
            "fov_support_bce_weight": 0.5,
            "fov_support_dice_weight": 0.5,
            "incorrect_evidence_weight": 0.05,
            "confidence_calibration_weight": 0.25,
            "guessed_incorrect_evidence_multiplier": 4.0,
            "guessed_confidence_calibration_multiplier": 8.0,
            "observation_relation_weight": 0.0,
            "evidence_relation_margin": 0.0,
            "confidence_regularizer_warmup_epochs": 0,
            "confidence_regularizer_ramp_epochs": 2,
            "scale_loss_weight": 1.0,
            "depth_scale_loss_weight": 0.5,
            "uncertainty_loss_weight": 0.05,
        },
    }
    contract_4096 = _resume_contract(config, "manifest-a")
    assert "cross_query_chunk_size" not in contract_4096["model"]
    config["model"]["cross_query_chunk_size"] = 8192
    assert _resume_contract(config, "manifest-a") == contract_4096

    legacy = dict(contract_4096)
    legacy["model"] = {
        **contract_4096["model"],
        "cross_query_chunk_size": 4096,
    }
    assert _normalized_resume_contract(legacy) == contract_4096


def test_head_compile_is_applied_in_place_with_explicit_settings() -> None:
    class StubAttention:
        def __init__(self) -> None:
            self.arguments = None

        def configure_query_chunk_compilation(self, **kwargs) -> None:
            self.arguments = kwargs

    class StubHead:
        def __init__(self) -> None:
            self.attention = StubAttention()

        def modules(self):
            return (self, self.attention)

    class StubSystem:
        def __init__(self) -> None:
            self.head = StubHead()

    model = StubSystem()
    settings = _configure_head_compilation(
        model,
        {
            "compile_head": True,
            "compile_head_backend": "inductor",
            "compile_head_mode": "default",
            "compile_head_dynamic": True,
        },
    )
    assert settings == {
        "enabled": True,
        "scope": "deformable_query_chunks",
        "backend": "inductor",
        "mode": "default",
        "dynamic": True,
        "compiled_modules": 1,
        "donated_buffer": False,
    }
    assert model.head.attention.arguments == {
        "backend": "inductor",
        "mode": "default",
        "dynamic": True,
    }


def test_surface_tolerance_tracks_latent_resolution() -> None:
    config = {
        "model": {
            "single_bev_output_size": 512,
            "merged_bev_output_size": 800,
            "single_latent_bev_size": 64,
            "merged_latent_bev_size": 80,
        },
        "training": {
            "surface_tolerance_latent_cell_fraction": 0.5,
        },
    }
    assert _surface_tolerance_pixels(config, "single") == 4
    assert _surface_tolerance_pixels(config, "merged") == 5
    config["model"]["single_latent_bev_size"] = 128
    config["model"]["merged_latent_bev_size"] = 160
    assert _surface_tolerance_pixels(config, "single") == 2
    assert _surface_tolerance_pixels(config, "merged") == 2


def test_step_losses_uses_branch_specific_surface_weights(
    monkeypatch,
) -> None:
    captured = []

    def fake_bev_loss(prediction, *args, weights, **kwargs):
        captured.append(weights.observed_surface)
        value = prediction.sum() * 0.0 + 1.0
        return {
            "loss": value,
            "observed_direct_loss": value,
            "direct_confidence_loss": value * 0.0,
            "guessed_completion_loss": value,
            "guessed_confidence_loss": value * 0.0,
        }

    monkeypatch.setattr(
        "vggt_bev_method1.cli_train_metric."
        "fov_complete_evidential_bev_loss",
        fake_bev_loss,
    )
    training = {
        "stage": "bev_only",
        "guessed_completion_class_weights": [1.0, 1.0],
        "bev_loss_weight": 1.0,
        "single_task_weight": 0.5,
        "merged_task_weight": 0.5,
        "observed_free_nll_weight": 1.0,
        "observed_surface_nll_weight": 0.35,
        "single_observed_surface_nll_weight": 0.40,
        "merged_observed_surface_nll_weight": 0.35,
        "guessed_completion_nll_weight": 1.0,
        "guessed_completion_dice_weight": 0.2,
        "observed_region_weight": 1.0,
        "guessed_region_weight": 0.25,
        "fov_support_bce_weight": 0.5,
        "fov_support_dice_weight": 0.5,
        "incorrect_evidence_weight": 0.05,
        "confidence_calibration_weight": 0.25,
        "guessed_incorrect_evidence_multiplier": 4.0,
        "guessed_confidence_calibration_multiplier": 8.0,
        "observation_relation_weight": 0.0,
        "evidence_relation_margin": 0.0,
        "scale_loss_weight": 1.0,
        "depth_scale_loss_weight": 0.5,
        "uncertainty_loss_weight": 0.05,
        "surface_tolerance_latent_cell_fraction": 0.5,
    }
    config = {
        "model": {
            "single_bev_output_size": 4,
            "merged_bev_output_size": 4,
            "single_latent_bev_size": 2,
            "merged_latent_bev_size": 2,
        },
        "training": training,
    }
    prediction = {
        "single_bev": torch.zeros(1),
        "merged_bev": torch.zeros(1),
        "scale": {"log_lambda_m_per_vggt": torch.zeros(1)},
    }
    target = torch.zeros(1)
    batch = {
        "images": torch.zeros(1),
        "single_fov_complete_target": target,
        "single_visible_target": target,
        "single_fov_support_target": target,
        "merged_fov_complete_target": target,
        "merged_visible_target": target,
        "merged_fov_support_target": target,
    }

    total, _, scale_target = _step_losses(
        prediction,
        batch,
        None,
        config,
        None,
        evidence_regularizer_scale=1.0,
        guessed_completion_scale=1.0,
    )

    assert captured == [0.40, 0.35]
    assert total.item() == 1.0
    assert scale_target is None


def test_single_plus_scale_stage_freezes_merged_and_skips_merged_loss(
    monkeypatch,
) -> None:
    head = Method1Head(
        cached_layers=(1,),
        spatial_scales=(1.0,),
        vggt_token_dim=8,
        hidden_dim=16,
        heads=4,
        decoder_layers=1,
        scale_decoder_layers=1,
        single_latent_bev_size=2,
        merged_latent_bev_size=2,
        single_output_size=4,
        merged_output_size=4,
        cross_query_chunk_size=4,
    )

    class StubSystem:
        def unwrapped_head(self):
            return head

    _set_training_stage(StubSystem(), "joint", ("single",))
    assert all(
        not parameter.requires_grad
        for parameter in head.merged_bev_decoder.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in head.single_bev_decoder.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in head.scale_decoder.parameters()
    )

    captured = []

    def fake_bev_loss(prediction, *args, **kwargs):
        captured.append(prediction)
        value = prediction["raw_output"].sum() * 0.0 + 1.0
        return {
            "loss": value,
            "observed_direct_loss": value,
            "direct_confidence_loss": value * 0.0,
            "guessed_completion_loss": value,
            "guessed_confidence_loss": value * 0.0,
        }

    monkeypatch.setattr(
        "vggt_bev_method1.cli_train_metric."
        "fov_complete_evidential_bev_loss",
        fake_bev_loss,
    )
    training = {
        "stage": "bev_only",
        "enabled_bev_branches": ["single"],
        "guessed_completion_class_weights": [1.0, 1.0],
        "bev_loss_weight": 1.0,
        "single_task_weight": 1.0,
        "merged_task_weight": 0.0,
        "observed_free_nll_weight": 1.0,
        "observed_surface_nll_weight": 1.5,
        "guessed_completion_nll_weight": 1.0,
        "guessed_completion_dice_weight": 0.2,
        "observed_region_weight": 1.0,
        "guessed_region_weight": 0.25,
        "fov_support_bce_weight": 0.5,
        "fov_support_dice_weight": 0.5,
        "incorrect_evidence_weight": 0.05,
        "confidence_calibration_weight": 0.25,
        "guessed_incorrect_evidence_multiplier": 4.0,
        "guessed_confidence_calibration_multiplier": 8.0,
        "observation_relation_weight": 0.0,
        "evidence_relation_margin": 0.0,
        "scale_loss_weight": 1.0,
        "depth_scale_loss_weight": 0.5,
        "uncertainty_loss_weight": 0.05,
        "surface_tolerance_latent_cell_fraction": 0.5,
    }
    config = {
        "model": {
            "single_bev_output_size": 4,
            "merged_bev_output_size": 4,
            "single_latent_bev_size": 2,
            "merged_latent_bev_size": 2,
        },
        "training": training,
    }
    single_prediction = torch.zeros(1, 3, 4, 4, requires_grad=True)
    prediction = {
        "single_bev": {
            "raw_output": single_prediction,
        },
    }
    target = torch.zeros(1)
    batch = {
        "images": torch.zeros(1),
        "single_fov_complete_target": target,
        "single_visible_target": target,
        "single_fov_support_target": target,
    }
    total, values, scale_target = _step_losses(
        prediction,
        batch,
        None,
        config,
        None,
        evidence_regularizer_scale=1.0,
        guessed_completion_scale=1.0,
    )
    assert captured == [prediction["single_bev"]]
    assert "single_bev_loss" in values
    assert all(not key.startswith("merged_bev_") for key in values)
    assert total.item() == 1.0
    assert scale_target is None


def test_guessed_curriculum_uses_optimizer_step_fraction() -> None:
    training = {
        "guessed_supervision_warmup_fraction": 0.05,
        "guessed_supervision_ramp_fraction": 0.10,
    }
    assert guessed_supervision_scale(0, 100, training) == 0.0
    assert guessed_supervision_scale(4, 100, training) == 0.0
    assert guessed_supervision_scale(5, 100, training) == 0.1
    assert guessed_supervision_scale(14, 100, training) == 1.0
    assert guessed_supervision_scale(99, 100, training) == 1.0


def test_direct_priority_pcgrad_projects_only_conflicting_guess() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    direct_loss = parameter[0]
    guessed_loss = -parameter[0] + parameter[1]
    correction, metrics = _direct_priority_gradient_correction(
        direct_loss,
        guessed_loss,
        [parameter],
    )
    direct_gradient = torch.tensor([1.0, 0.0])
    guessed_gradient = torch.tensor([-1.0, 1.0])
    projected = guessed_gradient + correction[0]
    assert torch.equal(correction[0], torch.tensor([1.0, 0.0]))
    assert torch.dot(direct_gradient, projected) >= 0
    assert float(metrics["pcgrad_conflict"]) == 1.0
    assert abs(float(metrics["pcgrad_projected_cosine"])) < 1e-6


def test_direct_priority_pcgrad_leaves_aligned_guess_unchanged() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    correction, metrics = _direct_priority_gradient_correction(
        parameter[0],
        parameter[0] + parameter[1],
        [parameter],
    )
    assert torch.equal(correction[0], torch.zeros_like(parameter))
    assert float(metrics["pcgrad_conflict"]) == 0.0


def test_direct_priority_pcgrad_caps_large_aligned_guess() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    correction, metrics = _direct_priority_gradient_correction(
        parameter[0],
        8.0 * parameter[0] + 6.0 * parameter[1],
        [parameter],
        maximum_guessed_to_direct_gradient_ratio=0.25,
    )
    guessed_gradient = torch.tensor([8.0, 6.0])
    projected = guessed_gradient + correction[0]
    assert torch.allclose(projected.norm(), torch.tensor(0.25))
    assert float(metrics["pcgrad_norm_cap_applied"]) == 1.0
    assert float(metrics["pcgrad_guessed_gradient_scale"]) < 1.0


def test_guessed_curriculum_can_be_disabled_for_direct_only_ablation() -> None:
    training = {
        "train_guessed_completion": False,
        "guessed_supervision_warmup_fraction": 0.05,
        "guessed_supervision_ramp_fraction": 0.10,
    }
    assert guessed_supervision_scale(99, 100, training) == 0.0


def test_distributed_sampler_is_disjoint_and_reconstructable() -> None:
    source = list(range(17))
    samplers = [
        DistributedEpochShuffleSampler(
            source,
            num_replicas=3,
            rank=rank,
            seed=11,
        )
        for rank in range(3)
    ]
    for sampler in samplers:
        sampler.set_epoch(2)
    partitions = [list(sampler) for sampler in samplers]
    assert all(len(partition) == 5 for partition in partitions)
    assert len(set().union(*map(set, partitions))) == 15
    repeated = DistributedEpochShuffleSampler(
        source,
        num_replicas=3,
        rank=1,
        seed=11,
    )
    repeated.set_epoch(2)
    assert list(repeated) == partitions[1]


@dataclass(frozen=True)
class _PrefixSample:
    session_index: int
    target_frame: int


class _PrefixDataset:
    def __init__(self) -> None:
        self.samples = [
            _PrefixSample(session_index, target_frame)
            for session_index, count in enumerate((3, 4, 2, 5, 3))
            for target_frame in range(count)
        ]

    def __len__(self) -> int:
        return len(self.samples)


def test_session_prefix_sampler_uses_every_session_once_per_epoch() -> None:
    source = _PrefixDataset()
    samplers = [
        SessionPrefixSampler(
            source,
            seed=17,
            num_replicas=2,
            rank=rank,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.set_epoch(0)
    selected = [index for sampler in samplers for index in sampler]
    selected_sessions = {
        source.samples[index].session_index for index in selected
    }
    assert selected_sessions == set(range(5))
    assert all(len(sampler) == 3 for sampler in samplers)


def test_session_prefix_sampler_rotates_prefixes_deterministically() -> None:
    source = _PrefixDataset()
    sampler = SessionPrefixSampler(source, seed=23, shuffle=False)
    choices = []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        choices.append(
            {
                source.samples[index].session_index: (
                    source.samples[index].target_frame
                )
                for index in sampler
            }
        )
    assert choices[0] != choices[1] != choices[2]
    assert len({choice[0] for choice in choices}) == 3

    repeated = SessionPrefixSampler(source, seed=23, shuffle=False)
    repeated.set_epoch(2)
    assert list(repeated) == list(sampler)


def test_session_prefix_sampler_batches_equal_history_across_ddp_ranks() -> None:
    source = _PrefixDataset()
    samplers = [
        SessionPrefixSampler(
            source,
            seed=29,
            num_replicas=2,
            rank=rank,
            batch_size=2,
        )
        for rank in range(2)
    ]
    expected_length = len(samplers[0])
    for epoch in range(4):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        partitions = [list(sampler) for sampler in samplers]
        assert all(len(partition) == expected_length for partition in partitions)
        assert expected_length % 2 == 0
        selected_sessions = {
            source.samples[index].session_index
            for partition in partitions
            for index in partition
        }
        assert selected_sessions == set(range(5))
        for start in range(0, expected_length, 2):
            synchronized_batch = [
                index
                for partition in partitions
                for index in partition[start : start + 2]
            ]
            assert len(
                {
                    source.samples[index].target_frame
                    for index in synchronized_batch
                }
            ) == 1

    repeated = SessionPrefixSampler(
        source,
        seed=29,
        num_replicas=2,
        rank=1,
        batch_size=2,
    )
    repeated.set_epoch(3)
    assert list(repeated) == list(samplers[1])


def test_session_prefix_batch_resume_offset_preserves_history_groups() -> None:
    source = _PrefixDataset()
    base = SessionPrefixSampler(
        source,
        seed=31,
        num_replicas=2,
        rank=0,
        batch_size=2,
    )
    base.set_epoch(2)
    full = list(base)
    resumed = EpochOffsetSampler(base)
    resumed.set_epoch(2)
    resumed.set_start_index(4)
    remaining = list(resumed)
    assert remaining == full[4:]
    for start in range(0, len(remaining), 2):
        assert len(
            {
                source.samples[index].target_frame
                for index in remaining[start : start + 2]
            }
        ) == 1


@dataclass(frozen=True)
class _ValidationSession:
    scene_key: str


class _ValidationDataset:
    def __init__(self) -> None:
        self.sessions = tuple(
            _ValidationSession(f"scene-{scene_index}")
            for scene_index in range(6)
            for _ in range(2)
        )
        self.samples = [
            _PrefixSample(session_index, target_frame)
            for session_index in range(len(self.sessions))
            for target_frame in range(8)
        ]

    def __len__(self) -> int:
        return len(self.samples)


def test_validation_subset_covers_scenes_and_history_not_first_records() -> None:
    source = _ValidationDataset()
    selected = stratified_validation_indices(
        source,
        maximum_samples=24,
        seed=17,
    )
    scenes = {
        source.sessions[source.samples[index].session_index].scene_key
        for index in selected
    }
    histories = {source.samples[index].target_frame for index in selected}
    assert scenes == {f"scene-{index}" for index in range(6)}
    assert min(histories) <= 1
    assert max(histories) >= 6
    assert selected != list(range(24))
    assert selected == stratified_validation_indices(
        source,
        maximum_samples=24,
        seed=17,
    )


def test_validation_subset_is_disjoint_across_ddp_ranks() -> None:
    source = _ValidationDataset()
    samplers = [
        StratifiedValidationSampler(
            source,
            maximum_samples=24,
            seed=17,
            num_replicas=4,
            rank=rank,
        )
        for rank in range(4)
    ]
    partitions = [list(sampler) for sampler in samplers]
    assert all(len(partition) == 6 for partition in partitions)
    assert len(set().union(*map(set, partitions))) == 24


def test_validation_subset_pads_tiny_smoke_data_for_every_rank() -> None:
    source = [object()]
    samplers = [
        StratifiedValidationSampler(
            source,
            maximum_samples=1,
            seed=17,
            num_replicas=4,
            rank=rank,
        )
        for rank in range(4)
    ]
    assert [list(sampler) for sampler in samplers] == [[0], [0], [0], [0]]
