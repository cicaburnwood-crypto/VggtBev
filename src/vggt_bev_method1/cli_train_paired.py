from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from vggt_bev_method1.cli_train import (
    build_datasets,
    describe_data,
    distributed_runtime,
    move_batch,
    report_rank_stage,
    seed_everything,
    smoke_subset,
)
from vggt_bev_method1.data import method1_collate
from vggt_bev_method1.losses import (
    EvidentialModelLossWeights,
    ObservedModelLossWeights,
    complete_evidential_model_loss,
    observed_model_loss,
)
from vggt_bev_method1.metrics import (
    evidential_complete_metrics,
    observed_categorical_metrics,
)
from vggt_bev_method1.models import LiveVGGTOmegaAdapter, PairedMethod1System
from vggt_bev_method1.models.geometry_conditioning import (
    GeometryBuilderConfig,
    validate_p1a_training_geometry,
)
from vggt_bev_method1.training_state import (
    DistributedEpochShuffleSampler,
    EpochShuffleSampler,
    SessionPrefixSampler,
    StratifiedValidationSampler,
    capture_rng_state,
    restore_rng_state,
)


def build_paired_model(config: dict, device: torch.device) -> PairedMethod1System:
    model_config = config["model"]
    cached_layers = tuple(int(value) for value in model_config["cached_layers"])
    spatial_scales = tuple(float(value) for value in model_config["spatial_scales"])
    adapter = LiveVGGTOmegaAdapter(
        model_config["vggt_source"],
        model_config["checkpoint"],
        device=device,
        patch_size=int(model_config["patch_size"]),
        cached_layers=cached_layers,
        maximum_history=int(config["data"]["maximum_history"]),
        geometry_builder_config=GeometryBuilderConfig(
            sample_stride=int(model_config.get("geometry_sample_stride", 4)),
            confidence_threshold=float(
                model_config.get("geometry_confidence_threshold", 0.25)
            ),
            minimum_points=int(
                model_config.get("geometry_minimum_points", 64)
            ),
            single_extent_normalized_scale=float(
                model_config["single_output_extent_normalized_scale"]
            ),
            merged_extent_normalized_scale=float(
                model_config["merged_output_extent_normalized_scale"]
            ),
            minimum_ground_quality=float(
                model_config.get("geometry_minimum_ground_quality", 0.05)
            ),
            ransac_hypotheses=int(
                model_config.get("geometry_ransac_hypotheses", 48)
            ),
            huber_iterations=int(
                model_config.get("geometry_huber_iterations", 4)
            ),
            fov_chunk_size=int(
                model_config.get("geometry_fov_chunk_size", 65536)
            ),
        ),
    )
    if int(model_config["geometry_cue_dim"]) != adapter.geometry_cue_dim:
        raise ValueError("configured geometry_cue_dim disagrees with live VGGT adapter")
    return PairedMethod1System(
        adapter,
        cached_layers=cached_layers,
        spatial_scales=spatial_scales,
        vggt_token_dim=int(model_config["vggt_token_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        geometry_cue_dim=int(model_config["geometry_cue_dim"]),
        heads=int(model_config["attention_heads"]),
        decoder_layers=int(model_config["decoder_layers"]),
        self_attention_mode=str(model_config["self_attention_mode"]),
        cross_attention_mode=str(model_config["cross_attention_mode"]),
        deformable_samples=int(model_config["deformable_samples"]),
        cross_query_chunk_size=int(model_config["cross_query_chunk_size"]),
        query_parameter_chunk_size=int(
            model_config.get("query_parameter_chunk_size", 32768)
        ),
        single_output_size=int(model_config["single_output_size"]),
        merged_output_size=int(model_config["merged_output_size"]),
        gradient_checkpointing=bool(model_config["gradient_checkpointing"]),
        enable_observed=(
            "observed"
            in config["training"].get(
                "enabled_models",
                ["observed", "complete_evidential"],
            )
        ),
    ).to(device)


def paired_resume_contract(config: dict, split_manifest_sha256: str) -> dict:
    training = config["training"]
    training_keys = (
        "pipeline",
        "enabled_models",
        "batch_size",
        "seed",
        "geometry_warmup_epochs",
        "geometry_ramp_epochs",
        "observed_learning_rate",
        "complete_learning_rate",
        "observed_known_weight",
        "observed_free_weight",
        "observed_surface_weight",
        "guessed_completion_class_weights",
        "guessed_completion_nll_weight",
        "guessed_completion_dice_weight",
        "observed_region_weight",
        "guessed_region_weight",
        "incorrect_evidence_weight",
        "observation_relation_weight",
        "confidence_calibration_weight",
        "guessed_incorrect_evidence_multiplier",
        "guessed_confidence_calibration_multiplier",
        "evidence_relation_margin",
        "surface_tolerance_m",
        "guessed_supervision_warmup_fraction",
        "guessed_supervision_ramp_fraction",
        "direct_priority_pcgrad",
        "maximum_guessed_to_direct_gradient_ratio",
        "confidence_regularizer_warmup_epochs",
        "confidence_regularizer_ramp_epochs",
        "single_task_weight",
        "merged_task_weight",
    )
    synchronization_keys = (
        "required_cuda_devices",
        "distributed_backend",
        "require_same_numa",
        "nccl_p2p_level",
        "ddp_bucket_cap_mb",
    )
    training_contract = {
        key: training[key]
        for key in (*training_keys, *synchronization_keys)
        if key in training
    }
    data_keys = (
        "supervision",
        "coordinate_mode",
        "single_target_extent_m",
        "merged_target_extent_m",
        "image_height",
        "image_width",
        "sample_stride",
        "minimum_history",
        "maximum_history",
        "sampling_mode",
    )
    return {
        "split_manifest_sha256": split_manifest_sha256,
        "data": {
            key: config["data"].get(
                key,
                "all_prefixes" if key == "sampling_mode" else None,
            )
            for key in data_keys
        },
        "model": config["model"],
        "training": training_contract,
    }


def _unwrapped_path_state(
    model: PairedMethod1System,
    model_kind: str,
) -> dict[str, torch.Tensor]:
    path = (
        model.unwrapped_observed_model()
        if model_kind == "observed"
        else model.unwrapped_complete_model()
    )
    return {
        key: value.detach().cpu()
        for key, value in path.state_dict().items()
    }


def _load_path_state(
    model: PairedMethod1System,
    model_kind: str,
    state: dict[str, torch.Tensor],
) -> None:
    path = (
        model.unwrapped_observed_model()
        if model_kind == "observed"
        else model.unwrapped_complete_model()
    )
    path.load_state_dict(state, strict=True)


def save_path_checkpoint(
    path: Path,
    *,
    model: PairedMethod1System,
    model_kind: str,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict,
    epoch: int,
    global_step: int,
    next_epoch: int,
    next_batch_in_epoch: int,
    split_manifest_sha256: str,
    checkpoint_pair_id: str,
) -> None:
    if model_kind not in ("observed", "complete_evidential"):
        raise ValueError(f"unsupported checkpoint model kind: {model_kind}")
    state_kind = "observed" if model_kind == "observed" else "complete"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 14,
            "pipeline_id": "P1A",
            "checkpoint_schema": "p1a-fov-complete-confidence-v3",
            "method": (
                "P1A cascaded fixed-grid geometry-conditioned paired models"
                if model.observed_model is not None
                else (
                    "P1A cascaded fixed-grid complete evidential Model B with "
                    "one live VGGT pass"
                )
            ),
            "model_kind": model_kind,
            "checkpoint_pair_id": checkpoint_pair_id,
            "pdf_contract": (
                "Method I cascaded decoder with frozen VGGT and explicit "
                "camera-height metric anchor; no Scale Token"
            ),
            "epoch": epoch,
            "global_step": global_step,
            "next_epoch": next_epoch,
            "next_batch_in_epoch": next_batch_in_epoch,
            "split_manifest_sha256": split_manifest_sha256,
            "resume_contract": paired_resume_contract(
                config,
                split_manifest_sha256,
            ),
            "config": config,
            "model_state_dict": _unwrapped_path_state(model, state_kind),
            "optimizer_state_dict": optimizer.state_dict(),
            "grad_scaler_state_dict": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "output_contract": (
                {
                    "single": "FOV-capped masked semantic BEV",
                    "merged": "history-FOV-union-capped masked semantic BEV",
                    "internal_tensor": "full-grid 3-class logits",
                    "final_semantic_key": "masked_observed_semantic",
                    "class_order": ["unknown", "free", "occupied"],
                    "confidence": False,
                    "coordinate_mode": (
                        "camera_height_anchored_fixed_normalized_scale"
                    ),
                    "extent_mode": "fixed_6p5_single_10_merged",
                    "single_extent_normalized_scale": 6.5,
                    "merged_extent_normalized_scale": 10.0,
                    "single_output_size": config["model"]["single_output_size"],
                    "merged_output_size": config["model"]["merged_output_size"],
                }
                if model_kind == "observed"
                else {
                    "single": "last-frame-FOV complete semantic BEV",
                    "merged": "history-FOV-union complete semantic BEV",
                    "internal_tensor": (
                        "full-grid Beta(alpha_occupied, beta_free)"
                    ),
                    "final_semantic_key": "fov_complete_semantic",
                    "outside_fov": "forced unknown",
                    "confidence_activation": "none",
                    "occupancy_mean": "alpha/(alpha+beta)",
                    "epistemic_uncertainty": "2/(alpha+beta)",
                    "runtime_fov_support": (
                        "explicit same-run VGGT K/E plus anchored ground"
                    ),
                    "outside_runtime_fov": "unknown",
                    "coordinate_mode": (
                        "camera_height_anchored_fixed_normalized_scale"
                    ),
                    "extent_mode": "fixed_6p5_single_10_merged",
                    "single_extent_normalized_scale": 6.5,
                    "merged_extent_normalized_scale": 10.0,
                    "single_output_size": config["model"]["single_output_size"],
                    "merged_output_size": config["model"]["merged_output_size"],
                }
            ),
            "runtime_input_contract": {
                "rgb_history": True,
                "live_vggt_tokens": True,
                "live_vggt_depth": True,
                "live_vggt_intrinsics": True,
                "live_vggt_extrinsics": True,
                "camera_height": True,
                "metric_calibration": True,
                "metric_anchor": "calibrated_camera_height_m",
                "coordinate_mode": (
                    "camera_height_anchored_fixed_normalized_scale"
                ),
                "internal_normalization_unit": (
                    "one_normalized_scale_unit_equals_one_metre_after_anchor"
                ),
                "extent_mode": "fixed_6p5_single_10_merged",
                "shared_live_vggt_forward_when_models_run_together": True,
                "ground_truth_bev": False,
                "ground_truth_depth": False,
                "ground_truth_intrinsics": False,
                "ground_truth_extrinsics": False,
                "ground_truth_trajectory": False,
                "training_scale_validation_uses_gt_trajectory": True,
                "training_target_resampling_uses_gt_trajectory": False,
            },
        },
        temporary,
    )
    temporary.replace(path)


def build_loss_configuration(
    config: dict,
    device: torch.device,
) -> tuple[
    ObservedModelLossWeights,
    EvidentialModelLossWeights,
    torch.Tensor,
]:
    training = config["training"]
    observed = ObservedModelLossWeights(
        known=float(training["observed_known_weight"]),
        free=float(training["observed_free_weight"]),
        surface=float(training["observed_surface_weight"]),
        single=float(training["single_task_weight"]),
        merged=float(training["merged_task_weight"]),
    )
    evidential = EvidentialModelLossWeights(
        observed_free=float(training["observed_free_weight"]),
        observed_surface=float(training["observed_surface_weight"]),
        guessed_nll=float(training["guessed_completion_nll_weight"]),
        guessed_overlap=float(training["guessed_completion_dice_weight"]),
        observed_region=float(training["observed_region_weight"]),
        guessed_region=float(training["guessed_region_weight"]),
        incorrect_evidence=float(training["incorrect_evidence_weight"]),
        observation_relation=float(training["observation_relation_weight"]),
        calibration=float(training["confidence_calibration_weight"]),
        guessed_incorrect_evidence_multiplier=float(
            training["guessed_incorrect_evidence_multiplier"]
        ),
        guessed_calibration_multiplier=float(
            training["guessed_confidence_calibration_multiplier"]
        ),
        relation_margin=float(training["evidence_relation_margin"]),
        single=float(training["single_task_weight"]),
        merged=float(training["merged_task_weight"]),
    )
    class_weights = torch.tensor(
        [
            float(value)
            for value in training["guessed_completion_class_weights"]
        ],
        device=device,
        dtype=torch.float32,
    )
    return observed, evidential, class_weights


def confidence_regularizer_scale(epoch: int, training: dict) -> float:
    warmup = int(training["confidence_regularizer_warmup_epochs"])
    ramp = int(training["confidence_regularizer_ramp_epochs"])
    if epoch < warmup:
        return 0.0
    if ramp == 0:
        return 1.0
    return max(0.0, min(1.0, (epoch - warmup + 1) / ramp))


def guessed_supervision_scale(
    step: int,
    total_steps: int,
    training: dict,
) -> float:
    warmup = round(
        total_steps
        * float(training["guessed_supervision_warmup_fraction"])
    )
    ramp = round(
        total_steps
        * float(training["guessed_supervision_ramp_fraction"])
    )
    if step < warmup:
        return 0.0
    if ramp == 0:
        return 1.0
    return min(1.0, (step - warmup + 1) / ramp)


def _surface_tolerance_pixels(config: dict, branch: str) -> int:
    output_size = int(config["model"][f"{branch}_output_size"])
    extent = float(config["data"][f"{branch}_target_extent_m"])
    return max(
        0,
        round(float(config["training"]["surface_tolerance_m"]) * output_size / extent),
    )


def _flatten_task_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    return torch.cat(
        [
            (
                gradient.detach().float().reshape(-1)
                if gradient is not None
                else torch.zeros_like(parameter).reshape(-1)
            )
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    )


def _direct_priority_gradient_correction(
    direct_loss: torch.Tensor,
    guessed_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    maximum_guessed_to_direct_gradient_ratio: float,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    direct = _flatten_task_gradients(direct_loss, parameters)
    guessed = _flatten_task_gradients(guessed_loss, parameters)
    if dist.is_initialized():
        dist.all_reduce(direct, op=dist.ReduceOp.SUM)
        dist.all_reduce(guessed, op=dist.ReduceOp.SUM)
        direct /= dist.get_world_size()
        guessed /= dist.get_world_size()
    direct_norm_square = direct.square().sum()
    direct_norm = direct_norm_square.sqrt()
    guessed_norm = guessed.norm()
    dot = (direct * guessed).sum()
    conflict = (dot < 0) & (direct_norm_square > 1e-12)
    coefficient = torch.where(
        conflict,
        dot / direct_norm_square.clamp_min(1e-12),
        dot.new_zeros(()),
    )
    projected = guessed - coefficient * direct
    projected_norm = projected.norm()
    maximum_norm = float(maximum_guessed_to_direct_gradient_ratio) * direct_norm
    norm_scale = torch.where(
        (direct_norm > 1e-12) & (projected_norm > maximum_norm),
        maximum_norm / projected_norm.clamp_min(1e-12),
        projected_norm.new_ones(()),
    )
    projected = projected * norm_scale
    correction = projected - guessed
    pieces = []
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        pieces.append(
            correction[offset : offset + count]
            .view_as(parameter)
            .to(dtype=parameter.dtype)
        )
        offset += count
    return pieces, {
        "pcgrad_conflict": conflict.to(direct.dtype),
        "pcgrad_direct_gradient_norm": direct_norm,
        "pcgrad_guessed_gradient_norm": guessed_norm,
        "pcgrad_projected_guessed_gradient_norm": projected.norm(),
        "pcgrad_guessed_gradient_scale": norm_scale,
    }


def _all_reduce_gradients(parameters: list[torch.nn.Parameter]) -> None:
    if not dist.is_initialized():
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad /= dist.get_world_size()


def _apply_gradient_correction(
    parameters: list[torch.nn.Parameter],
    correction: list[torch.Tensor],
) -> None:
    for parameter, value in zip(parameters, correction, strict=True):
        if parameter.grad is None:
            parameter.grad = value.clone()
        else:
            parameter.grad.add_(value)


@torch.no_grad()
def validate_paired(
    model: PairedMethod1System,
    loader: DataLoader,
    *,
    device: torch.device,
    observed_weights: ObservedModelLossWeights,
    evidential_weights: EvidentialModelLossWeights,
    class_weights: torch.Tensor,
    config: dict,
    maximum_batches: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        batch = move_batch(batch, device)
        extraction = model.extract(
            batch["images"],
            batch["camera_height_m"],
        )
        batch, geometry_validation = validate_p1a_training_geometry(
            batch,
            extraction,
        )
        complete_prediction = model.forward_complete(extraction)
        complete_losses = complete_evidential_model_loss(
            complete_prediction,
            batch,
            guessed_class_weights=class_weights,
            weights=evidential_weights,
            regularizer_scale=1.0,
            guessed_supervision_scale=1.0,
            surface_tolerance_single_pixels=_surface_tolerance_pixels(
                config,
                "single",
            ),
            surface_tolerance_merged_pixels=_surface_tolerance_pixels(
                config,
                "merged",
            ),
        )
        values: dict[str, torch.Tensor] = {
            f"complete_{key}": value
            for key, value in complete_losses.items()
        }
        observed_prediction = None
        if model.observed_model is not None:
            observed_prediction = model.forward_observed(extraction)
            observed_losses = observed_model_loss(
                observed_prediction,
                batch,
                weights=observed_weights,
                surface_tolerance_single_pixels=_surface_tolerance_pixels(
                    config,
                    "single",
                ),
                surface_tolerance_merged_pixels=_surface_tolerance_pixels(
                    config,
                    "merged",
                ),
            )
            values.update(
                {
                    f"observed_{key}": value
                    for key, value in observed_losses.items()
                }
            )
        for extent, extent_tensor in (
            ("single", geometry_validation["single_output_extent_m"]),
            ("merged", geometry_validation["merged_output_extent_m"]),
        ):
            complete_metrics = evidential_complete_metrics(
                complete_prediction[extent],
                batch[f"{extent}_fov_complete_target"],
                batch[f"{extent}_fov_visible_target"],
                batch[f"{extent}_fov_support_target"],
                target_extent_m=extent_tensor,
                surface_tolerance_pixels=_surface_tolerance_pixels(
                    config,
                    extent,
                ),
            )
            values.update(
                {
                    f"complete_{extent}_{key}": value
                    for key, value in complete_metrics.items()
                }
            )
            if observed_prediction is not None:
                observed_metrics = observed_categorical_metrics(
                    observed_prediction[extent],
                    batch[f"{extent}_fov_visible_target"],
                    target_extent_m=extent_tensor,
                )
                values.update(
                    {
                        f"observed_{extent}_{key}": value
                        for key, value in observed_metrics.items()
                    }
                )
        values.update(
            {
                "geometry_quality": geometry_validation[
                    "geometry_quality"
                ].mean(),
                "geometry_valid_fraction": geometry_validation[
                    "geometry_valid"
                ].float().mean(),
                "scale_validation_log_error": geometry_validation[
                    "scale_log_error"
                ][geometry_validation["scale_validation_valid"]].mean()
                if geometry_validation["scale_validation_valid"].any()
                else geometry_validation["scale_log_error"].new_zeros(()),
            }
        )
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        batches += 1
    if dist.is_initialized():
        reduced = {}
        for key, total in totals.items():
            value = torch.tensor(
                [total, float(batches)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            reduced[key] = float(value[0] / value[1].clamp_min(1.0))
        result = reduced
    else:
        result = {key: value / max(batches, 1) for key, value in totals.items()}
    model.train()
    return result


def _load_paired_resume(
    args: argparse.Namespace,
    *,
    model: PairedMethod1System,
    observed_optimizer: torch.optim.Optimizer | None,
    complete_optimizer: torch.optim.Optimizer,
    observed_scaler: torch.amp.GradScaler | None,
    complete_scaler: torch.amp.GradScaler,
    config: dict,
    split_manifest_sha256: str,
) -> tuple[int, int, int, dict | None]:
    observed_enabled = model.observed_model is not None
    if not observed_enabled:
        if args.resume_observed is not None:
            raise ValueError(
                "Model-B-only training does not accept --resume-observed"
            )
        if args.resume_complete is None:
            return 0, 0, 0, None
        complete_state = torch.load(
            args.resume_complete,
            map_location="cpu",
            weights_only=False,
        )
        if complete_state.get("model_kind") != "complete_evidential":
            raise ValueError("the complete resume checkpoint has the wrong model kind")
        contract = paired_resume_contract(config, split_manifest_sha256)
        if complete_state.get("resume_contract") != contract:
            raise ValueError(
                "complete resume contract does not match current training"
            )
        _load_path_state(
            model,
            "complete",
            complete_state["model_state_dict"],
        )
        complete_optimizer.load_state_dict(
            complete_state["optimizer_state_dict"]
        )
        complete_scaler.load_state_dict(
            complete_state["grad_scaler_state_dict"]
        )
        return (
            int(complete_state["next_epoch"]),
            int(complete_state["next_batch_in_epoch"]),
            int(complete_state["global_step"]),
            complete_state["rng_state"],
        )

    if (args.resume_observed is None) != (args.resume_complete is None):
        raise ValueError(
            "paired resume requires both --resume-observed and --resume-complete"
        )
    if args.resume_observed is None:
        return 0, 0, 0, None
    observed_state = torch.load(
        args.resume_observed,
        map_location="cpu",
        weights_only=False,
    )
    complete_state = torch.load(
        args.resume_complete,
        map_location="cpu",
        weights_only=False,
    )
    if observed_state.get("model_kind") != "observed":
        raise ValueError("the observed resume checkpoint has the wrong model kind")
    if complete_state.get("model_kind") != "complete_evidential":
        raise ValueError("the complete resume checkpoint has the wrong model kind")
    pair_fields = (
        "checkpoint_pair_id",
        "global_step",
        "next_epoch",
        "next_batch_in_epoch",
        "split_manifest_sha256",
    )
    for key in pair_fields:
        if observed_state.get(key) != complete_state.get(key):
            raise ValueError(f"paired resume checkpoints disagree on {key}")
    contract = paired_resume_contract(config, split_manifest_sha256)
    if (
        observed_state.get("resume_contract") != contract
        or complete_state.get("resume_contract") != contract
    ):
        raise ValueError("paired resume contract does not match current training")
    _load_path_state(model, "observed", observed_state["model_state_dict"])
    _load_path_state(model, "complete", complete_state["model_state_dict"])
    assert observed_optimizer is not None
    assert observed_scaler is not None
    observed_optimizer.load_state_dict(observed_state["optimizer_state_dict"])
    complete_optimizer.load_state_dict(complete_state["optimizer_state_dict"])
    observed_scaler.load_state_dict(observed_state["grad_scaler_state_dict"])
    complete_scaler.load_state_dict(complete_state["grad_scaler_state_dict"])
    return (
        int(observed_state["next_epoch"]),
        int(observed_state["next_batch_in_epoch"]),
        int(observed_state["global_step"]),
        observed_state["rng_state"],
    )


def run_paired_training(args: argparse.Namespace, config: dict) -> None:
    training = config["training"]
    enabled_models = tuple(
        training.get(
            "enabled_models",
            ["observed", "complete_evidential"],
        )
    )
    observed_enabled = "observed" in enabled_models
    sampling_mode = str(
        config["data"].get("sampling_mode", "all_prefixes")
    )
    if args.data_only:
        train_dataset, validation_dataset = build_datasets(config)
        report = describe_data(train_dataset, validation_dataset)
        report.update(
            {
                "pipeline": "P1A",
                "coordinate_mode": (
                    "camera_height_anchored_fixed_normalized_scale"
                ),
                "extent_mode": "fixed_6p5_single_10_merged",
                "gpu_initialized": False,
            }
        )
        sample = train_dataset[0]
        report["first_sample"] = {
            "images": list(sample["images"].shape),
            "alignment_camera_centers_m": list(
                sample["alignment_camera_centers_m"].shape
            ),
            "camera_height_m": float(sample["camera_height_m"]),
            "single_output_extent_normalized_scale": 6.5,
            "merged_output_extent_normalized_scale": 10.0,
            "target_shapes": {
                key: list(value.shape)
                for key, value in sample.items()
                if key.endswith("_target") and torch.is_tensor(value)
            },
        }
        print(json.dumps(report, indent=2), flush=True)
        return
    distributed, rank, world_size, local_rank, device, nccl_preflight = (
        distributed_runtime(training)
    )
    primary = rank == 0
    seed_everything(int(training["seed"]))

    if distributed:
        if primary:
            train_dataset, validation_dataset = build_datasets(
                config,
                verify_manifest=not args.smoke_first_sample,
            )
        dist.barrier()
        if not primary:
            train_dataset, validation_dataset = build_datasets(
                config,
                verify_manifest=False,
            )
    else:
        train_dataset, validation_dataset = build_datasets(config)

    if primary:
        description = describe_data(train_dataset, validation_dataset)
        description.update(
            {
                "pipeline": "p1a_cascade_paired_evidential",
                "coordinate_mode": (
                    "camera_height_anchored_fixed_normalized_scale"
                ),
                "extent_mode": "fixed_6p5_single_10_merged",
                "runtime_metric_anchor": "camera_height_m",
                "training_only_gt_scale_validation": True,
                "training_target_resampling": False,
                "trainable_models": len(enabled_models),
                "enabled_models": list(enabled_models),
                "shared_trainable_parameters": 0,
                "live_vggt_forwards_per_batch": 1,
                "checkpoint_models": list(enabled_models),
                "sampling_mode": sampling_mode,
                "optimizer_steps_per_epoch_per_rank": (
                    (
                        len(train_dataset.sessions) + world_size - 1
                    )
                    // world_size
                    if sampling_mode == "one_prefix_per_session"
                    else len(train_dataset) // world_size
                    if distributed
                    else len(train_dataset)
                ),
                "distributed": distributed,
                "world_size": world_size,
                "distributed_backend": nccl_preflight["backend"],
                "nccl_preflight": nccl_preflight,
            }
        )
        print(json.dumps(description, indent=2), flush=True)
    if args.data_only:
        if primary:
            sample = train_dataset[0]
            print(
                json.dumps(
                    {
                        "images": list(sample["images"].shape),
                        "target_shapes": {
                            key: list(value.shape)
                            for key, value in sample.items()
                            if key.endswith("_target") and torch.is_tensor(value)
                        },
                        "single_target_extent_m": float(
                            sample["single_target_extent_m"]
                        ),
                        "merged_target_extent_m": float(
                            sample["merged_target_extent_m"]
                        ),
                        "model_output_coordinate_mode": (
                            "camera_height_anchored_fixed_normalized_scale"
                        ),
                        "model_output_extent_mode": (
                            "fixed_6p5_single_10_merged"
                        ),
                        "single_output_extent_normalized_scale": 6.5,
                        "merged_output_extent_normalized_scale": 10.0,
                        "camera_height_used": True,
                    },
                    indent=2,
                ),
                flush=True,
            )
        if distributed:
            dist.destroy_process_group()
        return

    if distributed:
        report_rank_stage(rank, "before_paired_model_build", device)
    model = build_paired_model(config, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if distributed:
        report_rank_stage(rank, "before_independent_ddp_wrap", device)
        ddp_arguments = {
            "device_ids": [local_rank],
            "output_device": local_rank,
            "broadcast_buffers": False,
            "bucket_cap_mb": float(training.get("ddp_bucket_cap_mb", 25.0)),
            "gradient_as_bucket_view": True,
        }
        if observed_enabled:
            assert model.observed_model is not None
            model.observed_model = DistributedDataParallel(
                model.observed_model,
                **ddp_arguments,
            )
        model.complete_model = DistributedDataParallel(
            model.complete_model,
            **ddp_arguments,
        )
        torch.cuda.synchronize(device)
        report_rank_stage(rank, "after_independent_ddp_wrap", device)

    observed_parameters = (
        [
            parameter
            for parameter in model.unwrapped_observed_model().parameters()
            if parameter.requires_grad
        ]
        if observed_enabled
        else []
    )
    complete_parameters = [
        parameter
        for parameter in model.unwrapped_complete_model().parameters()
        if parameter.requires_grad
    ]
    if {id(value) for value in observed_parameters}.intersection(
        id(value) for value in complete_parameters
    ):
        raise RuntimeError("observed and complete models share trainable parameters")
    observed_optimizer = (
        torch.optim.AdamW(
            observed_parameters,
            lr=float(training["observed_learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        if observed_enabled
        else None
    )
    complete_optimizer = torch.optim.AdamW(
        complete_parameters,
        lr=float(training["complete_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler_enabled = device.type == "cuda" and amp_dtype == torch.float16
    observed_scaler = (
        torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        if observed_enabled
        else None
    )
    complete_scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    (
        starting_epoch,
        starting_batch_in_epoch,
        global_step,
        resume_rng_state,
    ) = _load_paired_resume(
        args,
        model=model,
        observed_optimizer=observed_optimizer,
        complete_optimizer=complete_optimizer,
        observed_scaler=observed_scaler,
        complete_scaler=complete_scaler,
        config=config,
        split_manifest_sha256=train_dataset.split_manifest_sha256,
    )

    train_source = (
        smoke_subset(train_dataset, world_size)
        if args.smoke_first_sample
        else train_dataset
    )
    if sampling_mode == "one_prefix_per_session" and not args.smoke_first_sample:
        train_sampler = SessionPrefixSampler(
            train_source,
            seed=int(training["seed"]),
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            shuffle=True,
        )
    elif distributed:
        train_sampler = DistributedEpochShuffleSampler(
            train_source,
            num_replicas=world_size,
            rank=rank,
            seed=int(training["seed"]),
            shuffle=not args.smoke_first_sample,
            drop_last=True,
        )
    else:
        train_sampler = EpochShuffleSampler(
            train_source,
            seed=int(training["seed"]),
            shuffle=not args.smoke_first_sample,
        )
    validation_sampler = StratifiedValidationSampler(
        validation_dataset,
        maximum_samples=int(training["validation_batches"]),
        seed=int(training["seed"]),
        num_replicas=world_size if distributed else 1,
        rank=rank if distributed else 0,
    )
    loader_generator = torch.Generator()
    train_loader = DataLoader(
        train_source,
        batch_size=int(training["batch_size"]),
        sampler=train_sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=method1_collate,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        sampler=validation_sampler,
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=method1_collate,
    )
    total_steps = max(1, len(train_loader) * int(training["epochs"]))
    observed_weights, evidential_weights, class_weights = build_loss_configuration(
        config,
        device,
    )
    output_dir = Path(training["output_dir"]).expanduser().resolve()
    if args.smoke_first_sample:
        output_dir = output_dir / "smoke"
    if primary:
        if observed_enabled:
            (output_dir / "observed").mkdir(parents=True, exist_ok=True)
        (output_dir / "complete_evidential").mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    model.train()
    first_source_reported = False
    stop = False
    resume_rng_pending = resume_rng_state is not None
    for epoch in range(starting_epoch, int(training["epochs"])):
        train_sampler.set_epoch(epoch)
        loader_generator.manual_seed(
            int(training["seed"]) + epoch * world_size + rank
        )
        geometry_gate = 1.0
        regularizer_scale = confidence_regularizer_scale(epoch, training)
        next_epoch = epoch + 1
        next_batch_in_epoch = 0
        for batch_index, batch in enumerate(train_loader):
            if epoch == starting_epoch and batch_index < starting_batch_in_epoch:
                continue
            if resume_rng_pending:
                restore_rng_state(resume_rng_state)
                resume_rng_pending = False
            step_started = time.monotonic()
            batch = move_batch(batch, device)

            # The expensive frozen VGGT extraction is computed once. Model B
            # consumes the same runtime-aligned bundle used at inference.
            extraction = model.extract(
                batch["images"],
                batch["camera_height_m"],
            )
            batch, geometry_validation = validate_p1a_training_geometry(
                batch,
                extraction,
            )
            geometry_valid = geometry_validation["geometry_valid"].all()
            if distributed:
                synchronized_valid = geometry_valid.to(torch.int32)
                dist.all_reduce(synchronized_valid, op=dist.ReduceOp.MIN)
                geometry_valid = synchronized_valid.bool()
            if not bool(geometry_valid):
                if primary:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "batch": batch_index,
                                "skipped_invalid_geometry": True,
                                "geometry_quality": float(
                                    geometry_validation[
                                        "geometry_quality"
                                    ].mean().cpu()
                                ),
                            }
                        ),
                        flush=True,
                    )
                del extraction
                continue

            observed_log: dict[str, float] | None = None
            if observed_enabled:
                assert observed_optimizer is not None
                assert observed_scaler is not None
                observed_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    observed_prediction = model.forward_observed(
                        extraction,
                        geometry_gate=geometry_gate,
                    )
                    observed_losses = observed_model_loss(
                        observed_prediction,
                        batch,
                        weights=observed_weights,
                        surface_tolerance_single_pixels=(
                            _surface_tolerance_pixels(config, "single")
                        ),
                        surface_tolerance_merged_pixels=(
                            _surface_tolerance_pixels(config, "merged")
                        ),
                    )
                observed_scaler.scale(observed_losses["loss"]).backward()
                observed_scaler.unscale_(observed_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    observed_parameters,
                    max_norm=1.0,
                )
                observed_scaler.step(observed_optimizer)
                observed_scaler.update()
                observed_log = {
                    key: float(value.detach().cpu())
                    for key, value in observed_losses.items()
                }
                del observed_prediction, observed_losses

            complete_optimizer.zero_grad(set_to_none=True)
            guessed_scale = guessed_supervision_scale(
                global_step,
                total_steps,
                training,
            )
            pcgrad_enabled = (
                bool(training["direct_priority_pcgrad"])
                and guessed_scale > 0.0
            )
            synchronization = (
                model.complete_model.no_sync()
                if distributed and pcgrad_enabled
                else nullcontext()
            )
            with synchronization:
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    complete_prediction = model.forward_complete(
                        extraction,
                        geometry_gate=geometry_gate,
                    )
                    complete_losses = complete_evidential_model_loss(
                        complete_prediction,
                        batch,
                        guessed_class_weights=class_weights,
                        weights=evidential_weights,
                        regularizer_scale=regularizer_scale,
                        guessed_supervision_scale=guessed_scale,
                        surface_tolerance_single_pixels=(
                            _surface_tolerance_pixels(config, "single")
                        ),
                        surface_tolerance_merged_pixels=(
                            _surface_tolerance_pixels(config, "merged")
                        ),
                    )
                correction = None
                if pcgrad_enabled:
                    correction, pcgrad_values = (
                        _direct_priority_gradient_correction(
                            complete_losses["direct_objective"],
                            complete_losses["guessed_objective"],
                            complete_parameters,
                            maximum_guessed_to_direct_gradient_ratio=float(
                                training[
                                    "maximum_guessed_to_direct_gradient_ratio"
                                ]
                            ),
                        )
                    )
                    complete_losses.update(pcgrad_values)
                complete_scaler.scale(complete_losses["loss"]).backward()
            complete_scaler.unscale_(complete_optimizer)
            if pcgrad_enabled:
                if distributed:
                    _all_reduce_gradients(complete_parameters)
                assert correction is not None
                _apply_gradient_correction(complete_parameters, correction)
            torch.nn.utils.clip_grad_norm_(complete_parameters, max_norm=1.0)
            complete_scaler.step(complete_optimizer)
            complete_scaler.update()
            complete_log = {
                key: float(value.detach().cpu())
                for key, value in complete_losses.items()
            }
            single_extent_report = float(
                complete_prediction["single"][
                    "extent_normalized_scale"
                ][0]
                .detach()
                .cpu()
            )
            merged_extent_report = float(
                complete_prediction["merged"][
                    "extent_normalized_scale"
                ][0]
                .detach()
                .cpu()
            )
            del complete_prediction, complete_losses

            global_step += 1
            if batch_index + 1 < len(train_loader):
                next_epoch = epoch
                next_batch_in_epoch = batch_index + 1
            else:
                next_epoch = epoch + 1
                next_batch_in_epoch = 0

            if not first_source_reported:
                if primary:
                    diagnostics = model.diagnostics(extraction)
                    print(
                        json.dumps(
                            {
                                "live_vggt_verified": True,
                                "live_vggt_forwards_this_batch": 1,
                                "geometry_source": diagnostics["geometry_source"],
                                "coordinate_mode": diagnostics[
                                    "coordinate_mode"
                                ],
                                "camera_height_used": diagnostics[
                                    "camera_height_used"
                                ],
                                "metric_scale_used": diagnostics[
                                    "metric_scale_used"
                                ],
                                "scene_radius_vggt": float(
                                    diagnostics["scene_radius_vggt"][0]
                                    .detach()
                                    .cpu()
                                ),
                                "single_extent_normalized_scale": (
                                    single_extent_report
                                ),
                                "merged_extent_normalized_scale": (
                                    merged_extent_report
                                ),
                                "metric_per_vggt_native_unit": float(
                                    diagnostics["p1a_geometry"][
                                        "metric_per_vggt_native_unit"
                                    ][0]
                                    .detach()
                                    .cpu()
                                ),
                                "ground_quality": float(
                                    diagnostics["p1a_geometry"][
                                        "ground_quality"
                                    ][0]
                                    .detach()
                                    .cpu()
                                ),
                                "model_a": (
                                    "direct observed categorical BEV"
                                    if observed_enabled
                                    else "disabled"
                                ),
                                "model_b": "complete Beta evidential BEV",
                                "shared_trainable_parameters": 0,
                                "confidence_sigmoid": False,
                                "gt_depth_pose_trajectory_in_forward": False,
                                "gt_trajectory_training_target_alignment": False,
                                "gt_trajectory_training_scale_validation": True,
                                "geometry_valid": bool(
                                    geometry_validation[
                                        "geometry_valid"
                                    ][0]
                                    .detach()
                                    .cpu()
                                ),
                                "scale_validation_valid": bool(
                                    geometry_validation[
                                        "scale_validation_valid"
                                    ][0]
                                    .detach()
                                    .cpu()
                                ),
                            }
                        ),
                        flush=True,
                    )
                first_source_reported = True
            del extraction

            if primary and global_step % int(training["log_every_steps"]) == 0:
                log_record = {
                    "epoch": epoch,
                    "step": global_step,
                    "complete_evidential_loss": complete_log["loss"],
                    "complete_direct_objective": complete_log[
                        "direct_objective"
                    ],
                    "complete_guessed_objective": complete_log[
                        "guessed_objective"
                    ],
                    "complete_single_observed_strength": complete_log[
                        "single_observed_mean_strength"
                    ],
                    "complete_single_guessed_strength": complete_log[
                        "single_guessed_mean_strength"
                    ],
                    "complete_merged_observed_strength": complete_log[
                        "merged_observed_mean_strength"
                    ],
                    "complete_merged_guessed_strength": complete_log[
                        "merged_guessed_mean_strength"
                    ],
                    "regularizer_scale": regularizer_scale,
                    "guessed_supervision_scale": guessed_scale,
                    "pcgrad_enabled": pcgrad_enabled,
                    "seconds": time.monotonic() - step_started,
                    "history_frames": int(batch["images"].shape[1]),
                    "effective_global_batch": (
                        int(training["batch_size"]) * world_size
                    ),
                    "geometry_gate": geometry_gate,
                    "geometry_quality": float(
                        geometry_validation["geometry_quality"].mean().cpu()
                    ),
                    "geometry_valid_fraction": float(
                        geometry_validation["geometry_valid"]
                        .float()
                        .mean()
                        .cpu()
                    ),
                    "sampling_mode": sampling_mode,
                }
                if observed_log is not None:
                    log_record["observed_loss"] = observed_log["loss"]
                print(
                    json.dumps(log_record),
                    flush=True,
                )
            if (
                primary
                and global_step % int(training["checkpoint_every_steps"]) == 0
            ):
                pair_id = (
                    f"{train_dataset.split_manifest_sha256[:12]}-"
                    f"step-{global_step:08d}"
                )
                checkpoint_paths = [
                    (
                        "complete_evidential",
                        complete_optimizer,
                        complete_scaler,
                    )
                ]
                if observed_enabled:
                    assert observed_optimizer is not None
                    assert observed_scaler is not None
                    checkpoint_paths.insert(
                        0,
                        ("observed", observed_optimizer, observed_scaler),
                    )
                for model_kind, optimizer, scaler in checkpoint_paths:
                    save_path_checkpoint(
                        output_dir
                        / model_kind
                        / f"step_{global_step:08d}.pt",
                        model=model,
                        model_kind=model_kind,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=config,
                        epoch=epoch,
                        global_step=global_step,
                        next_epoch=next_epoch,
                        next_batch_in_epoch=next_batch_in_epoch,
                        split_manifest_sha256=(
                            train_dataset.split_manifest_sha256
                        ),
                        checkpoint_pair_id=pair_id,
                    )
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                stop = True
                break

        epoch_completed = next_epoch > epoch
        if not args.skip_validation and epoch_completed:
            result = validate_paired(
                model,
                validation_loader,
                device=device,
                observed_weights=observed_weights,
                evidential_weights=evidential_weights,
                class_weights=class_weights,
                config=config,
                maximum_batches=int(training["validation_batches"]),
            )
            if primary:
                print(json.dumps({"epoch": epoch, "validation": result}), flush=True)
        if primary:
            pair_id = (
                f"{train_dataset.split_manifest_sha256[:12]}-"
                f"step-{global_step:08d}"
            )
            checkpoint_paths = [
                (
                    "complete_evidential",
                    complete_optimizer,
                    complete_scaler,
                )
            ]
            if observed_enabled:
                assert observed_optimizer is not None
                assert observed_scaler is not None
                checkpoint_paths.insert(
                    0,
                    ("observed", observed_optimizer, observed_scaler),
                )
            for model_kind, optimizer, scaler in checkpoint_paths:
                save_path_checkpoint(
                    output_dir / model_kind / "latest.pt",
                    model=model,
                    model_kind=model_kind,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    next_epoch=next_epoch,
                    next_batch_in_epoch=next_batch_in_epoch,
                    split_manifest_sha256=train_dataset.split_manifest_sha256,
                    checkpoint_pair_id=pair_id,
                )
        if stop:
            break
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    from vggt_bev_method1.cli_train import parse_args
    from vggt_bev_method1.config import load_config

    args = parse_args()
    if args.resume is not None:
        raise ValueError(
            "P1A uses --resume-observed and --resume-complete"
        )
    run_paired_training(args, load_config(args.config))


if __name__ == "__main__":
    main()
