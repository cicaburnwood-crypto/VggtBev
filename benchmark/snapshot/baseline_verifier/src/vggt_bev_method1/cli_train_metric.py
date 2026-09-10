from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from vggt_bev_method1.config import load_config
from vggt_bev_method1.data import method1_collate
from vggt_bev_method1.losses import (
    FOVCompleteEvidentialLossWeights,
    fov_complete_evidential_bev_loss,
)
from vggt_bev_method1.metrics import fov_complete_evidential_metrics
from vggt_bev_method1.models import (
    LiveVGGTOmegaAdapter,
    Method1System,
    ScaleFitConfig,
    fit_metric_scale_targets,
    metric_scale_losses,
    metric_scale_metrics,
)
from vggt_bev_method1.teacher_cache import TeacherCache
from vggt_bev_method1.train_utils import (
    build_datasets,
    distributed_runtime,
    move_batch,
    seed_everything,
    smoke_subset,
)
from vggt_bev_method1.training_state import (
    DistributedEpochShuffleSampler,
    EpochOffsetSampler,
    EpochShuffleSampler,
    SessionPrefixSampler,
    StratifiedValidationSampler,
)

EXECUTION_ONLY_MODEL_KEYS = frozenset({"cross_query_chunk_size"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train P1B FOV-complete fixed-metric BEV + metric scale token"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--smoke-first-sample", action="store_true")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(config: dict, device: torch.device) -> Method1System:
    model_config = config["model"]
    layers = tuple(int(value) for value in model_config["cached_layers"])
    adapter = LiveVGGTOmegaAdapter(
        model_config["vggt_source"],
        model_config["checkpoint"],
        device=device,
        patch_size=int(model_config["patch_size"]),
        cached_layers=layers,
    )
    return Method1System(
        adapter,
        cached_layers=layers,
        spatial_scales=tuple(
            float(value) for value in model_config["spatial_scales"]
        ),
        vggt_token_dim=int(model_config["vggt_token_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        heads=int(model_config["attention_heads"]),
        decoder_layers=int(model_config["decoder_layers"]),
        scale_decoder_layers=int(model_config["scale_decoder_layers"]),
        self_attention_mode=str(model_config["self_attention_mode"]),
        cross_attention_mode=str(model_config["cross_attention_mode"]),
        deformable_samples=int(model_config["deformable_samples"]),
        cross_query_chunk_size=int(model_config["cross_query_chunk_size"]),
        single_latent_bev_size=int(model_config["single_latent_bev_size"]),
        merged_latent_bev_size=int(model_config["merged_latent_bev_size"]),
        single_output_size=int(model_config["single_bev_output_size"]),
        merged_output_size=int(model_config["merged_bev_output_size"]),
        single_bev_extent_m=float(model_config["single_bev_extent_m"]),
        merged_bev_extent_m=float(model_config["merged_bev_extent_m"]),
        predict_scale_uncertainty=bool(
            model_config.get("predict_scale_uncertainty", True)
        ),
    ).to(device)


def _configure_head_compilation(
    model: Method1System,
    training: dict,
) -> dict[str, object]:
    enabled = bool(training.get("compile_head", False))
    settings: dict[str, object] = {
        "enabled": enabled,
        "scope": str(
            training.get(
                "compile_head_scope",
                "deformable_query_chunks",
            )
        ),
        "backend": str(training.get("compile_head_backend", "inductor")),
        "mode": str(training.get("compile_head_mode", "default")),
        "dynamic": bool(training.get("compile_head_dynamic", True)),
    }
    compiled_modules = 0
    if enabled:
        # PCGrad differentiates two objectives through the same forward graph.
        # AOTAutograd buffer donation assumes one non-retained backward and
        # otherwise fails before the optimizer step.
        import torch._functorch.config as functorch_config

        functorch_config.donated_buffer = False
        # Compiling the complete native-resolution head unrolls the Python
        # query loop into one impractically large graph. Compile bounded
        # deformable chunks and dense self-attention/FFN kernels separately.
        # These execution-only callables do not enter state_dicts.
        for module in model.head.modules():
            parameters = getattr(module, "parameters", None)
            if parameters is not None and not any(
                parameter.requires_grad
                for parameter in parameters(recurse=True)
            ):
                continue
            for method_name in (
                "configure_query_chunk_compilation",
                "configure_execution_compilation",
            ):
                configure = getattr(module, method_name, None)
                if configure is None:
                    continue
                configured = configure(
                    backend=settings["backend"],
                    mode=settings["mode"],
                    dynamic=settings["dynamic"],
                )
                if configured is not False:
                    compiled_modules += 1
        if compiled_modules == 0:
            raise RuntimeError(
                "head compile requested but no deformable query chunks exist"
            )
    settings["compiled_modules"] = compiled_modules
    settings["donated_buffer"] = False if enabled else None
    return settings


def scale_fit_config(config: dict) -> ScaleFitConfig:
    values = config["scale_fit"]
    return ScaleFitConfig(
        confidence_threshold=float(values["confidence_threshold"]),
        minimum_depth_m=float(values["minimum_depth_m"]),
        maximum_depth_m=float(values["maximum_depth_m"]),
        residual_threshold_log=float(values["residual_threshold_log"]),
        huber_delta_log=float(values["huber_delta_log"]),
        minimum_valid_pixels=int(values["minimum_valid_pixels"]),
        maximum_pixels_per_frame=int(values["maximum_pixels_per_frame"]),
        minimum_inlier_ratio=float(values["minimum_inlier_ratio"]),
        minimum_quality_weight=float(
            values.get("minimum_quality_weight", 0.10)
        ),
        quality_sigma_log=float(values["quality_sigma_log"]),
        desired_log_depth_range=float(values["desired_log_depth_range"]),
        irls_iterations=int(values["irls_iterations"]),
    )


def _enabled_bev_branches(training: dict) -> tuple[str, ...]:
    return tuple(
        training.get("enabled_bev_branches", ("single", "merged"))
    )


def _set_training_stage(
    model: Method1System,
    stage: str,
    enabled_bev_branches: tuple[str, ...] = ("single", "merged"),
) -> None:
    head = model.unwrapped_head()
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    if stage == "scale_only":
        for parameter in head.token_projector.parameters():
            parameter.requires_grad_(False)
        for decoder in (head.single_bev_decoder, head.merged_bev_decoder):
            for parameter in decoder.parameters():
                parameter.requires_grad_(False)
    else:
        for branch, decoder in (
            ("single", head.single_bev_decoder),
            ("merged", head.merged_bev_decoder),
        ):
            if branch not in enabled_bev_branches:
                for parameter in decoder.parameters():
                    parameter.requires_grad_(False)
    if stage == "bev_only":
        for parameter in head.scale_token_projector.parameters():
            parameter.requires_grad_(False)
        for parameter in head.scale_decoder.parameters():
            parameter.requires_grad_(False)


def _resume_contract(config: dict, manifest_sha256: str) -> dict:
    training = config["training"]
    loss_contract = {
        key: training[key]
        for key in (
            "guessed_completion_class_weights",
            "bev_loss_weight",
            "single_task_weight",
            "merged_task_weight",
            "observed_free_nll_weight",
            "observed_surface_nll_weight",
            "guessed_completion_nll_weight",
            "guessed_completion_dice_weight",
            "observed_region_weight",
            "guessed_region_weight",
            "surface_tolerance_latent_cell_fraction",
            "guessed_supervision_warmup_fraction",
            "guessed_supervision_ramp_fraction",
            "direct_priority_pcgrad",
            "train_guessed_completion",
            "maximum_guessed_to_direct_gradient_ratio",
            "fov_support_bce_weight",
            "fov_support_dice_weight",
            "incorrect_evidence_weight",
            "confidence_calibration_weight",
            "guessed_incorrect_evidence_multiplier",
            "guessed_confidence_calibration_multiplier",
            "observation_relation_weight",
            "evidence_relation_margin",
            "confidence_regularizer_warmup_epochs",
            "confidence_regularizer_ramp_epochs",
            "scale_loss_weight",
            "depth_scale_loss_weight",
            "uncertainty_loss_weight",
        )
    }
    has_branch_surface_weights = any(
        key in training
        for key in (
            "single_observed_surface_nll_weight",
            "merged_observed_surface_nll_weight",
        )
    )
    has_surface_continuity = (
        "observed_surface_continuity_weight" in training
    )
    if has_surface_continuity:
        loss_contract["observed_surface_continuity_weight"] = float(
            training["observed_surface_continuity_weight"]
        )
    if has_branch_surface_weights:
        loss_contract["single_observed_surface_nll_weight"] = float(
            training.get(
                "single_observed_surface_nll_weight",
                training["observed_surface_nll_weight"],
            )
        )
        loss_contract["merged_observed_surface_nll_weight"] = float(
            training.get(
                "merged_observed_surface_nll_weight",
                training["observed_surface_nll_weight"],
            )
        )
    return {
        "schema": "p1b-fixed-metric-fov-complete-evidential-v6",
        "loss_schema": (
            "observed-surface-continuity-confidence-pcgrad-v6"
            if has_surface_continuity
            else (
                "observed-pointwise-confidence-pcgrad-v5"
                if has_branch_surface_weights
                else "observed-pointwise-confidence-pcgrad-v4"
            )
        ),
        "split_manifest_sha256": manifest_sha256,
        "data": config["data"],
        "model": {
            key: value
            for key, value in config["model"].items()
            if key not in EXECUTION_ONLY_MODEL_KEYS
        },
        "scale_fit": config["scale_fit"],
        "stage": training["stage"],
        "enabled_bev_branches": list(_enabled_bev_branches(training)),
        "loss": loss_contract,
    }


def _normalized_resume_contract(contract: dict) -> dict:
    """Accept old checkpoints that recorded execution-only model fields."""

    normalized = dict(contract)
    model_contract = dict(normalized.get("model", {}))
    for key in EXECUTION_ONLY_MODEL_KEYS:
        model_contract.pop(key, None)
    normalized["model"] = model_contract
    return normalized


def save_checkpoint(
    path: Path,
    *,
    model: Method1System,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: dict,
    manifest_sha256: str,
    checkpoint_hash: str,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    validation_metrics: dict[str, float] | None = None,
    selection_name: str | None = None,
    selection_score: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    training = config["training"]
    enabled_bev_branches = _enabled_bev_branches(training)
    scale_trained = training["stage"] in ("scale_only", "joint")
    trained_outputs = [
        *(f"{branch}_bev" for branch in enabled_bev_branches),
        *(("scale",) if scale_trained else ()),
    ]
    disabled_outputs = [
        *(f"{branch}_bev" for branch in ("single", "merged")
          if branch not in enabled_bev_branches),
        *(("scale",) if not scale_trained else ()),
    ]
    output_contract = {
        "trained_outputs": trained_outputs,
        "disabled_untrained_outputs": disabled_outputs,
        "orientation": "latest ego centered; forward image-up",
        "confidence_activation": "none; confidence derived from Beta evidence",
        "content_supervision": (
            "collision-truth complete BEV clipped on the fly to "
            "unobstructed camera-FOV support"
        ),
        "confidence_supervision": (
            "visible masked cells versus occluded FOV-complete cells"
        ),
        "outside_fov_semantics": "unknown",
        "path_head": False,
        "runtime_inputs": ["RGB window"],
        "runtime_geometry_head_dependency": False,
    }
    bev_outputs = [
        "occupancy_probability",
        "evidence_confidence",
        "epistemic_uncertainty",
        "occupancy_distribution_variance",
        "fov_support_probability",
        "fov_complete_semantic",
        "navigation_confidence",
    ]
    if "single" in enabled_bev_branches:
        output_contract.update(
            {
                "single_bev": (
                    "FOV-complete occupied/free Beta evidence; unknown outside"
                ),
                "single_outputs": bev_outputs,
                "single_extent_m": 6.5,
                "single_size": [512, 512],
                "single_cell_size_m": 6.5 / 512,
                "single_bounds_m": [-3.25, 3.25, -3.25, 3.25],
            }
        )
    if "merged" in enabled_bev_branches:
        output_contract.update(
            {
                "merged_bev": (
                    "temporal-FOV-union complete occupied/free Beta evidence; "
                    "unknown outside"
                ),
                "merged_outputs": bev_outputs,
                "merged_extent_m": 10.0,
                "merged_size": [800, 800],
                "merged_cell_size_m": 10.0 / 800,
                "merged_bounds_m": [-5.0, 5.0, -5.0, 5.0],
                "merged_source": (
                    "direct RGB-window token decoding, not single-BEV fusion"
                ),
            }
        )
    if scale_trained:
        output_contract["scale"] = "lambda in meter per VGGT runtime unit"
    torch.save(
        {
            "format_version": 17,
            "pipeline_id": "P1B",
            "checkpoint_schema": "p1b-fixed-metric-fov-complete-evidential-v6",
            "method": (
                "frozen VGGT tokens -> direct FOV-complete evidential "
                f"fixed-metric {','.join(enabled_bev_branches)} BEV(s), "
                "RGB-only FOV support, plus parallel metric Scale Token"
            ),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
            "validation_metrics": validation_metrics,
            "checkpoint_selection": (
                None
                if selection_name is None
                else {
                    "name": selection_name,
                    "score": selection_score,
                    "lower_is_better": True,
                }
            ),
            "config": config,
            "resume_contract": _resume_contract(config, manifest_sha256),
            "vggt_checkpoint_sha256": checkpoint_hash,
            "preprocessing_version": "rgb-depth-resize-pad-v2",
            "head_state_dict": {
                key: value.detach().cpu()
                for key, value in model.unwrapped_head().state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "grad_scaler_state_dict": scaler.state_dict(),
            "output_contract": output_contract,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: Method1System,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    contract: dict,
) -> tuple[int, int, int | None]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        state.get("checkpoint_schema")
        != "p1b-fixed-metric-fov-complete-evidential-v6"
    ):
        raise ValueError(
            "resume checkpoint is not complete-evidential P1B v5; "
            "masked-BEV checkpoints cannot be resumed"
        )
    if _normalized_resume_contract(
        state.get("resume_contract", {})
    ) != _normalized_resume_contract(contract):
        raise ValueError("resume contract does not match current configuration")
    model.unwrapped_head().load_state_dict(state["head_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state.get("grad_scaler_state_dict", {}))
    batch_in_epoch = state.get("batch_in_epoch")
    return (
        int(state["epoch"]),
        int(state["global_step"]),
        None if batch_in_epoch is None else int(batch_in_epoch),
    )


def _validation_batches_per_rank(maximum_batches: int, world_size: int) -> int:
    if maximum_batches <= 0:
        return 0
    if world_size <= 0:
        raise ValueError("validation world size must be positive")
    return math.ceil(maximum_batches / world_size)


def validation_checkpoint_scores(
    metrics: dict[str, float],
) -> dict[str, float]:
    """Return gated, lower-is-better deployment-selection scores."""

    support_iou = metrics.get("single_bev_support_iou")
    surface_recall = metrics.get("single_bev_observed_surface_recall")
    if (
        support_iou is None
        or surface_recall is None
        or support_iou < 0.95
        or surface_recall < 0.92
    ):
        return {}
    required = (
        "single_bev_observed_free_false_occupied_rate",
        "single_bev_guessed_occupied_iou_gain_over_all_occupied",
        "single_bev_observed_confidence_ece",
        "single_bev_guessed_confidence_ece",
    )
    if any(key not in metrics for key in required):
        return {}
    return {
        "best_observed": float(
            metrics["single_bev_observed_free_false_occupied_rate"]
        ),
        "best_completion": -float(
            metrics[
                "single_bev_guessed_occupied_iou_gain_over_all_occupied"
            ]
        ),
        "best_calibrated": max(
            float(metrics["single_bev_observed_confidence_ece"]),
            float(metrics["single_bev_guessed_confidence_ece"]),
        ),
    }


def _load_best_checkpoint_scores(output_dir: Path) -> dict[str, float]:
    path = output_dir / "best_checkpoint_scores.json"
    if not path.exists():
        return {}
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("best checkpoint score record must be an object")
    return {str(key): float(value) for key, value in values.items()}


def _save_best_checkpoint_scores(
    output_dir: Path,
    scores: dict[str, float],
) -> None:
    path = output_dir / "best_checkpoint_scores.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(scores, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reduce_validation_totals(
    totals: dict[str, float],
    count: int,
    *,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    if not dist.is_initialized():
        return totals, count
    keys = tuple(sorted(totals))
    payload = torch.tensor(
        [*(totals[key] for key in keys), float(count)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    reduced_count = int(payload[-1].item())
    return (
        {key: float(payload[index].item()) for index, key in enumerate(keys)},
        reduced_count,
    )


def _sampler(dataset, config: dict, *, distributed: bool, rank: int, world_size: int):
    data = config["data"]
    seed = int(config["training"]["seed"])
    if (
        data.get("sampling_mode", "all_prefixes") == "one_prefix_per_session"
        and hasattr(dataset, "samples")
    ):
        return SessionPrefixSampler(
            dataset,
            seed=seed,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            batch_size=int(config["training"]["batch_size"]),
        )
    if distributed:
        return DistributedEpochShuffleSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
        )
    return EpochShuffleSampler(dataset, seed=seed)


def _base_dataset(dataset):
    return getattr(dataset, "dataset", dataset)


def _build_scale_target(
    batch: dict,
    geometry: dict,
    fit_config: ScaleFitConfig,
) -> dict[str, torch.Tensor]:
    return fit_metric_scale_targets(
        batch["scale_gt_depth_m"],
        batch["scale_gt_valid_mask"],
        geometry["estimated_depth_vggt"],
        geometry["estimated_depth_confidence"],
        fit_config,
    )


def _teacher_inputs(
    model: Method1System,
    batch: dict,
    *,
    cache: TeacherCache | None,
    cache_mode: str,
    need_geometry: bool,
) -> tuple[dict, dict | None]:
    if cache is not None and cache_mode in ("read", "write_through"):
        cached = cache.load(batch["metadata"], device=batch["images"].device)
        if cached is not None:
            extraction, geometry = cached
            return extraction, geometry if need_geometry else None
        if cache_mode == "read":
            sample = batch["metadata"][0]["sample_id"]
            raise FileNotFoundError(f"teacher cache miss in read mode: {sample}")
    extraction = model.extract(batch["images"])
    geometry = model.decode_teacher_geometry(extraction) if need_geometry else None
    if cache is not None and cache_mode == "write_through":
        # The required cache record includes depth/confidence/K/E even during
        # BEV-only curriculum stages.
        cache_geometry = (
            geometry
            if geometry is not None
            else model.decode_teacher_geometry(extraction)
        )
        cache.save(batch["metadata"], extraction, cache_geometry)
    return extraction, geometry


def _fov_complete_loss_weights(
    training: dict,
    branch: str,
) -> FOVCompleteEvidentialLossWeights:
    if branch not in ("single", "merged"):
        raise ValueError("branch must be single or merged")
    observed_surface = training.get(
        f"{branch}_observed_surface_nll_weight",
        training["observed_surface_nll_weight"],
    )
    return FOVCompleteEvidentialLossWeights(
        observed_free=float(training["observed_free_nll_weight"]),
        observed_surface=float(observed_surface),
        observed_surface_continuity=float(
            training.get("observed_surface_continuity_weight", 0.0)
        ),
        guessed_nll=float(training["guessed_completion_nll_weight"]),
        guessed_overlap=float(training["guessed_completion_dice_weight"]),
        observed_region=float(training["observed_region_weight"]),
        guessed_region=float(training["guessed_region_weight"]),
        support_bce=float(training["fov_support_bce_weight"]),
        support_dice=float(training["fov_support_dice_weight"]),
        incorrect_evidence=float(training["incorrect_evidence_weight"]),
        confidence_calibration=float(
            training["confidence_calibration_weight"]
        ),
        guessed_incorrect_evidence_multiplier=float(
            training["guessed_incorrect_evidence_multiplier"]
        ),
        guessed_confidence_calibration_multiplier=float(
            training["guessed_confidence_calibration_multiplier"]
        ),
        observation_relation=float(training["observation_relation_weight"]),
        relation_margin=float(training["evidence_relation_margin"]),
    )


def confidence_regularizer_scale(epoch: int, training: dict) -> float:
    warmup = int(training["confidence_regularizer_warmup_epochs"])
    ramp = int(training["confidence_regularizer_ramp_epochs"])
    if epoch < warmup:
        return 0.0
    if ramp == 0:
        return 1.0
    return min(1.0, (epoch - warmup + 1) / ramp)


def guessed_supervision_scale(
    step: int,
    total_steps: int,
    training: dict,
) -> float:
    if step < 0 or total_steps <= 0:
        raise ValueError("curriculum step/total_steps are invalid")
    if not bool(training.get("train_guessed_completion", True)):
        return 0.0
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


def learning_rate_factor(
    step: int,
    total_steps: int,
    training: dict,
) -> float:
    """Warm up, then cosine-decay without starving late sampler epochs."""

    if step < 0 or total_steps <= 0:
        raise ValueError("learning-rate step/total_steps are invalid")
    maximum = float(training["learning_rate"])
    minimum = float(training.get("minimum_learning_rate", 0.0))
    if not 0.0 <= minimum < maximum:
        raise ValueError(
            "minimum learning rate must be non-negative and below maximum"
        )
    floor = minimum / maximum
    warmup_steps = max(
        1,
        round(total_steps * float(training["warmup_fraction"])),
    )
    if step < warmup_steps:
        progress = (step + 1) / warmup_steps
        return floor + (1.0 - floor) * progress
    progress = (step - warmup_steps) / max(
        1,
        total_steps - warmup_steps,
    )
    cosine = 0.5 * (
        1.0 + math.cos(math.pi * min(progress, 1.0))
    )
    return floor + (1.0 - floor) * cosine


def _surface_tolerance_pixels(config: dict, branch: str) -> int:
    if branch not in ("single", "merged"):
        raise ValueError("surface-tolerance branch must be single or merged")
    model = config["model"]
    training = config["training"]
    output_size = int(model[f"{branch}_bev_output_size"])
    latent_size = int(model[f"{branch}_latent_bev_size"])
    fraction = float(training["surface_tolerance_latent_cell_fraction"])
    return max(0, round(fraction * output_size / latent_size))


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
    maximum_guessed_to_direct_gradient_ratio: float = 0.0,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    """Protect direct gradients from opposing or oversized guessed updates."""

    direct = _flatten_task_gradients(direct_loss, parameters)
    guessed = _flatten_task_gradients(guessed_loss, parameters)
    if dist.is_initialized():
        dist.all_reduce(direct, op=dist.ReduceOp.SUM)
        dist.all_reduce(guessed, op=dist.ReduceOp.SUM)
        direct /= dist.get_world_size()
        guessed /= dist.get_world_size()
    direct_norm_square = direct.square().sum()
    guessed_norm = guessed.norm()
    direct_norm = direct_norm_square.sqrt()
    dot = (direct * guessed).sum()
    conflict = (dot < 0) & (direct_norm_square > 1e-12)
    coefficient = torch.where(
        conflict,
        dot / direct_norm_square.clamp_min(1e-12),
        dot.new_zeros(()),
    )
    conflict_correction = -coefficient * direct
    projected_guessed = guessed + conflict_correction
    projected_norm_before_cap = projected_guessed.norm()
    ratio_limit = float(maximum_guessed_to_direct_gradient_ratio)
    if ratio_limit > 0:
        maximum_norm = ratio_limit * direct_norm
        norm_scale = torch.where(
            (direct_norm > 1e-12) & (projected_norm_before_cap > maximum_norm),
            maximum_norm / projected_norm_before_cap.clamp_min(1e-12),
            projected_norm_before_cap.new_ones(()),
        )
    else:
        norm_scale = projected_norm_before_cap.new_ones(())
    projected_guessed = projected_guessed * norm_scale
    correction = projected_guessed - guessed
    projected_cosine = (direct * projected_guessed).sum() / (
        direct_norm * projected_guessed.norm()
    ).clamp_min(1e-12)
    original_cosine = dot / (direct_norm * guessed_norm).clamp_min(1e-12)
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
        "pcgrad_original_cosine": original_cosine,
        "pcgrad_projected_cosine": projected_cosine,
        "pcgrad_direct_gradient_norm": direct_norm,
        "pcgrad_guessed_gradient_norm": guessed_norm,
        "pcgrad_projected_guessed_gradient_norm": projected_guessed.norm(),
        "pcgrad_guessed_gradient_scale": norm_scale,
        "pcgrad_norm_cap_applied": (norm_scale < 1.0).to(direct.dtype),
        "pcgrad_correction_norm": correction.norm(),
    }


def _all_reduce_gradients(parameters: list[torch.nn.Parameter]) -> None:
    if not dist.is_initialized():
        return
    flat = torch.cat(
        [
            (
                parameter.grad.detach().float().reshape(-1)
                if parameter.grad is not None
                else torch.zeros_like(parameter).reshape(-1)
            )
            for parameter in parameters
        ]
    )
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= dist.get_world_size()
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        value = flat[offset : offset + count].view_as(parameter)
        if parameter.grad is None:
            parameter.grad = value.to(dtype=parameter.dtype)
        else:
            parameter.grad.copy_(value)
        offset += count


def _apply_gradient_correction(
    parameters: list[torch.nn.Parameter],
    correction: list[torch.Tensor],
) -> None:
    for parameter, value in zip(parameters, correction, strict=True):
        if parameter.grad is None:
            parameter.grad = value.clone()
        else:
            parameter.grad.add_(value)


def _step_losses(
    prediction: dict,
    batch: dict,
    geometry: dict | None,
    config: dict,
    fit_config: ScaleFitConfig,
    *,
    evidence_regularizer_scale: float,
    guessed_completion_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict | None]:
    training = config["training"]
    stage = training["stage"]
    enabled_bev_branches = _enabled_bev_branches(training)
    if "scale" in prediction:
        zero = prediction["scale"]["log_lambda_m_per_vggt"].sum() * 0.0
    else:
        first_branch = prediction[f"{enabled_bev_branches[0]}_bev"]
        zero = first_branch["raw_output"].sum() * 0.0
    values: dict[str, torch.Tensor] = {}
    if stage in ("bev_only", "joint"):
        guessed_class_weights = torch.tensor(
            training["guessed_completion_class_weights"],
            device=batch["images"].device,
            dtype=torch.float32,
        )
        bev_loss = zero
        direct_bev_objective = zero
        guessed_bev_objective = zero
        for branch in enabled_bev_branches:
            loss_weights = _fov_complete_loss_weights(training, branch)
            branch_bev = fov_complete_evidential_bev_loss(
                prediction[f"{branch}_bev"],
                batch[f"{branch}_fov_complete_target"],
                batch[f"{branch}_visible_target"],
                batch[f"{branch}_fov_support_target"],
                guessed_class_weights=guessed_class_weights,
                weights=loss_weights,
                regularizer_scale=evidence_regularizer_scale,
                guessed_supervision_scale=guessed_completion_scale,
                surface_tolerance_pixels=_surface_tolerance_pixels(
                    config,
                    branch,
                ),
            )
            task_weight = float(training[f"{branch}_task_weight"])
            bev_loss = bev_loss + task_weight * branch_bev["loss"]
            values.update(
                {
                    f"{branch}_bev_{key}": value
                    for key, value in branch_bev.items()
                }
            )
            region_denominator = max(
                float(
                    loss_weights.observed_region
                    + loss_weights.guessed_region
                ),
                1e-6,
            )
            direct_bev_objective = direct_bev_objective + (
                float(training["bev_loss_weight"])
                * task_weight
                * (
                    float(loss_weights.observed_region)
                    / region_denominator
                    * branch_bev["observed_direct_loss"]
                    + branch_bev["direct_confidence_loss"]
                )
            )
            guessed_bev_objective = guessed_bev_objective + (
                float(training["bev_loss_weight"])
                * task_weight
                * (
                    float(loss_weights.guessed_region)
                    * float(guessed_completion_scale)
                    / region_denominator
                    * branch_bev["guessed_completion_loss"]
                    + branch_bev["guessed_confidence_loss"]
                )
            )
        values["bev_loss"] = bev_loss
        values["direct_bev_objective"] = direct_bev_objective
        values["guessed_bev_objective"] = guessed_bev_objective
        values["evidence_regularizer_scale"] = torch.tensor(
            evidence_regularizer_scale,
            device=batch["images"].device,
        )
    else:
        bev_loss = zero

    scale_target = None
    if stage in ("scale_only", "joint"):
        if geometry is None:
            raise RuntimeError("scale stage requires training-only VGGT depth")
        scale_target = _build_scale_target(batch, geometry, fit_config)
        scale = metric_scale_losses(prediction["scale"], scale_target)
        values.update({f"scale_{key}": value for key, value in scale.items()})
        valid = scale_target["target_valid"]

        def finite_mean(name: str) -> torch.Tensor:
            value = scale_target[name].float()
            selected = value[torch.isfinite(value)]
            return selected.mean() if selected.numel() else zero

        def supervised_mean(name: str) -> torch.Tensor:
            value = scale_target[name].float()
            selected = value[valid & torch.isfinite(value)]
            return selected.mean() if selected.numel() else zero

        values.update(
            {
                "scale_target_quality_mean": finite_mean("quality_weight"),
                "scale_target_supervised_quality_mean": supervised_mean(
                    "quality_weight"
                ),
                "scale_target_depth_residual": supervised_mean(
                    "depth_alignment_residual"
                ),
                "scale_target_inlier_ratio": finite_mean("inlier_ratio"),
                "scale_target_depth_range_coverage": finite_mean(
                    "depth_range_coverage"
                ),
                "scale_target_frame_agreement": finite_mean(
                    "frame_scale_agreement"
                ),
                "scale_target_valid_pixels": finite_mean("valid_pixel_count"),
            }
        )
    else:
        scale = {"scale": zero, "depth_scale": zero, "uncertainty": zero}

    total = (
        float(training["bev_loss_weight"]) * bev_loss
        + float(training["scale_loss_weight"]) * scale["scale"]
        + float(training["depth_scale_loss_weight"]) * scale["depth_scale"]
        + float(training["uncertainty_loss_weight"]) * scale["uncertainty"]
    )
    values["loss"] = total
    return total, values, scale_target


@torch.no_grad()
def validate(
    model: Method1System,
    loader: DataLoader,
    *,
    config: dict,
    device: torch.device,
    maximum_batches: int,
    checkpoint_hash: str,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    fit_config = scale_fit_config(config)
    stage = config["training"]["stage"]
    enabled_bev_branches = _enabled_bev_branches(config["training"])
    cache_config = config.get("teacher_cache", {"mode": "live"})
    cache_mode = str(cache_config.get("mode", "live"))
    cache = (
        TeacherCache(
            cache_config["root"],
            checkpoint_sha256=checkpoint_hash,
            preprocessing_version="rgb-depth-resize-pad-v2",
        )
        if cache_mode != "live"
        else None
    )
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        batch = move_batch(batch, device)
        extraction, geometry = _teacher_inputs(
            model,
            batch,
            cache=cache,
            cache_mode=cache_mode,
            need_geometry=stage in ("scale_only", "joint"),
        )
        prediction = model.forward_head(
            extraction,
            enabled_bev_branches=(
                enabled_bev_branches
                if stage in ("bev_only", "joint")
                else ()
            ),
            include_scale=stage in ("scale_only", "joint"),
        )
        _, values, target = _step_losses(
            prediction,
            batch,
            geometry,
            config,
            fit_config,
            evidence_regularizer_scale=1.0,
            guessed_completion_scale=(
                1.0
                if bool(
                    config["training"].get(
                        "train_guessed_completion",
                        True,
                    )
                )
                else 0.0
            ),
        )
        if stage in ("bev_only", "joint"):
            for branch in enabled_bev_branches:
                extent = torch.full(
                    (batch["images"].shape[0],),
                    float(config["data"][f"{branch}_bev_extent_m"]),
                    device=device,
                )
                values.update(
                    {
                        f"{branch}_bev_{key}": value
                        for key, value in fov_complete_evidential_metrics(
                            prediction[f"{branch}_bev"],
                            batch[f"{branch}_fov_complete_target"],
                            batch[f"{branch}_visible_target"],
                            batch[f"{branch}_fov_support_target"],
                            target_extent_m=extent,
                            surface_tolerance_pixels=(
                                _surface_tolerance_pixels(config, branch)
                            ),
                        ).items()
                    }
                )
        if target is not None:
            values.update(metric_scale_metrics(prediction["scale"], target))
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    model.train()
    totals, count = _reduce_validation_totals(totals, count, device=device)
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["training"]["seed"]))
    train_dataset, validation_dataset = build_datasets(
        config,
        verify_manifest=not args.smoke_first_sample,
    )
    if args.smoke_first_sample:
        train_dataset = smoke_subset(train_dataset, 1)
        validation_dataset = smoke_subset(validation_dataset, 1)
    if args.data_only:
        train_base = _base_dataset(train_dataset)
        validation_base = _base_dataset(validation_dataset)
        print(
            json.dumps(
                {
                    "pipeline": "p1b_fov_complete_metric_scale",
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "scene_overlap": len(
                        train_base.scene_keys.intersection(
                            validation_base.scene_keys
                        )
                    ),
                    "runtime_inputs": ["RGB window"],
                    "training_only_labels": [
                        "metric z-depth",
                        "camera horizontal FOV and GT planar poses",
                        "source complete and visibility-masked BEVs",
                        "online 6.5m single FOV-complete target",
                        "online 10m merged FOV-union-complete target",
                    ],
                },
                indent=2,
            )
        )
        return

    training = config["training"]
    enabled_bev_branches = _enabled_bev_branches(training)
    distributed, rank, world_size, _, device, preflight = distributed_runtime(training)
    primary = rank == 0
    model = build_model(config, device)
    _set_training_stage(
        model,
        str(training["stage"]),
        enabled_bev_branches,
    )
    head_compile_settings = _configure_head_compilation(model, training)
    if distributed:
        model.head = DistributedDataParallel(
            model.head,
            device_ids=[device.index],
            bucket_cap_mb=float(training.get("ddp_bucket_cap_mb", 25.0)),
        )
    trainable = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    unwrapped_head = model.unwrapped_head()
    bev_modules = [unwrapped_head.token_projector]
    bev_modules.extend(
        getattr(unwrapped_head, f"{branch}_bev_decoder")
        for branch in enabled_bev_branches
    )
    bev_trainable = [
        parameter
        for module in bev_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    pcgrad_enabled = (
        bool(training["direct_priority_pcgrad"])
        and bool(training["train_guessed_completion"])
        and training["stage"] in ("bev_only", "joint")
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    sampler = EpochOffsetSampler(
        _sampler(
            train_dataset,
            config,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        sampler=sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=method1_collate,
    )
    validation_sampler = StratifiedValidationSampler(
        validation_dataset,
        maximum_samples=int(training["validation_batches"]),
        seed=int(training["seed"]),
        num_replicas=world_size if distributed else 1,
        rank=rank if distributed else 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        sampler=validation_sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=method1_collate,
    )
    total_steps = max(1, len(loader) * int(training["epochs"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(step, total_steps, training),
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and not use_bf16,
    )
    manifest_sha256 = _base_dataset(train_dataset).split_manifest_sha256
    contract = _resume_contract(config, manifest_sha256)
    start_epoch = 0
    global_step = 0
    resume_batch_in_epoch: int | None = 0
    if args.resume:
        start_epoch, global_step, resume_batch_in_epoch = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            contract=contract,
        )
    full_steps_per_epoch = total_steps // int(training["epochs"])
    if resume_batch_in_epoch is None:
        # Format-14 checkpoints written before batch offsets were recorded can
        # still resume exactly because every rank has a deterministic sampler.
        resume_batch_in_epoch = global_step - start_epoch * full_steps_per_epoch
    if not 0 <= resume_batch_in_epoch <= full_steps_per_epoch:
        raise ValueError("resume batch offset is outside the checkpoint epoch")
    checkpoint_hash = _checkpoint_sha256(config["model"]["checkpoint"])
    fit_config = scale_fit_config(config)
    cache_config = config.get("teacher_cache", {"mode": "live"})
    cache_mode = str(cache_config.get("mode", "live"))
    teacher_cache = (
        TeacherCache(
            cache_config["root"],
            checkpoint_sha256=checkpoint_hash,
            preprocessing_version="rgb-depth-resize-pad-v2",
        )
        if cache_mode != "live"
        else None
    )
    output_dir = Path(training["output_dir"]).expanduser().resolve()
    best_checkpoint_scores: dict[str, float] = {}
    if primary:
        output_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_scores = _load_best_checkpoint_scores(output_dir)
        print(
            json.dumps(
                {
                    "pipeline": "p1b_fov_complete_metric_scale",
                    "stage": training["stage"],
                    "enabled_bev_branches": enabled_bev_branches,
                    "scale_enabled": (
                        training["stage"] in ("scale_only", "joint")
                    ),
                    "distributed": distributed,
                    "world_size": world_size,
                    "nccl": preflight,
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "sampling_mode": config["data"].get(
                        "sampling_mode",
                        "all_prefixes",
                    ),
                    "steps_per_epoch": len(loader),
                    "epochs": int(training["epochs"]),
                    "total_steps": total_steps,
                    "learning_rate": float(training["learning_rate"]),
                    "minimum_learning_rate": float(
                        training.get("minimum_learning_rate", 0.0)
                    ),
                    "warmup_fraction": float(training["warmup_fraction"]),
                    "vggt_checkpoint_sha256": checkpoint_hash,
                    "runtime_geometry_heads": False,
                    "head_compile": head_compile_settings,
                    "cross_query_chunk_size": int(
                        config["model"]["cross_query_chunk_size"]
                    ),
                    "checkpoint_every_steps": int(
                        training["checkpoint_every_steps"]
                    ),
                    "resumed_from": (
                        None if args.resume is None else str(args.resume)
                    ),
                    "resume_global_step": global_step,
                }
            ),
            flush=True,
        )

    model.train()
    training_started = time.monotonic()
    stop = False
    batch_in_epoch = resume_batch_in_epoch
    last_validation: dict[str, float] | None = None
    for epoch in range(start_epoch, int(training["epochs"])):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        batch_offset = batch_in_epoch if epoch == start_epoch else 0
        sampler.set_start_index(batch_offset * int(training["batch_size"]))
        for batch_index, batch in enumerate(loader, start=batch_offset):
            batch_started = time.monotonic()
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            extraction, geometry = _teacher_inputs(
                model,
                batch,
                cache=teacher_cache,
                cache_mode=cache_mode,
                need_geometry=training["stage"] in ("scale_only", "joint"),
            )
            synchronization = (
                model.head.no_sync()
                if distributed and pcgrad_enabled
                else nullcontext()
            )
            with synchronization:
                with torch.autocast(
                    device_type=device.type,
                    dtype=(
                        torch.bfloat16
                        if use_bf16
                        else torch.float16
                    ),
                    enabled=device.type == "cuda",
                ):
                    prediction = model.forward_head(
                        extraction,
                        enabled_bev_branches=(
                            enabled_bev_branches
                            if training["stage"] in ("bev_only", "joint")
                            else ()
                        ),
                        include_scale=(
                            training["stage"] in ("scale_only", "joint")
                        ),
                    )
                loss, values, _ = _step_losses(
                    prediction,
                    batch,
                    geometry,
                    config,
                    fit_config,
                    evidence_regularizer_scale=confidence_regularizer_scale(
                        epoch,
                        training,
                    ),
                    guessed_completion_scale=guessed_supervision_scale(
                        global_step,
                        total_steps,
                        training,
                    ),
                )
                correction = None
                if pcgrad_enabled:
                    correction, pcgrad_values = (
                        _direct_priority_gradient_correction(
                            values["direct_bev_objective"],
                            values["guessed_bev_objective"],
                            bev_trainable,
                            maximum_guessed_to_direct_gradient_ratio=float(
                                training[
                                    "maximum_guessed_to_direct_gradient_ratio"
                                ]
                            ),
                        )
                    )
                    values.update(pcgrad_values)
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if pcgrad_enabled:
                if distributed:
                    _all_reduce_gradients(trainable)
                if correction is None:
                    raise RuntimeError("PCGrad correction was not constructed")
                _apply_gradient_correction(bev_trainable, correction)
            torch.nn.utils.clip_grad_norm_(
                trainable,
                float(training["gradient_clip_norm"]),
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            batch_in_epoch = batch_index + 1
            if primary and global_step % int(training["log_every_steps"]) == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "batch_seconds": time.monotonic() - batch_started,
                            "elapsed_seconds": (
                                time.monotonic() - training_started
                            ),
                            "learning_rate": scheduler.get_last_lr()[0],
                            **{
                                key: float(value.detach().cpu())
                                for key, value in values.items()
                            },
                        }
                    ),
                    flush=True,
                )
            if (
                primary
                and global_step % int(training["checkpoint_every_steps"]) == 0
            ):
                save_checkpoint(
                    output_dir / f"p1b_metric_step_{global_step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    manifest_sha256=manifest_sha256,
                    checkpoint_hash=checkpoint_hash,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    global_step=global_step,
                )
            if args.max_train_steps and global_step >= args.max_train_steps:
                stop = True
                break
        if not args.skip_validation:
            if dist.is_initialized():
                dist.barrier()
            validation = validate(
                model,
                validation_loader,
                config=config,
                device=device,
                maximum_batches=_validation_batches_per_rank(
                    len(validation_sampler) * world_size,
                    world_size,
                ),
                checkpoint_hash=checkpoint_hash,
            )
            last_validation = validation
            if primary:
                print(
                    json.dumps({"epoch": epoch, "validation": validation}),
                    flush=True,
                )
                scores = validation_checkpoint_scores(validation)
                for name, score in scores.items():
                    if score >= best_checkpoint_scores.get(name, math.inf):
                        continue
                    best_checkpoint_scores[name] = score
                    save_checkpoint(
                        output_dir / f"{name}.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config=config,
                        manifest_sha256=manifest_sha256,
                        checkpoint_hash=checkpoint_hash,
                        epoch=epoch,
                        batch_in_epoch=batch_in_epoch,
                        global_step=global_step,
                        validation_metrics=validation,
                        selection_name=name,
                        selection_score=score,
                    )
                _save_best_checkpoint_scores(
                    output_dir,
                    best_checkpoint_scores,
                )
            if dist.is_initialized():
                dist.barrier()
        if stop:
            break
        if epoch + 1 < int(training["epochs"]):
            batch_in_epoch = 0
    if primary:
        save_checkpoint(
            output_dir / "p1b_metric_latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            manifest_sha256=manifest_sha256,
            checkpoint_hash=checkpoint_hash,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            global_step=global_step,
            validation_metrics=last_validation,
        )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
