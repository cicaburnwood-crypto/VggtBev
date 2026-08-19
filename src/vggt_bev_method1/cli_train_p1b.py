from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from vggt_bev_method1.cli_train_metric import (
    _base_dataset,
    _build_scale_target,
    _checkpoint_sha256,
    _configure_head_compilation,
    _enabled_bev_branches,
    _sampler,
    _teacher_inputs,
    learning_rate_factor,
    scale_fit_config,
)
from vggt_bev_method1.data import method1_collate
from vggt_bev_method1.data.p1b_targets import p1b_region_masks
from vggt_bev_method1.data.void_coverage import FINAL_GT_VOID_FILTER
from vggt_bev_method1.models import (
    LiveVGGTOmegaAdapter,
    P1BSystem,
    P1CSystem,
    branch_specific_projector_state_dict,
    metric_scale_losses,
    metric_scale_metrics,
)
from vggt_bev_method1.p1c_losses import (
    relative_se2_metric_totals,
    relative_se2_pose_losses,
)
from vggt_bev_method1.p1b_config import load_p1b_config
from vggt_bev_method1.p1b_losses import (
    P1BLossWeights,
    hidden_occupied_supervision_weight,
    p1b_bev_loss,
    p1b_fov_support_loss,
    p1b_routing_geometry_loss,
    wrong_evidence_kl_weight,
)
from vggt_bev_method1.p1b_metrics import (
    finalize_p1b_metrics,
    p1b_metric_totals,
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
    EpochOffsetSampler,
    StratifiedValidationSampler,
)

FORMAT_VERSION = 24
SCHEMAS = {
    "evidential": "p1b-branch-projectors-evidential-v7",
    "bce": "p1b-branch-projectors-bce-v7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train P1B or geometry-aware P1C BEV heads"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--smoke-first-sample", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        help="Load a compatible head checkpoint, run validation, and exit.",
    )
    arguments = parser.parse_args()
    if arguments.resume is not None and arguments.eval_checkpoint is not None:
        parser.error("--resume and --eval-checkpoint are mutually exclusive")
    return arguments


def build_model(config: dict, device: torch.device) -> P1BSystem | P1CSystem:
    values = config["model"]
    layers = tuple(int(value) for value in values["cached_layers"])
    adapter = LiveVGGTOmegaAdapter(
        values["vggt_source"],
        values["checkpoint"],
        device=device,
        patch_size=int(values["patch_size"]),
        cached_layers=layers,
    )
    system_class = (
        P1CSystem
        if values.get("pipeline_variant") == "P1C-NLL"
        else P1BSystem
    )
    p1c_arguments = (
        {
            "pose_hidden_dim": int(values["pose_hidden_dim"]),
            "pose_attention_heads": int(values["pose_attention_heads"]),
            "pose_layers": int(values["pose_layers"]),
            "pose_refinements": int(values["pose_refinements"]),
            "maximum_history": int(values["maximum_history"]),
            "pose_conditioning_detach": bool(
                values.get("pose_conditioning_detach", False)
            ),
        }
        if system_class is P1CSystem
        else {}
    )
    return system_class(
        adapter,
        probability_model=str(values["probability_model"]),
        cached_layers=layers,
        spatial_scales=tuple(float(value) for value in values["spatial_scales"]),
        vggt_token_dim=int(values["vggt_token_dim"]),
        hidden_dim=int(values["hidden_dim"]),
        heads=int(values["attention_heads"]),
        decoder_layers=int(values["decoder_layers"]),
        scale_decoder_layers=int(values["scale_decoder_layers"]),
        self_attention_mode=str(values["self_attention_mode"]),
        cross_attention_mode=str(values["cross_attention_mode"]),
        deformable_samples=int(values["deformable_samples"]),
        cross_query_chunk_size=int(values["cross_query_chunk_size"]),
        single_latent_bev_size=int(values["single_latent_bev_size"]),
        merged_latent_bev_size=int(values["merged_latent_bev_size"]),
        single_output_size=int(values["single_bev_output_size"]),
        merged_output_size=int(values["merged_bev_output_size"]),
        single_bev_extent_m=float(values["single_bev_extent_m"]),
        merged_bev_extent_m=float(values["merged_bev_extent_m"]),
        predict_scale_uncertainty=bool(values.get("predict_scale_uncertainty", True)),
        **p1c_arguments,
    ).to(device)


def _set_stage(
    model: P1BSystem | P1CSystem,
    stage: str,
    enabled: tuple[str, ...],
    bev_objective: str = "full",
) -> None:
    head = model.unwrapped_head()
    for parameter in head.parameters():
        parameter.requires_grad = False
    # ``hasattr`` is used instead of coupling the P1B training utilities to a
    # concrete subclass through their public signatures.
    is_p1c = hasattr(head, "relative_pose_head")
    if stage in ("bev_only", "joint"):
        if bev_objective == "pose_only":
            if not is_p1c or enabled != ("merged",):
                raise ValueError("pose_only requires the P1C Merged branch")
            for parameter in head.relative_pose_head.parameters():
                parameter.requires_grad = True
            return
        if bev_objective in (
            "fov_support_only",
            "fov_support_and_observed_gate",
        ):
            if enabled != ("merged",):
                raise ValueError("FOV-support-only stage requires Merged only")
            for module in (
                head.merged_routing_token_projector,
                head.merged_bev_decoder.routing,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True
            if is_p1c:
                for module in (
                    head.relative_pose_head,
                    head.merged_pose_embedding,
                ):
                    for parameter in module.parameters():
                        parameter.requires_grad = True
            return
        modules = []
        for branch in enabled:
            modules.extend(
                (
                    getattr(head, f"{branch}_guessed_token_projector"),
                    getattr(head, f"{branch}_routing_token_projector"),
                    getattr(head, f"{branch}_bev_decoder"),
                )
            )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        if is_p1c and "merged" in enabled:
            for module in (
                head.relative_pose_head,
                head.merged_pose_embedding,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = True
    if stage in ("scale_only", "joint"):
        for module in (head.scale_token_projector, head.scale_decoder):
            for parameter in module.parameters():
                parameter.requires_grad = True


def initialize_frozen_single_baseline(
    model: P1BSystem,
    checkpoint: str | Path,
    *,
    seed_merged_routing_from_single: bool = True,
) -> dict[str, int | str]:
    """Load only the trained Single/Scale path and seed Merged projectors.

    The historical baseline's Merged decoder is intentionally ignored because
    it was never trained and may have an incompatible latent query size. The
    new Merged projectors start from copies of the trained Single projectors,
    then become independently trainable.
    """

    resolved = Path(checkpoint).expanduser().resolve()
    state = torch.load(resolved, map_location="cpu", weights_only=False)
    source_probability = str(
        state.get("probability_model")
        or state.get("config", {}).get("model", {}).get("probability_model", "")
    )
    target_probability = model.unwrapped_head().probability_model
    if source_probability != target_probability:
        raise ValueError(
            "Single baseline probability model does not match target P1B model"
        )
    source = branch_specific_projector_state_dict(state["head"])
    head = model.unwrapped_head()
    initialized = head.state_dict()
    preserved_prefixes = (
        "single_guessed_token_projector.",
        "single_routing_token_projector.",
        "single_bev_decoder.",
        "scale_token_projector.",
        "scale_decoder.",
    )
    initialized_count = 0
    for key, target in tuple(initialized.items()):
        if not key.startswith(preserved_prefixes):
            continue
        if key not in source:
            raise ValueError(f"Single baseline is missing required tensor {key}")
        value = source[key]
        if value.shape != target.shape:
            raise ValueError(
                f"Single baseline tensor shape mismatch for {key}: "
                f"{tuple(value.shape)} != {tuple(target.shape)}"
            )
        initialized[key] = value
        initialized_count += 1
    projector_pairs = [
        (
            "single_guessed_token_projector.",
            "merged_guessed_token_projector.",
        ),
    ]
    if seed_merged_routing_from_single:
        projector_pairs.append(
            (
                "single_routing_token_projector.",
                "merged_routing_token_projector.",
            )
        )
    merged_projector_count = 0
    for source_prefix, target_prefix in projector_pairs:
        for target_key, target in tuple(initialized.items()):
            if not target_key.startswith(target_prefix):
                continue
            source_key = source_prefix + target_key.removeprefix(target_prefix)
            if source_key not in source:
                raise ValueError(
                    f"Single baseline is missing projector tensor {source_key}"
                )
            value = source[source_key]
            if value.shape != target.shape:
                raise ValueError(
                    f"Merged projector initialization shape mismatch for "
                    f"{target_key}"
                )
            initialized[target_key] = value.clone()
            merged_projector_count += 1
    head.load_state_dict(initialized, strict=True)
    return {
        "checkpoint": str(resolved),
        "checkpoint_sha256": _checkpoint_sha256(resolved),
        "source_epoch": int(state.get("epoch", -1)),
        "source_global_step": int(state.get("global_step", -1)),
        "preserved_tensor_count": initialized_count,
        "seeded_merged_projector_tensor_count": merged_projector_count,
        "seeded_merged_routing_from_single": seed_merged_routing_from_single,
    }


def _resize_square_query_content(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Resize a learned square query grid without interpolating metric buffers."""

    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("routing query tensors must be [square_cells, hidden_dim]")
    source_size = math.isqrt(int(source.shape[0]))
    target_size = math.isqrt(int(target.shape[0]))
    if source_size * source_size != source.shape[0]:
        raise ValueError("source routing query count is not square")
    if target_size * target_size != target.shape[0]:
        raise ValueError("target routing query count is not square")
    values = source.to(dtype=torch.float32).transpose(0, 1).reshape(
        1, source.shape[1], source_size, source_size
    )
    resized = F.interpolate(
        values,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(source.shape[1], -1).transpose(0, 1).to(
        dtype=target.dtype
    )


def initialize_merged_routing_warmstart(
    model: P1BSystem,
    checkpoint: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, int | str]:
    """Warm-start only trained Merged routing into a new native query grid."""

    resolved = Path(checkpoint).expanduser().resolve()
    state = torch.load(resolved, map_location="cpu", weights_only=False)
    if state.get("trained_outputs") != [
        "merged_fov_support",
        "merged_observed_gate",
    ]:
        raise ValueError("routing warm-start checkpoint has invalid trained outputs")
    if state.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("routing warm-start manifest does not match current data")
    source_probability = str(state.get("probability_model", ""))
    if source_probability != model.unwrapped_head().probability_model:
        raise ValueError("routing warm-start probability model does not match")

    source = branch_specific_projector_state_dict(state["head"])
    head = model.unwrapped_head()
    initialized = head.state_dict()
    prefixes = (
        "merged_routing_token_projector.",
        "merged_bev_decoder.routing.",
    )
    geometry_buffers = (
        "merged_bev_decoder.routing.metric_coordinates_m",
        "merged_bev_decoder.routing.reference_grid",
    )
    copied = 0
    resized = 0
    preserved_geometry = 0
    for key, target in tuple(initialized.items()):
        if not key.startswith(prefixes):
            continue
        if key in geometry_buffers:
            preserved_geometry += 1
            continue
        if key not in source:
            raise ValueError(f"routing warm-start is missing tensor {key}")
        value = source[key]
        if value.shape == target.shape:
            initialized[key] = value.to(dtype=target.dtype)
            copied += 1
            continue
        if key == "merged_bev_decoder.routing.query_content":
            initialized[key] = _resize_square_query_content(value, target)
            resized += 1
            continue
        raise ValueError(
            f"routing warm-start tensor shape mismatch for {key}: "
            f"{tuple(value.shape)} != {tuple(target.shape)}"
        )
    head.load_state_dict(initialized, strict=True)
    return {
        "checkpoint": str(resolved),
        "checkpoint_sha256": _checkpoint_sha256(resolved),
        "source_schema": str(state.get("checkpoint_schema", "")),
        "source_epoch": int(state.get("epoch", -1)),
        "source_global_step": int(state.get("global_step", -1)),
        "copied_tensor_count": copied,
        "resized_query_tensor_count": resized,
        "preserved_metric_buffer_count": preserved_geometry,
        "query_migration": "bilinear-query-content-preserve-native-metric-grid-v1",
    }


def _configure_attention_recomputation(
    model: P1BSystem,
    training: dict,
    enabled: tuple[str, ...],
) -> dict[str, dict[str, int | float | bool]]:
    """Configure execution-only rematerialization per BEV branch.

    This changes neither parameters nor forward values. The Single branch is
    small enough to retain activations, while the larger Merged branch keeps
    bounded-memory recomputation.
    """

    head = model.unwrapped_head()
    settings: dict[str, dict[str, int | float | bool]] = {}
    for branch in enabled:
        enabled_for_branch = bool(
            training.get(f"{branch}_attention_checkpoint", True)
        )
        checkpoint_layers = training.get(
            f"{branch}_attention_checkpoint_layers"
        )
        if checkpoint_layers is not None:
            checkpoint_layers = int(checkpoint_layers)
            if checkpoint_layers < 0:
                raise ValueError("attention checkpoint layer count cannot be negative")
        checkpoint_fraction = float(
            training.get(f"{branch}_attention_checkpoint_fraction", 1.0)
        )
        if not 0.0 <= checkpoint_fraction <= 1.0:
            raise ValueError("attention checkpoint fraction must be in [0, 1]")
        module_count = 0
        active_module_count = 0
        decoder = getattr(head, f"{branch}_bev_decoder")
        for module_name, module in decoder.named_modules():
            if hasattr(module, "memory_efficient_training"):
                module_enabled = enabled_for_branch
                if checkpoint_layers is not None:
                    path = module_name.split(".")
                    try:
                        block_index = int(path[path.index("blocks") + 1])
                    except (ValueError, IndexError):
                        block_index = 0
                    module_enabled = module_enabled and block_index < checkpoint_layers
                module.memory_efficient_training = module_enabled
                if hasattr(module, "memory_efficient_checkpoint_fraction"):
                    module.memory_efficient_checkpoint_fraction = checkpoint_fraction
                module_count += 1
                active_module_count += int(module_enabled)
        settings[branch] = {
            "enabled": active_module_count > 0,
            "modules": module_count,
            "active_modules": active_module_count,
            "checkpoint_layers_per_decoder": (
                checkpoint_layers if checkpoint_layers is not None else -1
            ),
            "checkpoint_fraction": checkpoint_fraction,
        }
    return settings


def _loss_weights(training: dict) -> P1BLossWeights:
    return P1BLossWeights(
        observed_gate_pixel=float(training.get("observed_gate_pixel_weight", 1.0)),
        guessed_pixel=float(training.get("guessed_pixel_weight", 1.0)),
        guessed_surface=float(training.get("guessed_surface_weight", 0.5)),
        guessed_free=float(training.get("guessed_free_weight", 0.35)),
        guessed_visible_surface=float(
            training.get("guessed_visible_surface_weight", 0.40)
        ),
        guessed_hidden_occupied=float(
            training.get("guessed_hidden_occupied_weight", 0.25)
        ),
        wrong_evidence_kl=float(
            training.get(
                "wrong_evidence_kl_weight",
                0.005 if training.get("pipeline") == "P1B-NLL" else 0.0,
            )
        ),
        support_bce=float(training.get("support_bce_weight", 0.5)),
        support_dice=float(training.get("support_dice_weight", 0.5)),
    )


def step_losses(
    prediction: dict,
    batch: dict,
    geometry: dict | None,
    config: dict,
    *,
    global_step: int,
    total_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict | None]:
    training = config["training"]
    stage = str(training["stage"])
    branches = _enabled_bev_branches(training)
    probability_model = str(config["model"]["probability_model"])
    bev_objective = str(training.get("bev_objective", "full"))
    if "pose" in prediction:
        zero = prediction["pose"]["relative_pose"].sum() * 0.0
    elif branches and f"{branches[0]}_bev" in prediction:
        branch_prediction = prediction[f"{branches[0]}_bev"]
        zero_source = (
            branch_prediction["fov_support_logit"]
            if bev_objective != "full"
            else branch_prediction["observed_gate_logit"]
        )
        zero = zero_source.sum() * 0.0
    else:
        zero = prediction["scale"]["log_lambda_m_per_vggt"].sum() * 0.0
    values: dict[str, torch.Tensor] = {}
    bev_total = zero
    if stage in ("bev_only", "joint"):
        if bev_objective == "pose_only":
            values["bev_loss"] = zero
        elif bev_objective in (
            "fov_support_only",
            "fov_support_and_observed_gate",
        ):
            if branches != ("merged",):
                raise ValueError("routing-geometry loss requires Merged only")
            support_arguments = {
                "variant": str(training["support_loss_variant"]),
                "bce_weight": float(training.get("support_bce_weight", 0.65)),
                "dice_weight": float(training.get("support_dice_weight", 0.35)),
                "tversky_weight": float(
                    training.get("support_tversky_weight", 0.0)
                ),
                "boundary_weight": float(
                    training.get("support_boundary_weight", 0.0)
                ),
                "boundary_radius": int(
                    training.get("support_boundary_radius", 2)
                ),
                "interior_weight": float(
                    training.get("support_interior_weight", 0.25)
                ),
                "edge_weight": float(
                    training.get("support_edge_weight", 0.45)
                ),
                "contour_weight": float(
                    training.get("support_contour_weight", 0.20)
                ),
                "region_dice_weight": float(
                    training.get("support_region_dice_weight", 0.10)
                ),
                "tversky_false_positive_weight": float(
                    training.get(
                        "support_tversky_false_positive_weight", 0.70
                    )
                ),
                "tversky_false_negative_weight": float(
                    training.get(
                        "support_tversky_false_negative_weight", 0.30
                    )
                ),
            }
            if bev_objective == "fov_support_and_observed_gate":
                gate_start_fraction = float(
                    training.get("observed_gate_start_fraction", 0.0)
                )
                gate_schedule_scale = float(
                    float(global_step) / float(max(total_steps, 1))
                    >= gate_start_fraction
                )
                configured_gate_weight = float(
                    training["observed_gate_pixel_weight"]
                )
                routing_loss = p1b_routing_geometry_loss(
                    prediction["merged_bev"],
                    batch["merged_fov_complete_target"],
                    batch["merged_visible_target"],
                    batch["merged_fov_support_target"],
                    gt_valid_mask=batch.get("merged_gt_valid_mask"),
                    observed_gate_weight=(
                        configured_gate_weight * gate_schedule_scale
                    ),
                    support_weight=float(training.get("support_task_weight", 1.0)),
                    gate_loss_variant=str(
                        training.get("observed_gate_loss_variant", "balanced_bce")
                    ),
                    gate_boundary_radius=int(
                        training.get("observed_gate_boundary_radius", 3)
                    ),
                    gate_interior_weight=float(
                        training.get("observed_gate_interior_weight", 0.20)
                    ),
                    gate_edge_weight=float(
                        training.get("observed_gate_edge_weight", 0.50)
                    ),
                    gate_contour_weight=float(
                        training.get("observed_gate_contour_weight", 0.20)
                    ),
                    gate_dice_weight=float(
                        training.get("observed_gate_dice_weight", 0.10)
                    ),
                    **support_arguments,
                )
                routing_loss["observed_gate_schedule_scale"] = torch.tensor(
                    gate_schedule_scale,
                    device=prediction["merged_bev"]["fov_support_logit"].device,
                )
            else:
                routing_loss = p1b_fov_support_loss(
                    prediction["merged_bev"],
                    batch["merged_fov_support_target"],
                    gt_valid_mask=batch.get("merged_gt_valid_mask"),
                    **support_arguments,
                )
            bev_total = float(training["merged_task_weight"]) * routing_loss["loss"]
            values.update(
                {
                    f"merged_bev_{key}": value
                    for key, value in routing_loss.items()
                }
            )
            values["bev_loss"] = bev_total
        else:
            kl_scale = wrong_evidence_kl_weight(
                global_step,
                total_steps,
                maximum=1.0,
                zero_fraction=float(
                    training.get("wrong_evidence_zero_fraction", 0.20)
                ),
                ramp_fraction=float(
                    training.get("wrong_evidence_ramp_fraction", 0.10)
                ),
            )
            hidden_occupied_scale = hidden_occupied_supervision_weight(
                global_step,
                total_steps,
                zero_fraction=float(
                    training.get("hidden_occupied_zero_fraction", 0.10)
                ),
                ramp_fraction=float(
                    training.get("hidden_occupied_ramp_fraction", 0.15)
                ),
            )
            for branch in branches:
                branch_loss = p1b_bev_loss(
                    prediction[f"{branch}_bev"],
                    batch[f"{branch}_fov_complete_target"],
                    batch[f"{branch}_visible_target"],
                    batch[f"{branch}_fov_support_target"],
                    gt_valid_mask=batch[f"{branch}_gt_valid_mask"],
                    probability_model=probability_model,
                    weights=_loss_weights(training),
                    wrong_evidence_scale=kl_scale,
                    hidden_occupied_scale=hidden_occupied_scale,
                )
                task_weight = float(training[f"{branch}_task_weight"])
                bev_total = bev_total + task_weight * branch_loss["loss"]
                values.update(
                    {
                        f"{branch}_bev_{key}": value
                        for key, value in branch_loss.items()
                    }
                )
            values["bev_loss"] = bev_total

    scale_target = None
    if stage in ("scale_only", "joint"):
        if geometry is None:
            raise RuntimeError("scale supervision requires training-only geometry")
        scale_target = _build_scale_target(batch, geometry, scale_fit_config(config))
        scale = metric_scale_losses(prediction["scale"], scale_target)
        values.update({f"scale_{key}": value for key, value in scale.items()})
    else:
        scale = {"scale": zero, "depth_scale": zero, "uncertainty": zero}
    if "pose" in prediction:
        pose = relative_se2_pose_losses(
            prediction["pose"],
            batch["relative_pose_target"],
            translation_weight=float(
                training.get("pose_translation_weight", 1.0)
            ),
            yaw_weight=float(training.get("pose_yaw_weight", 0.5)),
            refinement_gamma=float(
                training.get("pose_refinement_gamma", 1.5)
            ),
            smooth_l1_beta_m=float(
                training.get("pose_smooth_l1_beta_m", 0.10)
            ),
        )
        values.update({f"pose_{key}": value for key, value in pose.items()})
        pose_total = pose["loss"]
    else:
        pose_total = zero
    total = (
        float(training["bev_loss_weight"]) * bev_total
        + float(training["scale_loss_weight"]) * scale["scale"]
        + float(training["depth_scale_loss_weight"]) * scale["depth_scale"]
        + float(training.get("uncertainty_loss_weight", 0.0)) * scale["uncertainty"]
        + float(training.get("pose_loss_weight", 0.0)) * pose_total
    )
    values["loss"] = total
    return total, values, scale_target


def checkpoint_contract(config: dict, manifest_sha256: str) -> dict:
    probability_model = str(config["model"]["probability_model"])
    bev_objective = str(config["training"].get("bev_objective", "full"))
    is_p1c = config["training"]["pipeline"] == "P1C-NLL"
    support_variant = str(
        config["training"].get("support_loss_variant", "balanced_bce_dice")
    )
    contract = {
        "format_version": (
            FORMAT_VERSION + 2
            if is_p1c
            else FORMAT_VERSION + (1 if bev_objective != "full" else 0)
        ),
        "checkpoint_schema": (
            f"p1c-relative-se2-{bev_objective}-{support_variant}-v1"
            if is_p1c
            else (
                f"p1b-merged-routing-geometry-{support_variant}-v1"
                if bev_objective != "full"
                else SCHEMAS[probability_model]
            )
        ),
        "pipeline_id": config["training"]["pipeline"],
        "probability_model": probability_model,
        "manifest_sha256": manifest_sha256,
        "runtime_inputs": ["rgb_window"],
        "bev_architecture": (
            "pose-conditioned-observed-free-gate-plus-guessed-completion"
            if is_p1c
            else "observed-free-gate-plus-guessed-binary-completion"
        ),
        "projector_layout": "branch-specific-single-merged-v1",
        "loss_contract": "legacy-balanced-gate-guessed-only-surface-v2",
        "gate_loss": "per-sample-1to1-observed-free-vs-all-guessed",
        "surface_gradient_scope": "guessed-expert-only",
        "surface_target": "visible-target-equals-occupied",
        "hidden_occupied_curriculum": "0:10pct-ramp:10to25pct-full:25pct",
        "routing_classes": [
            "observed_free",
            "guessed_free",
            "guessed_occupied",
        ],
        "single_output": [512, 512, 6.5],
        "merged_output": (
            [
                int(config["model"]["merged_bev_output_size"]),
                int(config["model"]["merged_bev_output_size"]),
                float(config["model"]["merged_bev_extent_m"]),
            ]
            if is_p1c
            else [800, 800, 10.0]
        ),
        "bev_objective": bev_objective,
    }
    if is_p1c:
        contract.update(
            {
                "geometry_conditioning": "relative_se2_embedding",
                "pose_runtime_source": "frozen-vggt-camera-register-tokens",
                "pose_training_target": "gt-metric-latest-from-frame-se2",
                "pose_representation": ["tx_m", "tz_m", "sin_yaw", "cos_yaw"],
                "pose_refinements": int(config["model"]["pose_refinements"]),
                "runtime_inputs": ["rgb_window"],
                "loss_contract": "metric-relative-se2-plus-p1b-routing-v1",
            }
        )
    if bev_objective != "full":
        contract.update(
            {
                "trained_outputs": (
                    ["relative_se2_pose"]
                    if bev_objective == "pose_only"
                    else (
                        [
                            "relative_se2_pose",
                            "merged_fov_support",
                            "merged_observed_gate",
                        ]
                        if is_p1c
                        and bev_objective
                        == "fov_support_and_observed_gate"
                        else (
                            ["merged_fov_support", "merged_observed_gate"]
                            if bev_objective
                            == "fov_support_and_observed_gate"
                            else ["merged_fov_support"]
                        )
                    )
                ),
                "invalid_untrained_outputs": [
                    "merged_guessed_semantic",
                    "merged_confidence",
                    *(
                        ["merged_fov_support", "merged_observed_gate"]
                        if bev_objective == "pose_only"
                        else (
                            ["merged_observed_gate"]
                            if bev_objective == "fov_support_only"
                            else []
                        )
                    ),
                ],
                "trainable_scope": (
                    "relative-pose-head-only"
                    if bev_objective == "pose_only"
                    else (
                        "relative-pose-plus-merged-routing"
                        if is_p1c
                        else "merged-routing-projector-and-routing-decoder-only"
                    )
                ),
                "support_loss_variant": support_variant,
                "routing_initialization": (
                    "fresh-random"
                    if bool(
                        config["training"].get(
                            "fresh_merged_routing_initialization", False
                        )
                    )
                    else "configured"
                ),
            }
        )
    if bool(config["training"].get("freeze_single_to_baseline", False)):
        baseline = Path(
            config["training"]["single_baseline_checkpoint"]
        ).expanduser().resolve()
        contract.update(
            {
                "single_branch_source": "frozen-baseline",
                "single_baseline_sha256": _checkpoint_sha256(baseline),
                "merged_branch_source": "trainable-independent-projectors",
            }
        )
    else:
        contract["single_branch_source"] = "configured-training-stage"
    routing_warmstart = str(
        config["training"].get("routing_warmstart_checkpoint", "")
    ).strip()
    if routing_warmstart:
        contract.update(
            {
                "routing_parent_checkpoint_sha256": _checkpoint_sha256(
                    Path(routing_warmstart).expanduser().resolve()
                ),
                "routing_query_migration": (
                    "bilinear-query-content-preserve-native-metric-grid-v1"
                ),
            }
        )
    void_index = config["data"].get("void_coverage_index")
    if bool(config["data"].get("require_gt_void_mask", False)) and not void_index:
        raise ValueError("this run requires a GT Void coverage index")
    if void_index:
        payload = json.loads(
            Path(void_index).expanduser().resolve().read_text(encoding="utf-8")
        )
        algorithm = str(payload["algorithm"])
        if bool(config["data"].get("require_gt_void_mask", False)):
            required_algorithm = str(
                config["data"]["required_void_coverage_algorithm"]
            )
            if algorithm != required_algorithm:
                raise ValueError("Void coverage algorithm does not match config")
            if bool(payload.get("partial", True)):
                raise ValueError("training refuses a partial Void coverage index")
            if int(payload.get("missing_floor_band_count", -1)) != 0:
                raise ValueError("training refuses missing Void floor bands")
        contract_version = algorithm.rsplit("-", maxsplit=1)[-1]
        if contract_version not in {"v1", "v2", "v3", "v4"}:
            raise ValueError("unsupported Void coverage contract version")
        loss_contract = {
            "v1": "scene-geometry-only-gt-validity-v1",
            "v2": "scene-geometry-only-gt-validity-v2",
            "v3": "scene-geometry-only-gt-validity-v6",
            "v4": "scene-geometry-only-gt-validity-v7",
        }[contract_version]
        contract.update(
            {
                "loss_contract": loss_contract,
                "gt_void_filter": (
                    FINAL_GT_VOID_FILTER
                    if contract_version in {"v3", "v4"}
                    else algorithm
                ),
                "gt_validity_scope": (
                    "all-bev-losses-independent-of-fov-and-masked-target"
                ),
                "complete_gt_contract": (
                    "semantic-target-plus-independent-gt-valid-mask"
                ),
                "void_is_model_output_class": False,
                "void_loss_semantics": "hard-ignore-every-bev-loss-term",
                "scene_coverage_algorithm": algorithm,
                "void_coverage_index_sha256": payload["content_sha256"],
            }
        )
    return contract


def save_checkpoint(
    path: Path,
    *,
    model: P1BSystem,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    contract: dict,
    epoch: int,
    global_step: int,
    batch_in_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **contract,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "batch_in_epoch": int(batch_in_epoch),
        "head": model.unwrapped_head().state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: P1BSystem,
    optimizer: torch.optim.Optimizer,
    scheduler,
    contract: dict,
) -> tuple[int, int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in contract.items():
        if state.get(key) != expected:
            raise ValueError(f"checkpoint contract mismatch for {key}")
    model.unwrapped_head().load_state_dict(state["head"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return (
        int(state["epoch"]),
        int(state["global_step"]),
        int(state["batch_in_epoch"]),
    )


def load_head_checkpoint(
    path: Path,
    *,
    model: P1BSystem,
    contract: dict,
) -> tuple[int, int]:
    """Load a checkpoint for read-only evaluation without optimizer state."""

    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in contract.items():
        if state.get(key) != expected:
            raise ValueError(f"checkpoint contract mismatch for {key}")
    model.unwrapped_head().load_state_dict(state["head"], strict=True)
    return int(state["epoch"]), int(state["global_step"])


def _reduce_dict(values: dict[str, float], device: torch.device) -> dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return values
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return dict(zip(keys, tensor.cpu().tolist(), strict=True))


def _history_metric_keys(
    branches: tuple[str, ...],
    *,
    maximum_history: int,
    include_gate: bool,
) -> dict[str, float]:
    """Predeclare fixed DDP keys for history-conditioned binary metrics."""

    output: dict[str, float] = {}
    roles = ("support", "observed_gate") if include_gate else ("support",)
    for branch in branches:
        for history in range(1, maximum_history + 1):
            prefix = f"{branch}_history_{history:02d}"
            output[f"{prefix}_samples"] = 0.0
            for role in roles:
                for count in ("tp", "fp", "fn", "tn"):
                    output[f"{prefix}_{role}_{count}"] = 0.0
    return output


def _binary_metric_summary(
    values: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    tp = values.get(f"{prefix}_tp", 0.0)
    fp = values.get(f"{prefix}_fp", 0.0)
    fn = values.get(f"{prefix}_fn", 0.0)
    tn = values.get(f"{prefix}_tn", 0.0)
    return {
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1.0),
    }


@torch.no_grad()
def validate(
    model: P1BSystem,
    loader: DataLoader,
    *,
    config: dict,
    device: torch.device,
    checkpoint_hash: str,
) -> dict[str, float]:
    model.eval()
    training = config["training"]
    stage = str(training["stage"])
    branches = _enabled_bev_branches(training)
    bev_objective = str(training.get("bev_objective", "full"))
    metric_totals: dict[str, float] = {}
    maximum_history = int(config["data"]["maximum_history"])
    if stage in ("bev_only", "joint") and bev_objective in (
        "fov_support_only",
        "fov_support_and_observed_gate",
    ):
        metric_totals.update(
            _history_metric_keys(
                branches,
                maximum_history=maximum_history,
                include_gate=(
                    bev_objective == "fov_support_and_observed_gate"
                ),
            )
        )
    if config["training"]["pipeline"] == "P1C-NLL":
        metric_totals.update(
            {
                "pose_translation_error_sum_m": 0.0,
                "pose_yaw_error_sum_rad": 0.0,
                "pose_pose_count": 0.0,
            }
        )
        for history in range(1, maximum_history + 1):
            prefix = f"pose_history_{history:02d}"
            metric_totals[f"{prefix}_translation_error_sum_m"] = 0.0
            metric_totals[f"{prefix}_yaw_error_sum_rad"] = 0.0
            metric_totals[f"{prefix}_pose_count"] = 0.0
    scalar_totals: dict[str, float] = {}
    batches = 0
    cache_values = config.get("teacher_cache", {"mode": "live"})
    cache_mode = str(cache_values.get("mode", "live"))
    cache = (
        TeacherCache(
            cache_values["root"],
            checkpoint_sha256=checkpoint_hash,
            preprocessing_version="rgb-depth-resize-pad-v2",
        )
        if cache_mode != "live"
        else None
    )
    for batch_index, batch in enumerate(loader):
        if batch_index >= int(training["validation_batches"]):
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
            enabled_bev_branches=branches if stage in ("bev_only", "joint") else (),
            include_scale=stage in ("scale_only", "joint"),
            bev_objective=bev_objective,
        )
        _, losses, scale_target = step_losses(
            prediction,
            batch,
            geometry,
            config,
            global_step=1,
            total_steps=1,
        )
        for key, value in losses.items():
            scalar_totals[key] = scalar_totals.get(key, 0.0) + float(value.cpu())
        if "pose" in prediction:
            pose_totals = relative_se2_metric_totals(
                prediction["pose"]["relative_pose"],
                batch["relative_pose_target"],
            )
            for key, value in pose_totals.items():
                metric_totals[f"pose_{key}"] += float(value.cpu())
            for sample_index, metadata in enumerate(batch["metadata"]):
                history = int(metadata["history_frame_count"])
                sample_totals = relative_se2_metric_totals(
                    prediction["pose"]["relative_pose"][
                        sample_index : sample_index + 1
                    ],
                    batch["relative_pose_target"][
                        sample_index : sample_index + 1
                    ],
                )
                prefix = f"pose_history_{history:02d}"
                for key, value in sample_totals.items():
                    metric_totals[f"{prefix}_{key}"] += float(value.cpu())
        if (
            stage in ("bev_only", "joint")
            and bev_objective != "pose_only"
        ):
            for branch in branches:
                if bev_objective != "full":
                    probability = prediction[f"{branch}_bev"][
                        "fov_support_probability"
                    ]
                    truth = batch[f"{branch}_fov_support_target"].bool()
                    valid = batch[f"{branch}_gt_valid_mask"].bool()
                    hard = probability >= 0.5
                    counts = {
                        "support_tp": (hard & truth & valid).sum(),
                        "support_fp": (hard & ~truth & valid).sum(),
                        "support_fn": (~hard & truth & valid).sum(),
                        "support_tn": (~hard & ~truth & valid).sum(),
                    }
                    for key, value in counts.items():
                        name = f"{branch}_{key}"
                        metric_totals[name] = metric_totals.get(name, 0.0) + float(
                            value.cpu()
                        )
                    for sample_index, metadata in enumerate(batch["metadata"]):
                        history = int(metadata["history_frame_count"])
                        if not 1 <= history <= maximum_history:
                            raise ValueError(
                                f"history length {history} lies outside 1.."
                                f"{maximum_history}"
                            )
                        prefix = f"{branch}_history_{history:02d}"
                        sample_valid = valid[sample_index]
                        sample_truth = truth[sample_index]
                        sample_hard = hard[sample_index]
                        history_counts = {
                            "support_tp": (
                                sample_hard & sample_truth & sample_valid
                            ).sum(),
                            "support_fp": (
                                sample_hard & ~sample_truth & sample_valid
                            ).sum(),
                            "support_fn": (
                                ~sample_hard & sample_truth & sample_valid
                            ).sum(),
                            "support_tn": (
                                ~sample_hard & ~sample_truth & sample_valid
                            ).sum(),
                        }
                        metric_totals[f"{prefix}_samples"] += 1.0
                        for key, value in history_counts.items():
                            metric_totals[f"{prefix}_{key}"] += float(
                                value.cpu()
                            )
                    if bev_objective == "fov_support_and_observed_gate":
                        masks = p1b_region_masks(
                            batch[f"{branch}_fov_complete_target"],
                            batch[f"{branch}_visible_target"],
                            batch[f"{branch}_fov_support_target"],
                        )
                        gate_truth = masks.observed_free
                        gate_domain = masks.valid & valid
                        gate_hard = prediction[f"{branch}_bev"][
                            "observed_gate_probability"
                        ] >= 0.5
                        gate_counts = {
                            "observed_gate_tp": (
                                gate_hard & gate_truth & gate_domain
                            ).sum(),
                            "observed_gate_fp": (
                                gate_hard & ~gate_truth & gate_domain
                            ).sum(),
                            "observed_gate_fn": (
                                ~gate_hard & gate_truth & gate_domain
                            ).sum(),
                            "observed_gate_tn": (
                                ~gate_hard & ~gate_truth & gate_domain
                            ).sum(),
                        }
                        for key, value in gate_counts.items():
                            name = f"{branch}_{key}"
                            metric_totals[name] = metric_totals.get(
                                name, 0.0
                            ) + float(value.cpu())
                        for sample_index, metadata in enumerate(
                            batch["metadata"]
                        ):
                            history = int(metadata["history_frame_count"])
                            prefix = f"{branch}_history_{history:02d}"
                            sample_truth = gate_truth[sample_index]
                            sample_domain = gate_domain[sample_index]
                            sample_hard = gate_hard[sample_index]
                            history_counts = {
                                "observed_gate_tp": (
                                    sample_hard
                                    & sample_truth
                                    & sample_domain
                                ).sum(),
                                "observed_gate_fp": (
                                    sample_hard
                                    & ~sample_truth
                                    & sample_domain
                                ).sum(),
                                "observed_gate_fn": (
                                    ~sample_hard
                                    & sample_truth
                                    & sample_domain
                                ).sum(),
                                "observed_gate_tn": (
                                    ~sample_hard
                                    & ~sample_truth
                                    & sample_domain
                                ).sum(),
                            }
                            for key, value in history_counts.items():
                                metric_totals[f"{prefix}_{key}"] += float(
                                    value.cpu()
                                )
                    continue
                totals = p1b_metric_totals(
                    prediction[f"{branch}_bev"],
                    batch[f"{branch}_fov_complete_target"],
                    batch[f"{branch}_visible_target"],
                    batch[f"{branch}_fov_support_target"],
                    gt_valid_mask=batch[f"{branch}_gt_valid_mask"],
                )
                for key, value in totals.items():
                    name = f"{branch}_{key}"
                    metric_totals[name] = metric_totals.get(name, 0.0) + float(value.cpu())
        if scale_target is not None:
            for key, value in metric_scale_metrics(
                prediction["scale"], scale_target
            ).items():
                name = f"scale_{key}"
                scalar_totals[name] = scalar_totals.get(name, 0.0) + float(value.cpu())
        batches += 1
    reduced_metrics = _reduce_dict(metric_totals, device)
    reduced_scalars = _reduce_dict({**scalar_totals, "__batches": float(batches)}, device)
    count = max(reduced_scalars.pop("__batches", 0.0), 1.0)
    output = {key: value / count for key, value in reduced_scalars.items()}
    if config["training"]["pipeline"] == "P1C-NLL":
        pose_count = reduced_metrics.get("pose_pose_count", 0.0)
        output["pose_count"] = pose_count
        output["pose_translation_mae_m"] = (
            reduced_metrics.get("pose_translation_error_sum_m", 0.0)
            / max(pose_count, 1.0)
        )
        output["pose_yaw_mae_rad"] = (
            reduced_metrics.get("pose_yaw_error_sum_rad", 0.0)
            / max(pose_count, 1.0)
        )
        for history in range(1, maximum_history + 1):
            prefix = f"pose_history_{history:02d}"
            history_count = reduced_metrics.get(f"{prefix}_pose_count", 0.0)
            if history_count <= 0.0:
                continue
            output[f"{prefix}_pose_count"] = history_count
            output[f"{prefix}_translation_mae_m"] = (
                reduced_metrics[f"{prefix}_translation_error_sum_m"]
                / history_count
            )
            output[f"{prefix}_yaw_mae_rad"] = (
                reduced_metrics[f"{prefix}_yaw_error_sum_rad"]
                / history_count
            )
    if stage in ("bev_only", "joint") and bev_objective != "pose_only":
        for branch in branches:
            prefix = f"{branch}_"
            raw = {
                key.removeprefix(prefix): value
                for key, value in reduced_metrics.items()
                if key.startswith(prefix)
            }
            if bev_objective != "full":
                tp = raw.get("support_tp", 0.0)
                fp = raw.get("support_fp", 0.0)
                fn = raw.get("support_fn", 0.0)
                tn = raw.get("support_tn", 0.0)
                output.update(
                    {
                        f"{branch}_support_precision": tp / max(tp + fp, 1.0),
                        f"{branch}_support_recall": tp / max(tp + fn, 1.0),
                        f"{branch}_support_iou": tp / max(tp + fp + fn, 1.0),
                        f"{branch}_support_accuracy": (tp + tn)
                        / max(tp + fp + fn + tn, 1.0),
                    }
                )
                if bev_objective == "fov_support_and_observed_gate":
                    gate_tp = raw.get("observed_gate_tp", 0.0)
                    gate_fp = raw.get("observed_gate_fp", 0.0)
                    gate_fn = raw.get("observed_gate_fn", 0.0)
                    gate_tn = raw.get("observed_gate_tn", 0.0)
                    output.update(
                        {
                            f"{branch}_observed_gate_precision": gate_tp
                            / max(gate_tp + gate_fp, 1.0),
                            f"{branch}_observed_gate_recall": gate_tp
                            / max(gate_tp + gate_fn, 1.0),
                            f"{branch}_observed_gate_iou": gate_tp
                            / max(gate_tp + gate_fp + gate_fn, 1.0),
                            f"{branch}_observed_gate_accuracy": (
                                gate_tp + gate_tn
                            )
                            / max(gate_tp + gate_fp + gate_fn + gate_tn, 1.0),
                        }
                    )
                for history in range(1, maximum_history + 1):
                    history_prefix = f"history_{history:02d}"
                    samples = raw.get(f"{history_prefix}_samples", 0.0)
                    if samples <= 0.0:
                        continue
                    output[f"{branch}_{history_prefix}_samples"] = samples
                    roles = ["support"]
                    if bev_objective == "fov_support_and_observed_gate":
                        roles.append("observed_gate")
                    for role in roles:
                        summary = _binary_metric_summary(
                            raw,
                            f"{history_prefix}_{role}",
                        )
                        output.update(
                            {
                                f"{branch}_{history_prefix}_{role}_{key}": value
                                for key, value in summary.items()
                            }
                        )
            else:
                output.update(
                    {
                        f"{branch}_{key}": value
                        for key, value in finalize_p1b_metrics(raw).items()
                    }
                )
    model.train()
    return output


def main() -> None:
    args = parse_args()
    config = load_p1b_config(args.config)
    training = config["training"]
    seed_everything(int(training["seed"]))
    train_dataset, validation_dataset = build_datasets(
        config, verify_manifest=not args.smoke_first_sample
    )
    if args.smoke_first_sample:
        train_dataset = smoke_subset(train_dataset, 1)
        validation_dataset = smoke_subset(validation_dataset, 1)
    if args.data_only:
        print(
            json.dumps(
                {
                    "pipeline": training["pipeline"],
                    "probability_model": config["model"]["probability_model"],
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "runtime_inputs": ["rgb_window"],
                },
                indent=2,
            )
        )
        return

    branches = _enabled_bev_branches(training)
    bev_objective = str(training.get("bev_objective", "full"))
    distributed, rank, world_size, _, device, preflight = distributed_runtime(training)
    primary = rank == 0
    model = build_model(config, device)
    baseline_initialization = None
    if bool(training.get("freeze_single_to_baseline", False)):
        baseline_initialization = initialize_frozen_single_baseline(
            model,
            training["single_baseline_checkpoint"],
            seed_merged_routing_from_single=bool(
                training.get("seed_merged_routing_from_single", True)
            ),
        )
    routing_warmstart = None
    routing_warmstart_path = str(
        training.get("routing_warmstart_checkpoint", "")
    ).strip()
    if routing_warmstart_path:
        routing_warmstart = initialize_merged_routing_warmstart(
            model,
            routing_warmstart_path,
            expected_manifest_sha256=_base_dataset(train_dataset).split_manifest_sha256,
        )
    _set_stage(model, str(training["stage"]), branches, bev_objective)
    attention_checkpointing = _configure_attention_recomputation(
        model,
        training,
        branches,
    )
    compile_settings = _configure_head_compilation(model, training)
    if distributed:
        model.head = DistributedDataParallel(
            model.head,
            device_ids=[device.index],
            bucket_cap_mb=float(training.get("ddp_bucket_cap_mb", 25.0)),
            broadcast_buffers=bool(training.get("ddp_broadcast_buffers", False)),
            gradient_as_bucket_view=bool(
                training.get("ddp_gradient_as_bucket_view", True)
            ),
            static_graph=bool(training.get("ddp_static_graph", True)),
        )
    trainable = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    trainable_names = [
        name
        for name, parameter in model.unwrapped_head().named_parameters()
        if parameter.requires_grad
    ]
    if bev_objective != "full":
        allowed = (
            ("relative_pose_head.",)
            if bev_objective == "pose_only"
            else (
                "merged_routing_token_projector.",
                "merged_bev_decoder.routing.",
                "relative_pose_head.",
                "merged_pose_embedding.",
            )
        )
        unexpected = [name for name in trainable_names if not name.startswith(allowed)]
        if unexpected or not trainable_names:
            raise RuntimeError(
                "FOV-support-only trainable scope violation: "
                f"unexpected={unexpected[:5]}, count={len(trainable_names)}"
            )
    fused_optimizer = device.type == "cuda" and bool(
        training.get("fused_optimizer", True)
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.02)),
        fused=fused_optimizer,
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
    train_worker_count = int(training["num_workers"])
    train_loader_worker_options: dict[str, object] = {}
    if train_worker_count > 0:
        train_loader_worker_options.update(
            persistent_workers=bool(training.get("persistent_workers", True)),
            prefetch_factor=int(training.get("prefetch_factor", 4)),
        )
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        sampler=sampler,
        num_workers=train_worker_count,
        pin_memory=device.type == "cuda",
        collate_fn=method1_collate,
        **train_loader_worker_options,
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
    manifest_hash = _base_dataset(train_dataset).split_manifest_sha256
    contract = checkpoint_contract(config, manifest_hash)
    start_epoch = 0
    global_step = 0
    batch_offset = 0
    if args.resume is not None:
        start_epoch, global_step, batch_offset = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            contract=contract,
        )
    checkpoint_hash = _checkpoint_sha256(config["model"]["checkpoint"])
    if args.eval_checkpoint is not None:
        checkpoint_epoch, checkpoint_step = load_head_checkpoint(
            args.eval_checkpoint,
            model=model,
            contract=contract,
        )
        metrics = validate(
            model,
            validation_loader,
            config=config,
            device=device,
            checkpoint_hash=checkpoint_hash,
        )
        if primary:
            print(
                json.dumps(
                    {
                        "evaluation_checkpoint": str(
                            args.eval_checkpoint.expanduser().resolve()
                        ),
                        "checkpoint_epoch": checkpoint_epoch,
                        "checkpoint_step": checkpoint_step,
                        "validation": metrics,
                    }
                ),
                flush=True,
            )
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return
    output_dir = Path(training["output_dir"]).expanduser().resolve()
    if primary:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "pipeline": training["pipeline"],
                    "schema": contract["checkpoint_schema"],
                    "world_size": world_size,
                    "nccl": preflight,
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                    "total_steps": total_steps,
                    "head_compile": compile_settings,
                    "attention_checkpointing": attention_checkpointing,
                    "ddp_execution": {
                        "broadcast_buffers": bool(
                            training.get("ddp_broadcast_buffers", False)
                        ),
                        "gradient_as_bucket_view": bool(
                            training.get("ddp_gradient_as_bucket_view", True)
                        ),
                        "static_graph": bool(
                            training.get("ddp_static_graph", True)
                        ),
                    },
                    "fused_optimizer": fused_optimizer,
                    "training_minimal_outputs": bool(
                        training.get("training_minimal_outputs", True)
                    ),
                    "projector_layout": "branch-specific-single-merged-v1",
                    "single_baseline_initialization": baseline_initialization,
                    "routing_warmstart": routing_warmstart,
                    "vggt_runs_per_batch": 1,
                    "bev_objective": bev_objective,
                    "trained_output": (
                        "relative_se2_pose"
                        if bev_objective == "pose_only"
                        else (
                            (
                                "relative_se2_pose+merged_fov_support+"
                                "merged_observed_gate"
                            )
                            if training["pipeline"] == "P1C-NLL"
                            and bev_objective
                            == "fov_support_and_observed_gate"
                            else (
                                "merged_fov_support+merged_observed_gate"
                                if bev_objective
                                == "fov_support_and_observed_gate"
                                else (
                                    "merged_fov_support"
                                    if bev_objective == "fov_support_only"
                                    else "configured_full_bev"
                                )
                            )
                        )
                    ),
                    "trainable_parameter_tensors": len(trainable_names),
                    "trainable_parameter_prefixes": sorted(
                        {name.split(".", 1)[0] for name in trainable_names}
                    ),
                }
            ),
            flush=True,
        )
    cache_values = config.get("teacher_cache", {"mode": "live"})
    cache_mode = str(cache_values.get("mode", "live"))
    cache = (
        TeacherCache(
            cache_values["root"],
            checkpoint_sha256=checkpoint_hash,
            preprocessing_version="rgb-depth-resize-pad-v2",
        )
        if cache_mode != "live"
        else None
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    stop = False
    for epoch in range(start_epoch, int(training["epochs"])):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        offset = batch_offset if epoch == start_epoch else 0
        sampler.set_start_index(offset * int(training["batch_size"]))
        for batch_index, batch in enumerate(loader, start=offset):
            batch_started = time.monotonic()
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            extraction, geometry = _teacher_inputs(
                model,
                batch,
                cache=cache,
                cache_mode=cache_mode,
                need_geometry=training["stage"] in ("scale_only", "joint"),
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=device.type == "cuda",
            ):
                prediction = model.forward_head(
                    extraction,
                    enabled_bev_branches=(
                        branches if training["stage"] in ("bev_only", "joint") else ()
                    ),
                    include_scale=training["stage"] in ("scale_only", "joint"),
                    assemble_runtime_outputs=not bool(
                        training.get("training_minimal_outputs", True)
                    ),
                    bev_objective=bev_objective,
                )
                loss, values, _ = step_losses(
                    prediction,
                    batch,
                    geometry,
                    config,
                    global_step=global_step,
                    total_steps=total_steps,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable,
                float(training["gradient_clip_norm"]),
                foreach=True,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            if primary and global_step % int(training["log_every_steps"]) == 0:
                memory = {}
                if device.type == "cuda":
                    gib = float(1024**3)
                    memory = {
                        "cuda_allocated_gib": torch.cuda.memory_allocated(device) / gib,
                        "cuda_reserved_gib": torch.cuda.memory_reserved(device) / gib,
                        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                        / gib,
                        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device)
                        / gib,
                    }
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "batch_seconds": time.monotonic() - batch_started,
                            "elapsed_seconds": time.monotonic() - started,
                            "learning_rate": scheduler.get_last_lr()[0],
                            **memory,
                            **{key: float(value.detach().cpu()) for key, value in values.items()},
                        }
                    ),
                    flush=True,
                )
            if primary and global_step % int(training["checkpoint_every_steps"]) == 0:
                save_checkpoint(
                    output_dir / f"p1b_step_{global_step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    contract=contract,
                    epoch=epoch,
                    global_step=global_step,
                    batch_in_epoch=batch_index + 1,
                )
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                stop = True
                break
        batch_offset = 0
        if not args.skip_validation:
            metrics = validate(
                model,
                validation_loader,
                config=config,
                device=device,
                checkpoint_hash=checkpoint_hash,
            )
            if primary:
                print(json.dumps({"epoch": epoch, "validation": metrics}), flush=True)
        if primary:
            save_checkpoint(
                output_dir / "p1b_latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                contract=contract,
                epoch=epoch + 1,
                global_step=global_step,
                batch_in_epoch=0,
            )
        if distributed:
            dist.barrier()
        if stop:
            break


if __name__ == "__main__":
    main()
