from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 on managed GPU servers.
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Supervision = Literal["metric_fov_complete_evidential"]


@dataclass(frozen=True)
class LabelValues:
    occupied: int = 0
    unknown: int = 112
    free: int = 255


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def _positive(section: dict, keys: tuple[str, ...], prefix: str) -> None:
    for key in keys:
        if float(section.get(key, 0.0)) <= 0:
            raise ValueError(f"{prefix}.{key} must be positive")


def validate_config(config: dict[str, Any]) -> None:
    for name in ("data", "model", "scale_fit", "training"):
        if name not in config:
            raise ValueError(f"configuration is missing [{name}]")

    data = config["data"]
    if data.get("supervision") != "metric_fov_complete_evidential":
        raise ValueError(
            "data.supervision must be metric_fov_complete_evidential"
        )
    if data.get("coordinate_mode") != "p1b_fixed_metric":
        raise ValueError("data.coordinate_mode must be p1b_fixed_metric")
    if str(data.get("gt_depth_convention")) != "camera_axis_z_depth_m":
        raise ValueError("data.gt_depth_convention must be camera_axis_z_depth_m")
    if float(data.get("single_bev_extent_m", 0.0)) != 6.5:
        raise ValueError("data.single_bev_extent_m must be 6.5")
    if int(data.get("single_bev_output_size", 0)) != 512:
        raise ValueError("data.single_bev_output_size must be 512")
    if float(data.get("merged_bev_extent_m", 0.0)) != 10.0:
        raise ValueError("data.merged_bev_extent_m must be 10.0")
    if int(data.get("merged_bev_output_size", 0)) != 800:
        raise ValueError("data.merged_bev_output_size must be 800")
    if not str(data.get("split_manifest", "")).strip():
        raise ValueError("data.split_manifest is required")
    if str(data.get("sampling_mode", "all_prefixes")) not in (
        "all_prefixes",
        "one_prefix_per_session",
    ):
        raise ValueError("data.sampling_mode is invalid")
    if data.get("maximum_sessions") is not None and int(
        data["maximum_sessions"]
    ) <= 0:
        raise ValueError("data.maximum_sessions must be positive")
    if bool(data.get("allow_active_writer_completed_snapshot", False)) and (
        data.get("maximum_sessions") is None
    ):
        raise ValueError(
            "active-writer snapshots require a finite data.maximum_sessions"
        )
    forbidden = {
        "camera_height_m",
        "metric_scale",
        "single_canonical_extent",
        "merged_canonical_extent",
    }
    leaked = forbidden.intersection(data)
    if leaked:
        raise ValueError(f"new P1B forbids legacy fields: {sorted(leaked)}")

    model = config["model"]
    if model.get("pipeline_variant") != "p1b_fov_complete_metric_scale":
        raise ValueError(
            "model.pipeline_variant must be p1b_fov_complete_metric_scale"
        )
    if float(model.get("single_bev_extent_m", 0.0)) != 6.5:
        raise ValueError("model.single_bev_extent_m must be 6.5")
    if int(model.get("single_bev_output_size", 0)) != 512:
        raise ValueError("model.single_bev_output_size must be 512")
    if float(model.get("merged_bev_extent_m", 0.0)) != 10.0:
        raise ValueError("model.merged_bev_extent_m must be 10.0")
    if int(model.get("merged_bev_output_size", 0)) != 800:
        raise ValueError("model.merged_bev_output_size must be 800")
    for name, output in (
        ("single_latent_bev_size", 512),
        ("merged_latent_bev_size", 800),
    ):
        latent_size = int(model.get(name, 0))
        if latent_size <= 0 or latent_size > output:
            raise ValueError(
                f"model.{name} must be positive and no larger than {output}"
            )
    cached_layers = model.get("cached_layers", [])
    spatial_scales = model.get("spatial_scales", [])
    if len(cached_layers) != len(spatial_scales) or not cached_layers:
        raise ValueError("cached_layers/spatial_scales must be non-empty and aligned")
    _positive(
        model,
        (
            "patch_size",
            "vggt_token_dim",
            "hidden_dim",
            "attention_heads",
            "decoder_layers",
            "scale_decoder_layers",
            "deformable_samples",
            "cross_query_chunk_size",
        ),
        "model",
    )
    if int(model["hidden_dim"]) % int(model["attention_heads"]):
        raise ValueError("model.hidden_dim must be divisible by attention_heads")
    if model.get("self_attention_mode") not in ("linear", "exact"):
        raise ValueError("model.self_attention_mode must be linear or exact")
    if model.get("cross_attention_mode") not in (
        "linear",
        "exact",
        "deformable",
    ):
        raise ValueError("model.cross_attention_mode is invalid")

    scale = config["scale_fit"]
    _positive(
        scale,
        (
            "minimum_depth_m",
            "maximum_depth_m",
            "residual_threshold_log",
            "huber_delta_log",
            "minimum_valid_pixels",
            "maximum_pixels_per_frame",
            "quality_sigma_log",
            "desired_log_depth_range",
            "irls_iterations",
        ),
        "scale_fit",
    )
    if not 0 <= float(scale.get("confidence_threshold", -1)) <= 1:
        raise ValueError("scale_fit.confidence_threshold must be in [0,1]")
    if not 0 < float(scale.get("minimum_inlier_ratio", 0)) <= 1:
        raise ValueError("scale_fit.minimum_inlier_ratio must be in (0,1]")
    if not 0 < float(scale.get("minimum_quality_weight", 0.10)) <= 1:
        raise ValueError("scale_fit.minimum_quality_weight must be in (0,1]")

    training = config["training"]
    if training.get("pipeline") != "p1b_fov_complete_metric_scale":
        raise ValueError(
            "training.pipeline must be p1b_fov_complete_metric_scale"
        )
    if training.get("stage") not in ("scale_only", "bev_only", "joint"):
        raise ValueError("training.stage must be scale_only, bev_only, or joint")
    enabled_bev_branches = training.get(
        "enabled_bev_branches",
        ["single", "merged"],
    )
    if (
        not isinstance(enabled_bev_branches, list)
        or not enabled_bev_branches
        or len(set(enabled_bev_branches)) != len(enabled_bev_branches)
        or set(enabled_bev_branches).difference(("single", "merged"))
    ):
        raise ValueError(
            "training.enabled_bev_branches must be a non-empty unique subset "
            "of ['single', 'merged']"
        )
    obsolete_training = {
        "complete_class_weights",
        "complete_nll_weight",
        "complete_dice_weight",
        "fov_complete_class_weights",
        "fov_complete_nll_weight",
        "fov_complete_dice_weight",
        "guessed_confidence_ceiling",
        "invalid_evidence_weight",
        "guessed_supervision_warmup_epochs",
        "guessed_supervision_ramp_epochs",
    }.intersection(training)
    if obsolete_training:
        raise ValueError(
            "FOV-complete P1B forbids obsolete loss fields: "
            f"{sorted(obsolete_training)}"
        )
    _positive(
        training,
        (
            "required_cuda_devices",
            "epochs",
            "batch_size",
            "learning_rate",
            "bev_loss_weight",
            "observed_free_nll_weight",
            "observed_surface_nll_weight",
            "guessed_completion_nll_weight",
            "guessed_completion_dice_weight",
            "observed_region_weight",
            "guessed_region_weight",
            "fov_support_bce_weight",
            "fov_support_dice_weight",
            "scale_loss_weight",
            "depth_scale_loss_weight",
            "gradient_clip_norm",
            "checkpoint_every_steps",
            "log_every_steps",
            "validation_batches",
        ),
        "training",
    )
    minimum_learning_rate = float(
        training.get("minimum_learning_rate", 0.0)
    )
    learning_rate = float(training["learning_rate"])
    if not 0.0 <= minimum_learning_rate < learning_rate:
        raise ValueError(
            "training.minimum_learning_rate must be non-negative and smaller "
            "than training.learning_rate"
        )
    warmup_fraction = float(training.get("warmup_fraction", 0.0))
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("training.warmup_fraction must be in [0,1)")
    for branch in ("single", "merged"):
        key = f"{branch}_task_weight"
        value = float(training.get(key, -1.0))
        if branch in enabled_bev_branches and value <= 0:
            raise ValueError(f"training.{key} must be positive when enabled")
        if branch not in enabled_bev_branches and value != 0:
            raise ValueError(f"training.{key} must be zero when disabled")
    _positive(
        training,
        tuple(
            key
            for key in (
                "single_observed_surface_nll_weight",
                "merged_observed_surface_nll_weight",
            )
            if key in training
        ),
        "training",
    )
    for key in (
        "observed_surface_continuity_weight",
        "incorrect_evidence_weight",
        "confidence_calibration_weight",
        "guessed_incorrect_evidence_multiplier",
        "guessed_confidence_calibration_multiplier",
        "maximum_guessed_to_direct_gradient_ratio",
        "observation_relation_weight",
        "evidence_relation_margin",
        "uncertainty_loss_weight",
        "weight_decay",
    ):
        if float(training.get(key, 0.0)) < 0:
            raise ValueError(f"training.{key} cannot be negative")
    for key in (
        "confidence_regularizer_warmup_epochs",
        "confidence_regularizer_ramp_epochs",
    ):
        if int(training.get(key, -1)) < 0:
            raise ValueError(f"training.{key} cannot be negative")
    for key in (
        "guessed_supervision_warmup_fraction",
        "guessed_supervision_ramp_fraction",
    ):
        value = float(training.get(key, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"training.{key} must be in [0,1]")
    if (
        float(training["guessed_supervision_warmup_fraction"])
        + float(training["guessed_supervision_ramp_fraction"])
        > 1.0
    ):
        raise ValueError(
            "guessed supervision warm-up and ramp fractions cannot exceed 1"
        )
    tolerance_fraction = float(
        training.get("surface_tolerance_latent_cell_fraction", 0.0)
    )
    if not 0.0 < tolerance_fraction <= 1.0:
        raise ValueError(
            "training.surface_tolerance_latent_cell_fraction must be in (0,1]"
        )
    if not isinstance(training.get("direct_priority_pcgrad"), bool):
        raise ValueError("training.direct_priority_pcgrad must be boolean")
    if not isinstance(training.get("train_guessed_completion"), bool):
        raise ValueError("training.train_guessed_completion must be boolean")
    if not isinstance(training.get("compile_head", False), bool):
        raise ValueError("training.compile_head must be boolean")
    if not isinstance(training.get("compile_head_dynamic", True), bool):
        raise ValueError("training.compile_head_dynamic must be boolean")
    if str(training.get("compile_head_backend", "inductor")) != "inductor":
        raise ValueError("training.compile_head_backend must be inductor")
    if str(
        training.get("compile_head_scope", "deformable_query_chunks")
    ) != "deformable_query_chunks":
        raise ValueError(
            "training.compile_head_scope must be deformable_query_chunks"
        )
    if str(training.get("compile_head_mode", "default")) not in (
        "default",
        "reduce-overhead",
        "max-autotune-no-cudagraphs",
    ):
        raise ValueError("training.compile_head_mode is invalid")
    weights = training.get("guessed_completion_class_weights", [])
    if len(weights) != 2 or any(float(value) <= 0 for value in weights):
        raise ValueError(
            "training.guessed_completion_class_weights must contain positive "
            "[free, occupied] weights"
        )
    if str(training.get("distributed_backend", "nccl")) != "nccl":
        raise ValueError("training.distributed_backend must be nccl")
    maximum_distributed_batch_size = (
        4 if enabled_bev_branches == ["single"] else 2
    )
    if (
        int(training["required_cuda_devices"]) > 1
        and not (
            1
            <= int(training["batch_size"])
            <= maximum_distributed_batch_size
        )
    ):
        raise ValueError(
            "distributed P1B per-rank batch_size exceeds the validated "
            f"limit of {maximum_distributed_batch_size} for branches "
            f"{enabled_bev_branches}"
        )
    if (
        int(training["batch_size"]) > 1
        and data.get("sampling_mode", "all_prefixes")
        != "one_prefix_per_session"
    ):
        raise ValueError(
            "P1B batch_size > 1 requires one_prefix_per_session "
            "history-length bucketing"
        )
    cache = config.get("teacher_cache", {"mode": "live"})
    if cache.get("mode", "live") not in ("live", "read", "write_through"):
        raise ValueError(
            "teacher_cache.mode must be live, read, or write_through"
        )
    if cache.get("mode", "live") != "live" and not str(cache.get("root", "")).strip():
        raise ValueError("teacher_cache.root is required for cache read/write")
