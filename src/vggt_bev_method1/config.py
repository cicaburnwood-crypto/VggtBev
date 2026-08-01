from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility on managed GPU servers.
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Supervision = Literal["observed", "complete", "joint"]


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


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "training"):
        if section not in config:
            raise ValueError(f"configuration is missing [{section}]")

    data = config["data"]
    if not str(data.get("split_manifest", "")).strip():
        raise ValueError("data.split_manifest is required for reproducible training")
    if data.get("supervision") not in ("observed", "complete", "joint"):
        raise ValueError("data.supervision must be 'observed', 'complete', or 'joint'")
    if data.get("coordinate_mode") != (
        "camera_height_anchored_fixed_normalized_scale"
    ):
        raise ValueError(
            "data.coordinate_mode must be "
            "'camera_height_anchored_fixed_normalized_scale'"
        )
    forbidden_metric_keys = {
        "extent_key",
        "single_extent_m",
        "merged_extent_m",
        "camera_height_m",
    }
    leaked_metric_keys = forbidden_metric_keys.intersection(data)
    if leaked_metric_keys:
        raise ValueError(
            "runtime metric-calibration fields are forbidden in Method I configs: "
            f"{sorted(leaked_metric_keys)}"
        )
    if float(data.get("single_target_extent_m", 0)) != 6.5:
        raise ValueError("data.single_target_extent_m must be 6.5")
    if float(data.get("merged_target_extent_m", 0)) != 10.0:
        raise ValueError("data.merged_target_extent_m must be 10.0")
    if int(data.get("maximum_history", 0)) != 10:
        raise ValueError("P1A data.maximum_history must be 10")
    image_height = int(data.get("image_height", 0))
    image_width = int(data.get("image_width", 0))
    patch_size = int(config["model"].get("patch_size", 16))
    if image_height <= 0 or image_width <= 0:
        raise ValueError("configured image dimensions must be positive")
    if image_height % patch_size or image_width % patch_size:
        raise ValueError("configured image dimensions must be divisible by patch_size")
    forbidden_output_keys = {
        "output_size",
        "extent_predictor",
        "single_scale_x",
        "single_scale_z",
        "merged_scale_x",
        "merged_scale_z",
    }
    leaked_output_keys = forbidden_output_keys.intersection(config["model"])
    if leaked_output_keys:
        raise ValueError(
            "learned/per-axis P1A output scale fields are forbidden: "
            f"{sorted(leaked_output_keys)}"
        )
    forbidden_p1b_keys = {
        "scale_token",
        "scale_token_dim",
        "learned_fov_head",
        "canonical_to_vggt_scale",
    }
    leaked_p1b_keys = forbidden_p1b_keys.intersection(config["model"])
    if leaked_p1b_keys:
        raise ValueError(
            "P1B-only learned-scale/support fields are forbidden in P1A: "
            f"{sorted(leaked_p1b_keys)}"
        )
    if config["model"].get("pipeline_variant", "p1a_cascade") != "p1a_cascade":
        raise ValueError("model.pipeline_variant must be p1a_cascade")
    if config["model"].get("runtime_metric_anchor") != "camera_height_m":
        raise ValueError(
            "model.runtime_metric_anchor must be camera_height_m"
        )
    if float(
        config["model"].get("single_output_extent_normalized_scale", 0)
    ) != 6.5:
        raise ValueError(
            "model.single_output_extent_normalized_scale must be 6.5"
        )
    if float(
        config["model"].get("merged_output_extent_normalized_scale", 0)
    ) != 10.0:
        raise ValueError(
            "model.merged_output_extent_normalized_scale must be 10.0"
        )
    if int(config["model"].get("single_output_size", 0)) != 512:
        raise ValueError("model.single_output_size must be 512")
    if int(config["model"].get("merged_output_size", 0)) != 800:
        raise ValueError("model.merged_output_size must be 800")
    cached_layers = config["model"].get("cached_layers", [])
    spatial_scales = config["model"].get("spatial_scales", [])
    if len(cached_layers) != len(spatial_scales) or not cached_layers:
        raise ValueError("model.cached_layers and model.spatial_scales must have equal length")
    if any(float(scale) <= 0 for scale in spatial_scales):
        raise ValueError("model.spatial_scales must contain positive values")
    if int(config["model"].get("geometry_cue_dim", 0)) != 24:
        raise ValueError(
            "model.geometry_cue_dim must be 24 for VGGT + metric-anchor cues"
        )
    if int(config["model"].get("geometry_sample_stride", 4)) <= 0:
        raise ValueError("model.geometry_sample_stride must be positive")
    confidence_threshold = float(
        config["model"].get("geometry_confidence_threshold", 0.25)
    )
    if not 0.0 <= confidence_threshold < 1.0:
        raise ValueError(
            "model.geometry_confidence_threshold must be in [0, 1)"
        )
    if int(config["model"].get("geometry_minimum_points", 64)) < 3:
        raise ValueError("model.geometry_minimum_points must be at least three")
    if int(config["model"].get("geometry_ransac_hypotheses", 48)) <= 0:
        raise ValueError("model.geometry_ransac_hypotheses must be positive")
    if int(config["model"].get("geometry_huber_iterations", 4)) <= 0:
        raise ValueError("model.geometry_huber_iterations must be positive")
    if int(config["model"].get("geometry_fov_chunk_size", 65536)) <= 0:
        raise ValueError("model.geometry_fov_chunk_size must be positive")
    minimum_quality = float(
        config["model"].get("geometry_minimum_ground_quality", 0.05)
    )
    if not 0.0 <= minimum_quality <= 1.0:
        raise ValueError(
            "model.geometry_minimum_ground_quality must be in [0, 1]"
        )
    if config["model"].get("self_attention_mode") not in ("linear", "exact"):
        raise ValueError("model.self_attention_mode must be linear or exact")
    if config["model"].get("cross_attention_mode") not in (
        "linear",
        "exact",
        "deformable",
    ):
        raise ValueError(
            "model.cross_attention_mode must be linear, exact, or deformable"
        )
    if int(config["model"].get("deformable_samples", 0)) <= 0:
        raise ValueError("model.deformable_samples must be positive")
    if int(config["model"].get("cross_query_chunk_size", 0)) <= 0:
        raise ValueError("model.cross_query_chunk_size must be positive")
    if int(config["model"].get("query_parameter_chunk_size", 32768)) <= 0:
        raise ValueError("model.query_parameter_chunk_size must be positive")

    training = config["training"]
    pipeline = str(training.get("pipeline", "paired_evidential"))
    if pipeline != "paired_evidential":
        raise ValueError(
            "P1A uses only training.pipeline='paired_evidential'; the legacy "
            "fixed-target trainer is intentionally disabled"
        )
    if data.get("supervision") != "joint":
        raise ValueError(
            "paired evidential training requires data.supervision='joint' "
            "to load masked and complete targets together"
        )
    sampling_mode = str(data.get("sampling_mode", "all_prefixes"))
    if sampling_mode not in ("all_prefixes", "one_prefix_per_session"):
        raise ValueError(
            "data.sampling_mode must be all_prefixes or "
            "one_prefix_per_session"
        )
    required_cuda_devices = int(training.get("required_cuda_devices", 1))
    if required_cuda_devices <= 0:
        raise ValueError("training.required_cuda_devices must be positive")
    if required_cuda_devices > 1 and int(training.get("batch_size", 0)) != 1:
        raise ValueError("distributed Method I training requires per-rank batch_size=1")
    if str(training.get("distributed_backend", "nccl")) != "nccl":
        raise ValueError("training.distributed_backend must be nccl")
    if str(training.get("nccl_p2p_level", "AUTO")).upper() not in (
        "AUTO",
        "LOC",
        "NVL",
        "PIX",
        "PXB",
        "PHB",
    ):
        raise ValueError(
            "training.nccl_p2p_level must be AUTO or must not permit "
            "cross-NUMA SYS P2P"
        )
    if required_cuda_devices > 1 and not bool(
        training.get("require_same_numa", True)
    ):
        raise ValueError(
            "distributed Method I training requires same-NUMA GPUs to preserve "
            "the full-speed NCCL path"
        )
    if float(training.get("ddp_bucket_cap_mb", 25.0)) <= 0:
        raise ValueError("training.ddp_bucket_cap_mb must be positive")
    if int(training.get("nccl_timeout_seconds", 300)) <= 0:
        raise ValueError("training.nccl_timeout_seconds must be positive")
    warmup_epochs = int(training.get("geometry_warmup_epochs", 0))
    ramp_epochs = int(training.get("geometry_ramp_epochs", 0))
    if warmup_epochs != 0 or ramp_epochs != 0:
        raise ValueError(
            "P1A robust geometry is always active; geometry warm-up/ramp "
            "must both be zero"
        )
    if pipeline == "paired_evidential":
        enabled_models = training.get(
            "enabled_models",
            ["observed", "complete_evidential"],
        )
        if not isinstance(enabled_models, list) or not enabled_models:
            raise ValueError("training.enabled_models must be a non-empty list")
        if len(set(enabled_models)) != len(enabled_models) or not set(
            enabled_models
        ).issubset({"observed", "complete_evidential"}):
            raise ValueError(
                "training.enabled_models may contain observed and/or "
                "complete_evidential without duplicates"
            )
        if "complete_evidential" not in enabled_models:
            raise ValueError(
                "paired evidential training must keep complete_evidential enabled"
            )
        positive_keys = (
            "complete_learning_rate",
            "observed_known_weight",
            "observed_free_weight",
            "observed_surface_weight",
            "guessed_completion_nll_weight",
            "observed_region_weight",
            "single_task_weight",
            "merged_task_weight",
        )
        if "observed" in enabled_models:
            positive_keys += (
                "observed_learning_rate",
            )
        for key in positive_keys:
            if float(training.get(key, 0.0)) <= 0:
                raise ValueError(f"training.{key} must be positive")
        nonnegative_keys = (
            "guessed_completion_dice_weight",
            "guessed_region_weight",
            "incorrect_evidence_weight",
            "observation_relation_weight",
            "confidence_calibration_weight",
            "confidence_regularizer_warmup_epochs",
            "confidence_regularizer_ramp_epochs",
            "surface_tolerance_m",
        )
        for key in nonnegative_keys:
            if float(training.get(key, 0.0)) < 0:
                raise ValueError(f"training.{key} cannot be negative")
        class_weights = training.get("guessed_completion_class_weights", [])
        if len(class_weights) != 2 or any(float(value) <= 0 for value in class_weights):
            raise ValueError(
                "training.guessed_completion_class_weights must contain "
                "positive [free, occupied] weights"
            )
        if abs(float(class_weights[0]) - float(class_weights[1])) > 1e-6:
            raise ValueError(
                "P1A guessed completion must use balanced [free, occupied] "
                "class weights; data imbalance is handled by macro Dice"
            )
        for key in (
            "guessed_incorrect_evidence_multiplier",
            "guessed_confidence_calibration_multiplier",
            "maximum_guessed_to_direct_gradient_ratio",
        ):
            if float(training.get(key, 0.0)) <= 0:
                raise ValueError(f"training.{key} must be positive")
        for key in (
            "guessed_supervision_warmup_fraction",
            "guessed_supervision_ramp_fraction",
        ):
            value = float(training.get(key, -1.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"training.{key} must be in [0, 1]")
        if (
            float(training["guessed_supervision_warmup_fraction"])
            + float(training["guessed_supervision_ramp_fraction"])
            > 1.0
        ):
            raise ValueError("guessed supervision warm-up plus ramp cannot exceed 1")
        if training.get("direct_priority_pcgrad") is not True:
            raise ValueError(
                "P1A requires training.direct_priority_pcgrad=true so guessed "
                "completion cannot oppose directly observed supervision"
            )
        observed_region_weight = float(training["observed_region_weight"])
        guessed_region_weight = float(training["guessed_region_weight"])
        if guessed_region_weight > 0.25 * observed_region_weight:
            raise ValueError(
                "P1A guessed_region_weight cannot exceed 25% of "
                "observed_region_weight"
            )
        if (
            float(training["maximum_guessed_to_direct_gradient_ratio"])
            > 0.25
        ):
            raise ValueError(
                "P1A maximum_guessed_to_direct_gradient_ratio cannot exceed 0.25"
            )
        if float(training["guessed_supervision_warmup_fraction"]) < 0.05:
            raise ValueError(
                "P1A guessed supervision requires at least a 5% direct-only warm-up"
            )
        if float(training["guessed_supervision_ramp_fraction"]) < 0.10:
            raise ValueError(
                "P1A guessed supervision requires at least a 10% ramp"
            )
