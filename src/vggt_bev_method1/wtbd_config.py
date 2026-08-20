from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path
from typing import Any


PIPELINE_ID = "WTBD-MERGE-SCALE-NLL"
CHECKPOINT_SCHEMA = "wtbd-merge-only-native-geometry-vggt-unit-scale-v1"


def load_wtbd_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        config = tomllib.load(stream)
    validate_wtbd_config(config)
    config["_config_path"] = str(resolved)
    return config


def validate_wtbd_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "scale_fit", "training"):
        if section not in config:
            raise ValueError(f"configuration is missing [{section}]")
    data = config["data"]
    model = config["model"]
    training = config["training"]
    if str(model.get("pipeline_variant")) != PIPELINE_ID:
        raise ValueError(f"model.pipeline_variant must be {PIPELINE_ID}")
    if str(training.get("pipeline")) != PIPELINE_ID:
        raise ValueError(f"training.pipeline must be {PIPELINE_ID}")
    if str(model.get("probability_model")) != "evidential":
        raise ValueError("WTBD Merge-Scale requires evidential NLL")
    if str(model.get("geometry_conditioning")) != "native_vggt_extrinsics":
        raise ValueError(
            "model.geometry_conditioning must be native_vggt_extrinsics"
        )
    if str(data.get("coordinate_mode")) != "metric_source_to_vggt_units":
        raise ValueError("data.coordinate_mode must be metric_source_to_vggt_units")
    if str(data.get("supervision")) != "metric_fov_complete_evidential":
        raise ValueError("existing FOV-complete Merged data contract is required")
    if int(data.get("maximum_history", 0)) > 10:
        raise ValueError("maximum_history must not exceed 10")
    if int(data.get("minimum_history", 0)) < 1:
        raise ValueError("minimum_history must be positive")
    if float(data.get("merged_source_extent_m", 0.0)) != 10.0:
        raise ValueError("existing Merged source GT must remain 10 m")
    if int(data.get("merged_source_output_size", 0)) != 800:
        raise ValueError("training must load the 10 m source GT at 800x800")
    canonical_extent = float(model.get("merged_bev_extent_vggt", 0.0))
    if canonical_extent <= 0.0:
        raise ValueError("model.merged_bev_extent_vggt must be positive")
    output_size = int(model.get("merged_bev_output_size", 0))
    latent_size = int(model.get("merged_latent_bev_size", 0))
    if output_size <= 0 or latent_size <= 0 or latent_size > output_size:
        raise ValueError("Merged latent/output sizes are invalid")
    if bool(model.get("native_query_resolution", False)) and latent_size != output_size:
        raise ValueError("native_query_resolution requires latent_size == output_size")
    if any(
        key in model
        for key in (
            "single_bev_extent_m",
            "single_bev_output_size",
            "single_latent_bev_size",
        )
    ):
        raise ValueError("WTBD Merge-Scale config must not define Single BEV")
    if str(training.get("stage")) not in ("merged_only", "scale_only", "joint"):
        raise ValueError("training.stage must be merged_only, scale_only or joint")
    if str(training.get("initialization", "fresh")) != "fresh":
        raise ValueError("WTBD heads must use fresh initialization")
    forbidden = (
        "single_baseline_checkpoint",
        "routing_warmstart_checkpoint",
        "pose_warmstart_checkpoint",
        "scale_input_checkpoint",
    )
    present = [key for key in forbidden if str(training.get(key, "")).strip()]
    if present:
        raise ValueError(f"fresh WTBD training forbids warm-start inputs: {present}")
    for key in (
        "learning_rate",
        "bev_loss_weight",
        "scale_loss_weight",
        "depth_scale_loss_weight",
        "gradient_clip_norm",
    ):
        if float(training.get(key, 0.0)) <= 0.0:
            raise ValueError(f"training.{key} must be positive")
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")
    if str(config.get("teacher_cache", {}).get("mode", "live")) != "live":
        raise ValueError(
            "WTBD v1 requires teacher_cache.mode=live; cached geometry is not "
            "silently accepted until its native-extrinsic contract is versioned"
        )
