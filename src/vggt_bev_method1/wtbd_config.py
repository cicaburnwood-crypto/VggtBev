from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path
from typing import Any


PIPELINE_ID = "WTBD-IMPLICIT-MERGE-SCALE-NLL"
CHECKPOINT_SCHEMA = (
    "wtbd-merge-only-implicit-geometry-cross-attention-scale-v2"
)


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
    if str(model.get("geometry_conditioning")) != (
        "implicit_multiview_token_cross_attention"
    ):
        raise ValueError(
            "model.geometry_conditioning must be "
            "implicit_multiview_token_cross_attention"
        )
    if str(model.get("cross_attention_mode")) != "linear":
        raise ValueError(
            "WTBD v2 requires global linear cross-attention over all frames"
        )
    if str(data.get("coordinate_mode")) != "metric_source_to_vggt_units":
        raise ValueError("data.coordinate_mode must be metric_source_to_vggt_units")
    if str(data.get("supervision")) != "metric_fov_complete_evidential":
        raise ValueError("existing FOV-complete Merged data contract is required")
    if int(data.get("maximum_history", 0)) != 10:
        raise ValueError("WTBD v2 maximum_history must be exactly 10")
    if int(data.get("minimum_history", 0)) != 10:
        raise ValueError("WTBD v2 trains on complete 10-frame windows")
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
    if int(model.get("maximum_history", 0)) != int(data["maximum_history"]):
        raise ValueError(
            "model.maximum_history must match data.maximum_history"
        )
    for key in (
        "implicit_geometry_hidden_dim",
        "implicit_geometry_heads",
        "implicit_geometry_layers",
        "maximum_prefix_tokens",
    ):
        if int(model.get(key, 0)) <= 0:
            raise ValueError(f"model.{key} must be positive")
    if int(model["implicit_geometry_hidden_dim"]) % int(
        model["implicit_geometry_heads"]
    ):
        raise ValueError(
            "implicit geometry hidden dim must be divisible by its heads"
        )
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
            "WTBD v2 requires teacher_cache.mode=live; frozen tokens and "
            "training-only scale labels must use the runtime VGGT checkpoint"
        )
