from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path
from typing import Any


PIPELINE_ID = "P1D-DIRECT-MERGED-SCALE-NLL"
CHECKPOINT_SCHEMA = "p1d-direct-merged-temporal-reliability-scale-v1"


def load_p1d_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        config = tomllib.load(stream)
    validate_p1d_config(config)
    config["_config_path"] = str(resolved)
    return config


def validate_p1d_config(config: dict[str, Any]) -> None:
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
        raise ValueError("P1D requires evidential NLL")
    if str(model.get("geometry_conditioning")) != (
        "implicit_temporal_cross_attention_with_learned_reliability"
    ):
        raise ValueError("P1D geometry conditioning contract is invalid")
    if str(model.get("cross_attention_mode")) != "linear":
        raise ValueError("P1D reliability requires global linear cross-attention")
    if str(data.get("coordinate_mode")) != "metric_source_to_vggt_units":
        raise ValueError("P1D target coordinate contract is invalid")
    if str(data.get("supervision")) != "metric_fov_complete_evidential":
        raise ValueError("P1D requires existing FOV-complete Merged GT")
    minimum_history = int(data.get("minimum_history", 0))
    maximum_history = int(data.get("maximum_history", 0))
    if minimum_history < 1 or maximum_history < minimum_history:
        raise ValueError("P1D history bounds are invalid")
    if maximum_history > 64:
        raise ValueError("P1D configured history cannot exceed 64 frames")
    if int(model.get("maximum_history", 0)) != maximum_history:
        raise ValueError("model/data maximum_history must match")
    if float(data.get("merged_source_extent_m", 0.0)) != 10.0:
        raise ValueError("P1D uses the existing 10m Merged source GT")
    if int(data.get("merged_source_output_size", 0)) != 800:
        raise ValueError("P1D must load the existing 800x800 Merged source GT")
    output_size = int(model.get("merged_bev_output_size", 0))
    latent_size = int(model.get("merged_latent_bev_size", 0))
    if output_size <= 0 or not 0 < latent_size <= output_size:
        raise ValueError("P1D Merged query/output sizes are invalid")
    if float(model.get("merged_bev_extent_vggt", 0.0)) <= 0.0:
        raise ValueError("P1D VGGT-unit extent must be positive")
    if bool(model.get("native_query_resolution", False)) and latent_size != output_size:
        raise ValueError("native_query_resolution requires latent_size == output_size")
    for key in (
        "implicit_geometry_hidden_dim",
        "implicit_geometry_heads",
        "implicit_geometry_layers",
        "maximum_prefix_tokens",
        "frame_reliability_hidden_dim",
    ):
        if int(model.get(key, 0)) <= 0:
            raise ValueError(f"model.{key} must be positive")
    reliability_minimum = float(model.get("frame_reliability_minimum", 0.0))
    reliability_maximum = float(model.get("frame_reliability_maximum", 0.0))
    if not 0.0 < reliability_minimum < 1.0 < reliability_maximum:
        raise ValueError("P1D frame reliability bounds must straddle one")
    dropout = float(model.get("training_frame_dropout_probability", -1.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("P1D training frame dropout must be in [0,1)")
    if str(training.get("stage")) != "joint":
        raise ValueError("P1D is trained jointly in one stage")
    if str(training.get("initialization", "fresh")) != "fresh":
        raise ValueError("P1D requires fresh head initialization")
    forbidden = (
        "single_baseline_checkpoint",
        "routing_warmstart_checkpoint",
        "pose_warmstart_checkpoint",
        "scale_input_checkpoint",
        "extrinsic_input",
    )
    present = [key for key in forbidden if str(training.get(key, "")).strip()]
    if present:
        raise ValueError(f"P1D forbids sequential/warm-start inputs: {present}")
    for key in (
        "learning_rate",
        "bev_loss_weight",
        "scale_loss_weight",
        "depth_scale_loss_weight",
        "gradient_clip_norm",
    ):
        if float(training.get(key, 0.0)) <= 0.0:
            raise ValueError(f"training.{key} must be positive")
    for key in (
        "history_observed_gate_weight",
        "history_support_weight",
        "history_guessed_weight",
        "guessed_hard_pixel_weight",
    ):
        if float(training.get(key, -1.0)) < 0.0:
            raise ValueError(f"training.{key} cannot be negative")
    if not 0.0 < float(training.get("guessed_hard_fraction", 0.0)) <= 1.0:
        raise ValueError("training.guessed_hard_fraction must be in (0,1]")
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")
    if str(config.get("teacher_cache", {}).get("mode", "live")) != "live":
        raise ValueError("P1D requires one live frozen-VGGT pass per batch")
