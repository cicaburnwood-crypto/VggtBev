from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


PIPELINE_ID = "M04-PARALLEL-ANCHOR-HISTORY-MERGED-SCALE-NLL"
CHECKPOINT_SCHEMA = "m04-parallel-anchor-history-merged-scale-512-v3"


def load_m04_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        config = tomllib.load(stream)
    validate_m04_config(config)
    config["_config_path"] = str(resolved)
    return config


def validate_m04_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "scale_fit", "training"):
        if section not in config:
            raise ValueError(f"M04 configuration is missing [{section}]")
    data = config["data"]
    model = config["model"]
    training = config["training"]
    if str(model.get("pipeline_variant")) != PIPELINE_ID:
        raise ValueError(f"model.pipeline_variant must be {PIPELINE_ID}")
    if str(training.get("pipeline")) != PIPELINE_ID:
        raise ValueError(f"training.pipeline must be {PIPELINE_ID}")
    if str(data.get("coordinate_mode")) != "metric_source_to_vggt_units":
        raise ValueError("M04 target coordinate contract is invalid")
    if str(data.get("supervision")) != "metric_fov_complete_evidential":
        raise ValueError("M04 requires metric FOV-complete evidential GT")
    # The user explicitly selected the existing 10x10 m supervision.  Do not
    # silently point canonical M04 at a newly generated or relabelled extent.
    if float(data.get("merged_source_extent_m", 0.0)) != 10.0:
        raise ValueError("canonical M04 must use the existing 10 m Merged GT")
    if int(data.get("merged_source_image_size", 0)) != 512:
        raise ValueError("canonical M04 source PNGs must remain 512x512")
    if str(data.get("merged_complete_directory")) != "merged_complete_10m":
        raise ValueError("canonical M04 complete GT directory is invalid")
    if str(data.get("merged_masked_directory")) != "merged_masked_10m":
        raise ValueError("canonical M04 masked GT directory is invalid")
    if int(data.get("merged_source_output_size", 0)) != 512:
        raise ValueError(
            "canonical M04 must inverse-sample the native 512x512 GT directly"
        )
    minimum_history = int(data.get("minimum_history", 0))
    maximum_history = int(data.get("maximum_history", 0))
    if minimum_history != 1 or not 1 <= maximum_history <= 10:
        raise ValueError("M04 histories must span from one through at most ten frames")
    if str(data.get("sampling_mode")) != "one_prefix_per_session":
        raise ValueError("M04 requires one balanced temporal prefix per session")

    if str(model.get("probability_model")) != "evidential":
        raise ValueError("M04 requires Beta evidential occupancy")
    if str(model.get("geometry_conditioning")) != (
        "parallel_latest_anchor_and_implicit_history"
    ):
        raise ValueError("M04 geometry conditioning contract is invalid")
    if int(model.get("maximum_history", 0)) != maximum_history:
        raise ValueError("M04 model/data maximum_history must match")
    if int(model.get("merged_bev_output_size", 0)) != 512:
        raise ValueError("canonical M04 output must remain 512x512")
    if not bool(model.get("native_query_resolution", False)):
        raise ValueError("M04 requires a native-resolution query grid")
    if float(model.get("merged_bev_extent_vggt", 0.0)) != 6.5:
        raise ValueError("canonical M04 extent must remain 6.5 VGGT units")
    for key in (
        "hidden_dim",
        "attention_heads",
        "latest_decoder_layers",
        "history_decoder_layers",
        "scale_decoder_layers",
        "query_fourier_bands",
        "refinement_layers",
        "implicit_geometry_hidden_dim",
        "implicit_geometry_heads",
        "implicit_geometry_layers",
        "maximum_prefix_tokens",
        "frame_reliability_hidden_dim",
    ):
        if int(model.get(key, 0)) <= 0:
            raise ValueError(f"model.{key} must be positive")
    if int(model["hidden_dim"]) % int(model["attention_heads"]):
        raise ValueError("M04 hidden_dim must be divisible by attention_heads")
    if str(training.get("stage")) != "joint":
        raise ValueError("M04 trains BEV and Scale jointly in one stage")
    if str(training.get("initialization", "fresh")) != "fresh":
        raise ValueError("canonical M04 requires a fresh M04 head")
    if float(training.get("evidence_kl_weight", -1.0)) < 0.0:
        raise ValueError("M04 evidence KL weight cannot be negative")
    for key in (
        "learning_rate",
        "gradient_clip_norm",
    ):
        if float(training.get(key, 0.0)) <= 0.0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("epochs", "batch_size"):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if float(training.get("minimum_mean_source_coverage", -1.0)) < 0.0:
        raise ValueError("minimum_mean_source_coverage cannot be negative")
    if bool(training.get("ddp_static_graph", False)):
        raise ValueError(
            "M04 cannot use DDP static_graph because N=1 skips history"
        )
    if not bool(training.get("ddp_find_unused_parameters", False)):
        raise ValueError(
            "M04 requires DDP unused-parameter discovery for the N=1 branch"
        )
    forbidden = (
        "single_baseline_checkpoint",
        "routing_warmstart_checkpoint",
        "pose_warmstart_checkpoint",
        "camera_height_input",
        "extrinsic_input",
        "navigation_loss",
        "planner_loss",
    )
    present = [key for key in forbidden if str(training.get(key, "")).strip()]
    if present:
        raise ValueError(f"M04 forbids runtime/sequence/navigation inputs: {present}")
    if str(config.get("teacher_cache", {}).get("mode", "live")) != "live":
        raise ValueError("M04 requires one live frozen-VGGT pass per batch")
