from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


PIPELINE_ID = "M05-LATEST-ANCHORED-REVERSE-GATED-MERGED-SCALE-NLL"
CHECKPOINT_SCHEMA = "m05-role-preserving-reverse-gated-merged-scale-512-v2"


def load_m05_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        config = tomllib.load(stream)
    validate_m05_config(config)
    config["_config_path"] = str(resolved)
    return config


def validate_m05_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "scale_fit", "training"):
        if section not in config:
            raise ValueError(f"M05 configuration is missing [{section}]")
    data = config["data"]
    model = config["model"]
    training = config["training"]
    if str(model.get("pipeline_variant")) != PIPELINE_ID:
        raise ValueError(f"model.pipeline_variant must be {PIPELINE_ID}")
    if str(training.get("pipeline")) != PIPELINE_ID:
        raise ValueError(f"training.pipeline must be {PIPELINE_ID}")
    if str(data.get("coordinate_mode")) != "metric_source_to_vggt_units":
        raise ValueError("M05 target coordinate contract is invalid")
    if str(data.get("supervision")) != "metric_fov_complete_evidential":
        raise ValueError("M05 requires metric FOV-complete evidential GT")
    if float(data.get("merged_source_extent_m", 0.0)) != 10.0:
        raise ValueError("canonical M05 must reuse the existing 10 m Merged GT")
    if int(data.get("merged_source_image_size", 0)) != 512:
        raise ValueError("canonical M05 source rasters must remain 512x512")
    if str(data.get("merged_complete_directory")) != "merged_complete_10m":
        raise ValueError("canonical M05 complete GT directory is invalid")
    if str(data.get("merged_masked_directory")) != "merged_masked_10m":
        raise ValueError("canonical M05 masked GT directory is invalid")
    if int(data.get("merged_source_output_size", 0)) != 512:
        raise ValueError("canonical M05 must use native 512x512 supervision")
    minimum_history = int(data.get("minimum_history", 0))
    maximum_history = int(data.get("maximum_history", 0))
    if minimum_history != 1 or not 1 <= maximum_history <= 10:
        raise ValueError("M05 histories must span one through at most ten frames")
    if str(data.get("sampling_mode")) != "one_prefix_per_session":
        raise ValueError("M05 requires one balanced temporal prefix per session")

    if str(model.get("probability_model")) != "evidential":
        raise ValueError("M05 requires Beta evidential occupancy")
    if str(model.get("geometry_conditioning")) != (
        "latest_anchor_reverse_gated_history"
    ):
        raise ValueError("M05 geometry conditioning contract is invalid")
    if int(model.get("maximum_history", 0)) != maximum_history:
        raise ValueError("M05 model/data maximum_history must match")
    if int(model.get("merged_bev_output_size", 0)) != 512:
        raise ValueError("canonical M05 output must remain 512x512")
    if float(model.get("merged_bev_extent_vggt", 0.0)) != 6.5:
        raise ValueError("canonical M05 extent must remain 6.5 VGGT units")
    if not bool(model.get("native_query_resolution", False)):
        raise ValueError("M05 requires native-resolution queries")
    if not bool(model.get("full_per_pixel_query", False)):
        raise ValueError("M05 requires the full-capacity native query table")
    if not bool(model.get("reverse_history_weight_sharing", False)):
        raise ValueError("M05 requires shared newest-to-oldest history updates")
    if not bool(model.get("structured_prefix_readout", False)):
        raise ValueError("M05 must keep camera/register readout roles separate")
    if not bool(model.get("structured_frame_reliability", False)):
        raise ValueError("M05 frame reliability must preserve prefix-token roles")
    for key in (
        "hidden_dim",
        "attention_heads",
        "latest_decoder_layers",
        "history_update_layers",
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
        raise ValueError("M05 hidden_dim must be divisible by attention_heads")
    if int(model.get("cross_query_chunk_size", 0)) <= 0:
        raise ValueError("model.cross_query_chunk_size must be positive")

    if str(training.get("stage")) != "joint":
        raise ValueError("M05 trains BEV and Scale jointly")
    if str(training.get("initialization", "fresh")) != "fresh":
        raise ValueError("canonical M05 requires a fresh head")
    if str(training.get("bev_objective")) != "single_baseline_dual_supervision":
        raise ValueError("M05 must supervise latest and Merged with the Single loss")
    latest_weight = float(training.get("latest_auxiliary_loss_weight", 0.50))
    if not 0.0 <= latest_weight <= 1.0:
        raise ValueError("training.latest_auxiliary_loss_weight must be in [0,1]")
    for key in ("learning_rate", "gradient_clip_norm"):
        if float(training.get(key, 0.0)) <= 0.0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("epochs", "batch_size"):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if bool(training.get("ddp_static_graph", False)):
        raise ValueError("M05 variable history cannot use DDP static_graph")
    if not bool(training.get("ddp_find_unused_parameters", False)):
        raise ValueError("M05 requires DDP unused-parameter discovery at N=1")
    fractions = (
        "wrong_evidence_zero_fraction",
        "wrong_evidence_ramp_fraction",
        "hidden_occupied_zero_fraction",
        "hidden_occupied_ramp_fraction",
    )
    for key in fractions:
        value = float(training.get(key, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"training.{key} must be in [0,1]")
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
        raise ValueError(f"M05 forbids runtime/navigation inputs: {present}")
    if str(config.get("teacher_cache", {}).get("mode", "live")) != "live":
        raise ValueError("M05 requires one live frozen-VGGT pass per batch")
