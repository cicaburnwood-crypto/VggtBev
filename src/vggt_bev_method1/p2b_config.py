from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path
from typing import Any


def load_p2b_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        config = tomllib.load(stream)
    validate_p2b_config(config)
    config["_config_path"] = str(resolved)
    return config


def _positive(section: dict, names: tuple[str, ...], prefix: str) -> None:
    for name in names:
        if float(section.get(name, 0.0)) <= 0.0:
            raise ValueError(f"{prefix}.{name} must be positive")


def validate_p2b_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "scale_fit", "training"):
        if section not in config:
            raise ValueError(f"configuration is missing [{section}]")
    data = config["data"]
    model = config["model"]
    training = config["training"]
    if data.get("supervision") != "metric_fov_complete_evidential":
        raise ValueError("P2B reuses the existing FOV-complete dataset contract")
    if data.get("coordinate_mode") != "p2b_fixed_metric":
        raise ValueError("data.coordinate_mode must be p2b_fixed_metric")
    if int(data.get("maximum_history", 0)) > 10:
        raise ValueError("P2B maximum_history must not exceed 10 RGB frames")
    if float(data.get("single_bev_extent_m", 0.0)) != 6.5:
        raise ValueError("single BEV extent must remain 6.5 m")
    if int(data.get("single_bev_output_size", 0)) != 512:
        raise ValueError("single BEV output must remain 512x512")
    if float(data.get("merged_bev_extent_m", 0.0)) != 10.0:
        raise ValueError("merged BEV extent must remain 10 m")
    if int(data.get("merged_bev_output_size", 0)) != 800:
        raise ValueError("merged BEV output must remain 800x800")
    grid_contract = (
        ("single_bev_extent_m", 6.5),
        ("single_bev_output_size", 512),
        ("merged_bev_extent_m", 10.0),
        ("merged_bev_output_size", 800),
    )
    for name, expected in grid_contract:
        if float(model.get(name, 0.0)) != float(expected):
            raise ValueError(f"model.{name} must remain {expected}")
        if float(model[name]) != float(data[name]):
            raise ValueError(f"data.{name} and model.{name} must match")
    if int(model.get("single_latent_bev_size", 0)) != int(
        model["single_bev_output_size"]
    ):
        raise ValueError(
            "P2B single_latent_bev_size must equal the native 512 output size"
        )
    probability_model = str(model.get("probability_model", ""))
    expected_pipeline = {
        "evidential": "P2B-NLL",
        "bce": "P2B-BCE",
    }.get(probability_model)
    if expected_pipeline is None:
        raise ValueError("model.probability_model must be evidential or bce")
    if model.get("pipeline_variant") != expected_pipeline:
        raise ValueError(
            f"model.pipeline_variant must be {expected_pipeline} for {probability_model}"
        )
    if training.get("pipeline") != expected_pipeline:
        raise ValueError(f"training.pipeline must be {expected_pipeline}")
    if training.get("stage") not in ("scale_only", "bev_only", "joint"):
        raise ValueError("training.stage is invalid")
    enabled = training.get("enabled_bev_branches", ["single", "merged"])
    if not enabled or set(enabled).difference(("single", "merged")):
        raise ValueError("enabled_bev_branches must select single and/or merged")
    forbidden = {
        "direct_priority_pcgrad",
        "guessed_completion_dice_weight",
        "confidence_calibration_weight",
        "observation_relation_weight",
        "surface_tolerance_latent_cell_fraction",
        "ray_sequence_weight",
        "first_hit_weight",
        "first_hit_distance_weight",
        "surface_continuity_weight",
        "gate_bce_weight",
        "gate_monotonic_weight",
        "fusion_gate_weight",
        "routing_pixel_weight",
        "surface_gate_pixel_weight",
    }.intersection(training)
    if forbidden:
        raise ValueError(f"P2B forbids legacy loss fields: {sorted(forbidden)}")
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
            "single_latent_bev_size",
            "merged_latent_bev_size",
        ),
        "model",
    )
    if int(model["hidden_dim"]) % int(model["attention_heads"]):
        raise ValueError("model.hidden_dim must be divisible by attention_heads")
    _positive(
        training,
        (
            "required_cuda_devices",
            "epochs",
            "batch_size",
            "learning_rate",
            "bev_loss_weight",
            "scale_loss_weight",
            "depth_scale_loss_weight",
            "gradient_clip_norm",
            "checkpoint_every_steps",
            "log_every_steps",
            "validation_batches",
        ),
        "training",
    )
    for name in (
        "observed_gate_pixel_weight",
        "guessed_pixel_weight",
        "guessed_surface_weight",
        "guessed_free_weight",
        "guessed_visible_surface_weight",
        "guessed_hidden_occupied_weight",
        "wrong_evidence_kl_weight",
        "support_bce_weight",
        "support_dice_weight",
        "single_task_weight",
        "merged_task_weight",
    ):
        value = float(training.get(name, 0.0))
        if value < 0.0:
            raise ValueError(f"training.{name} cannot be negative")
    deprecated_gate_keys = (
        "gate_observed_free_weight",
        "gate_visible_surface_weight",
        "gate_hidden_weight",
        "fused_surface_weight",
    )
    configured_deprecated = [
        name for name in deprecated_gate_keys if name in training
    ]
    if configured_deprecated:
        raise ValueError(
            "surface-v2 forbids Gate-coupled loss keys: "
            f"{configured_deprecated}; use guessed_surface_weight"
        )
    group_weight_sets = {
        "guessed": (
            "guessed_free_weight",
            "guessed_visible_surface_weight",
            "guessed_hidden_occupied_weight",
        ),
    }
    defaults = {
        "guessed_free_weight": 0.35,
        "guessed_visible_surface_weight": 0.40,
        "guessed_hidden_occupied_weight": 0.25,
    }
    for group, names in group_weight_sets.items():
        if sum(float(training.get(name, defaults[name])) for name in names) <= 0.0:
            raise ValueError(f"training.{group} group weights must have positive sum")
    schedules = (
        ("wrong_evidence_zero_fraction", "wrong_evidence_ramp_fraction", 0.20, 0.10),
        (
            "hidden_occupied_zero_fraction",
            "hidden_occupied_ramp_fraction",
            0.10,
            0.15,
        ),
    )
    for zero_name, ramp_name, zero_default, ramp_default in schedules:
        zero_fraction = float(training.get(zero_name, zero_default))
        ramp_fraction = float(training.get(ramp_name, ramp_default))
        if not 0.0 <= zero_fraction <= 1.0:
            raise ValueError(f"training.{zero_name} must be in [0,1]")
        if not 0.0 <= ramp_fraction <= 1.0:
            raise ValueError(f"training.{ramp_name} must be in [0,1]")
        if zero_fraction + ramp_fraction > 1.0:
            raise ValueError(
                f"training.{zero_name} + training.{ramp_name} cannot exceed 1"
            )
    if probability_model == "bce" and float(
        training.get("wrong_evidence_kl_weight", 0.0)
    ) != 0.0:
        raise ValueError("P2B-BCE must not configure an evidence KL")
    if probability_model == "evidential" and not 0.0 <= float(
        training.get("wrong_evidence_kl_weight", 0.0)
    ) <= 0.05:
        raise ValueError("P2B-NLL wrong-evidence KL must be in [0,0.05]")
