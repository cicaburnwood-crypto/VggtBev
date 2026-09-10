from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path
from typing import Any


def load_p1b_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        config = tomllib.load(stream)
    validate_p1b_config(config)
    config["_config_path"] = str(resolved)
    return config


def _positive(section: dict, names: tuple[str, ...], prefix: str) -> None:
    for name in names:
        if float(section.get(name, 0.0)) <= 0.0:
            raise ValueError(f"{prefix}.{name} must be positive")


def validate_p1b_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "scale_fit", "training"):
        if section not in config:
            raise ValueError(f"configuration is missing [{section}]")
    data = config["data"]
    model = config["model"]
    training = config["training"]
    if bool(data.get("require_gt_void_mask", False)):
        if not str(data.get("void_coverage_index", "")).strip():
            raise ValueError(
                "data.void_coverage_index is required when "
                "data.require_gt_void_mask=true"
            )
        expected_void_algorithm = str(
            data.get("required_void_coverage_algorithm", "")
        )
        if expected_void_algorithm != "strict-solid-voxel-or-navmesh-coverage-v4":
            raise ValueError(
                "data.required_void_coverage_algorithm must select strict v4"
            )
    if data.get("supervision") != "metric_fov_complete_evidential":
        raise ValueError("P1B reuses the existing FOV-complete dataset contract")
    if data.get("coordinate_mode") != "p1b_fixed_metric":
        raise ValueError("data.coordinate_mode must be p1b_fixed_metric")
    if int(data.get("maximum_history", 0)) > 10:
        raise ValueError("P1B maximum_history must not exceed 10 RGB frames")
    if data.get("session_selection_order", "lexicographic_session_key") not in (
        "lexicographic_session_key",
        "completion_time",
    ):
        raise ValueError("data.session_selection_order is invalid")
    if float(data.get("single_bev_extent_m", 0.0)) != 6.5:
        raise ValueError("single BEV extent must remain 6.5 m")
    if int(data.get("single_bev_output_size", 0)) != 512:
        raise ValueError("single BEV output must remain 512x512")
    if float(data.get("merged_source_extent_m", 10.0)) != 10.0:
        raise ValueError("merged source GT extent must remain 10 m")
    merged_contract = (
        float(data.get("merged_bev_extent_m", 0.0)),
        int(data.get("merged_bev_output_size", 0)),
    )
    if merged_contract not in ((10.0, 800), (6.5, 512)):
        raise ValueError(
            "merged BEV grid must be either native 10 m/800 or the "
            "latest-ego center-cropped 6.5 m/512 variant"
        )
    grid_contract = (
        ("single_bev_extent_m", 6.5),
        ("single_bev_output_size", 512),
        ("merged_bev_extent_m", merged_contract[0]),
        ("merged_bev_output_size", merged_contract[1]),
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
            "P1B single_latent_bev_size must equal the native 512 output size"
        )
    if int(model.get("merged_latent_bev_size", 0)) != int(
        model["merged_bev_output_size"]
    ):
        raise ValueError(
            "P1B Merged routing geometry must use native output resolution"
        )
    probability_model = str(model.get("probability_model", ""))
    expected_pipeline = {
        "evidential": "P1B-NLL",
        "bce": "P1B-BCE",
    }.get(probability_model)
    if expected_pipeline is None:
        raise ValueError("model.probability_model must be evidential or bce")
    if model.get("pipeline_variant") != expected_pipeline:
        raise ValueError(
            f"model.pipeline_variant must be {expected_pipeline} for {probability_model}"
        )
    projector_layout = str(
        model.get("projector_layout", "branch-specific-single-merged-v1")
    )
    if projector_layout != "branch-specific-single-merged-v1":
        raise ValueError(
            "model.projector_layout must be branch-specific-single-merged-v1"
        )
    if training.get("pipeline") != expected_pipeline:
        raise ValueError(f"training.pipeline must be {expected_pipeline}")
    if training.get("stage") not in ("scale_only", "bev_only", "joint"):
        raise ValueError("training.stage is invalid")
    bev_objective = str(training.get("bev_objective", "full"))
    if bev_objective not in (
        "full",
        "fov_support_only",
        "fov_support_and_observed_gate",
    ):
        raise ValueError("training.bev_objective is invalid")
    enabled = training.get("enabled_bev_branches", ["single", "merged"])
    if not enabled or set(enabled).difference(("single", "merged")):
        raise ValueError("enabled_bev_branches must select single and/or merged")
    freeze_single = training.get("freeze_single_to_baseline", False)
    if not isinstance(freeze_single, bool):
        raise ValueError("training.freeze_single_to_baseline must be a boolean")
    if freeze_single:
        if not str(training.get("single_baseline_checkpoint", "")).strip():
            raise ValueError(
                "training.single_baseline_checkpoint is required when Single "
                "is frozen to a baseline"
            )
        if training.get("stage") != "bev_only":
            raise ValueError(
                "frozen-Single training must use training.stage=bev_only"
            )
        if tuple(enabled) != ("merged",):
            raise ValueError(
                "frozen-Single training must enable only the merged BEV branch"
            )
        if float(training.get("single_task_weight", 0.0)) != 0.0:
            raise ValueError(
                "frozen-Single training requires single_task_weight=0"
            )
        if float(training.get("merged_task_weight", 0.0)) <= 0.0:
            raise ValueError(
                "frozen-Single training requires merged_task_weight>0"
            )
    seed_merged_routing = training.get(
        "seed_merged_routing_from_single", True
    )
    fresh_merged_routing = training.get(
        "fresh_merged_routing_initialization", False
    )
    if not isinstance(seed_merged_routing, bool):
        raise ValueError(
            "training.seed_merged_routing_from_single must be a boolean"
        )
    if not isinstance(fresh_merged_routing, bool):
        raise ValueError(
            "training.fresh_merged_routing_initialization must be a boolean"
        )
    if not isinstance(
        training.get("routing_warmstart_allow_manifest_change", False), bool
    ):
        raise ValueError(
            "training.routing_warmstart_allow_manifest_change must be a boolean"
        )
    if fresh_merged_routing:
        if seed_merged_routing:
            raise ValueError(
                "fresh Merged Routing forbids Single-projector seeding"
            )
        if str(training.get("routing_warmstart_checkpoint", "")).strip():
            raise ValueError(
                "fresh Merged Routing forbids a routing warm-start checkpoint"
            )
    if bev_objective in (
        "fov_support_only",
        "fov_support_and_observed_gate",
    ):
        if not freeze_single:
            raise ValueError("FOV-support-only training must freeze Single")
        if training.get("stage") != "bev_only" or tuple(enabled) != ("merged",):
            raise ValueError(
                "FOV-support-only training requires bev_only Merged branch"
            )
        variant = str(training.get("support_loss_variant", ""))
        if variant not in (
            "balanced_bce_dice",
            "balanced_bce_dice_boundary",
            "boundary_tversky",
            "role_balanced_contour",
        ):
            raise ValueError("training.support_loss_variant is invalid")
        forbidden_objective_weights = (
            "guessed_pixel_weight",
            "guessed_surface_weight",
            "guessed_free_weight",
            "guessed_visible_surface_weight",
            "guessed_hidden_occupied_weight",
            "wrong_evidence_kl_weight",
        )
        nonzero = [
            name
            for name in forbidden_objective_weights
            if float(training.get(name, 0.0)) != 0.0
        ]
        if nonzero:
            raise ValueError(
                "FOV-support-only training forbids non-support loss weights: "
                f"{nonzero}"
            )
        observed_gate_weight = float(
            training.get("observed_gate_pixel_weight", 0.0)
        )
        if (
            bev_objective == "fov_support_only"
            and observed_gate_weight != 0.0
        ):
            raise ValueError("FOV-support-only training forbids Observed Gate loss")
        if (
            bev_objective == "fov_support_and_observed_gate"
            and observed_gate_weight <= 0.0
        ):
            raise ValueError(
                "combined routing-geometry training requires Observed Gate loss"
            )
        gate_start_fraction = float(
            training.get("observed_gate_start_fraction", 0.0)
        )
        if not 0.0 <= gate_start_fraction < 1.0:
            raise ValueError(
                "training.observed_gate_start_fraction must be in [0, 1)"
            )
        if float(training.get("support_task_weight", 1.0)) <= 0.0:
            raise ValueError("routing-geometry support_task_weight must be positive")
        support_weights = (
            float(training.get("support_bce_weight", 0.0)),
            float(training.get("support_dice_weight", 0.0)),
            float(training.get("support_tversky_weight", 0.0)),
            float(training.get("support_boundary_weight", 0.0)),
        )
        role_support_weights = (
            float(training.get("support_interior_weight", 0.0)),
            float(training.get("support_edge_weight", 0.0)),
            float(training.get("support_contour_weight", 0.0)),
            float(training.get("support_region_dice_weight", 0.0)),
        )
        if variant != "role_balanced_contour" and sum(support_weights) <= 0.0:
            raise ValueError("FOV-support-only loss weights must have positive sum")
        if variant == "balanced_bce_dice" and any(support_weights[2:]):
            raise ValueError(
                "balanced_bce_dice forbids Tversky and boundary weights"
            )
        if variant == "balanced_bce_dice_boundary" and (
            support_weights[1] <= 0.0
            or support_weights[2] != 0.0
            or support_weights[3] <= 0.0
        ):
            raise ValueError(
                "balanced_bce_dice_boundary requires positive Dice/boundary "
                "weights and zero Tversky weight"
            )
        if variant == "boundary_tversky" and (
            support_weights[2] <= 0.0 or support_weights[3] <= 0.0
        ):
            raise ValueError(
                "boundary_tversky requires positive Tversky and boundary weights"
            )
        if variant == "role_balanced_contour" and (
            any(support_weights) or min(role_support_weights) < 0.0
            or abs(sum(role_support_weights) - 1.0) > 1e-6
        ):
            raise ValueError(
                "role_balanced_contour requires zero legacy support weights "
                "and non-negative role weights summing to 1"
            )
        gate_variant = str(
            training.get("observed_gate_loss_variant", "balanced_bce")
        )
        if gate_variant not in ("balanced_bce", "role_balanced_contour"):
            raise ValueError("training.observed_gate_loss_variant is invalid")
        gate_role_weights = (
            float(training.get("observed_gate_interior_weight", 0.0)),
            float(training.get("observed_gate_edge_weight", 0.0)),
            float(training.get("observed_gate_contour_weight", 0.0)),
            float(training.get("observed_gate_dice_weight", 0.0)),
        )
        if (
            bev_objective == "fov_support_and_observed_gate"
            and gate_variant == "role_balanced_contour"
            and (
                min(gate_role_weights) < 0.0
                or abs(sum(gate_role_weights) - 1.0) > 1e-6
            )
        ):
            raise ValueError(
                "role-balanced Gate weights must be non-negative and sum to 1"
            )
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
        raise ValueError(f"P1B forbids legacy loss fields: {sorted(forbidden)}")
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
        "support_tversky_weight",
        "support_boundary_weight",
        "support_interior_weight",
        "support_edge_weight",
        "support_contour_weight",
        "support_region_dice_weight",
        "observed_gate_interior_weight",
        "observed_gate_edge_weight",
        "observed_gate_contour_weight",
        "observed_gate_dice_weight",
        "single_task_weight",
        "merged_task_weight",
    ):
        value = float(training.get(name, 0.0))
        if value < 0.0:
            raise ValueError(f"training.{name} cannot be negative")
    if int(training.get("support_boundary_radius", 1)) < 1:
        raise ValueError("training.support_boundary_radius must be at least one")
    if int(training.get("observed_gate_boundary_radius", 1)) < 1:
        raise ValueError(
            "training.observed_gate_boundary_radius must be at least one"
        )
    tversky_pair = (
        float(training.get("support_tversky_false_positive_weight", 0.70)),
        float(training.get("support_tversky_false_negative_weight", 0.30)),
    )
    if min(tversky_pair) <= 0.0 or abs(sum(tversky_pair) - 1.0) > 1e-6:
        raise ValueError("FOV-support Tversky FP/FN weights must be positive and sum to 1")
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
        if bev_objective != "full":
            continue
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
        raise ValueError("P1B-BCE must not configure an evidence KL")
    if probability_model == "evidential" and not 0.0 <= float(
        training.get("wrong_evidence_kl_weight", 0.0)
    ) <= 0.05:
        raise ValueError("P1B-NLL wrong-evidence KL must be in [0,0.05]")
