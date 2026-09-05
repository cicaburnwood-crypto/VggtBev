from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .m05_config import validate_m05_config
from .m05_plus_contract import (
    BEV_REFERENCE_FRAME,
    CHECKPOINT_SCHEMA,
    DATASET_INPUT_ORDER,
    EXPECTED_PREFIX_TOKENS,
    GEOMETRY_CONDITIONING,
    PATCH_FUSION,
    PATCH_TOKEN_STREAMS,
    PIPELINE_ID,
    PREFIX_CONDITIONING,
    PREFIX_CONTEXT_TRAINING,
    VGGT_INPUT_ORDER,
)

__all__ = [
    "CHECKPOINT_SCHEMA",
    "PIPELINE_ID",
    "load_m05_plus_config",
    "validate_m05_plus_config",
]


def load_m05_plus_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        config = tomllib.load(stream)
    validate_m05_plus_config(config)
    config["_config_path"] = str(resolved)
    return config


def validate_m05_plus_config(config: dict[str, Any]) -> None:
    """Validate M05+ while retaining the proven M05 data/loss contract."""
    model = config.get("model", {})
    training = config.get("training", {})
    if model.get("pipeline_variant") != PIPELINE_ID:
        raise ValueError(f"model.pipeline_variant must be {PIPELINE_ID}")
    if training.get("pipeline") != PIPELINE_ID:
        raise ValueError(f"training.pipeline must be {PIPELINE_ID}")

    # Reuse every invariant concerning fixed-metric data and the established
    # loss. Translate only architecture identity fields for base validation.
    compatible = deepcopy(config)
    compatible["model"]["pipeline_variant"] = (
        "M05-LATEST-ANCHORED-REVERSE-GATED-METRIC-MERGED-SCALE-NLL"
    )
    compatible["model"]["geometry_conditioning"] = (
        "latest_anchor_reverse_gated_history"
    )
    compatible["model"]["history_update_layers"] = int(
        model.get("temporal_proposal_layers", 0)
    )
    compatible["model"]["refinement_layers"] = int(
        model.get("shared_refinement_layers", 0)
    )
    compatible["model"]["structured_prefix_readout"] = True
    compatible["model"]["implicit_geometry_hidden_dim"] = int(
        model.get("prefix_context_hidden_dim", 0)
    )
    compatible["model"]["implicit_geometry_heads"] = int(
        model.get("prefix_context_heads", 0)
    )
    compatible["model"]["implicit_geometry_layers"] = int(
        model.get("prefix_context_layers", 0)
    )
    compatible["training"]["pipeline"] = compatible["model"][
        "pipeline_variant"
    ]
    validate_m05_config(compatible)

    if model.get("geometry_conditioning") != GEOMETRY_CONDITIONING:
        raise ValueError("M05+ forbids explicit geometry conditioning")
    if model.get("patch_fusion") != PATCH_FUSION:
        raise ValueError("M05+ requires DPT-lite top-down patch fusion")
    if model.get("patch_token_streams") != PATCH_TOKEN_STREAMS:
        raise ValueError("M05+ must preserve local/global patch-token streams")
    if model.get("prefix_conditioning") != PREFIX_CONDITIONING:
        raise ValueError("M05+ requires per-cell role-separated prefix attention")
    if model.get("prefix_context_training") != PREFIX_CONTEXT_TRAINING:
        raise ValueError("M05+ prefix context must train jointly through BEV only")
    if bool(model.get("explicit_geometry_module", True)):
        raise ValueError("M05+ forbids explicit geometry modules")
    if bool(model.get("geometry_auxiliary_loss", True)):
        raise ValueError("M05+ forbids a separately supervised geometry head")
    if model.get("temporal_attention_mode") != "per_query_frame_softmax":
        raise ValueError("M05+ temporal attention must softmax across frames per query")
    if not bool(model.get("temporal_null_enabled", False)):
        raise ValueError("M05+ requires the learned no-history/null candidate")
    if model.get("dataset_input_order") != DATASET_INPUT_ORDER:
        raise ValueError("M05+ dataset windows must be chronological")
    if model.get("vggt_input_order") != VGGT_INPUT_ORDER:
        raise ValueError("M05+ must give the latest RGB frame to VGGT first")
    if model.get("bev_reference_frame") != BEV_REFERENCE_FRAME:
        raise ValueError("M05+ BEV must remain latest-frame ego-centric")
    required = {
        "hidden_dim": 96,
        "query_content_dim": 64,
        "latest_decoder_layers": 3,
        "temporal_proposal_layers": 2,
        "shared_refinement_layers": 2,
        "routing_refinement_layers": 1,
        "evidence_refinement_layers": 1,
        "patch_stream_dim": 48,
        "prefix_context_hidden_dim": 1024,
        "prefix_context_heads": 16,
        "prefix_context_layers": 8,
        "frame_reliability_hidden_dim": 256,
        "scale_hidden_dim": 64,
        "scale_decoder_layers": 2,
    }
    for key, minimum in required.items():
        if int(model.get(key, 0)) < minimum:
            raise ValueError(f"model.{key} must be at least {minimum} for M05+")
    if int(model["hidden_dim"]) % int(model["attention_heads"]):
        raise ValueError("M05+ hidden_dim must be divisible by attention_heads")
    if int(model["prefix_context_hidden_dim"]) % int(
        model["prefix_context_heads"]
    ):
        raise ValueError("prefix context width must divide evenly across heads")
    if int(model["vggt_token_dim"]) % 2:
        raise ValueError("VGGT local/global concatenated token width must be even")
    if int(model["scale_hidden_dim"]) % int(model["attention_heads"]):
        raise ValueError("scale_hidden_dim must divide evenly across attention heads")
    if int(model.get("maximum_prefix_tokens", 0)) != EXPECTED_PREFIX_TOKENS:
        raise ValueError(
            f"M05+ production prefix contract requires {EXPECTED_PREFIX_TOKENS} tokens"
        )
    null_probability = float(
        model.get("temporal_null_initial_probability", 0.0)
    )
    if not 0.0 < null_probability < 1.0:
        raise ValueError(
            "model.temporal_null_initial_probability must be in (0,1)"
        )
    if "temporal_null_initial_bias" in model:
        raise ValueError(
            "temporal_null_initial_bias is ambiguous; use "
            "temporal_null_initial_probability"
        )
    retired_geometry_keys = (
        "implicit_geometry_hidden_dim",
        "implicit_geometry_heads",
        "implicit_geometry_layers",
    )
    present_retired = [key for key in retired_geometry_keys if key in model]
    if present_retired:
        raise ValueError(
            "M05+ v3 replaces the implicit geometry trunk with a jointly "
            f"trained prefix-token context path: {present_retired}"
        )

    if training.get("latest_auxiliary_loss_mode") != "merged_primary_additive":
        raise ValueError(
            "M05+ requires merged-primary additive latest supervision"
        )
    if "latest_auxiliary_loss_weight" in training:
        raise ValueError(
            "latest_auxiliary_loss_weight is the retired convex-mix field"
        )
    latest_multiplier = float(
        training.get("latest_auxiliary_multiplier", -1.0)
    )
    if not 0.0 <= latest_multiplier <= 1.0:
        raise ValueError(
            "training.latest_auxiliary_multiplier must be in [0,1]"
        )
