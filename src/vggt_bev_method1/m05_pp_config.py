from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .m05_plus_config import validate_m05_plus_config
from .m05_plus_contract import PIPELINE_ID as M05_PLUS_PIPELINE_ID
from .m05_pp_contract import (
    CHECKPOINT_SCHEMA,
    EXPECTED_PREFIX_TOKENS,
    PIPELINE_ID,
    RESOLUTION_HIERARCHY,
)

__all__ = [
    "CHECKPOINT_SCHEMA",
    "PIPELINE_ID",
    "load_m05_pp_config",
    "validate_m05_pp_config",
]


def load_m05_pp_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        config = tomllib.load(stream)
    validate_m05_pp_config(config)
    config["_config_path"] = str(resolved)
    return config


def validate_m05_pp_config(config: dict[str, Any]) -> None:
    """Validate the coarse-reasoning/high-resolution-correction M05++ contract."""
    model = config.get("model", {})
    training = config.get("training", {})
    data = config.get("data", {})
    if model.get("pipeline_variant") != PIPELINE_ID:
        raise ValueError(f"model.pipeline_variant must be {PIPELINE_ID}")
    if training.get("pipeline") != PIPELINE_ID:
        raise ValueError(f"training.pipeline must be {PIPELINE_ID}")

    # Reuse the complete M05+ data, loss, token-role and no-geometry contract.
    # Only normalize the architecture identity and the deliberately smaller
    # prefix trunk while the parent validator checks all shared invariants.
    compatible = deepcopy(config)
    compatible["model"]["pipeline_variant"] = M05_PLUS_PIPELINE_ID
    compatible["training"]["pipeline"] = M05_PLUS_PIPELINE_ID
    compatible["model"]["prefix_context_hidden_dim"] = 1024
    compatible["model"]["prefix_context_heads"] = 16
    compatible["model"]["prefix_context_layers"] = 8
    validate_m05_plus_config(compatible)

    exact = {
        "coarse_bev_size": 256,
        "merged_bev_output_size": 512,
        "hidden_dim": 96,
        "query_content_dim": 64,
        "patch_stream_dim": 96,
        "prefix_context_hidden_dim": 512,
        "prefix_context_heads": 8,
        "prefix_context_layers": 4,
        "fine_correction_layers": 1,
    }
    for key, expected in exact.items():
        if int(model.get(key, 0)) != expected:
            raise ValueError(f"M05++ model.{key} must be {expected}")
    if int(data.get("merged_source_output_size", 0)) != 512:
        raise ValueError("M05++ final supervision must remain native 512x512")
    if model.get("resolution_hierarchy") != RESOLUTION_HIERARCHY:
        raise ValueError("M05++ resolution hierarchy is invalid")
    if not bool(model.get("latest_high_resolution_skip", False)):
        raise ValueError("M05++ requires the latest-patch high-resolution skip")
    if model.get("final_refinement_mode") != "depthwise_separable":
        raise ValueError("M05++ final 512 refinement must remain lightweight")
    if int(model.get("maximum_prefix_tokens", 0)) != EXPECTED_PREFIX_TOKENS:
        raise ValueError("M05++ requires one Camera and sixteen Register tokens")
    if str(data.get("missing_depth_policy", "error")) not in {
        "error",
        "invalid",
    }:
        raise ValueError("data.missing_depth_policy must be error or invalid")

