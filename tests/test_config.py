from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from vggt_bev_method1.config import load_config, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _valid_config() -> dict:
    failures = []
    for path in sorted((PROJECT_ROOT / "configs").glob("p1a_*.toml")):
        try:
            return load_config(path)
        except ValueError as error:
            failures.append(f"{path.name}: {error}")
    raise AssertionError(
        "deployment has no valid P1A v3 config:\n" + "\n".join(failures)
    )


def test_active_p1a_config_enforces_direct_observation_priority() -> None:
    config = _valid_config()
    training = config["training"]
    assert training["direct_priority_pcgrad"] is True
    assert training["guessed_region_weight"] <= (
        0.25 * training["observed_region_weight"]
    )
    assert training["maximum_guessed_to_direct_gradient_ratio"] <= 0.25
    assert training["guessed_completion_class_weights"] == [1.0, 1.0]
    assert training["guessed_supervision_warmup_fraction"] >= 0.05
    assert training["guessed_supervision_ramp_fraction"] >= 0.10


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("direct_priority_pcgrad", False, "direct_priority_pcgrad=true"),
        ("guessed_region_weight", 0.26, "guessed_region_weight"),
        (
            "maximum_guessed_to_direct_gradient_ratio",
            0.26,
            "maximum_guessed_to_direct_gradient_ratio",
        ),
        (
            "guessed_supervision_warmup_fraction",
            0.04,
            "direct-only warm-up",
        ),
        ("guessed_supervision_ramp_fraction", 0.09, "at least a 10% ramp"),
    ],
)
def test_direct_priority_contract_rejects_unsafe_overrides(
    key: str,
    value: float | bool,
    message: str,
) -> None:
    config = deepcopy(_valid_config())
    config["training"][key] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_direct_priority_contract_rejects_guessed_class_bias() -> None:
    config = deepcopy(_valid_config())
    config["training"]["guessed_completion_class_weights"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="balanced"):
        validate_config(config)
