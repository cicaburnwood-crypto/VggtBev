"""Legacy import aliases for checkpoints/config tooling created before P1B renaming."""

from .p1b_config import load_p1b_config, validate_p1b_config

load_p2b_config = load_p1b_config
validate_p2b_config = validate_p1b_config

__all__ = ["load_p2b_config", "validate_p2b_config"]
