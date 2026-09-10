"""Legacy probability imports for pre-rename P1B tooling."""

from .p1b_probability import (
    ProbabilityModel,
    compose_semantic,
    decode_binary_prediction,
    fuse_pixel_routing,
)

__all__ = [
    "ProbabilityModel",
    "compose_semantic",
    "decode_binary_prediction",
    "fuse_pixel_routing",
]
