"""Legacy import aliases; new code must import :mod:`p1b_targets`."""

from .p1b_targets import (
    ROUTING_CLASS_COUNT,
    ROUTING_CLASS_NAMES,
    ROUTING_GUESSED_FREE,
    ROUTING_GUESSED_OCCUPIED,
    ROUTING_OBSERVED_FREE,
    P1BRegionMasks,
    p1b_region_masks,
)

P2BRegionMasks = P1BRegionMasks
p2b_region_masks = p1b_region_masks

__all__ = [
    "P2BRegionMasks",
    "ROUTING_CLASS_COUNT",
    "ROUTING_CLASS_NAMES",
    "ROUTING_GUESSED_FREE",
    "ROUTING_GUESSED_OCCUPIED",
    "ROUTING_OBSERVED_FREE",
    "p2b_region_masks",
]
