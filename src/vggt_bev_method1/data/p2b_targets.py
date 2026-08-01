from __future__ import annotations

from dataclasses import dataclass

import torch

from vggt_bev_method1.config import LabelValues

ROUTING_OBSERVED_FREE = 0
ROUTING_OBSERVED_SURFACE = 1
ROUTING_GUESSED = 2
ROUTING_CLASS_COUNT = 3
ROUTING_CLASS_NAMES = (
    "observed_free",
    "observed_surface",
    "guessed",
)


@dataclass(frozen=True)
class P2BRegionMasks:
    valid: torch.Tensor
    observed: torch.Tensor
    guessed: torch.Tensor
    observed_free: torch.Tensor
    observed_surface: torch.Tensor
    occupied: torch.Tensor
    observed_gate_target: torch.Tensor
    surface_gate_target: torch.Tensor
    routing_target: torch.Tensor


def p2b_region_masks(
    complete: torch.Tensor,
    visible: torch.Tensor,
    support: torch.Tensor,
    *,
    labels: LabelValues = LabelValues(),
) -> P2BRegionMasks:
    """Create exact per-pixel routing labels from existing BEV targets.

    No ray, contour, dilation or morphology operation is used. The masked
    visible target defines observed-free and observed-surface pixels directly;
    valid complete pixels absent from that target are completion pixels.
    """

    if not (complete.shape == visible.shape == support.shape):
        raise ValueError("complete, visible and support targets must align")
    valid = complete != labels.unknown
    if not torch.equal(valid, support.bool()):
        raise ValueError("P2B support must equal the valid complete target")
    observed = (visible != labels.unknown) & valid
    guessed = valid & ~observed
    if bool((observed & (visible != complete)).any()):
        raise ValueError("visible target disagrees with complete target")
    occupied = complete == labels.occupied
    observed_free = observed & ~occupied
    observed_surface = observed & occupied

    routing_target = torch.full_like(
        complete,
        ROUTING_GUESSED,
        dtype=torch.long,
    )
    routing_target[observed_free] = ROUTING_OBSERVED_FREE
    routing_target[observed_surface] = ROUTING_OBSERVED_SURFACE
    return P2BRegionMasks(
        valid=valid,
        observed=observed,
        guessed=guessed,
        observed_free=observed_free,
        observed_surface=observed_surface,
        occupied=occupied,
        observed_gate_target=observed.to(torch.float32),
        surface_gate_target=observed_surface.to(torch.float32),
        routing_target=routing_target,
    )
