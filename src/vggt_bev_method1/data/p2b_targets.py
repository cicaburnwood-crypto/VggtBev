from __future__ import annotations

from dataclasses import dataclass

import torch

from vggt_bev_method1.config import LabelValues

ROUTING_OBSERVED_FREE = 0
ROUTING_GUESSED_FREE = 1
ROUTING_GUESSED_OCCUPIED = 2
ROUTING_CLASS_COUNT = 3
ROUTING_CLASS_NAMES = (
    "observed_free",
    "guessed_free",
    "guessed_occupied",
)


@dataclass(frozen=True)
class P2BRegionMasks:
    valid: torch.Tensor
    observed_free: torch.Tensor
    guessed: torch.Tensor
    guessed_free: torch.Tensor
    guessed_occupied: torch.Tensor
    occupied: torch.Tensor
    observed_gate_target: torch.Tensor
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
    visible target defines observed-free pixels directly. Every other valid
    complete pixel, including visible occupied boundary pixels, is assigned to
    the guessed expert as either guessed-free or guessed-occupied.
    """

    if not (complete.shape == visible.shape == support.shape):
        raise ValueError("complete, visible and support targets must align")
    valid = complete != labels.unknown
    if not torch.equal(valid, support.bool()):
        raise ValueError("P2B support must equal the valid complete target")
    visible_known = (visible != labels.unknown) & valid
    if bool((visible_known & (visible != complete)).any()):
        raise ValueError("visible target disagrees with complete target")
    occupied = complete == labels.occupied
    observed_free = visible_known & ~occupied
    guessed = valid & ~observed_free
    guessed_free = guessed & ~occupied
    guessed_occupied = guessed & occupied

    routing_target = torch.full_like(
        complete,
        ROUTING_GUESSED_FREE,
        dtype=torch.long,
    )
    routing_target[observed_free] = ROUTING_OBSERVED_FREE
    routing_target[guessed_free] = ROUTING_GUESSED_FREE
    routing_target[guessed_occupied] = ROUTING_GUESSED_OCCUPIED
    return P2BRegionMasks(
        valid=valid,
        observed_free=observed_free,
        guessed=guessed,
        guessed_free=guessed_free,
        guessed_occupied=guessed_occupied,
        occupied=occupied,
        observed_gate_target=observed_free.to(torch.float32),
        routing_target=routing_target,
    )
