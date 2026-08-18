from __future__ import annotations

from dataclasses import dataclass

import torch

from vggt_bev_method1.config import LabelValues

DEFAULT_LABEL_VALUES = LabelValues()

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
class P1BRegionMasks:
    valid: torch.Tensor
    observed_free: torch.Tensor
    visible_surface: torch.Tensor
    guessed: torch.Tensor
    hidden_guessed: torch.Tensor
    guessed_free: torch.Tensor
    guessed_occupied: torch.Tensor
    hidden_guessed_occupied: torch.Tensor
    occupied: torch.Tensor
    observed_gate_target: torch.Tensor
    routing_target: torch.Tensor


def p1b_region_masks(
    complete: torch.Tensor,
    visible: torch.Tensor,
    support: torch.Tensor,
    *,
    labels: LabelValues = DEFAULT_LABEL_VALUES,
) -> P1BRegionMasks:
    """Create exact per-pixel routing labels from existing BEV targets.

    No ray, contour, dilation or morphology operation is used. The masked
    visible target defines observed-free pixels directly. Every other valid
    complete pixel, including visible occupied boundary pixels, is assigned to
    the guessed expert as either guessed-free or guessed-occupied.  Exact
    ``masked == occupied`` cells are additionally exposed as a loss-only
    visible-surface subset.  This does not add a routing class or model output.
    """

    if not (complete.shape == visible.shape == support.shape):
        raise ValueError("complete, visible and support targets must align")
    valid = complete != labels.unknown
    if not torch.equal(valid, support.bool()):
        raise ValueError("P1B support must equal the valid complete target")
    visible_known = (visible != labels.unknown) & valid
    if bool((visible_known & (visible != complete)).any()):
        raise ValueError("visible target disagrees with complete target")
    occupied = complete == labels.occupied
    observed_free = visible_known & ~occupied
    visible_surface = visible_known & occupied
    guessed = valid & ~observed_free
    hidden_guessed = guessed & ~visible_surface
    guessed_free = guessed & ~occupied
    guessed_occupied = guessed & occupied
    hidden_guessed_occupied = guessed_occupied & ~visible_surface

    if not torch.equal(
        observed_free | visible_surface | hidden_guessed,
        valid,
    ):
        raise ValueError("P1B direct/hidden loss subsets must partition support")
    if bool(
        (observed_free & visible_surface).any()
        or (observed_free & hidden_guessed).any()
        or (visible_surface & hidden_guessed).any()
    ):
        raise ValueError("P1B direct/hidden loss subsets overlap")

    routing_target = torch.full_like(
        complete,
        ROUTING_GUESSED_FREE,
        dtype=torch.long,
    )
    routing_target[observed_free] = ROUTING_OBSERVED_FREE
    routing_target[guessed_free] = ROUTING_GUESSED_FREE
    routing_target[guessed_occupied] = ROUTING_GUESSED_OCCUPIED
    return P1BRegionMasks(
        valid=valid,
        observed_free=observed_free,
        visible_surface=visible_surface,
        guessed=guessed,
        hidden_guessed=hidden_guessed,
        guessed_free=guessed_free,
        guessed_occupied=guessed_occupied,
        hidden_guessed_occupied=hidden_guessed_occupied,
        occupied=occupied,
        observed_gate_target=observed_free.to(torch.float32),
        routing_target=routing_target,
    )
