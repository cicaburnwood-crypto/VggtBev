"""Legacy import aliases; new code must import :mod:`p1b_losses`."""

from .p1b_losses import (
    P1BLossWeights,
    hidden_occupied_supervision_weight,
    p1b_bev_loss,
    wrong_evidence_kl_weight,
)

P2BLossWeights = P1BLossWeights
p2b_bev_loss = p1b_bev_loss

__all__ = [
    "P2BLossWeights",
    "hidden_occupied_supervision_weight",
    "p2b_bev_loss",
    "wrong_evidence_kl_weight",
]
