from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from vggt_bev_method1.config import LabelValues

DEFAULT_LABELS = LabelValues()


@dataclass(frozen=True)
class FOVCompleteEvidentialLossWeights:
    """Weights for FOV-complete content, support and evidence calibration.

    The visibility-masked BEV partitions FOV-complete truth into observed free
    ray space, sparse observed surface hits and occluded/inferred completion.
    Both observed tasks are pointwise and task-balanced. Area overlap belongs
    only to completion; the observed branch has no hole, smoothness or area
    objective.
    """

    observed_free: float = 1.0
    observed_surface: float = 1.0
    observed_surface_continuity: float = 0.0
    guessed_nll: float = 1.0
    guessed_overlap: float = 0.50
    observed_region: float = 1.0
    guessed_region: float = 0.50
    support_bce: float = 0.5
    support_dice: float = 0.5
    incorrect_evidence: float = 0.05
    confidence_calibration: float = 0.25
    guessed_incorrect_evidence_multiplier: float = 1.0
    guessed_confidence_calibration_multiplier: float = 1.0
    observation_relation: float = 0.10
    relation_margin: float = 0.10


def _mean_or_zero(
    values: torch.Tensor,
    mask: torch.Tensor,
    cell_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not bool(mask.any()):
        return values.sum() * 0.0
    if cell_weights is None:
        return values[mask].mean()
    selected_weights = cell_weights[mask]
    return (values[mask] * selected_weights).sum() / selected_weights.sum().clamp_min(
        1e-6
    )


def _balanced_observed_guessed_mean(
    values: torch.Tensor,
    observed: torch.Tensor,
    guessed: torch.Tensor,
    cell_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Give observed evidence and inferred content equal region importance."""

    observed_mean = _mean_or_zero(values, observed, cell_weights)
    guessed_mean = _mean_or_zero(values, guessed, cell_weights)
    has_observed = bool(observed.any())
    has_guessed = bool(guessed.any())
    if has_observed and has_guessed:
        combined = 0.5 * (observed_mean + guessed_mean)
    elif has_observed:
        combined = observed_mean
    else:
        combined = guessed_mean
    return combined, observed_mean, guessed_mean


def _beta_kl_to_uniform(
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    strength = alpha + beta
    return (
        torch.lgamma(strength)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength))
        + (beta - 1.0) * (torch.digamma(beta) - torch.digamma(strength))
    )


def _region_weighted_pair(
    observed_value: torch.Tensor,
    guessed_value: torch.Tensor,
    *,
    observed_weight: float,
    guessed_weight: float,
    guessed_scale: float,
) -> torch.Tensor:
    """Keep observed gradient weight fixed while guessed supervision ramps."""

    denominator = max(float(observed_weight + guessed_weight), 1e-6)
    return (
        float(observed_weight) * observed_value
        + float(guessed_weight) * float(guessed_scale) * guessed_value
    ) / denominator


def _weighted_available_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    first_available: bool,
    second_available: bool,
    first_weight: float,
    second_weight: float,
) -> torch.Tensor:
    """Average independent direct-observation tasks without pixel-count bias."""

    denominator = (
        float(first_weight) * float(first_available)
        + float(second_weight) * float(second_available)
    )
    if denominator <= 0:
        return (first + second) * 0.0
    return (
        float(first_weight) * first * float(first_available)
        + float(second_weight) * second * float(second_available)
    ) / denominator


def _observed_surface_pair_continuity(
    occupied_nll: torch.Tensor,
    observed_surface: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize weak points along exact 8-connected GT surface pairs.

    Each undirected pair is visited once. The maximum endpoint NLL makes a
    single weak prediction expensive on every connected surface segment that
    it interrupts, without dilating the target or supervising nearby free
    cells as occupied.
    """

    pair_slices = (
        ((slice(None), slice(None), slice(None, -1)),
         (slice(None), slice(None), slice(1, None))),
        ((slice(None), slice(None, -1), slice(None)),
         (slice(None), slice(1, None), slice(None))),
        ((slice(None), slice(None, -1), slice(None, -1)),
         (slice(None), slice(1, None), slice(1, None))),
        ((slice(None), slice(None, -1), slice(1, None)),
         (slice(None), slice(1, None), slice(None, -1))),
    )
    pair_loss_sum = occupied_nll.sum() * 0.0
    pair_count = torch.zeros((), device=occupied_nll.device, dtype=torch.long)
    for first_slice, second_slice in pair_slices:
        pair_mask = (
            observed_surface[first_slice] & observed_surface[second_slice]
        )
        pair_losses = torch.maximum(
            occupied_nll[first_slice],
            occupied_nll[second_slice],
        )
        pair_loss_sum = pair_loss_sum + pair_losses[pair_mask].sum()
        pair_count = pair_count + pair_mask.sum()
    pair_loss = pair_loss_sum / pair_count.to(occupied_nll.dtype).clamp_min(1.0)
    return pair_loss, pair_count


def _local_minimum(values: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError("surface tolerance radius cannot be negative")
    if radius == 0:
        return values
    return -F.max_pool2d(
        -values[:, None],
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )[:, 0]


def _local_maximum(values: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 0:
        raise ValueError("surface tolerance radius cannot be negative")
    if radius == 0:
        return values
    return F.max_pool2d(
        values[:, None],
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )[:, 0]


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return _local_maximum(mask.float(), radius) > 0


def _target_regions(
    complete_target: torch.Tensor,
    observed_target: torch.Tensor,
    *,
    labels: LabelValues,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if complete_target.shape != observed_target.shape:
        raise ValueError("complete and masked-observed BEV shapes do not match")
    allowed = (
        (complete_target == labels.unknown)
        | (complete_target == labels.free)
        | (complete_target == labels.occupied)
    )
    if not bool(allowed.all()):
        raise ValueError("complete BEV contains an unsupported label")
    observed_allowed = (
        (observed_target == labels.unknown)
        | (observed_target == labels.free)
        | (observed_target == labels.occupied)
    )
    if not bool(observed_allowed.all()):
        raise ValueError("masked-observed BEV contains an unsupported label")

    valid = complete_target != labels.unknown
    observed = (observed_target != labels.unknown) & valid
    guessed = (observed_target == labels.unknown) & valid
    if bool(((observed_target != labels.unknown) & ~valid).any()):
        raise ValueError("masked-observed BEV has known cells outside complete GT")
    if bool((observed & (observed_target != complete_target)).any()):
        raise ValueError("masked-observed labels disagree with complete GT")
    occupied = complete_target == labels.occupied
    return valid, observed, guessed, occupied


def fov_complete_evidential_bev_loss(
    prediction: dict[str, torch.Tensor],
    fov_complete_target: torch.Tensor,
    visible_target: torch.Tensor,
    fov_support_target: torch.Tensor,
    *,
    guessed_class_weights: torch.Tensor,
    weights: FOVCompleteEvidentialLossWeights,
    regularizer_scale: float,
    guessed_supervision_scale: float,
    surface_tolerance_pixels: int,
    labels: LabelValues = DEFAULT_LABELS,
) -> dict[str, torch.Tensor]:
    """Train FOV support, occupancy and an evidential confidence distribution.

    Observed free cells and sparse first-hit obstacle surfaces receive separate
    pointwise evidential NLLs with task-level balancing. Only guessed cells
    receive occupied-area overlap loss. Outside-FOV cells are unknown and are
    handled by the support branch.
    """

    alpha = prediction["alpha_occupied"].float()
    beta = prediction["beta_free"].float()
    support_logit = prediction["fov_support_logit"].float()
    if (
        alpha.shape != fov_complete_target.shape
        or beta.shape != fov_complete_target.shape
        or support_logit.shape != fov_complete_target.shape
        or fov_support_target.shape != fov_complete_target.shape
    ):
        raise ValueError("prediction and FOV-complete target shapes do not match")
    valid, observed, guessed, occupied = _target_regions(
        fov_complete_target,
        visible_target,
        labels=labels,
    )
    support_truth = fov_support_target.bool()
    if not torch.equal(support_truth, valid):
        raise ValueError("FOV support must equal valid FOV-complete cells")
    strength = alpha + beta
    occupancy_probability = alpha / strength
    support_probability = torch.sigmoid(support_logit)

    support_bce_map = F.binary_cross_entropy_with_logits(
        support_logit,
        support_truth.to(support_logit.dtype),
        reduction="none",
    )
    support_bce, _, _ = _balanced_observed_guessed_mean(
        support_bce_map,
        support_truth,
        ~support_truth,
    )
    support_truth_float = support_truth.to(support_probability.dtype)
    support_dice = 1.0 - (
        2.0 * (support_probability * support_truth_float).sum() + 1.0
    ) / (
        support_probability.sum() + support_truth_float.sum() + 1.0
    )
    support_loss = (
        weights.support_bce * support_bce
        + weights.support_dice * support_dice
    )

    if guessed_class_weights.numel() != 2:
        raise ValueError("guessed class weights must be [free, occupied]")
    guessed_class_weights = guessed_class_weights.to(
        device=alpha.device,
        dtype=alpha.dtype,
    )
    guessed_cell_weights = torch.where(
        occupied,
        guessed_class_weights[1],
        guessed_class_weights[0],
    )

    occupied_nll_map = torch.digamma(strength) - torch.digamma(alpha)
    free_nll_map = torch.digamma(strength) - torch.digamma(beta)
    expected_nll_map = torch.where(occupied, occupied_nll_map, free_nll_map)
    observed_free = observed & ~occupied
    observed_surface = observed & occupied
    # Direct observations are exact labels. The former latent-cell dilation
    # removed valid free supervision around every sparse obstacle surface and
    # created an artificial unsupervised band. Keep every observed free pixel
    # and supervise the surface itself pointwise.
    surface_tolerance_band = torch.zeros_like(observed_surface)
    observed_free_supervised = observed_free
    observed_free_nll = _mean_or_zero(
        free_nll_map,
        observed_free_supervised,
    )
    observed_surface_nll = _mean_or_zero(
        occupied_nll_map,
        observed_surface,
    )
    (
        observed_surface_continuity_loss,
        observed_surface_pair_count,
    ) = _observed_surface_pair_continuity(
        occupied_nll_map,
        observed_surface,
    )
    observed_surface_task_loss = (
        observed_surface_nll
        + float(weights.observed_surface_continuity)
        * observed_surface_continuity_loss
    )
    guessed_nll = _mean_or_zero(
        expected_nll_map,
        guessed,
        guessed_cell_weights,
    )

    occupied_truth = occupied.to(occupancy_probability.dtype)
    observed_direct_loss = _weighted_available_pair(
        observed_free_nll,
        observed_surface_task_loss,
        first_available=bool(observed_free_supervised.any()),
        second_available=bool(observed_surface.any()),
        first_weight=weights.observed_free,
        second_weight=weights.observed_surface,
    )

    def region_dice(
        probability: torch.Tensor,
        truth: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(mask.any()):
            return occupancy_probability.sum() * 0.0
        selected_probability = probability[mask]
        selected_truth = truth[mask]
        return 1.0 - (
            2.0 * (selected_probability * selected_truth).sum() + 1.0
        ) / (
            selected_probability.sum() + selected_truth.sum() + 1.0
        )

    guessed_occupied_dice = region_dice(
        occupancy_probability,
        occupied_truth,
        guessed,
    )
    guessed_free_dice = region_dice(
        1.0 - occupancy_probability,
        1.0 - occupied_truth,
        guessed,
    )
    guessed_macro_dice = _weighted_available_pair(
        guessed_occupied_dice,
        guessed_free_dice,
        first_available=bool((guessed & occupied).any()),
        second_available=bool((guessed & ~occupied).any()),
        first_weight=1.0,
        second_weight=1.0,
    )
    guessed_completion_loss = (
        weights.guessed_nll * guessed_nll
        + weights.guessed_overlap * guessed_macro_dice
    )
    guessed_supervision_scale = float(
        max(0.0, min(1.0, guessed_supervision_scale))
    )
    occupancy_loss = _region_weighted_pair(
        observed_direct_loss,
        guessed_completion_loss,
        observed_weight=weights.observed_region,
        guessed_weight=weights.guessed_region,
        guessed_scale=guessed_supervision_scale,
    )
    direct_evidential_nll = _weighted_available_pair(
        observed_free_nll,
        observed_surface_nll,
        first_available=bool(observed_free_supervised.any()),
        second_available=bool(observed_surface.any()),
        first_weight=weights.observed_free,
        second_weight=weights.observed_surface,
    )
    expected_nll = _region_weighted_pair(
        direct_evidential_nll,
        guessed_nll,
        observed_weight=weights.observed_region,
        guessed_weight=weights.guessed_region,
        guessed_scale=guessed_supervision_scale,
    )

    # Keep only evidence assigned to the wrong class, then shrink that
    # evidence toward the no-evidence Beta(1, 1) prior.
    free_incorrect_evidence_map = _beta_kl_to_uniform(
        alpha,
        torch.ones_like(beta),
    )
    occupied_incorrect_evidence_map = _beta_kl_to_uniform(
        torch.ones_like(alpha),
        beta,
    )
    incorrect_evidence_map = torch.where(
        occupied,
        occupied_incorrect_evidence_map,
        free_incorrect_evidence_map,
    )
    observed_free_incorrect_evidence = _mean_or_zero(
        free_incorrect_evidence_map,
        observed_free_supervised,
    )
    observed_surface_incorrect_evidence = _mean_or_zero(
        occupied_incorrect_evidence_map,
        observed_surface,
    )
    observed_incorrect_evidence = _weighted_available_pair(
        observed_free_incorrect_evidence,
        observed_surface_incorrect_evidence,
        first_available=bool(observed_free_supervised.any()),
        second_available=bool(observed_surface.any()),
        first_weight=weights.observed_free,
        second_weight=weights.observed_surface,
    )
    guessed_incorrect_evidence = _mean_or_zero(
        incorrect_evidence_map,
        guessed,
    )
    weighted_guessed_incorrect_evidence = (
        float(weights.guessed_incorrect_evidence_multiplier)
        * guessed_incorrect_evidence
    )
    incorrect_evidence = _region_weighted_pair(
        observed_incorrect_evidence,
        weighted_guessed_incorrect_evidence,
        observed_weight=weights.observed_region,
        guessed_weight=weights.guessed_region,
        guessed_scale=guessed_supervision_scale,
    )

    predicted_occupied = occupancy_probability >= 0.5
    correct = (predicted_occupied == occupied).detach().to(alpha.dtype)
    confidence = (1.0 - 2.0 / strength).clamp(0.0, 1.0)
    # Calibrate both regions to correctness. No morphology-derived confidence
    # target and no surface-area objective are introduced.
    calibration_map = (confidence - correct).square()
    observed_free_calibration = _mean_or_zero(
        calibration_map,
        observed_free_supervised,
    )
    observed_surface_calibration = _mean_or_zero(
        calibration_map,
        observed_surface,
    )
    observed_calibration = _weighted_available_pair(
        observed_free_calibration,
        observed_surface_calibration,
        first_available=bool(observed_free_supervised.any()),
        second_available=bool(observed_surface.any()),
        first_weight=weights.observed_free,
        second_weight=weights.observed_surface,
    )
    guessed_calibration = _mean_or_zero(
        calibration_map,
        guessed,
    )
    weighted_guessed_calibration = (
        float(weights.guessed_confidence_calibration_multiplier)
        * guessed_calibration
    )
    confidence_calibration = _region_weighted_pair(
        observed_calibration,
        weighted_guessed_calibration,
        observed_weight=weights.observed_region,
        guessed_weight=weights.guessed_region,
        guessed_scale=guessed_supervision_scale,
    )

    observed_free_mean_confidence = _mean_or_zero(
        confidence,
        observed_free_supervised,
    )
    observed_surface_mean_confidence = _mean_or_zero(
        confidence,
        observed_surface,
    )
    observed_mean_confidence = _weighted_available_pair(
        observed_free_mean_confidence,
        observed_surface_mean_confidence,
        first_available=bool(observed_free_supervised.any()),
        second_available=bool(observed_surface.any()),
        # Confidence semantics are task-balanced and must not disappear when
        # a content-loss coefficient is ablated.
        first_weight=1.0,
        second_weight=1.0,
    )
    guessed_mean_confidence = _mean_or_zero(confidence, guessed)

    if bool(observed.any()) and bool(guessed.any()):
        observation_relation = guessed_supervision_scale * F.relu(
            float(weights.relation_margin)
            - (observed_mean_confidence - guessed_mean_confidence)
        )
    else:
        observation_relation = confidence.sum() * 0.0

    regularizer_scale = float(max(0.0, min(1.0, regularizer_scale)))
    content_loss = occupancy_loss + support_loss
    region_denominator = max(
        float(weights.observed_region + weights.guessed_region),
        1e-6,
    )
    direct_confidence_loss = regularizer_scale * (
        float(weights.observed_region)
        / region_denominator
        * (
            weights.incorrect_evidence * observed_incorrect_evidence
            + weights.confidence_calibration * observed_calibration
        )
    )
    guessed_confidence_loss = regularizer_scale * (
        float(weights.guessed_region)
        * guessed_supervision_scale
        / region_denominator
        * (
            weights.incorrect_evidence
            * weighted_guessed_incorrect_evidence
            + weights.confidence_calibration
            * weighted_guessed_calibration
        )
        + weights.observation_relation * observation_relation
    )
    confidence_loss = direct_confidence_loss + guessed_confidence_loss
    return {
        "loss": content_loss + confidence_loss,
        "content_loss": content_loss,
        "occupancy_loss": occupancy_loss,
        "support_loss": support_loss,
        "support_bce_loss": support_bce,
        "support_dice_loss": support_dice,
        "confidence_loss": confidence_loss,
        "direct_confidence_loss": direct_confidence_loss,
        "guessed_confidence_loss": guessed_confidence_loss,
        "evidential_nll": expected_nll,
        "observed_evidential_nll": direct_evidential_nll,
        "observed_free_nll": observed_free_nll,
        "observed_surface_nll": observed_surface_nll,
        "observed_surface_continuity_loss": (
            observed_surface_continuity_loss
        ),
        "observed_surface_task_loss": observed_surface_task_loss,
        "observed_surface_pair_count": observed_surface_pair_count,
        "guessed_evidential_nll": guessed_nll,
        "observed_direct_loss": observed_direct_loss,
        "guessed_completion_loss": guessed_completion_loss,
        "guessed_dice_loss": guessed_macro_dice,
        "guessed_macro_dice_loss": guessed_macro_dice,
        "guessed_occupied_dice_loss": guessed_occupied_dice,
        "guessed_free_dice_loss": guessed_free_dice,
        "incorrect_evidence_loss": incorrect_evidence,
        "observed_incorrect_evidence_loss": observed_incorrect_evidence,
        "observed_free_incorrect_evidence_loss": (
            observed_free_incorrect_evidence
        ),
        "observed_surface_incorrect_evidence_loss": (
            observed_surface_incorrect_evidence
        ),
        "guessed_incorrect_evidence_loss": guessed_incorrect_evidence,
        "weighted_guessed_incorrect_evidence_loss": (
            weighted_guessed_incorrect_evidence
        ),
        "confidence_calibration_loss": confidence_calibration,
        "observed_confidence_calibration_loss": observed_calibration,
        "observed_free_confidence_calibration_loss": (
            observed_free_calibration
        ),
        "observed_surface_confidence_calibration_loss": (
            observed_surface_calibration
        ),
        "guessed_confidence_calibration_loss": guessed_calibration,
        "weighted_guessed_confidence_calibration_loss": (
            weighted_guessed_calibration
        ),
        "observation_relation_loss": observation_relation,
        "guessed_supervision_scale": torch.tensor(
            guessed_supervision_scale,
            device=alpha.device,
        ),
        "effective_observed_region_weight": torch.tensor(
            weights.observed_region
            / max(weights.observed_region + weights.guessed_region, 1e-6),
            device=alpha.device,
        ),
        "effective_guessed_region_weight": torch.tensor(
            weights.guessed_region
            * guessed_supervision_scale
            / max(weights.observed_region + weights.guessed_region, 1e-6),
            device=alpha.device,
        ),
        "observed_mean_confidence": observed_mean_confidence,
        "observed_free_mean_confidence": observed_free_mean_confidence,
        "observed_surface_mean_confidence": observed_surface_mean_confidence,
        "guessed_mean_confidence": guessed_mean_confidence,
        "confidence_gap": observed_mean_confidence - guessed_mean_confidence,
        "observed_fraction": observed.float().mean(),
        "observed_free_fraction": observed_free.float().mean(),
        "observed_free_supervised_fraction": (
            observed_free_supervised.float().mean()
        ),
        "observed_surface_fraction": observed_surface.float().mean(),
        "surface_tolerance_band_fraction": (
            surface_tolerance_band.float().mean()
        ),
        "surface_tolerance_pixels": torch.tensor(
            int(surface_tolerance_pixels),
            device=alpha.device,
        ),
        "guessed_fraction": guessed.float().mean(),
        "valid_fraction": valid.float().mean(),
        "predicted_support_fraction": support_probability.mean(),
        "target_support_fraction": support_truth_float.mean(),
        "predicted_occupied_fraction": _mean_or_zero(
            occupancy_probability,
            valid,
        ),
        "target_occupied_fraction": _mean_or_zero(
            occupied_truth,
            valid,
        ),
    }
