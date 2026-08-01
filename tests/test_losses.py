from __future__ import annotations

import torch

from vggt_bev_method1.losses import (
    FOVCompleteEvidentialLossWeights,
    fov_complete_evidential_bev_loss,
)


def _prediction(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    evidence = torch.nn.functional.softplus(raw)
    return {
        "alpha_occupied": evidence[:, 0] + 1.0,
        "beta_free": evidence[:, 1] + 1.0,
        "fov_support_logit": torch.full_like(raw[:, 0], 6.0),
    }


def _targets() -> tuple[torch.Tensor, torch.Tensor]:
    complete = torch.full((1, 8, 8), 255, dtype=torch.uint8)
    complete[:, 2:6, 2:6] = 0
    observed = torch.full_like(complete, 112)
    observed[:, :4] = complete[:, :4]
    return complete, observed


def _loss(
    raw: torch.Tensor,
    complete: torch.Tensor,
    observed: torch.Tensor,
    *,
    weights: FOVCompleteEvidentialLossWeights | None = None,
    guessed_supervision_scale: float = 1.0,
    regularizer_scale: float = 0.0,
    surface_tolerance_pixels: int = 0,
) -> dict[str, torch.Tensor]:
    return fov_complete_evidential_bev_loss(
        _prediction(raw),
        complete,
        observed,
        complete != 112,
        guessed_class_weights=torch.tensor([1.0, 1.0]),
        weights=weights or FOVCompleteEvidentialLossWeights(),
        regularizer_scale=regularizer_scale,
        guessed_supervision_scale=guessed_supervision_scale,
        surface_tolerance_pixels=surface_tolerance_pixels,
    )


def _occupancy_only_weights() -> FOVCompleteEvidentialLossWeights:
    return FOVCompleteEvidentialLossWeights(
        support_bce=0.0,
        support_dice=0.0,
        incorrect_evidence=0.0,
        confidence_calibration=0.0,
        observation_relation=0.0,
    )


def test_loss_backpropagates_observed_and_guessed_content() -> None:
    raw = torch.randn(1, 2, 8, 8, requires_grad=True)
    complete, observed = _targets()
    losses = _loss(
        raw,
        complete,
        observed,
        weights=_occupancy_only_weights(),
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert raw.grad is not None
    assert raw.grad[:, :, :4].abs().sum() > 0
    assert raw.grad[:, :, 4:].abs().sum() > 0


def test_guessed_completion_uses_complete_target() -> None:
    complete, observed = _targets()
    correct_raw = torch.full((1, 2, 8, 8), -4.0)
    wrong_raw = correct_raw.clone()
    guessed_occupied = (complete == 0) & (observed == 112)
    correct_raw[:, 0][guessed_occupied] = 6.0
    wrong_raw[:, 1][guessed_occupied] = 6.0
    weights = FOVCompleteEvidentialLossWeights(
        guessed_overlap=0.0,
        support_bce=0.0,
        support_dice=0.0,
        incorrect_evidence=0.0,
        confidence_calibration=0.0,
        observation_relation=0.0,
    )
    correct = _loss(correct_raw, complete, observed, weights=weights)
    wrong = _loss(wrong_raw, complete, observed, weights=weights)
    assert correct["guessed_evidential_nll"] < wrong["guessed_evidential_nll"]


def test_guessed_curriculum_cannot_change_observed_gradient() -> None:
    complete, observed = _targets()
    weights = _occupancy_only_weights()
    first = torch.randn(1, 2, 8, 8, requires_grad=True)
    second = first.detach().clone()
    guessed = observed == 112
    second[:, 0][guessed] = 9.0
    second[:, 1][guessed] = -9.0
    second.requires_grad_(True)

    first_loss = _loss(
        first,
        complete,
        observed,
        weights=weights,
        guessed_supervision_scale=0.0,
    )["loss"]
    second_loss = _loss(
        second,
        complete,
        observed,
        weights=weights,
        guessed_supervision_scale=0.0,
    )["loss"]
    first_loss.backward()
    second_loss.backward()

    direct = observed != 112
    assert torch.allclose(first_loss, second_loss)
    assert torch.allclose(
        first.grad[:, :, direct[0]],
        second.grad[:, :, direct[0]],
    )
    assert first.grad[:, :, guessed[0]].abs().sum() == 0
    assert second.grad[:, :, guessed[0]].abs().sum() == 0


def test_sparse_observed_surface_is_not_drowned_by_observed_free_area() -> None:
    complete = torch.full((1, 10, 10), 255, dtype=torch.uint8)
    complete[:, 5, 5] = 0
    observed = complete.clone()
    raw = torch.zeros(1, 2, 10, 10, requires_grad=True)
    losses = _loss(
        raw,
        complete,
        observed,
        weights=_occupancy_only_weights(),
        guessed_supervision_scale=0.0,
    )
    losses["occupancy_loss"].backward()

    surface = complete == 0
    free = complete == 255
    surface_gradient = raw.grad[:, :, surface[0]].abs().sum()
    free_gradient = raw.grad[:, :, free[0]].abs().sum()
    ratio = surface_gradient / free_gradient
    assert losses["observed_surface_fraction"] < 0.02
    assert 0.9 < float(ratio) < 1.1


def test_observed_surface_is_pointwise_not_area_dice() -> None:
    complete = torch.full((1, 8, 8), 255, dtype=torch.uint8)
    complete[:, 3, 4] = 0
    observed = complete.clone()
    correct = torch.full((1, 2, 8, 8), -4.0)
    wrong = correct.clone()
    correct[:, 0, 3, 4] = 6.0
    wrong[:, 1, 3, 4] = 6.0
    weights = FOVCompleteEvidentialLossWeights(
        guessed_nll=0.0,
        guessed_overlap=0.0,
        support_bce=0.0,
        support_dice=0.0,
        incorrect_evidence=0.0,
        confidence_calibration=0.0,
        observation_relation=0.0,
    )
    correct_loss = _loss(
        correct,
        complete,
        observed,
        weights=weights,
        guessed_supervision_scale=0.0,
    )
    wrong_loss = _loss(
        wrong,
        complete,
        observed,
        weights=weights,
        guessed_supervision_scale=0.0,
    )
    assert correct_loss["observed_surface_nll"] < wrong_loss[
        "observed_surface_nll"
    ]
    assert "observed_dice_loss" not in correct_loss


def test_observed_surface_accepts_a_hit_inside_decoder_tolerance() -> None:
    complete = torch.full((1, 9, 9), 255, dtype=torch.uint8)
    complete[:, 4, 4] = 0
    observed = complete.clone()
    near = torch.full((1, 2, 9, 9), -4.0)
    far = near.clone()
    near[:, 0, 4, 6] = 7.0
    far[:, 0, 4, 7] = 7.0
    weights = FOVCompleteEvidentialLossWeights(
        guessed_nll=0.0,
        guessed_overlap=0.0,
        support_bce=0.0,
        support_dice=0.0,
        incorrect_evidence=0.0,
        confidence_calibration=0.0,
        observation_relation=0.0,
    )
    near_loss = _loss(
        near,
        complete,
        observed,
        weights=weights,
        guessed_supervision_scale=0.0,
        surface_tolerance_pixels=2,
    )
    far_loss = _loss(
        far,
        complete,
        observed,
        weights=weights,
        guessed_supervision_scale=0.0,
        surface_tolerance_pixels=2,
    )
    assert near_loss["observed_surface_nll"] < far_loss[
        "observed_surface_nll"
    ]


def test_surface_tolerance_band_is_not_also_supervised_as_free() -> None:
    complete = torch.full((1, 9, 9), 255, dtype=torch.uint8)
    complete[:, 4, 4] = 0
    observed = complete.clone()
    first = torch.zeros(1, 2, 9, 9)
    second = first.clone()
    second[:, 0, 2:7, 2:7] = 8.0
    # Preserve the same best surface hit in both predictions.
    first[:, 0, 4, 4] = 8.0
    weights = FOVCompleteEvidentialLossWeights(
        guessed_nll=0.0,
        guessed_overlap=0.0,
        support_bce=0.0,
        support_dice=0.0,
        incorrect_evidence=0.0,
        confidence_calibration=0.0,
        observation_relation=0.0,
    )
    first_loss = _loss(
        first,
        complete,
        observed,
        weights=weights,
        guessed_supervision_scale=0.0,
        surface_tolerance_pixels=2,
    )
    second_loss = _loss(
        second,
        complete,
        observed,
        weights=weights,
        guessed_supervision_scale=0.0,
        surface_tolerance_pixels=2,
    )
    assert torch.allclose(
        first_loss["observed_free_nll"],
        second_loss["observed_free_nll"],
    )
    assert torch.allclose(
        first_loss["observed_surface_nll"],
        second_loss["observed_surface_nll"],
    )
    assert first_loss["observed_free_supervised_fraction"] < first_loss[
        "observed_free_fraction"
    ]


def test_guessed_overlap_is_macro_occupied_and_free_dice() -> None:
    complete = torch.full((1, 8, 8), 255, dtype=torch.uint8)
    complete[:, :, :4] = 0
    observed = torch.full_like(complete, 112)
    raw = torch.zeros(1, 2, 8, 8)
    losses = _loss(
        raw,
        complete,
        observed,
        weights=FOVCompleteEvidentialLossWeights(
            observed_free=0.0,
            observed_surface=0.0,
            guessed_nll=0.0,
            guessed_overlap=1.0,
            support_bce=0.0,
            support_dice=0.0,
            incorrect_evidence=0.0,
            confidence_calibration=0.0,
            observation_relation=0.0,
        ),
    )
    expected = 0.5 * (
        losses["guessed_occupied_dice_loss"]
        + losses["guessed_free_dice_loss"]
    )
    assert torch.allclose(losses["guessed_macro_dice_loss"], expected)
    assert torch.allclose(losses["guessed_completion_loss"], expected)


def test_observation_relationship_prefers_lower_confidence_for_guesses() -> None:
    complete, observed = _targets()
    observed_mask = observed != 112
    guessed_mask = observed == 112
    aligned_raw = torch.zeros(1, 2, 8, 8)
    reversed_raw = torch.zeros_like(aligned_raw)
    aligned_raw[:, :, observed_mask[0]] = 6.0
    aligned_raw[:, :, guessed_mask[0]] = -3.0
    reversed_raw[:, :, observed_mask[0]] = -3.0
    reversed_raw[:, :, guessed_mask[0]] = 6.0
    weights = FOVCompleteEvidentialLossWeights(
        observed_free=0.0,
        observed_surface=0.0,
        guessed_nll=0.0,
        guessed_overlap=0.0,
        support_bce=0.0,
        support_dice=0.0,
        incorrect_evidence=0.0,
        confidence_calibration=0.0,
        observation_relation=1.0,
        relation_margin=0.15,
    )
    aligned = _loss(
        aligned_raw,
        complete,
        observed,
        weights=weights,
        guessed_supervision_scale=1.0,
        regularizer_scale=1.0,
    )
    reversed_result = _loss(
        reversed_raw,
        complete,
        observed,
        weights=weights,
        guessed_supervision_scale=1.0,
        regularizer_scale=1.0,
    )
    assert aligned["confidence_gap"] > 0
    assert reversed_result["confidence_gap"] < 0
    assert (
        aligned["observation_relation_loss"]
        < reversed_result["observation_relation_loss"]
    )


def test_guessed_confidence_multiplier_targets_only_guessed_regularizer() -> None:
    complete, observed = _targets()
    raw = torch.full((1, 2, 8, 8), -3.0)
    guessed = observed == 112
    raw[:, 0][guessed] = 8.0
    common = dict(
        observed_free=0.0,
        observed_surface=0.0,
        guessed_nll=0.0,
        guessed_overlap=0.0,
        support_bce=0.0,
        support_dice=0.0,
        incorrect_evidence=0.0,
        confidence_calibration=1.0,
        observation_relation=0.0,
    )
    baseline = _loss(
        raw,
        complete,
        observed,
        weights=FOVCompleteEvidentialLossWeights(
            **common,
            guessed_confidence_calibration_multiplier=1.0,
        ),
        regularizer_scale=1.0,
    )
    strengthened = _loss(
        raw,
        complete,
        observed,
        weights=FOVCompleteEvidentialLossWeights(
            **common,
            guessed_confidence_calibration_multiplier=4.0,
        ),
        regularizer_scale=1.0,
    )
    assert torch.allclose(
        baseline["direct_confidence_loss"],
        strengthened["direct_confidence_loss"],
    )
    assert strengthened["guessed_confidence_loss"] > baseline[
        "guessed_confidence_loss"
    ]


def test_direct_only_scale_removes_all_guessed_confidence_gradient() -> None:
    complete, observed = _targets()
    raw = torch.randn(1, 2, 8, 8, requires_grad=True)
    losses = _loss(
        raw,
        complete,
        observed,
        weights=FOVCompleteEvidentialLossWeights(
            guessed_nll=0.0,
            guessed_overlap=0.0,
            support_bce=0.0,
            support_dice=0.0,
            incorrect_evidence=1.0,
            confidence_calibration=1.0,
            guessed_incorrect_evidence_multiplier=4.0,
            guessed_confidence_calibration_multiplier=4.0,
            observation_relation=1.0,
        ),
        guessed_supervision_scale=0.0,
        regularizer_scale=1.0,
    )
    losses["loss"].backward()
    guessed = observed == 112
    assert float(losses["guessed_confidence_loss"].detach()) == 0.0
    assert raw.grad[:, :, guessed[0]].abs().sum() == 0
    assert raw.grad[:, :, ~guessed[0]].abs().sum() > 0


def test_wrong_high_confidence_surface_is_penalized() -> None:
    complete = torch.full((1, 5, 5), 255, dtype=torch.uint8)
    complete[:, 2, 2] = 0
    observed = complete.clone()
    raw = torch.full((1, 2, 5, 5), -4.0)
    raw[:, 1] = 8.0
    losses = _loss(
        raw,
        complete,
        observed,
        weights=FOVCompleteEvidentialLossWeights(
            observed_free=0.0,
            observed_surface=0.0,
            guessed_nll=0.0,
            guessed_overlap=0.0,
            support_bce=0.0,
            support_dice=0.0,
            incorrect_evidence=0.0,
            confidence_calibration=1.0,
            observation_relation=0.0,
        ),
        guessed_supervision_scale=0.0,
        regularizer_scale=1.0,
        surface_tolerance_pixels=1,
    )
    assert losses["observed_surface_confidence_calibration_loss"] > 0.5


def test_outside_fov_cells_are_trained_by_support_not_beta_evidence() -> None:
    complete, observed = _targets()
    complete[:, -2:] = 112
    observed[:, -2:] = 112
    raw = torch.full((1, 2, 8, 8), -2.0)
    aligned = _prediction(raw)
    reversed_support = _prediction(raw)
    aligned["fov_support_logit"] = torch.where(
        complete != 112,
        torch.full_like(raw[:, 0], 6.0),
        torch.full_like(raw[:, 0], -6.0),
    )
    reversed_support["fov_support_logit"] = -aligned["fov_support_logit"]
    weights = FOVCompleteEvidentialLossWeights(
        observed_free=0.0,
        observed_surface=0.0,
        guessed_nll=0.0,
        guessed_overlap=0.0,
        support_bce=1.0,
        support_dice=1.0,
        incorrect_evidence=0.0,
        confidence_calibration=0.0,
        observation_relation=0.0,
    )
    kwargs = {
        "fov_complete_target": complete,
        "visible_target": observed,
        "fov_support_target": complete != 112,
        "guessed_class_weights": torch.tensor([1.0, 1.0]),
        "weights": weights,
        "regularizer_scale": 1.0,
        "guessed_supervision_scale": 0.0,
        "surface_tolerance_pixels": 0,
    }
    aligned_loss = fov_complete_evidential_bev_loss(aligned, **kwargs)
    reversed_loss = fov_complete_evidential_bev_loss(
        reversed_support,
        **kwargs,
    )
    assert aligned_loss["support_loss"] < reversed_loss["support_loss"]
