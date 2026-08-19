from vggt_bev_method1.cli_train_p1b import (
    _binary_metric_summary,
    _history_metric_keys,
)


def test_history_metric_keys_are_fixed_across_ddp_ranks() -> None:
    keys = _history_metric_keys(
        ("merged",),
        maximum_history=3,
        include_gate=True,
    )

    assert len(keys) == 3 * (1 + 2 * 4)
    assert keys["merged_history_01_samples"] == 0.0
    assert keys["merged_history_03_support_tp"] == 0.0
    assert keys["merged_history_02_observed_gate_tn"] == 0.0


def test_binary_metric_summary_uses_pixel_counts() -> None:
    values = {
        "history_04_support_tp": 80.0,
        "history_04_support_fp": 20.0,
        "history_04_support_fn": 40.0,
        "history_04_support_tn": 60.0,
    }

    summary = _binary_metric_summary(values, "history_04_support")

    assert summary == {
        "precision": 0.8,
        "recall": 2.0 / 3.0,
        "iou": 80.0 / 140.0,
        "accuracy": 0.7,
    }
