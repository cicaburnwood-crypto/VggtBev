from __future__ import annotations

import csv

import pytest

from p1b_void_audit.thresholds import (
    TARGET_NAMES,
    build_threshold_index,
    load_threshold_index,
    save_threshold_index,
)


def _write_rows(path, sessions):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("session_key", "dataset", "gt_bev", "void_fraction_of_grid"),
        )
        writer.writeheader()
        for session_key, source, fractions in sessions:
            for target, fraction in zip(TARGET_NAMES, fractions, strict=True):
                writer.writerow(
                    {
                        "session_key": session_key,
                        "dataset": source,
                        "gt_bev": target,
                        "void_fraction_of_grid": fraction,
                    }
                )


def test_threshold_is_per_single_bev_and_any_single_failure_excludes_session(tmp_path):
    source = tmp_path / "audit.csv"
    _write_rows(
        source,
        [
            ("session_a", "hm3d", (0.10, 0.20, 0.30, 0.40)),
            ("session_b", "hm3d", (0.30, 0.05, 0.10, 0.20)),
            ("session_c", "procthor-10k", (0.31, 0.00, 0.00, 0.70)),
        ],
    )
    index = build_threshold_index(source)
    result = index.evaluate(30.0)
    assert result["total_sessions"] == 3
    assert result["retained_sessions"] == 2
    assert result["excluded_sessions"] == 1
    assert result["by_gt_bev"]["merged_complete"]["above_source_threshold"] == 0
    assert result["by_gt_bev"]["merged_observed"]["above_source_threshold"] == 2


def test_each_dataset_source_has_an_independent_threshold(tmp_path):
    source = tmp_path / "audit.csv"
    _write_rows(
        source,
        [
            ("session_a", "hm3d", (0.10, 0.20, 0.30, 0.40)),
            ("session_b", "hm3d", (0.30, 0.05, 0.10, 0.20)),
            ("session_c", "procthor-10k", (0.31, 0.00, 0.00, 0.70)),
        ],
    )
    index = build_threshold_index(source)
    result = index.evaluate_by_source({"hm3d": 30.0, "procthor-10k": 80.0})
    assert result["retained_sessions"] == 3
    assert result["by_source"]["hm3d"]["retained_sessions"] == 2
    assert result["by_source"]["hm3d"]["total_sessions"] == 2
    assert result["by_source"]["procthor-10k"]["retained_sessions"] == 1
    assert result["by_source"]["procthor-10k"]["total_sessions"] == 1


def test_merged_bevs_are_diagnostic_only_for_session_filtering(tmp_path):
    source = tmp_path / "audit.csv"
    _write_rows(
        source,
        [("session_a", "hm3d", (0.10, 0.10, 0.95, 0.99))],
    )
    index = build_threshold_index(source)
    assert index.evaluate(10.0)["retained_sessions"] == 1


def test_threshold_index_round_trip(tmp_path):
    source = tmp_path / "audit.csv"
    _write_rows(source, [("session_a", "hm3d", (0.4, 0.2, 0.3, 0.1))])
    destination = tmp_path / "index.npz"
    save_threshold_index(build_threshold_index(source), destination)
    loaded = load_threshold_index(destination)
    assert loaded.evaluate(39.9)["retained_sessions"] == 0
    assert loaded.evaluate(40.0)["retained_sessions"] == 1


def test_missing_bev_row_is_rejected(tmp_path):
    source = tmp_path / "audit.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("session_key", "dataset", "gt_bev", "void_fraction_of_grid"),
        )
        writer.writeheader()
        for target in TARGET_NAMES[:-1]:
            writer.writerow(
                {
                    "session_key": "session_a",
                    "dataset": "hm3d",
                    "gt_bev": target,
                    "void_fraction_of_grid": 0.1,
                }
            )
    with pytest.raises(ValueError, match="exactly four"):
        build_threshold_index(source)
