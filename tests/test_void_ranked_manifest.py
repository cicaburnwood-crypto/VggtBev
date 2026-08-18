from __future__ import annotations

from vggt_bev_method1.data.void_ranked_manifest import (
    RankedCandidate,
    select_ranked_sessions,
)


def _candidate(number: int, scene: str, source: str, fraction: float):
    return RankedCandidate(
        session_key=f"session_{number:03d}",
        scene_key=f"{source}:{scene}",
        source=source,
        void_fraction=fraction,
    )


def test_lowest_ranked_train_then_post_cut_scene_disjoint_validation():
    rows = {
        "hm3d": [
            _candidate(1, "a", "hm3d", 0.01),
            _candidate(2, "b", "hm3d", 0.02),
            _candidate(3, "a", "hm3d", 0.03),
            _candidate(4, "c", "hm3d", 0.04),
            _candidate(5, "d", "hm3d", 0.05),
        ]
    }
    result = select_ranked_sessions(rows, {"hm3d": 2}, validation_fraction=0.5)
    assert [item.session_key for item in result.train] == ["session_001", "session_002"]
    assert [item.session_key for item in result.validation] == ["session_004"]
    assert not {item.scene_key for item in result.train}.intersection(
        item.scene_key for item in result.validation
    )
    stats = result.source_statistics["hm3d"]
    assert stats["post_cut_candidates_skipped_for_scene_leakage"] == 1
    assert stats["validation_first_source_rank"] == 4


def test_source_quotas_are_independent_and_keep_declared_order():
    rows = {
        "hm3d": [
            _candidate(1, "a", "hm3d", 0.02),
            _candidate(2, "b", "hm3d", 0.01),
            _candidate(3, "c", "hm3d", 0.03),
            _candidate(4, "d", "hm3d", 0.04),
        ],
        "hssd": [
            _candidate(5, "a", "hssd", 0.10),
            _candidate(6, "b", "hssd", 0.20),
            _candidate(7, "c", "hssd", 0.30),
            _candidate(8, "d", "hssd", 0.40),
        ],
    }
    result = select_ranked_sessions(
        rows,
        {"hm3d": 2, "hssd": 2},
        validation_fraction=0.5,
    )
    assert [item.source for item in result.train] == [
        "hm3d",
        "hm3d",
        "hssd",
        "hssd",
    ]
    assert [item.session_key for item in result.train] == [
        "session_002",
        "session_001",
        "session_005",
        "session_006",
    ]
