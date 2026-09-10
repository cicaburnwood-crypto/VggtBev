from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence

from .dataset import load_session_records
from .manifest import FORMAT_VERSION, _canonical_digest, _session_entry


@dataclass(frozen=True)
class RankedCandidate:
    session_key: str
    scene_key: str
    source: str
    void_fraction: float
    source_rank: int = 0


@dataclass(frozen=True)
class RankedSelection:
    train: tuple[RankedCandidate, ...]
    validation: tuple[RankedCandidate, ...]
    source_statistics: dict[str, dict]


def _validation_count(quota: int, validation_fraction: float) -> int:
    exact = Decimal(quota) * Decimal(str(validation_fraction))
    if exact != exact.to_integral_value():
        raise ValueError(
            f"validation fraction {validation_fraction} does not produce an exact "
            f"session count for quota {quota}"
        )
    return int(exact)


def select_ranked_sessions(
    candidates_by_source: Mapping[str, Sequence[RankedCandidate]],
    quotas: Mapping[str, int],
    *,
    validation_fraction: float,
) -> RankedSelection:
    """Take the lowest-Void train quota, then scene-disjoint post-cut validation."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    selected_train: list[RankedCandidate] = []
    selected_validation: list[RankedCandidate] = []
    statistics: dict[str, dict] = {}
    for source, quota in quotas.items():
        if quota <= 0:
            raise ValueError(f"quota for {source!r} must be positive")
        ordered = sorted(
            candidates_by_source.get(source, ()),
            key=lambda item: (item.void_fraction, item.session_key),
        )
        ordered = [replace(item, source_rank=index) for index, item in enumerate(ordered, 1)]
        validation_count = _validation_count(quota, validation_fraction)
        if len(ordered) < quota + validation_count:
            raise ValueError(f"source {source!r} has too few ranked sessions")
        train = ordered[:quota]
        train_scenes = {item.scene_key for item in train}
        validation: list[RankedCandidate] = []
        skipped_train_scene = 0
        for item in ordered[quota:]:
            if item.scene_key in train_scenes:
                skipped_train_scene += 1
                continue
            validation.append(item)
            if len(validation) == validation_count:
                break
        if len(validation) != validation_count:
            raise ValueError(
                f"source {source!r} cannot supply {validation_count} post-cut "
                "scene-disjoint validation sessions"
            )
        validation_scenes = {item.scene_key for item in validation}
        selected_train.extend(train)
        selected_validation.extend(validation)
        statistics[source] = {
            "available_sessions": len(ordered),
            "train_sessions": len(train),
            "validation_sessions": len(validation),
            "train_scenes": len(train_scenes),
            "validation_scenes": len(validation_scenes),
            "train_minimum_void_fraction": train[0].void_fraction,
            "train_maximum_void_fraction": train[-1].void_fraction,
            "validation_minimum_void_fraction": min(
                item.void_fraction for item in validation
            ),
            "validation_maximum_void_fraction": max(
                item.void_fraction for item in validation
            ),
            "validation_first_source_rank": validation[0].source_rank,
            "validation_last_source_rank": validation[-1].source_rank,
            "post_cut_candidates_skipped_for_scene_leakage": skipped_train_scene,
        }
    train_keys = {item.session_key for item in selected_train}
    validation_keys = {item.session_key for item in selected_validation}
    if len(train_keys) != len(selected_train) or len(validation_keys) != len(
        selected_validation
    ):
        raise ValueError("ranked selection contains duplicate session keys")
    if train_keys.intersection(validation_keys):
        raise ValueError("ranked selection leaks sessions")
    train_scenes = {item.scene_key for item in selected_train}
    validation_scenes = {item.scene_key for item in selected_validation}
    if train_scenes.intersection(validation_scenes):
        raise ValueError("ranked selection leaks scenes")
    return RankedSelection(
        train=tuple(selected_train),
        validation=tuple(selected_validation),
        source_statistics=statistics,
    )


def read_single_void_candidates(
    audit_csv: str | Path,
    wanted_sources: set[str],
) -> tuple[dict[str, list[RankedCandidate]], int, int]:
    source = Path(audit_csv).expanduser().resolve()
    result = {name: [] for name in wanted_sources}
    seen_sessions: set[str] = set()
    single_record_count = 0
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "session_key",
            "scene_key",
            "dataset",
            "gt_bev",
            "void_fraction_of_grid",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Void audit CSV lacks ranked-selection columns")
        for row in reader:
            if row["gt_bev"] != "single_complete":
                continue
            single_record_count += 1
            session_key = str(row["session_key"])
            if session_key in seen_sessions:
                raise ValueError(f"duplicate Single record for {session_key!r}")
            seen_sessions.add(session_key)
            dataset = str(row["dataset"])
            if dataset not in wanted_sources:
                continue
            fraction = float(row["void_fraction_of_grid"])
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"invalid Single Void fraction for {session_key!r}")
            result[dataset].append(
                RankedCandidate(
                    session_key=session_key,
                    scene_key=str(row["scene_key"]),
                    source=dataset,
                    void_fraction=fraction,
                )
            )
    if not single_record_count:
        raise ValueError("Void audit CSV contains no Single records")
    return result, len(seen_sessions), single_record_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def create_ranked_training_manifest(
    *,
    dataset_root: str | Path,
    audit_csv: str | Path,
    output_path: str | Path,
    quotas: Mapping[str, int],
    validation_fraction: float,
    split_seed: int,
    inventory_workers: int = 16,
) -> dict:
    root = Path(dataset_root).expanduser().resolve()
    csv_path = Path(audit_csv).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")
    if inventory_workers <= 0:
        raise ValueError("inventory_workers must be positive")
    candidates, source_session_count, single_record_count = read_single_void_candidates(
        csv_path,
        set(quotas),
    )
    selection = select_ranked_sessions(
        candidates,
        quotas,
        validation_fraction=validation_fraction,
    )
    selected = selection.train + selection.validation
    print(f"loading {len(selected):,} selected completed sessions", flush=True)
    records = load_session_records(root, [item.session_key for item in selected])
    record_by_key = {record.key: record for record in records}

    def inventory(item: RankedCandidate) -> dict:
        entry = _session_entry(record_by_key[item.session_key])
        if entry["scene_key"] != item.scene_key:
            raise ValueError(f"scene changed for {item.session_key!r}")
        entry.update(
            {
                "dataset": item.source,
                "selection_single_void_fraction_of_grid": item.void_fraction,
                "selection_source_rank": item.source_rank,
            }
        )
        return entry

    entries: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=inventory_workers) as executor:
        for number, (item, entry) in enumerate(
            zip(selected, executor.map(inventory, selected), strict=True),
            start=1,
        ):
            entries[item.session_key] = entry
            if number % 1000 == 0:
                print(f"inventoried {number:,}/{len(selected):,} sessions", flush=True)
    train_entries = [entries[item.session_key] for item in selection.train]
    validation_entries = [
        entries[item.session_key] for item in selection.validation
    ]
    payload = {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "source_session_count": source_session_count,
        "maximum_sessions": None,
        "source_writer_count_at_freeze": 0,
        "snapshot_scope": "completed-session-single-void-ranked-selection-v1",
        "selection_order": (
            "per source ascending single_complete void_fraction_of_grid, then "
            "session key; exact train quota first; validation continues after "
            "the cut while skipping train scenes"
        ),
        "selection_metric": "single_complete.void_fraction_of_grid",
        "merged_bev_used_for_selection": False,
        "validation_scene_disjoint": True,
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "session_count": len(selected),
        "scene_count": len({item.scene_key for item in selected}),
        "train_session_count": len(selection.train),
        "validation_session_count": len(selection.validation),
        "source_quotas": dict(quotas),
        "source_statistics": selection.source_statistics,
        "void_audit_csv": str(csv_path),
        "void_audit_csv_sha256": _sha256(csv_path),
        "void_audit_single_record_count": single_record_count,
        "train": train_entries,
        "validation": validation_entries,
    }
    payload["content_sha256"] = _canonical_digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        os.link(temporary, output)
    except FileExistsError:
        raise FileExistsError(f"manifest appeared while writing: {output}") from None
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"wrote {len(selection.train):,} train + "
        f"{len(selection.validation):,} validation sessions to {output}",
        flush=True,
    )
    return payload
