from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TARGET_NAMES = (
    "single_complete",
    "single_observed",
    "merged_complete",
    "merged_observed",
)
TARGET_SET = frozenset(TARGET_NAMES)
FILTER_TARGET_NAMES = (
    "single_complete",
    "single_observed",
)
INDEX_FORMAT_VERSION = 3


@dataclass(frozen=True)
class ThresholdIndex:
    session_max: np.ndarray
    by_target: dict[str, np.ndarray]
    by_source: dict[str, np.ndarray]
    by_source_target: dict[str, dict[str, np.ndarray]]
    metadata: dict

    @property
    def session_count(self) -> int:
        return int(self.session_max.size)

    def evaluate(self, threshold_percent: float) -> dict:
        return self.evaluate_by_source(
            {source: threshold_percent for source in self.by_source}
        )

    def evaluate_by_source(self, thresholds_percent: dict[str, float]) -> dict:
        unknown = set(thresholds_percent).difference(self.by_source)
        missing = set(self.by_source).difference(thresholds_percent)
        if unknown or missing:
            raise ValueError(
                f"source thresholds must match index sources; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        normalized: dict[str, float] = {}
        for source, threshold in thresholds_percent.items():
            if not np.isfinite(threshold) or not 0.0 <= threshold <= 100.0:
                raise ValueError(
                    f"threshold for {source!r} must be finite and in [0, 100]"
                )
            normalized[source] = float(threshold)
        retained_total = 0
        by_source_result = {}
        target_retained = {name: 0 for name in TARGET_NAMES}
        for source in sorted(self.by_source):
            values = self.by_source[source]
            total = int(values.size)
            threshold_fraction = normalized[source] / 100.0
            retained = int(np.searchsorted(values, threshold_fraction, side="right"))
            retained_total += retained
            by_source_result[source] = {
                "threshold_percent": normalized[source],
                "total_sessions": total,
                "retained_sessions": retained,
                "excluded_sessions": total - retained,
                "retained_fraction": retained / total if total else 0.0,
            }
            for name in TARGET_NAMES:
                target_retained[name] += int(
                    np.searchsorted(
                        self.by_source_target[source][name],
                        threshold_fraction,
                        side="right",
                    )
                )
        total = self.session_count
        by_target = {
            name: {
                "at_or_below_source_threshold": target_retained[name],
                "above_source_threshold": total - target_retained[name],
            }
            for name in TARGET_NAMES
        }
        return {
            "thresholds_percent": normalized,
            "comparison": "exclude_when_any_gt_bev_void_fraction_is_strictly_greater",
            "total_sessions": total,
            "retained_sessions": retained_total,
            "excluded_sessions": total - retained_total,
            "retained_fraction": retained_total / total if total else 0.0,
            "excluded_fraction": (total - retained_total) / total if total else 0.0,
            "by_source": by_source_result,
            "by_gt_bev": by_target,
        }

def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _finish_session(
    *,
    session_key: str,
    fractions: dict[str, float],
    source: str,
    maxima: list[float],
    by_target: dict[str, list[float]],
    by_source: dict[str, list[float]],
    by_source_target: dict[str, dict[str, list[float]]],
) -> None:
    missing = TARGET_SET.difference(fractions)
    extra = set(fractions).difference(TARGET_SET)
    if missing or extra:
        raise ValueError(
            f"session {session_key!r} does not contain exactly four GT BEVs; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    session_maximum = max(fractions[name] for name in FILTER_TARGET_NAMES)
    maxima.append(session_maximum)
    for name in TARGET_NAMES:
        by_target[name].append(fractions[name])
    by_source.setdefault(source, []).append(session_maximum)
    source_targets = by_source_target.setdefault(
        source,
        {name: [] for name in TARGET_NAMES},
    )
    for name in TARGET_NAMES:
        source_targets[name].append(fractions[name])


def build_threshold_index(csv_path: str | Path) -> ThresholdIndex:
    """Stream a grouped audit CSV into sorted per-session threshold arrays."""

    source = Path(csv_path).expanduser().resolve()
    maxima: list[float] = []
    by_target_lists = {name: [] for name in TARGET_NAMES}
    by_source_lists: dict[str, list[float]] = {}
    by_source_target_lists: dict[str, dict[str, list[float]]] = {}
    seen_sessions: set[str] = set()
    current_key: str | None = None
    current_source: str | None = None
    current_fractions: dict[str, float] = {}
    record_count = 0
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"session_key", "dataset", "gt_bev", "void_fraction_of_grid"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"audit CSV lacks required columns: {sorted(required)}")
        for row in reader:
            record_count += 1
            session_key = str(row["session_key"])
            source_name = str(row["dataset"])
            target = str(row["gt_bev"])
            if not session_key:
                raise ValueError(f"record {record_count} has an empty session_key")
            if not source_name:
                raise ValueError(f"record {record_count} has an empty dataset source")
            if target not in TARGET_SET:
                raise ValueError(f"record {record_count} has unknown gt_bev {target!r}")
            try:
                fraction = float(row["void_fraction_of_grid"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"record {record_count} has an invalid Void fraction"
                ) from error
            if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"record {record_count} Void fraction must be in [0, 1]"
                )
            if current_key is None:
                current_key = session_key
                current_source = source_name
                seen_sessions.add(session_key)
            elif session_key != current_key:
                assert current_source is not None
                _finish_session(
                    session_key=current_key,
                    fractions=current_fractions,
                    source=current_source,
                    maxima=maxima,
                    by_target=by_target_lists,
                    by_source=by_source_lists,
                    by_source_target=by_source_target_lists,
                )
                current_fractions = {}
                current_key = session_key
                current_source = source_name
                if session_key in seen_sessions:
                    raise ValueError(
                        "audit CSV rows must keep the four records for each session "
                        f"contiguous; repeated session {session_key!r}"
                    )
                seen_sessions.add(session_key)
            elif source_name != current_source:
                raise ValueError(
                    f"session {session_key!r} contains multiple dataset sources"
                )
            if target in current_fractions:
                raise ValueError(
                    f"session {session_key!r} contains duplicate {target!r} rows"
                )
            current_fractions[target] = fraction
    if current_key is not None:
        assert current_source is not None
        _finish_session(
            session_key=current_key,
            fractions=current_fractions,
            source=current_source,
            maxima=maxima,
            by_target=by_target_lists,
            by_source=by_source_lists,
            by_source_target=by_source_target_lists,
        )
    if not maxima:
        raise ValueError("audit CSV contains no sessions")
    session_max = np.sort(np.asarray(maxima, dtype=np.float64))
    by_target = {
        name: np.sort(np.asarray(values, dtype=np.float64))
        for name, values in by_target_lists.items()
    }
    by_source = {
        source_name: np.sort(np.asarray(values, dtype=np.float64))
        for source_name, values in by_source_lists.items()
    }
    by_source_target = {
        source_name: {
            name: np.sort(np.asarray(values, dtype=np.float64))
            for name, values in source_targets.items()
        }
        for source_name, source_targets in by_source_target_lists.items()
    }
    histogram, edges = np.histogram(session_max, bins=100, range=(0.0, 1.0))
    metadata = {
        "format_version": INDEX_FORMAT_VERSION,
        "metric": "void_fraction_of_grid",
        "metric_definition": "void_pixels / (output_height * output_width)",
        "session_rule": "exclude if either Single GT BEV is strictly above threshold",
        "filter_target_names": list(FILTER_TARGET_NAMES),
        "diagnostic_only_target_names": [
            name for name in TARGET_NAMES if name not in FILTER_TARGET_NAMES
        ],
        "equality_rule": "a GT BEV exactly equal to threshold is retained",
        "csv": str(source),
        "csv_size_bytes": source.stat().st_size,
        "csv_mtime_ns": source.stat().st_mtime_ns,
        "csv_sha256": _sha256(source),
        "session_count": int(session_max.size),
        "record_count": record_count,
        "target_names": list(TARGET_NAMES),
        "sources": [
            {"name": source_name, "session_count": int(by_source[source_name].size)}
            for source_name in sorted(by_source)
        ],
        "histogram": {
            "bin_edges_percent": (edges * 100.0).tolist(),
            "session_counts": histogram.astype(np.int64).tolist(),
        },
    }
    return ThresholdIndex(
        session_max=session_max,
        by_target=by_target,
        by_source=by_source,
        by_source_target=by_source_target,
        metadata=metadata,
    )


def save_threshold_index(index: ThresholdIndex, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            session_max=index.session_max,
            metadata=np.asarray(json.dumps(index.metadata, separators=(",", ":"))),
            **{f"target_{name}": index.by_target[name] for name in TARGET_NAMES},
            **{
                f"source_{source_name}": values
                for source_name, values in index.by_source.items()
            },
            **{
                f"source_target_{source_name}_{name}": values
                for source_name, source_targets in index.by_source_target.items()
                for name, values in source_targets.items()
            },
        )
    temporary.replace(destination)
    return destination


def load_threshold_index(path: str | Path) -> ThresholdIndex:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        if int(metadata.get("format_version", -1)) != INDEX_FORMAT_VERSION:
            raise ValueError("unsupported Void threshold index format")
        session_max = np.asarray(payload["session_max"], dtype=np.float64)
        by_target = {
            name: np.asarray(payload[f"target_{name}"], dtype=np.float64)
            for name in TARGET_NAMES
        }
        source_names = [str(item["name"]) for item in metadata["sources"]]
        by_source = {
            source_name: np.asarray(payload[f"source_{source_name}"], dtype=np.float64)
            for source_name in source_names
        }
        by_source_target = {
            source_name: {
                name: np.asarray(
                    payload[f"source_target_{source_name}_{name}"],
                    dtype=np.float64,
                )
                for name in TARGET_NAMES
            }
            for source_name in source_names
        }
    if any(values.size != session_max.size for values in by_target.values()):
        raise ValueError("Void threshold index arrays have inconsistent lengths")
    if sum(values.size for values in by_source.values()) != session_max.size:
        raise ValueError("Void threshold source arrays have inconsistent lengths")
    for source_name, values in by_source.items():
        if any(
            target_values.size != values.size
            for target_values in by_source_target[source_name].values()
        ):
            raise ValueError(
                f"Void threshold target arrays are inconsistent for {source_name}"
            )
    return ThresholdIndex(
        session_max=session_max,
        by_target=by_target,
        by_source=by_source,
        by_source_target=by_source_target,
        metadata=metadata,
    )


def load_or_build_threshold_index(
    csv_path: str | Path,
    cache_path: str | Path,
) -> ThresholdIndex:
    source = Path(csv_path).expanduser().resolve()
    cache = Path(cache_path).expanduser().resolve()
    if cache.is_file():
        loaded = load_threshold_index(cache)
        metadata = loaded.metadata
        if (
            int(metadata.get("csv_size_bytes", -1)) == source.stat().st_size
            and int(metadata.get("csv_mtime_ns", -1)) == source.stat().st_mtime_ns
        ):
            return loaded
    built = build_threshold_index(source)
    save_threshold_index(built, cache)
    return built
