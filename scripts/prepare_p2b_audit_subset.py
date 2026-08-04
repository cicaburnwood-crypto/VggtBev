#!/usr/bin/env python3
"""Export a P2B runtime audit and freeze a ranked, scene-safe retraining subset.

The source manifest already owns the scene-grouped train/validation split.  This
tool ranks sessions *inside each existing split*, so a lower-loss training
selection cannot move a validation scene into training.  Sessions without a
guessed region are never allowed to win the guessed-quality ranking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flatten(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}_{key}" if prefix else str(key)
            flattened.update(_flatten(child, prefix=child_prefix))
        return flattened
    if isinstance(value, (list, tuple)):
        return {prefix: json.dumps(value, separators=(",", ":"))}
    return {prefix: value}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on {path}:{number}") from exc
    return records


def _split_target_counts(*, train_count: int, validation_count: int, target: int) -> tuple[int, int]:
    total = train_count + validation_count
    if target <= 0 or target > total:
        raise ValueError(f"target must be in [1, {total}], got {target}")
    exact_train = target * train_count / total
    exact_validation = target * validation_count / total
    counts = [math.floor(exact_train), math.floor(exact_validation)]
    remaining = target - sum(counts)
    fractions = [exact_train - counts[0], exact_validation - counts[1]]
    for index in sorted(range(2), key=lambda item: (-fractions[item], item))[:remaining]:
        counts[index] += 1
    return counts[0], counts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--target-sessions", type=int, default=36_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_records = _load_jsonl(args.audit)
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    expected_digest = source_manifest.pop("content_sha256", None)
    if expected_digest != _canonical_digest(source_manifest):
        raise ValueError("source manifest checksum mismatch")
    source_manifest["content_sha256"] = expected_digest

    partitions = {
        "train": source_manifest["train"],
        "validation": source_manifest["validation"],
    }
    split_by_key: dict[str, str] = {}
    entry_by_key: dict[str, dict[str, Any]] = {}
    for split, entries in partitions.items():
        for entry in entries:
            key = str(entry["key"])
            if key in split_by_key:
                raise ValueError(f"duplicate source-manifest session key: {key}")
            split_by_key[key] = split
            entry_by_key[key] = entry

    audit_by_key: dict[str, dict[str, Any]] = {}
    for record in audit_records:
        key = str(record["session_key"])
        if key in audit_by_key:
            raise ValueError(f"duplicate audit session key: {key}")
        if key not in split_by_key:
            raise ValueError(f"audit session is absent from source manifest: {key}")
        audit_by_key[key] = record
    if set(audit_by_key) != set(split_by_key):
        missing = sorted(set(split_by_key).difference(audit_by_key))
        extra = sorted(set(audit_by_key).difference(split_by_key))
        raise ValueError(
            f"audit/source session mismatch: missing={len(missing)} extra={len(extra)}"
        )

    ranking: dict[str, list[tuple[float, str]]] = {"train": [], "validation": []}
    score_by_key: dict[str, float] = {}
    for key, record in audit_by_key.items():
        losses = record.get("loss", {})
        guessed_fraction = float(losses.get("single_bev_guessed_fraction", 0.0))
        guessed_loss = float(losses["single_bev_guessed_pixel_loss"])
        # No inferred cells means there was no guessed task to evaluate; such a
        # session cannot qualify as a "best guessed" example merely because a
        # masked reduction returned zero.
        score = guessed_loss if guessed_fraction > 0.0 and math.isfinite(guessed_loss) else math.inf
        score_by_key[key] = score
        ranking[split_by_key[key]].append((score, key))
    for values in ranking.values():
        values.sort(key=lambda item: (item[0], item[1]))

    train_target, validation_target = _split_target_counts(
        train_count=len(partitions["train"]),
        validation_count=len(partitions["validation"]),
        target=args.target_sessions,
    )
    selected_keys = {
        "train": {key for _, key in ranking["train"][:train_target]},
        "validation": {key for _, key in ranking["validation"][:validation_target]},
    }
    if any(math.isinf(score_by_key[key]) for keys in selected_keys.values() for key in keys):
        raise ValueError("target selection would include a session with no guessed pixels")

    rank_by_key = {
        key: rank
        for split, values in ranking.items()
        for rank, (_, key) in enumerate(values, start=1)
    }
    csv_rows: list[dict[str, Any]] = []
    for key in sorted(audit_by_key):
        row = _flatten(audit_by_key[key])
        split = split_by_key[key]
        row.update(
            {
                "source_split": split,
                "guessed_selection_score": score_by_key[key],
                "guessed_rank_within_source_split": rank_by_key[key],
                "selected_for_retraining": key in selected_keys[split],
            }
        )
        csv_rows.append(row)
    fields = sorted({field for row in csv_rows for field in row})
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    selected_train = [entry for entry in partitions["train"] if entry["key"] in selected_keys["train"]]
    selected_validation = [entry for entry in partitions["validation"] if entry["key"] in selected_keys["validation"]]
    payload = dict(source_manifest)
    payload.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "maximum_sessions": args.target_sessions,
            "selection_order": (
                "lowest audit single_bev_guessed_pixel_loss within the pre-existing "
                "scene-grouped split; sessions with zero guessed pixels excluded"
            ),
            "session_count": args.target_sessions,
            "scene_count": len(
                {entry["scene_key"] for entry in selected_train + selected_validation}
            ),
            "train": selected_train,
            "validation": selected_validation,
            "selection_source_manifest_sha256": expected_digest,
            "selection_audit": str(args.audit.expanduser().resolve()),
            "selection_score": "single_bev_guessed_pixel_loss",
            "selection_excludes_zero_guessed_fraction": True,
        }
    )
    payload.pop("content_sha256", None)
    payload["content_sha256"] = _canonical_digest(payload)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "csv": str(args.csv.resolve()),
                "output_manifest": str(args.output_manifest.resolve()),
                "audit_sessions": len(audit_by_key),
                "selected_sessions": len(selected_train) + len(selected_validation),
                "selected_train_sessions": len(selected_train),
                "selected_validation_sessions": len(selected_validation),
                "source_manifest_sha256": expected_digest,
                "output_manifest_sha256": payload["content_sha256"],
                "train_score_range": [
                    ranking["train"][0][0],
                    ranking["train"][train_target - 1][0],
                ],
                "validation_score_range": [
                    ranking["validation"][0][0],
                    ranking["validation"][validation_target - 1][0],
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
