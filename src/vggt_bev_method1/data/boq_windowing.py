"""Deterministic BoQ sliding windows for offline trajectory conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BoQCandidateDecision:
    frame_id: int
    selected: bool
    similarity_to_last_selected: float | None


@dataclass(frozen=True)
class BoQWindow:
    start_frame_id: int
    selected_frame_ids: tuple[int, ...]
    decisions: tuple[BoQCandidateDecision, ...]
    required_frame_count: int

    @property
    def complete(self) -> bool:
        return len(self.selected_frame_ids) == self.required_frame_count

    @property
    def rejected_frame_ids(self) -> tuple[int, ...]:
        return tuple(item.frame_id for item in self.decisions if not item.selected)


def _normalized_descriptor(value: np.ndarray, *, frame_id: int) -> np.ndarray:
    descriptor = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(descriptor))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"BoQ descriptor is invalid for frame {frame_id}")
    return descriptor / norm


def build_boq_window(
    frame_ids: Sequence[int],
    descriptors: Mapping[int, np.ndarray],
    *,
    start_frame_id: int,
    similarity_threshold: float = 0.60,
    required_frame_count: int = 10,
) -> BoQWindow:
    """Select one window, comparing each candidate to the last accepted frame.

    The start frame is always accepted. Later candidates are accepted only
    when cosine similarity is strictly lower than the threshold. Therefore a
    score equal to or above 0.60 is redundant and rejects only that candidate,
    not the whole window.
    """

    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("BoQ similarity threshold must be within [-1, 1]")
    if required_frame_count <= 0:
        raise ValueError("required_frame_count must be positive")
    ordered = tuple(sorted({int(value) for value in frame_ids}))
    if start_frame_id not in ordered:
        raise ValueError(f"window start frame is unavailable: {start_frame_id}")

    selected: list[int] = []
    decisions: list[BoQCandidateDecision] = []
    last_selected: np.ndarray | None = None
    for frame_id in ordered:
        if frame_id < start_frame_id:
            continue
        if frame_id not in descriptors:
            raise KeyError(f"missing BoQ descriptor for frame {frame_id}")
        descriptor = _normalized_descriptor(descriptors[frame_id], frame_id=frame_id)
        similarity = (
            None
            if last_selected is None
            else float(np.clip(last_selected @ descriptor, -1.0, 1.0))
        )
        accepted = similarity is None or similarity < similarity_threshold
        decisions.append(
            BoQCandidateDecision(
                frame_id=frame_id,
                selected=accepted,
                similarity_to_last_selected=similarity,
            )
        )
        if accepted:
            selected.append(frame_id)
            last_selected = descriptor
            if len(selected) == required_frame_count:
                break

    return BoQWindow(
        start_frame_id=start_frame_id,
        selected_frame_ids=tuple(selected),
        decisions=tuple(decisions),
        required_frame_count=required_frame_count,
    )


def build_boq_sliding_windows(
    frame_ids: Sequence[int],
    descriptors: Mapping[int, np.ndarray],
    *,
    start_interval: int = 5,
    similarity_threshold: float = 0.60,
    required_frame_count: int = 10,
    anchor_frame_id: int = 0,
) -> tuple[BoQWindow, ...]:
    """Start a fresh BoQ history at raw frames 0, 5, 10, ... by default."""

    if start_interval <= 0:
        raise ValueError("BoQ window start interval must be positive")
    ordered = tuple(sorted({int(value) for value in frame_ids}))
    starts = [
        frame_id
        for frame_id in ordered
        if frame_id >= anchor_frame_id
        and (frame_id - anchor_frame_id) % start_interval == 0
    ]
    return tuple(
        build_boq_window(
            ordered,
            descriptors,
            start_frame_id=start,
            similarity_threshold=similarity_threshold,
            required_frame_count=required_frame_count,
        )
        for start in starts
    )
