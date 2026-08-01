from __future__ import annotations

import random
from collections.abc import Iterator, Sized

import numpy as np
import torch
from torch.utils.data import Sampler


class EpochShuffleSampler(Sampler[int]):
    """Deterministic per-epoch ordering that can be reconstructed on resume."""

    def __init__(self, data_source: Sized, *, seed: int, shuffle: bool = True) -> None:
        self.data_source = data_source
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        size = len(self.data_source)
        if not self.shuffle:
            return iter(range(size))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(size, generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


class DistributedEpochShuffleSampler(Sampler[int]):
    """Deterministic disjoint per-rank samples for exact DDP resume."""

    def __init__(
        self,
        data_source: Sized,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank/world size")
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        size = len(data_source)
        if drop_last:
            self.num_samples = size // num_replicas
        else:
            self.num_samples = (size + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        size = len(self.data_source)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(size, generator=generator).tolist()
        else:
            indices = list(range(size))
        if self.drop_last:
            indices = indices[: self.total_size]
        elif len(indices) < self.total_size:
            indices.extend(indices[: self.total_size - len(indices)])
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples


class SessionPrefixSampler(Sampler[int]):
    """Choose one runtime-aligned temporal prefix per session and epoch.

    The selected target advances deterministically through each session's
    available prefixes across epochs.  Under DDP, samples with similar history
    lengths are grouped into the same synchronized step to reduce straggler
    time while every session remains represented.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
    ) -> None:
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid session-prefix sampler rank/world size")
        samples = getattr(data_source, "samples", None)
        if not isinstance(samples, list) or not samples:
            raise ValueError(
                "session-prefix sampling requires a dataset with sample records"
            )
        grouped: dict[int, list[int]] = {}
        for sample_index, sample in enumerate(samples):
            session_index = getattr(sample, "session_index", None)
            target_frame = getattr(sample, "target_frame", None)
            if not isinstance(session_index, int) or not isinstance(
                target_frame,
                int,
            ):
                raise ValueError(
                    "session-prefix sample records require integer "
                    "session_index and target_frame"
                )
            grouped.setdefault(session_index, []).append(sample_index)
        self.data_source = data_source
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self._samples = samples
        self._session_candidates = tuple(
            tuple(
                sorted(
                    grouped[session_index],
                    key=lambda index: samples[index].target_frame,
                )
            )
            for session_index in sorted(grouped)
        )
        offset_generator = random.Random(seed)
        self._session_offsets = tuple(
            offset_generator.randrange(len(candidates))
            for candidates in self._session_candidates
        )
        self.num_samples = (
            len(self._session_candidates) + num_replicas - 1
        ) // num_replicas
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        selected: list[int] = []
        for session_index, candidates in enumerate(self._session_candidates):
            # Give sessions different phase offsets, then visit every available
            # prefix once before repeating one in a later epoch.
            offset = self._session_offsets[session_index]
            selected.append(candidates[(offset + self.epoch) % len(candidates)])

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle and self.num_replicas > 1:
            selected.sort(
                key=lambda index: self._samples[index].target_frame
            )
            if len(selected) < self.total_size:
                # Complete the final synchronized group with a same-length
                # duplicate so its ranks do not become temporal stragglers.
                selected.extend(
                    [selected[-1]] * (self.total_size - len(selected))
                )
            groups = [
                selected[start : start + self.num_replicas]
                for start in range(0, len(selected), self.num_replicas)
            ]
            group_order = torch.randperm(
                len(groups),
                generator=generator,
            ).tolist()
            ordered = []
            for group_index in group_order:
                group = groups[group_index]
                if len(group) > 1:
                    within = torch.randperm(
                        len(group),
                        generator=generator,
                    ).tolist()
                    group = [group[index] for index in within]
                ordered.extend(group)
            selected = ordered
        elif self.shuffle:
            order = torch.randperm(
                len(selected),
                generator=generator,
            ).tolist()
            selected = [selected[index] for index in order]

        if len(selected) < self.total_size:
            selected.extend(selected[: self.total_size - len(selected)])
        return iter(
            selected[self.rank : self.total_size : self.num_replicas]
        )

    def __len__(self) -> int:
        return self.num_samples


def stratified_validation_indices(
    data_source: Sized,
    *,
    maximum_samples: int,
    seed: int,
    history_bins: int = 4,
) -> list[int]:
    """Choose a fixed scene- and history-balanced validation subset."""

    if maximum_samples <= 0 or len(data_source) <= 0:
        return []
    budget = min(maximum_samples, len(data_source))
    samples = getattr(data_source, "samples", None)
    sessions = getattr(data_source, "sessions", None)
    if not isinstance(samples, list) or not sessions:
        positions = np.linspace(0, len(data_source) - 1, budget)
        return [int(round(position)) for position in positions]

    scene_groups: dict[str, list[int]] = {}
    session_max_target: dict[int, int] = {}
    for sample_index, sample in enumerate(samples):
        scene_key = sessions[sample.session_index].scene_key
        scene_groups.setdefault(scene_key, []).append(sample_index)
        session_max_target[sample.session_index] = max(
            session_max_target.get(sample.session_index, 0),
            sample.target_frame,
        )
    scene_order = sorted(scene_groups)
    random.Random(seed).shuffle(scene_order)
    bucket_order = (0, history_bins - 1, *range(1, history_bins - 1))
    per_scene: dict[str, list[int]] = {}
    for scene_key in scene_order:
        buckets: list[list[int]] = [[] for _ in range(history_bins)]
        for sample_index in scene_groups[scene_key]:
            sample = samples[sample_index]
            maximum = max(1, session_max_target[sample.session_index])
            bucket = min(
                history_bins - 1,
                int(sample.target_frame / maximum * history_bins),
            )
            buckets[bucket].append(sample_index)
        rng = random.Random(f"{seed}:{scene_key}")
        for bucket in buckets:
            rng.shuffle(bucket)
        ordered: list[int] = []
        while any(buckets):
            for bucket_index in bucket_order:
                if buckets[bucket_index]:
                    ordered.append(buckets[bucket_index].pop())
        per_scene[scene_key] = ordered

    selected: list[int] = []
    while len(selected) < budget:
        for scene_key in scene_order:
            if per_scene[scene_key]:
                selected.append(per_scene[scene_key].pop(0))
                if len(selected) == budget:
                    break
    return selected


class StratifiedValidationSampler(Sampler[int]):
    """Shard a representative validation subset without first-scene bias."""

    def __init__(
        self,
        data_source: Sized,
        *,
        maximum_samples: int,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        selected = stratified_validation_indices(
            data_source,
            maximum_samples=maximum_samples,
            seed=seed,
        )
        if not selected:
            raise ValueError("validation sampler received an empty budget")
        self.num_samples = (len(selected) + num_replicas - 1) // num_replicas
        total_size = self.num_samples * num_replicas
        original = tuple(selected)
        selected.extend(
            original[index % len(original)]
            for index in range(total_size - len(selected))
        )
        self.indices = tuple(selected[rank:total_size:num_replicas])

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
