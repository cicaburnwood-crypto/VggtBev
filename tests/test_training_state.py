from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from vggt_bev_method1.cli_train import resume_contract
from vggt_bev_method1.cli_train_paired import (
    _direct_priority_gradient_correction,
)
from vggt_bev_method1.nccl import (
    parse_gpu_inventory,
    parse_topology_numa,
    resolve_visible_devices,
)
from vggt_bev_method1.training_state import (
    DistributedEpochShuffleSampler,
    EpochShuffleSampler,
    SessionPrefixSampler,
    StratifiedValidationSampler,
    capture_rng_state,
    restore_rng_state,
)


def test_nccl_preflight_resolves_uuid_and_parses_numa_topology() -> None:
    inventory = parse_gpu_inventory(
        "\n".join(
            (
                "0, GPU-aaaa",
                "1, GPU-bbbb",
                "2, GPU-cccc",
                "3, GPU-dddd",
            )
        )
    )
    assert resolve_visible_devices("GPU-bbbb,2", inventory) == [1, 2]
    topology = """\
        GPU0 GPU1 GPU2 GPU3 CPU Affinity NUMA Affinity GPU NUMA ID
GPU0    X    NODE SYS  SYS  0-31         0             N/A
GPU1    NODE X    SYS  SYS  0-31         0             N/A
GPU2    SYS  SYS  X    NODE 32-63        1             N/A
GPU3    SYS  SYS  NODE X    32-63        1             N/A
"""
    assert parse_topology_numa(topology, gpu_count=4) == {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
    }


def test_epoch_sampler_reconstructs_order_without_storing_permutation() -> None:
    source = list(range(17))
    first = EpochShuffleSampler(source, seed=23)
    first.set_epoch(4)
    order = list(first)

    resumed = EpochShuffleSampler(source, seed=23)
    resumed.set_epoch(4)
    assert list(resumed) == order

    resumed.set_epoch(5)
    assert list(resumed) != order
    assert sorted(order) == list(range(len(source)))


def test_rng_state_restores_python_numpy_and_torch() -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))

    random.random()
    np.random.rand()
    torch.rand(3)
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_resume_contract_rejects_order_or_model_changes() -> None:
    config = {
        "data": {
            "supervision": "observed",
            "coordinate_mode": (
                "camera_height_anchored_fixed_normalized_scale"
            ),
            "single_target_extent_m": 6.5,
            "merged_target_extent_m": 10.0,
            "image_height": 384,
            "image_width": 512,
            "sample_stride": 1,
            "minimum_history": 1,
            "maximum_history": 34,
        },
        "model": {"hidden_dim": 64},
        "training": {
            "batch_size": 1,
            "seed": 17,
            "geometry_warmup_epochs": 1,
            "geometry_ramp_epochs": 2,
        },
    }
    original = resume_contract(config, "manifest-a")
    assert original == resume_contract(config, "manifest-a")
    config["training"]["seed"] = 18
    assert original != resume_contract(config, "manifest-a")


def test_distributed_sampler_is_disjoint_and_reconstructable() -> None:
    source = list(range(17))
    samplers = [
        DistributedEpochShuffleSampler(
            source,
            num_replicas=3,
            rank=rank,
            seed=11,
        )
        for rank in range(3)
    ]
    for sampler in samplers:
        sampler.set_epoch(2)
    partitions = [list(sampler) for sampler in samplers]
    assert all(len(partition) == 5 for partition in partitions)
    assert len(set().union(*map(set, partitions))) == 15
    repeated = DistributedEpochShuffleSampler(
        source,
        num_replicas=3,
        rank=1,
        seed=11,
    )
    repeated.set_epoch(2)
    assert list(repeated) == partitions[1]


@dataclass(frozen=True)
class _PrefixSample:
    session_index: int
    target_frame: int


class _PrefixDataset:
    def __init__(self) -> None:
        self.samples = [
            _PrefixSample(session_index, target_frame)
            for session_index, count in enumerate((3, 4, 2, 5, 3))
            for target_frame in range(count)
        ]

    def __len__(self) -> int:
        return len(self.samples)


def test_session_prefix_sampler_uses_every_session_once_per_epoch() -> None:
    source = _PrefixDataset()
    samplers = [
        SessionPrefixSampler(
            source,
            seed=17,
            num_replicas=2,
            rank=rank,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.set_epoch(0)
    selected = [index for sampler in samplers for index in sampler]
    selected_sessions = {
        source.samples[index].session_index for index in selected
    }
    assert selected_sessions == set(range(5))
    assert all(len(sampler) == 3 for sampler in samplers)


def test_session_prefix_sampler_rotates_prefixes_deterministically() -> None:
    source = _PrefixDataset()
    sampler = SessionPrefixSampler(source, seed=23, shuffle=False)
    choices = []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        choices.append(
            {
                source.samples[index].session_index: (
                    source.samples[index].target_frame
                )
                for index in sampler
            }
        )
    assert choices[0] != choices[1] != choices[2]
    assert len({choice[0] for choice in choices}) == 3

    repeated = SessionPrefixSampler(source, seed=23, shuffle=False)
    repeated.set_epoch(2)
    assert list(repeated) == list(sampler)


def test_direct_priority_pcgrad_removes_conflict_and_caps_guess() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    direct = parameter[0]
    guessed = -parameter[0] + 10.0 * parameter[1]
    correction, values = _direct_priority_gradient_correction(
        direct,
        guessed,
        [parameter],
        maximum_guessed_to_direct_gradient_ratio=0.25,
    )
    guessed_gradient = torch.tensor([-1.0, 10.0])
    projected = guessed_gradient + correction[0]
    assert torch.dot(torch.tensor([1.0, 0.0]), projected) >= 0
    assert projected.norm() <= 0.250001
    assert values["pcgrad_conflict"] == 1


@dataclass(frozen=True)
class _Scene:
    scene_key: str


class _ValidationDataset(_PrefixDataset):
    def __init__(self) -> None:
        super().__init__()
        self.sessions = [
            _Scene("scene-a"),
            _Scene("scene-a"),
            _Scene("scene-b"),
            _Scene("scene-c"),
            _Scene("scene-d"),
        ]


def test_stratified_validation_visits_scenes_before_repeating() -> None:
    source = _ValidationDataset()
    sampler = StratifiedValidationSampler(
        source,
        maximum_samples=4,
        seed=17,
    )
    scenes = {
        source.sessions[source.samples[index].session_index].scene_key
        for index in sampler
    }
    assert scenes == {"scene-a", "scene-b", "scene-c", "scene-d"}
