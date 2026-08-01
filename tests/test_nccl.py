from __future__ import annotations

from types import SimpleNamespace

import pytest

from vggt_bev_method1.nccl import (
    MINIMUM_NCCL_RUNTIME_VERSION,
    configure_and_validate_nccl,
    format_nccl_version,
    parse_gpu_inventory,
    parse_topology_numa,
    resolve_visible_devices,
)

INVENTORY = """\
0, GPU-00000000-0000-0000-0000-000000000000
1, GPU-11111111-1111-1111-1111-111111111111
2, GPU-22222222-2222-2222-2222-222222222222
3, GPU-33333333-3333-3333-3333-333333333333
"""

TOPOLOGY = """\
        GPU0 GPU1 GPU2 GPU3 CPU Affinity NUMA Affinity GPU NUMA ID
GPU0     X   NODE SYS  SYS  0-31         0             N/A
GPU1    NODE  X   SYS  SYS  0-31         0             N/A
GPU2    SYS  SYS   X   NODE 32-63        1             N/A
GPU3    SYS  SYS  NODE  X   32-63        1             N/A
"""


def fake_nvidia_smi(command, **_kwargs):
    if command[1:3] == ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"]:
        return SimpleNamespace(stdout=INVENTORY)
    if command[1:] == ["topo", "-m"]:
        return SimpleNamespace(stdout=TOPOLOGY)
    raise AssertionError(command)


def test_inventory_and_topology_resolve_indices_and_uuids() -> None:
    inventory = parse_gpu_inventory(INVENTORY)
    assert resolve_visible_devices("1,GPU-33333333", inventory) == [1, 3]
    assert parse_topology_numa(TOPOLOGY, 4) == {0: 0, 1: 0, 2: 1, 3: 1}


def test_auto_p2p_keeps_fast_nccl_autotuning(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("NCCL_P2P_LEVEL", "PHB")
    monkeypatch.setattr("vggt_bev_method1.nccl.subprocess.run", fake_nvidia_smi)
    monkeypatch.setattr(
        "vggt_bev_method1.nccl.nccl_runtime_version",
        lambda: MINIMUM_NCCL_RUNTIME_VERSION,
    )
    result = configure_and_validate_nccl(
        world_size=2,
        require_same_numa=True,
        p2p_level="AUTO",
    )
    assert result["physical_devices"] == [2, 3]
    assert result["numa_affinities"] == [1, 1]
    assert result["nccl_p2p_auto_tuned"] is True
    assert result["nccl_runtime_version_text"] == "2.26.5"
    assert "NCCL_P2P_LEVEL" not in __import__("os").environ


def test_old_nccl_runtime_is_rejected_before_process_group(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr("vggt_bev_method1.nccl.subprocess.run", fake_nvidia_smi)
    monkeypatch.setattr("vggt_bev_method1.nccl.nccl_runtime_version", lambda: 22602)
    with pytest.raises(RuntimeError, match="2.26.2 is blocked"):
        configure_and_validate_nccl(
            world_size=2,
            require_same_numa=True,
            p2p_level="AUTO",
        )


def test_nccl_version_format() -> None:
    assert format_nccl_version(22605) == "2.26.5"


def test_cross_numa_is_rejected_before_nccl_initialization(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    monkeypatch.setattr("vggt_bev_method1.nccl.subprocess.run", fake_nvidia_smi)
    with pytest.raises(RuntimeError, match="cross-NUMA"):
        configure_and_validate_nccl(
            world_size=2,
            require_same_numa=True,
            p2p_level="AUTO",
        )


def test_sys_p2p_override_is_never_accepted(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(ValueError, match="cross-NUMA SYS"):
        configure_and_validate_nccl(
            world_size=2,
            require_same_numa=True,
            p2p_level="SYS",
        )
