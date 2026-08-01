from __future__ import annotations

import ctypes
import os
import re
import subprocess

NCCL_P2P_AUTO = "AUTO"
NCCL_P2P_LEVELS = frozenset(
    (NCCL_P2P_AUTO, "LOC", "NVL", "PIX", "PXB", "PHB")
)
MINIMUM_NCCL_RUNTIME_VERSION = 22605


def nccl_runtime_version() -> int:
    """Return the version exported by the NCCL library loaded by this process."""
    try:
        library = ctypes.CDLL("libnccl.so.2")
    except OSError as error:
        raise RuntimeError(
            "cannot resolve the NCCL runtime library loaded by PyTorch"
        ) from error
    version = ctypes.c_int()
    result = library.ncclGetVersion(ctypes.byref(version))
    if result != 0:
        raise RuntimeError(f"ncclGetVersion failed with status {result}")
    return int(version.value)


def format_nccl_version(version: int) -> str:
    major = version // 10000
    minor = (version % 10000) // 100
    patch = version % 100
    return f"{major}.{minor}.{patch}"


def parse_gpu_inventory(text: str) -> dict[str, int]:
    inventory: dict[str, int] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        index_text, uuid = (part.strip() for part in line.split(",", maxsplit=1))
        index = int(index_text)
        inventory[str(index)] = index
        inventory[uuid] = index
    if not inventory:
        raise RuntimeError("nvidia-smi returned an empty GPU inventory")
    return inventory


def resolve_visible_devices(
    cuda_visible_devices: str,
    inventory: dict[str, int],
) -> list[int]:
    tokens = [token.strip() for token in cuda_visible_devices.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain explicit GPU indices or UUIDs")
    resolved = []
    for token in tokens:
        if token in inventory:
            resolved.append(inventory[token])
            continue
        uuid_matches = {
            index
            for name, index in inventory.items()
            if name.startswith("GPU-") and name.startswith(token)
        }
        if len(uuid_matches) != 1:
            raise ValueError(f"cannot uniquely resolve CUDA-visible GPU {token!r}")
        resolved.append(uuid_matches.pop())
    if len(set(resolved)) != len(resolved):
        raise ValueError("CUDA_VISIBLE_DEVICES resolves to duplicate physical GPUs")
    return resolved


def parse_topology_numa(text: str, gpu_count: int) -> dict[int, int]:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    result: dict[int, int] = {}
    for line in clean.splitlines():
        fields = line.split()
        if (
            not fields
            or not re.fullmatch(r"GPU\d+", fields[0])
            or len(fields) < 2
            or not re.fullmatch(r"X|NV\d+|PIX|PXB|PHB|NODE|SYS", fields[1])
        ):
            continue
        if len(fields) < gpu_count + 3:
            raise RuntimeError(f"unexpected nvidia-smi topology row: {line}")
        physical_index = int(fields[0][3:])
        numa_text = fields[gpu_count + 2]
        if not re.fullmatch(r"\d+", numa_text):
            raise RuntimeError(f"GPU {physical_index} has no numeric NUMA affinity")
        result[physical_index] = int(numa_text)
    if len(result) != gpu_count:
        raise RuntimeError("could not parse every GPU NUMA affinity from nvidia-smi")
    return result


def configure_and_validate_nccl(
    *,
    world_size: int,
    require_same_numa: bool,
    p2p_level: str,
) -> dict:
    p2p_level = p2p_level.strip().upper()
    if p2p_level not in NCCL_P2P_LEVELS:
        raise ValueError(
            "NCCL P2P level must be AUTO or a level that cannot enable "
            "cross-NUMA SYS P2P"
        )
    visible_text = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_text:
        raise RuntimeError(
            "distributed training requires an explicit CUDA_VISIBLE_DEVICES list"
        )
    inventory_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    inventory = parse_gpu_inventory(inventory_result.stdout)
    physical_devices = resolve_visible_devices(visible_text, inventory)
    if len(physical_devices) != world_size:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES count must match the torchrun world size"
        )
    topology_result = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    gpu_count = len({index for index in inventory.values()})
    numa_by_device = parse_topology_numa(topology_result.stdout, gpu_count)
    selected_numa = [numa_by_device[index] for index in physical_devices]
    if require_same_numa and len(set(selected_numa)) != 1:
        raise RuntimeError(
            "refusing cross-NUMA NCCL training; selected physical GPUs "
            f"{physical_devices} have NUMA affinities {selected_numa}"
        )
    existing_p2p_level = os.environ.get("NCCL_P2P_LEVEL")
    if p2p_level == NCCL_P2P_AUTO:
        # The same-NUMA guard removes the unsafe topology. Leaving this unset
        # lets NCCL choose the fastest remaining P2P path (including NODE).
        os.environ.pop("NCCL_P2P_LEVEL", None)
    else:
        if existing_p2p_level is not None and existing_p2p_level != p2p_level:
            raise RuntimeError(
                f"NCCL_P2P_LEVEL={existing_p2p_level!r} conflicts with required "
                f"value {p2p_level!r}"
            )
        os.environ["NCCL_P2P_LEVEL"] = p2p_level
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    runtime_version = nccl_runtime_version()
    if runtime_version < MINIMUM_NCCL_RUNTIME_VERSION:
        raise RuntimeError(
            "NCCL runtime "
            f"{format_nccl_version(runtime_version)} is blocked for distributed "
            "Method I training because communicator initialization is unstable; "
            "run through scripts/torchrun_method1.sh with the project-local "
            f"NCCL >= {format_nccl_version(MINIMUM_NCCL_RUNTIME_VERSION)}"
        )
    return {
        "backend": "nccl",
        "nccl_runtime_version": runtime_version,
        "nccl_runtime_version_text": format_nccl_version(runtime_version),
        "physical_devices": physical_devices,
        "numa_affinities": selected_numa,
        "same_numa_required": require_same_numa,
        "nccl_p2p_level": p2p_level,
        "nccl_p2p_auto_tuned": p2p_level == NCCL_P2P_AUTO,
    }
