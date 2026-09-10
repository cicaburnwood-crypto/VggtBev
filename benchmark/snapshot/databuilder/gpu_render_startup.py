"""Cross-process GPU renderer startup coordination and device validation.

The expensive renderer *startup* is serialized per physical GPU.  The lock is
released as soon as the renderer is ready, so already initialized Habitat and
AI2-THOR instances continue collecting concurrently at full throughput.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator


def _normalize_uuid(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized.startswith("gpu-"):
        normalized = f"gpu-{normalized}"
    return normalized


def nvidia_gpu_inventory() -> dict[int, str]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    ).stdout
    inventory: dict[int, str] = {}
    for line in output.splitlines():
        index_text, uuid_text = [part.strip() for part in line.split(",", 1)]
        inventory[int(index_text)] = _normalize_uuid(uuid_text)
    if not inventory:
        raise RuntimeError("nvidia-smi returned an empty GPU inventory")
    return inventory


def physical_gpu_from_visible_device(fallback: int = 0) -> int:
    """Resolve the stable CUDA_VISIBLE_DEVICES token back to a physical index."""

    token = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    inventory = nvidia_gpu_inventory()
    if token:
        if token.isdigit() and int(token) in inventory:
            return int(token)
        normalized = _normalize_uuid(token)
        for index, uuid in inventory.items():
            if uuid == normalized:
                return index
        raise RuntimeError(f"visible GPU token {token!r} is absent from nvidia-smi")
    if fallback not in inventory:
        raise RuntimeError(f"fallback physical GPU {fallback} is absent from nvidia-smi")
    return fallback


def _queue_root() -> Path:
    configured = os.environ.get("VGGT_RENDER_STARTUP_LOCK_ROOT")
    if configured:
        return Path(configured).expanduser()
    super_root = Path("/home/liudiwen/data/BEV")
    if super_root.is_dir():
        return super_root / ".renderer_startup_queue"
    medical_root = Path("/data/disk_7t/diwen/VGGT_DATA/data_build_unlimited")
    return medical_root / ".renderer_startup_queue"


@contextlib.contextmanager
def renderer_startup_gate(
    physical_gpu: int,
    *,
    label: str,
    timeout_s: float = 900.0,
) -> Iterator[None]:
    """Allow only one renderer initialization at a time on one physical GPU."""

    root = _queue_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"GPU{physical_gpu}.lock"
    started = time.monotonic()
    next_report = started
    with path.open("a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if now - started >= timeout_s:
                    raise TimeoutError(
                        f"GPU{physical_gpu} renderer startup queue exceeded {timeout_s:.0f}s"
                    )
                if now >= next_report:
                    print(
                        f"renderer-startup queue waiting GPU{physical_gpu} "
                        f"label={label} elapsed={now - started:.1f}s",
                        flush=True,
                    )
                    next_report = now + 30.0
                time.sleep(0.25)
        acquired = time.monotonic()
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "physical_gpu": physical_gpu,
                    "label": label,
                    "acquired_at": time.time(),
                }
            )
            + "\n"
        )
        handle.flush()
        print(
            f"renderer-startup queue acquired GPU{physical_gpu} label={label} "
            f"wait={acquired - started:.1f}s",
            flush=True,
        )
        try:
            yield
        finally:
            held = time.monotonic() - acquired
            print(
                f"renderer-startup queue released GPU{physical_gpu} label={label} "
                f"held={held:.1f}s",
                flush=True,
            )
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _vulkan_uuid_to_index(runtime_root: Path) -> dict[str, int]:
    bundled = (
        Path(runtime_root)
        / "tools/vulkan-tools/root/usr/bin/vulkaninfo"
    )
    executable = shutil.which("vulkaninfo")
    if executable is None and bundled.is_file():
        executable = str(bundled)
    if executable is None:
        raise FileNotFoundError("vulkaninfo was not found")
    output = subprocess.run(
        [executable, "--summary"],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    ).stdout
    result: dict[str, int] = {}
    for index_text, uuid_text in re.findall(
        r"GPU(\d+):.*?deviceUUID\s*=\s*([0-9a-fA-F-]+)",
        output,
        flags=re.DOTALL,
    ):
        normalized = _normalize_uuid(uuid_text)
        if normalized.startswith("gpu-6d657361-"):
            continue
        result[normalized] = int(index_text)
    return result


def refresh_ai2thor_cuda_vulkan_mapping(runtime_root: Path) -> dict[int, int]:
    """Refresh AI2-THOR's derived mapping once per node boot, atomically."""

    cache_root = Path(runtime_root) / "runtime_home/.ai2thor"
    cache_root.mkdir(parents=True, exist_ok=True)
    mapping_path = cache_root / "cuda-vulkan-mapping.json"
    stamp_path = cache_root / "cuda-vulkan-mapping.boot.json"
    lock_path = cache_root / "cuda-vulkan-mapping.refresh.lock"
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        inventory = nvidia_gpu_inventory()
        inventory_json = {str(index): uuid for index, uuid in inventory.items()}
        try:
            stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
            cached = json.loads(mapping_path.read_text(encoding="utf-8"))
            if (
                stamp.get("boot_id") == boot_id
                and stamp.get("nvidia_inventory") == inventory_json
                and stamp.get("mapping") == cached
            ):
                return {int(key): int(value) for key, value in cached.items()}
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            pass

        vulkan = _vulkan_uuid_to_index(runtime_root)
        missing = [uuid for uuid in inventory.values() if uuid not in vulkan]
        if missing:
            raise RuntimeError(f"Vulkan inventory is missing NVIDIA UUIDs: {missing}")
        mapping = {index: vulkan[uuid] for index, uuid in inventory.items()}
        if len(set(mapping.values())) != len(mapping):
            raise RuntimeError(f"CUDA-to-Vulkan mapping is not one-to-one: {mapping}")
        payload = {str(index): value for index, value in mapping.items()}

        def atomic_json(path: Path, value: object) -> None:
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(value, stream, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary)

        atomic_json(mapping_path, payload)
        atomic_json(
            stamp_path,
            {
                "boot_id": boot_id,
                "nvidia_inventory": inventory_json,
                "mapping": payload,
                "refreshed_at": time.time(),
            },
        )
        print(f"refreshed AI2-THOR CUDA-to-Vulkan mapping: {mapping}", flush=True)
        return mapping


def assert_process_gpu_binding(
    pid: int,
    expected_uuid: str,
    *,
    timeout_s: float = 20.0,
) -> None:
    """Fail closed if a newly started Unity process is attached to another GPU."""

    expected = _normalize_uuid(expected_uuid)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=20,
        ).stdout
        for line in output.splitlines():
            uuid_text, pid_text = [part.strip() for part in line.split(",", 1)]
            if int(pid_text) != int(pid):
                continue
            actual = _normalize_uuid(uuid_text)
            if actual != expected:
                raise RuntimeError(
                    f"renderer PID {pid} bound to {actual}, expected {expected}"
                )
            print(f"renderer PID {pid} binding verified on {expected}", flush=True)
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"renderer PID {pid} did not appear in nvidia-smi within {timeout_s:.0f}s"
    )
