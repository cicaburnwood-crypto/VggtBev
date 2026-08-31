#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


GPU_FIELDS = (
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "pstate",
    "clocks.sm",
    "clocks.mem",
)


def _number(value: str) -> int | float | None:
    text = value.strip()
    if not text or text.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _command(arguments: list[str], timeout: float = 3.0) -> tuple[str, str | None]:
    try:
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return "", f"{type(error).__name__}: {error}"
    if result.returncode:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        return result.stdout, detail
    return result.stdout, None


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        parts = raw.split()
        if parts:
            values[key] = int(parts[0]) * 1024
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    return {
        "total_bytes": total,
        "used_bytes": max(0, total - available),
        "available_bytes": available,
        "free_bytes": values.get("MemFree", 0),
        "buffers_bytes": values.get("Buffers", 0),
        "cached_bytes": values.get("Cached", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
        "swap_free_bytes": values.get("SwapFree", 0),
    }


def _cpu_counters() -> tuple[int, int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    numbers = [int(value) for value in fields[1:]]
    total = sum(numbers)
    idle = numbers[3] + (numbers[4] if len(numbers) > 4 else 0)
    iowait = numbers[4] if len(numbers) > 4 else 0
    return total, idle, iowait


def _cpu_usage(
    previous: tuple[int, int, int] | None,
    current: tuple[int, int, int],
) -> dict[str, float | None]:
    if previous is None or current[0] <= previous[0]:
        return {"used_percent": None, "idle_percent": None, "iowait_percent": None}
    elapsed = current[0] - previous[0]
    idle = current[1] - previous[1]
    iowait = current[2] - previous[2]
    return {
        "used_percent": 100.0 * (elapsed - idle) / elapsed,
        "idle_percent": 100.0 * idle / elapsed,
        "iowait_percent": 100.0 * iowait / elapsed,
    }


def _gpu_state() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    output, error = _command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if error:
        errors.append(f"gpu_query: {error}")
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(GPU_FIELDS):
            errors.append(f"gpu_query malformed row: {line}")
            continue
        row = dict(zip(GPU_FIELDS, parts))
        gpus.append(
            {
                "index": _number(row["index"]),
                "uuid": row["uuid"],
                "name": row["name"],
                "memory_total_mib": _number(row["memory.total"]),
                "memory_used_mib": _number(row["memory.used"]),
                "memory_free_mib": _number(row["memory.free"]),
                "gpu_util_percent": _number(row["utilization.gpu"]),
                "memory_util_percent": _number(row["utilization.memory"]),
                "temperature_c": _number(row["temperature.gpu"]),
                "power_draw_w": _number(row["power.draw"]),
                "power_limit_w": _number(row["power.limit"]),
                "pstate": row["pstate"],
                "sm_clock_mhz": _number(row["clocks.sm"]),
                "memory_clock_mhz": _number(row["clocks.mem"]),
            }
        )

    app_output, app_error = _command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if app_error:
        errors.append(f"compute_query: {app_error}")
    applications: list[dict[str, Any]] = []
    for line in app_output.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            errors.append(f"compute_query malformed row: {line}")
            continue
        applications.append(
            {
                "gpu_uuid": parts[0],
                "pid": _number(parts[1]),
                "process_name": parts[2],
                "used_memory_mib": _number(parts[3]),
            }
        )
    return gpus, applications, errors


def _latest_training_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 512 * 1024))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in record and "batch_seconds" in record:
            return {
                key: record.get(key)
                for key in (
                    "epoch",
                    "step",
                    "batch_seconds",
                    "elapsed_seconds",
                    "learning_rate",
                    "loss",
                )
            }
    return None


def _tmux_exists(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _disk_state(paths: list[Path]) -> list[dict[str, Any]]:
    output = []
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
        except OSError as error:
            output.append({"path": str(path), "error": str(error)})
            continue
        output.append(
            {
                "path": str(path),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "used_percent": 100.0 * usage.used / max(usage.total, 1),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append-only server health watchdog")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training-log", required=True, type=Path)
    parser.add_argument("--tmux-session", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--disk-path", action="append", type=Path, default=[])
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.interval <= 0:
        raise ValueError("interval must be positive")
    output = arguments.output.expanduser().resolve()
    training_log = arguments.training_log.expanduser().resolve()
    disk_paths = [path.expanduser().resolve() for path in arguments.disk_path]
    output.parent.mkdir(parents=True, exist_ok=True)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    previous_cpu: tuple[int, int, int] | None = None
    sequence = 0
    with output.open("a", encoding="utf-8", buffering=1) as stream:
        while True:
            started = time.monotonic()
            current_cpu = _cpu_counters()
            gpus, applications, errors = _gpu_state()
            load_1m, load_5m, load_15m = os.getloadavg()
            sample = {
                "format_version": 2,
                "sequence": sequence,
                "timestamp": dt.datetime.now().astimezone().isoformat(),
                "monotonic_seconds": started,
                "host": socket.gethostname(),
                "boot_id": boot_id,
                "uptime_seconds": float(Path("/proc/uptime").read_text().split()[0]),
                "load_average": {"1m": load_1m, "5m": load_5m, "15m": load_15m},
                "cpu": _cpu_usage(previous_cpu, current_cpu),
                "memory": _meminfo(),
                "disk": _disk_state(disk_paths),
                "gpus": gpus,
                "compute_applications": applications,
                "training": {
                    "tmux_session": arguments.tmux_session,
                    "tmux_alive": _tmux_exists(arguments.tmux_session),
                    "log": str(training_log),
                    "latest": _latest_training_record(training_log),
                },
                "errors": errors,
            }
            stream.write(json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            previous_cpu = current_cpu
            sequence += 1
            if arguments.once:
                break
            delay = arguments.interval - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
