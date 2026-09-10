"""Fixed 500 scene groups x five routes, plus fail-closed GPU launch guards."""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import threading
from protocol import CONFIG, stamp, require_stamp, prepare_root

HERE = Path(__file__).resolve().parent
FATAL_XIDS = {62, 79, 119, 120, 154}
SOURCES = ("hm3d", "hssd", "mp3d", "procthor")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def command(argv, timeout=15):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {result.stderr[-1500:]}")
    return result.stdout


def groups():
    result = [dict(source=source, source_group=i)
              for source, count in CONFIG['sampling']['groups_per_source'].items() for i in range(count)]
    random.Random(CONFIG['sampling']['master_seed']).shuffle(result)
    return result


def preflight(args):
    """Exactly ten one-second samples immediately before GPU processes launch."""
    root = Path(os.environ["REALTIME_OUTPUT"])
    if (root / "CIRCUIT_BREAKER.json").exists():
        raise RuntimeError("Circuit breaker is latched; inspect it before explicit recovery")
    samples = []
    for _ in range(10):
        raw = command(["nvidia-smi", "-i", str(args.gpu),
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"])
        index, uuid, used, total, util = [v.strip() for v in raw.strip().split(",")]
        sample = dict(utc=utc(), index=int(index), uuid=uuid,
                      vram_percent=100 * float(used) / float(total), compute_percent=float(util))
        samples.append(sample)
        time.sleep(1)
    apps = command(["nvidia-smi", "-i", str(args.gpu),
        "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    eligible = all(s["vram_percent"] < 30 and s["compute_percent"] < 20
                   and s["uuid"] == args.uuid for s in samples) and not apps.strip()
    save(root / "logs" / f"preflight_gpu{args.gpu}_{time.time_ns()}.json",
         dict(samples=samples, compute_apps=apps, eligible=eligible,
              policy="all samples VRAM<30%, compute<20%, no existing compute process"))
    if not eligible:
        raise RuntimeError(f"GPU {args.gpu} is no longer idle; no workload was launched")
    recovery = command(["nvidia-smi", "-i", args.uuid,
        "--query-gpu=gpu_recovery_action", "--format=csv,noheader,nounits"]).strip()
    save(root / "logs" / f"recovery_gpu{args.gpu}_{time.time_ns()}.json",
         dict(utc=utc(), uuid=args.uuid, recovery_action=recovery,
              source="narrow assigned-GPU preflight query; live Xid subscription follows"))
    if recovery.lower() not in {"none", "n/a"}:
        raise RuntimeError(f"GPU {args.gpu} requests recovery: {recovery}")


class NvmlEvents:
    """Read-only native NVML Xid subscription; does not require kernel-log access."""
    class Event(ctypes.Structure):
        _fields_ = [("device", ctypes.c_void_p), ("eventType", ctypes.c_ulonglong),
                    ("eventData", ctypes.c_ulonglong), ("gpuInstanceId", ctypes.c_uint),
                    ("computeInstanceId", ctypes.c_uint)]

    class ProcessV2(ctypes.Structure):
        # nvmlProcessInfo_v2_t, not the newer, ABI-different v3 structure.
        _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                    ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.event_set = ctypes.c_void_p()
        self.check(self.lib.nvmlInit_v2(), "initialize")
        self.check(self.lib.nvmlEventSetCreate(ctypes.byref(self.event_set)), "create event set")
        self.devices = {}
        self.handles = {}
        count = ctypes.c_uint()
        self.check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), "get GPU count")
        for index in range(count.value):
            handle = ctypes.c_void_p()
            self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index),
                ctypes.byref(handle)), "get GPU handle")
            uuid = ctypes.create_string_buffer(96)
            self.check(self.lib.nvmlDeviceGetUUID(handle, uuid, ctypes.c_uint(len(uuid))), "get UUID")
            self.devices[handle.value] = uuid.value.decode()
            self.handles[uuid.value.decode()] = handle
            self.check(self.lib.nvmlDeviceRegisterEvents(handle, ctypes.c_ulonglong(8),
                self.event_set), "subscribe to Xid errors")

    @staticmethod
    def check(code, operation):
        if code != 0:
            raise RuntimeError(f"NVML {operation} failed with status {code}")

    def wait(self):
        event = self.Event()
        code = self.lib.nvmlEventSetWait_v2(self.event_set, ctypes.byref(event), ctypes.c_uint(500))
        if code == 10:  # NVML_ERROR_TIMEOUT: no new event, not a GPU error.
            return None
        self.check(code, "wait for Xid event")
        return dict(xid=int(event.eventData), uuid=self.devices.get(event.device))

    def close(self):
        self.lib.nvmlEventSetFree(self.event_set)
        self.lib.nvmlShutdown()

    def processes(self, uuid=None):
        """Only the driver process tables: no power/clocks/firmware/full -q."""
        rows = []
        for device_uuid, handle in self.handles.items():
            if uuid is not None and device_uuid != uuid:
                continue
            for name, kind in (("nvmlDeviceGetComputeRunningProcesses_v2", "C"),
                               ("nvmlDeviceGetGraphicsRunningProcesses_v2", "G")):
                function = getattr(self.lib, name)
                capacity = 64
                for _ in range(3):
                    count = ctypes.c_uint(capacity)
                    infos = (self.ProcessV2 * capacity)()
                    code = function(handle, ctypes.byref(count), infos)
                    if code == 7:  # NVML_ERROR_INSUFFICIENT_SIZE: process-list race.
                        capacity = max(capacity * 2, count.value + 8)
                        continue
                    self.check(code, name)
                    rows.extend(dict(pid=int(info.pid), uuid=device_uuid, type=kind)
                                for info in infos[:count.value])
                    break
                else:
                    raise RuntimeError("NVML process table kept growing during inspection")
        return rows


def descendants(root_pid):
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(entry.name)] = int(stat[1])
        except (OSError, ValueError, IndexError):
            continue
    owned = {root_pid}
    while True:
        found = {pid for pid, parent in parents.items() if parent in owned}
        if found <= owned:
            return owned
        owned.update(found)


def process_snapshot(events, root):
    """One lightweight global sweep / 30 seconds, shared by all six guards."""
    cache = root / "logs" / "nvml_process_snapshot.json"
    with (root / "locks" / "nvml_process_snapshot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache.exists():
            snapshot = json.loads(cache.read_text())
            if 0 <= time.monotonic() - snapshot["monotonic"] < 30:
                return snapshot["processes"]
        rows = events.processes()
        save(cache, dict(utc=utc(), monotonic=time.monotonic(), processes=rows))
        return rows


def inspect_mapping(owner_pid, expected_uuid, events, root):
    # Assigned-GPU live query plus cached global table catches wrong GPU binding.
    rows = events.processes(expected_uuid) + process_snapshot(events, root)
    owned = descendants(owner_pid)
    seen = {}
    for row in rows:
        if row["pid"] in owned:
            if row["uuid"] != expected_uuid:
                raise RuntimeError(f'Owned PID {row["pid"]} mapped to {row["uuid"]}, expected {expected_uuid}')
            seen[(row["pid"], row["type"])] = row
    return sorted(seen.values(), key=lambda row: (row["pid"], row["type"]))


def latch(root, reason, lane):
    path = root / "CIRCUIT_BREAKER.json"
    try:
        with path.open("x") as stream:
            json.dump(dict(utc=utc(), lane=lane, reason=reason, automatic_retry=False), stream, indent=2)
    except FileExistsError:
        pass


def guard(args):
    root = Path(os.environ["REALTIME_OUTPUT"])
    ready = root / "logs" / f"guard{args.lane}.ready.json"
    events = None
    mapping_thread = None
    try:
        events = NvmlEvents()
        save(ready, dict(pid=os.getpid(), owner_pid=args.owner_pid, utc=utc(),
                         xid_source="NVML XidCriticalError events on all devices", uuid=args.uuid,
                         mapping_source="native NVML v2 process tables; 30-second shared global cache"))
        next_mapping = time.monotonic() + args.lane * 5
        previous = None
        mapping_result = {}

        def inspect_in_background():
            try:
                mapping_result["rows"] = inspect_mapping(args.owner_pid, args.uuid, events, root)
            except BaseException as error:
                mapping_result["error"] = error

        while Path(f"/proc/{args.owner_pid}").exists():
            if (root / "CIRCUIT_BREAKER.json").exists():
                raise RuntimeError("Shared circuit breaker was latched")
            event = events.wait()
            if event is not None:
                print(json.dumps(dict(utc=utc(), gpu_event=event)), flush=True)
                if event["xid"] in FATAL_XIDS:
                    raise RuntimeError(f"Fatal GPU Xid: {event}")
            if mapping_thread is not None and not mapping_thread.is_alive():
                if "error" in mapping_result:
                    raise mapping_result["error"]
                mapping = mapping_result["rows"]
                if mapping != previous:
                    save(root / "logs" / f"gpu_mapping_lane{args.lane}.json",
                         dict(utc=utc(), expected_uuid=args.uuid, processes=mapping))
                    previous = mapping
                mapping_thread = None
                next_mapping = time.monotonic() + 30
            if mapping_thread is None and time.monotonic() >= next_mapping:
                mapping_result = {}
                mapping_started = time.monotonic()
                mapping_thread = threading.Thread(target=inspect_in_background, daemon=True)
                mapping_thread.start()
            elif mapping_thread is not None and time.monotonic() - mapping_started > 90:
                raise RuntimeError("Native NVML process inspection stalled for 90 seconds")
    except BaseException as error:
        if not isinstance(error, KeyboardInterrupt):
            latch(root, repr(error), args.lane)
            print(f"GPU GUARD STOP: {error!r}", flush=True)
            try:
                os.kill(args.owner_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            raise
    finally:
        if events is not None and (mapping_thread is None or not mapping_thread.is_alive()):
            events.close()
        ready.unlink(missing_ok=True)


def terminate_child(child):
    # Its renderer can remain in the owned process group after a failed parent.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if child.poll() is not None:
        return
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def rejected_before_routes(out):
    marker = out / "REJECTED.json"
    if not marker.exists():
        return False
    row = json.loads(marker.read_text())
    if not row.get("replacement_allowed") or row.get("stage") not in {"before_routes", "before_episode"}:
        raise RuntimeError(f"Unqualified rejection cannot be resampled: {out}")
    if (out / "tasks.json").exists() or (out / "routes.json").exists() or list(out.glob("route_*")) or (out / "results.json").exists():
        raise RuntimeError(f"Rejected group already contains frozen/evaluated routes: {out}")
    return True


def dispatch(args):
    root = Path(os.environ["REALTIME_OUTPUT"])
    prepare_root(root)
    completed = 0
    for number, group in enumerate(groups()):
        if number % args.lanes != args.lane:
            continue
        if args.limit_groups and completed >= args.limit_groups:
            break
        if (root / "CIRCUIT_BREAKER.json").exists():
            raise RuntimeError("GPU safety circuit breaker is latched")
        groupdir = root / "groups" / f'group_{number:04d}_{group["source"]}'
        groupdir.mkdir(parents=True, exist_ok=True)
        if (groupdir / "COMPLETE.json").exists():
            require_stamp(json.loads((groupdir/'COMPLETE.json').read_text()))
            completed += 1
            continue
        for attempt in range(CONFIG['sampling']['max_scene_attempts_per_group']):
            out = groupdir / f"attempt_{attempt:03d}"
            if rejected_before_routes(out):
                continue
            result = out / "COMPLETE.json"
            if not result.exists():
                if out.exists() and not args.resume:
                    raise RuntimeError(f"Incomplete attempt preserved, refusing automatic overwrite/retry: {out}")
                python = os.environ["WORKER_PYTHON"] if group["source"] == "procthor" else os.environ["HABITAT_PYTHON"]
                argv = [python, str(HERE / "run_group.py"), "--source", group["source"],
                    "--seed", str(CONFIG['sampling']['master_seed'] + number * CONFIG['sampling']['max_scene_attempts_per_group'] + attempt), "--output", str(out),
                    "--gpu-index", str(args.gpu)]
                if out.exists() and args.resume:
                    argv.append('--resume')
                save(root / "logs" / f"current_lane{args.lane}.json",
                     dict(group_number=number, attempt=attempt, source=group["source"],
                          argv=argv, output=str(out), utc=utc()))
                child = subprocess.Popen(argv, start_new_session=True)
                try:
                    code = child.wait()
                    if code:
                        raise RuntimeError(f"Group worker exited {code}; stopped without retries: {out}")
                finally:
                    terminate_child(child)
                if rejected_before_routes(out):
                    continue
                if not result.exists():
                    raise RuntimeError(f"Group exited without COMPLETE or safe rejection: {out}")
            row = json.loads(result.read_text())
            require_stamp(row)
            if row.get("route_count") != 5:
                raise RuntimeError(f"Expected exactly five shared routes in completed group: {out}")
            save(groupdir / "COMPLETE.json", dict(**stamp(), result=str(result), group_number=number,
                 scene_attempt=attempt, route_count=5, **group))
            completed += 1
            print(f'COMPLETE group={number} source={group["source"]} routes=5', flush=True)
            break
        else:
            raise RuntimeError(f"200 pre-route scene attempts exhausted in {groupdir}")
    save(root / f"COMPLETE_LANE_{args.lane}.json",
         dict(utc=utc(), completed_groups=completed, routes=completed * 5,
              smoke=bool(args.limit_groups), lane=args.lane, lanes=args.lanes))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--gpu", type=int, required=True)
    pre.add_argument("--uuid", required=True)
    watch = sub.add_parser("guard")
    watch.add_argument("--owner-pid", type=int, required=True)
    watch.add_argument("--uuid", required=True)
    watch.add_argument("--lane", type=int, required=True)
    run = sub.add_parser("run")
    run.add_argument("lane", type=int)
    run.add_argument("gpu", type=int)
    run.add_argument("--lanes", type=int, default=6)
    run.add_argument("--limit-groups", type=int, default=0)
    run.add_argument("--resume", action="store_true")
    trip = sub.add_parser("trip")
    trip.add_argument("--lane", type=int, required=True)
    trip.add_argument("--reason", required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("terminated")))
    if args.operation == "run" and not 0 <= args.lane < args.lanes:
        parser.error("lane must be in [0, lanes)")
    try:
        if args.operation == "trip":
            latch(Path(os.environ["REALTIME_OUTPUT"]), args.reason, args.lane)
        else:
            {"preflight": preflight, "guard": guard, "run": dispatch}[args.operation](args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        if args.operation == "run":
            latch(Path(os.environ["REALTIME_OUTPUT"]), repr(error), args.lane)
        raise


if __name__ == "__main__":
    main()
