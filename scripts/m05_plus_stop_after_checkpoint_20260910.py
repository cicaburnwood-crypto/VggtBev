import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import zipfile

ROOT = Path('/mnt/data/benyun/Leju-Kuavo5W/vbev/m05_plus_train')
RUN = ROOT / 'runs/m05_plus_a100_8gpu_10e_256_frozen_20260906'
DEADLINE = dt.datetime.fromisoformat('2026-09-10T08:00:00+08:00').timestamp()
STATUS = ROOT / 'watchdog/m05_plus_stop_after_checkpoint_20260910.json'
CONFIG = 'm05_plus_a100_8gpu_10e_256_frozen.toml'
WATCHDOGS = {str(ROOT / 'scripts/m05_plus_idle_watchdog_after2100_20260909.sh'), str(ROOT / 'scripts/m05_plus_idle_watchdog_a100.sh')}

def report(status, **extra):
    row = dict(time=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(), status=status, **extra)
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(STATUS)
    print(json.dumps(row), flush=True)

def args_for(pid):
    try:
        p = Path('/proc') / str(pid)
        if p.stat().st_uid != os.getuid():
            return []
        return (p / 'cmdline').read_bytes().decode().strip('\0').split('\0')
    except (OSError, UnicodeError):
        return []

def matching(kind):
    found = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        args = args_for(int(p.name))
        if kind == 'watchdog':
            match = bool(WATCHDOGS.intersection(args)) and args and Path(args[0]).name == 'bash'
        else:
            match = ('vggt_bev_method1.cli_train_m05_plus' in args and
                     any(a.endswith(CONFIG) for a in args) and
                     args and args[0].startswith('/mnt/data/benyun/Leju-Kuavo5W/vbev/'))
            if kind == 'launcher':
                match = match and 'torch.distributed.run' in args
        if match:
            found[int(p.name)] = args
    return found

def terminate_verified(processes):
    for pid, args in processes.items():
        if args_for(pid) == args:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

def checkpoints():
    result = {}
    for p in RUN.glob('m05_plus_step_*.pt'):
        try:
            st = p.stat()
            result[p.name] = (st.st_size, st.st_mtime_ns)
        except FileNotFoundError:
            pass
    return result

lock = (ROOT / 'watchdog/m05_plus_stop_after_checkpoint_20260910.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
report('waiting_until_0800', deadline='2026-09-10T08:00:00+08:00')
while time.time() < DEADLINE:
    time.sleep(max(0.1, min(60, DEADLINE - time.time())))

baseline = checkpoints()
terminate_verified(matching('watchdog'))
time.sleep(2)
launchers = matching('launcher')
if not matching('train'):
    report('no_training_at_deadline_no_action', automatic_recovery_disabled=True)
    raise SystemExit(0)
if len(launchers) != 1:
    report('error_ambiguous_launcher', launchers=list(launchers), automatic_recovery_disabled=True)
    raise SystemExit(2)
report('waiting_for_next_checkpoint', launcher=list(launchers), baseline_latest=max(baseline, default=None))
previous = {}
while True:
    if not any(args_for(pid) == args for pid, args in launchers.items()):
        report('training_ended_before_checkpoint_no_restart')
        break
    current = checkpoints()
    ready = [name for name, stat in current.items() if baseline.get(name) != stat and previous.get(name) == stat and stat[0] > 1000000000 and stat[1] >= int(DEADLINE * 1e9)]
    for name in sorted(ready, reverse=True):
        try:
            with zipfile.ZipFile(RUN / name) as archive:
                if not any(n.endswith('/data.pkl') for n in archive.namelist()):
                    continue
        except (OSError, zipfile.BadZipFile):
            continue
        report('checkpoint_complete_stopping_training', checkpoint=name, launcher=list(launchers))
        terminate_verified(launchers)
        for attempt in range(60):
            if not matching('train'):
                report('stopped_after_checkpoint', checkpoint=name, automatic_recovery_disabled=True)
                raise SystemExit(0)
            time.sleep(2)
        report('error_training_not_exited_after_sigterm', checkpoint=name, remaining=list(matching('train')))
        raise SystemExit(3)
    previous = current
    time.sleep(10)
