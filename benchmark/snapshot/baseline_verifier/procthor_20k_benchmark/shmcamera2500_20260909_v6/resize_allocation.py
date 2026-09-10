"""Explicit user-authorized operational resize; benchmark source/contract unchanged."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import re
import subprocess
import time

from dispatch import descendants
from protocol import check_model_acceptance, verify_root
from summary import summarize

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/liudiwen/VGGT/baseline_verifier/procthor_20k_runs/shmcamera2500_20260909_v6')
AUDIT = None
GPUS = []


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def command(args):
    return subprocess.check_output(args, text=True, timeout=20)


def token(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if stat[0] == 'Z' else stat[19]
    except (OSError, IndexError):
        return None


def fingerprint_files():
    files = list(ROOT.glob('groups/group_*/attempt_*/route_*/*/result.json'))
    files += list(ROOT.glob('groups/group_*/attempt_*/tasks.json'))
    files += list(ROOT.glob('groups/group_*/attempt_*/route_*/*/trajectory.npz'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def check_hashes(before):
    for rel, digest in before.items():
        p = ROOT / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Committed result/task changed: ' + rel)


def stop():
    verify_root(ROOT)
    if (AUDIT / 'STOPPED.json').exists():
        raise RuntimeError('Stop audit already exists; inspect instead of repeating')
    if (ROOT / 'CIRCUIT_BREAKER.json').exists():
        raise RuntimeError('Circuit breaker exists; resize cannot bypass it')
    targets, tracked = [], {}
    allocation = json.loads((ROOT/'CURRENT_ALLOCATION.json').read_text())
    for record in allocation['lanes']:
        lane=record['lane'];gpu=record.get('gpu_index',record.get('gpu'))
        ready = json.loads((ROOT / 'logs' / f'guard{lane}.ready.json').read_text())
        pid = int(ready['owner_pid'])
        process = Path(f'/proc/{pid}')
        argv = (process / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        if (process.stat().st_uid != os.getuid() or
                argv != ['bash', str(HERE / 'launch.sh'), str(gpu), str(lane)]):
            raise RuntimeError('Unexpected launcher identity; refusing signal: ' + str(pid))
        targets.append(dict(lane=lane, pid=pid, start_token=token(pid), argv=argv))
        for child in descendants(pid):
            if Path(f'/proc/{child}').stat().st_uid != os.getuid():
                raise RuntimeError('Foreign descendant; refusing signal')
            tracked[child] = token(child)
    save(AUDIT / 'STOP_REQUEST.json', dict(utc=now(), reason='User requested resource resize', requested_gpus=GPUS, targets=targets))
    for item in targets:
        if token(item['pid']) != item['start_token']:
            raise RuntimeError('Launcher identity changed before signal')
        os.kill(item['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 125
    while True:
        live = [pid for pid, old in tracked.items() if old is not None and token(pid) == old]
        if not live:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Owned descendants did not exit; no restart: ' + str(live))
        print('Waiting for owned cleanup:', len(live), flush=True)
        time.sleep(5)
    if (ROOT / 'CIRCUIT_BREAKER.json').exists():
        raise RuntimeError('Circuit tripped during stop; do not reset automatically')
    report = summarize(ROOT)
    if report['pending_result_commits']:
        raise RuntimeError('Uncommitted result requires inspection before resume')
    proof = dict(utc=now(), completed_episodes=report['completed_episodes'],
                 completed_groups=report['complete_scene_groups'],
                 completed_routes=report['fully_compared_routes'], sha256=fingerprint_files())
    save(AUDIT / 'STOPPED.json', proof)
    print(json.dumps({k:v for k,v in proof.items() if k != 'sha256'}), flush=True)


def resume():
    verify_root(ROOT)
    check_model_acceptance()
    proof = json.loads((AUDIT / 'STOPPED.json').read_text())
    check_hashes(proof['sha256'])
    if (AUDIT / 'RESUMED.json').exists():
        raise RuntimeError('Already resumed; refusing duplicate launch')
    if (ROOT / 'CIRCUIT_BREAKER.json').exists():
        raise RuntimeError('Circuit breaker remains latched')
    records = []
    for _ in range(10):
        raw = command(['nvidia-smi', '-i', ','.join(map(str,GPUS)),
                       '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                       '--format=csv,noheader,nounits'])
        rows = []
        for line in raw.strip().splitlines():
            i,u,used,total,util = [x.strip() for x in line.split(',')]
            rows.append(dict(index=int(i), uuid=u, memory_percent=100*float(used)/float(total), compute_percent=float(util)))
        records.append(dict(utc=now(), rows=rows))
        time.sleep(1)
    apps = command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'])
    busy = {line.split(',')[0].strip() for line in apps.splitlines()}
    gpus = records[-1]['rows']
    if len(gpus) != len(GPUS) or any(g['uuid'] in busy for g in gpus) or any(
            row['memory_percent'] >= 30 or row['compute_percent'] >= 20
            for sample in records for row in sample['rows']):
        raise RuntimeError('Requested GPUs are not idle; no restart')
    save(AUDIT / 'RESUME_PREFLIGHT.json', dict(samples=records, compute_apps=apps))
    allocation = dict(utc=now(), operation=AUDIT.name,
                      lanes_count=len(GPUS), groups_total=500, original_started_utc='2026-09-09T12:10:24.099852+00:00',
                      assignment=f'group_number modulo {len(GPUS)}', lanes=[], immutable_baseline=str(AUDIT/'STOPPED.json'))
    for lane,gpu in enumerate(gpus):
        physical=gpu['index']
        name = f'shm_v6_gpu{physical}_{AUDIT.name}'
        cmd = f'env REALTIME_LANES={len(GPUS)} REALTIME_RESUME=1 REALTIME_OUTPUT={ROOT} bash {HERE}/launch.sh {physical} {lane}'
        subprocess.run(['tmux','new-session','-d','-s',name,cmd], check=True)
        allocation['lanes'].append(dict(lane=lane, gpu_index=physical, uuid=gpu['uuid'], tmux=name,
                                       command=cmd, assigned_groups=len(range(lane,500,len(GPUS)))))
        save(ROOT/'CURRENT_ALLOCATION.json', allocation)
    save(AUDIT/'RESUMED.json', allocation)
    print(json.dumps(allocation), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stop','resume','verify'])
    parser.add_argument('--gpus',required=True,help='Comma-separated authorized physical GPU indices')
    parser.add_argument('--operation',required=True,help='Unique safe audit directory name')
    args=parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',args.operation): raise ValueError('Invalid operation name')
    GPUS=[int(v) for v in args.gpus.split(',')]
    if not GPUS or len(set(GPUS))!=len(GPUS) or any(v not in range(8) for v in GPUS): raise ValueError('Invalid GPUs')
    AUDIT=ROOT/'operations'/args.operation
    action=args.action
    if action == 'verify':
        check_hashes(json.loads((AUDIT/'STOPPED.json').read_text())['sha256'])
        print('All committed result/task/trajectory hashes unchanged', flush=True)
    else:
        {'stop':stop, 'resume':resume}[action]()

