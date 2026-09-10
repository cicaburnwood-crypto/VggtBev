"""One-shot deletion of the explicitly enumerated obsolete realtime outputs.

Never matches arbitrary runs, assets, training data or checkpoints by wildcard.
The small safety incident and deletion manifest survive outside deleted trees.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('host', choices=['local', 'super_5090'])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    report_root = Path(__file__).resolve().parent
    if args.host == 'super_5090':
        root = Path('/home/liudiwen/VGGT/baseline_verifier/procthor_20k_runs')
        names = ['realtime2500_20260909', 'realtime2500_20260909_r1',
                 'realtime2500_repair_20260909']
    else:
        root = Path('/home/wolfie/VPN/.work/realtime_benchmark_20260909')
        names = ['failure_audit', 'first_episode_evidence', '__pycache__']
    if root.is_symlink() or root.resolve() != root:
        raise RuntimeError('Unexpected deletion root')
    # Inspect actual same-user Python processes, not a shell command containing
    # these strings. No signals are sent, even when the check refuses cleanup.
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry/'cmdline').read_bytes().split(b'\0')
            if not argv or b'python' not in Path(os.fsdecode(argv[0])).name.encode():
                continue
            if any(b'realtime2500' in x or b'virtualcamera2500' in x for x in argv[1:]):
                active.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if active:
        raise RuntimeError(f'Benchmark Python processes still active: {active}')
    rows = []
    for name in names:
        path = root/name
        if path.is_symlink():
            raise RuntimeError(f'Refusing linked deletion root: {path}')
        if not path.exists():
            rows.append(dict(path=str(path), status='already_absent'))
            continue
        if path.resolve().parent != root or path.stat().st_uid != os.getuid():
            raise RuntimeError(f'Unexpected path or owner: {path}')
        allocated = int(subprocess.check_output(['du', '-s', '-B1', '--', str(path)], text=True).split()[0])
        incidents = [json.loads(f.read_text()) for f in path.glob('CIRCUIT_BREAKER.json')]
        rows.append(dict(path=str(path), status='planned', allocated_bytes=allocated,
                         safety_incidents=incidents))
    report = dict(host=args.host, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='only obsolete 20260909 realtime experiment outputs; source and other experiments retained',
        model_benchmark_started=False, rows=rows)
    report_path = report_root/f'CLEANUP_{args.host}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    if args.apply:
        report_path.write_text(json.dumps(report, indent=2)+'\n')
        for row in rows:
            if row['status'] != 'planned':
                continue
            shutil.rmtree(row['path'])
            row['status'] = 'deleted' if not Path(row['path']).exists() else 'deletion_failed'
            report_path.write_text(json.dumps(report, indent=2)+'\n')
        report['deleted_allocated_bytes'] = sum(r.get('allocated_bytes', 0) for r in rows if r['status']=='deleted')
        report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
