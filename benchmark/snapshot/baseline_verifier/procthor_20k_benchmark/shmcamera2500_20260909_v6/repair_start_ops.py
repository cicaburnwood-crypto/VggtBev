"""One-shot, audited software repair/resume; not an automatic retry loop."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'procthor_20k_runs/shmcamera2500_20260909_v6'
AUDIT=ROOT/'operations/start_body_repair_20260910'


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p): return json.loads(p.read_text())
def save(p,row):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(p)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(argv): return subprocess.check_output(argv,text=True,timeout=20).strip()


def snapshot():
    if (AUDIT/'BEFORE.json').exists(): raise RuntimeError('Already snapshotted')
    AUDIT.mkdir(parents=True,exist_ok=True)
    for p in [ROOT/'RUN_PROTOCOL.json', ROOT/'CIRCUIT_BREAKER.json',
              HERE/'run_group.py', HERE/'MODEL_INTERFACE_ACCEPTED.json']:
        shutil.copy2(p,AUDIT/p.name)
    files=[]
    for pattern in ('groups/group_*/attempt_*/tasks.json',
                    'groups/group_*/attempt_*/route_*/*/result.json',
                    'groups/group_*/attempt_*/route_*/*/trajectory.npz'):
        files.extend(ROOT.glob(pattern))
    from summary import summarize
    save(AUDIT/'BEFORE.json',dict(utc=utc(),files={str(p.relative_to(ROOT)):sha(p) for p in files},
                               summary=summarize(ROOT)))
    print(json.dumps(dict(snapshot=str(AUDIT),files=len(files))))


def migrate():
    from protocol import manifest, check_model_acceptance
    old=read(AUDIT/'RUN_PROTOCOL.json');new=manifest()
    if new['config']!=old['config']: raise RuntimeError('Scientific protocol changed')
    changed=[k for k in new['code_sha256'] if new['code_sha256'][k]!=old['code_sha256'][k]]
    if changed!=['run_group.py']: raise RuntimeError('Unexpected implementation changes: '+repr(changed))
    tests=read(AUDIT/'CPU_REGRESSION.json')
    if tests['exit_code'] or tests['run_group_sha256']!=sha(HERE/'run_group.py'):
        raise RuntimeError('Current sampler regression proof missing/failed')
    check_model_acceptance()
    proof=read(HERE/'MODEL_INTERFACE_ACCEPTED.json')
    if not all('startfix_20260910' in p for p in proof['evidence_paths']):
        raise RuntimeError('Need actual fresh interface evidence for this repair')
    before=read(AUDIT/'BEFORE.json')
    for name,digest in before['files'].items():
        if sha(ROOT/name)!=digest: raise RuntimeError('Old result/task/trajectory changed: '+name)
    if read(ROOT/'RUN_PROTOCOL.json')!=old: raise RuntimeError('Concurrent protocol migration')
    save(AUDIT/'SOURCE_MIGRATION.json',dict(utc=utc(),changed=changed,old=old,new=new,
        interface_proof=proof,preserved_files=len(before['files']),
        reason='Sampling-only exact-body initial-pose validity correction; models, controllers and scoring unchanged'))
    save(ROOT/'RUN_PROTOCOL.json',new)
    from summary import summarize
    current=summarize(ROOT)
    if current['completed_episodes']!=before['summary']['completed_episodes']:
        raise RuntimeError('Committed outcome count changed')
    save(AUDIT/'MIGRATED.json',dict(utc=utc(),summary=current))
    print(json.dumps(dict(migrated=True,committed=current['completed_episodes'])))


def regress():
    import sys
    tests=['test_start_sampling','test_exact_executor','test_virtual_episode',
           'test_native_execution','test_adapter_v5','test_shm_contract','test_protocol']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=HERE,
                          capture_output=True,text=True,timeout=60)
    save(AUDIT/'CPU_REGRESSION.json',dict(utc=utc(),exit_code=result.returncode,
        tests=tests,output=result.stdout+result.stderr,run_group_sha256=sha(HERE/'run_group.py')))
    print(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError('Regression failed')


def resume():
    from protocol import check_model_acceptance, verify_root
    check_model_acceptance();verify_root(ROOT)
    if not (AUDIT/'MIGRATED.json').exists() or (AUDIT/'RESUMED.json').exists():
        raise RuntimeError('Migration missing or resume already performed')
    breaker=read(ROOT/'CIRCUIT_BREAKER.json')
    if breaker!=read(AUDIT/'CIRCUIT_BREAKER.json'):
        raise RuntimeError('Circuit breaker changed; re-audit required')
    failure=ROOT/'groups/group_0051_mp3d/attempt_001/INFRASTRUCTURE_ERROR.json'
    if read(failure)['error']!="RuntimeError('Invalid benchmark start: robot body collides')":
        raise RuntimeError('Not the audited software failure')
    recovery=run(['nvidia-smi','-i','0,1,2,3',
        '--query-gpu=index,uuid,gpu_recovery_action','--format=csv,noheader,nounits'])
    if len(recovery.splitlines())!=4 or any(line.split(',')[-1].strip().lower()!='none' for line in recovery.splitlines()):
        raise RuntimeError('Assigned GPU requests recovery')
    apps=run(['nvidia-smi','-i','0,1,2,3','--query-compute-apps=pid,process_name','--format=csv,noheader'])
    if apps: raise RuntimeError('Assigned GPU still occupied')
    save(AUDIT/'RESUME_PREFLIGHT.json',dict(utc=utc(),recovery=recovery,compute_apps=apps,
        kernel_log_access='unavailable; old guard logs show shared software latch, live NVML Xid monitoring retained',
        policy='Each launcher separately repeats 10-second idle and recovery checks'))
    (ROOT/'CIRCUIT_BREAKER.json').rename(AUDIT/'CLEARED_SOFTWARE_CIRCUIT_BREAKER.json')
    lanes=[]
    for lane in range(4):
        name=f'shm_v6_gpu{lane}_20260910_startfix'
        cmd=f'env REALTIME_LANES=4 REALTIME_RESUME=1 REALTIME_OUTPUT={ROOT} bash {HERE}/launch.sh {lane} {lane}'
        subprocess.run(['tmux','new-session','-d','-s',name,cmd],check=True)
        lanes.append(dict(lane=lane,gpu=lane,tmux=name))
    previous=read(ROOT/'CURRENT_ALLOCATION.json')
    record=dict(utc=utc(),lanes=lanes,physical_gpus=[0,1,2,3],lane_count=4,
        assignment='group_number modulo 4',resume=True,previous_allocation=previous,
        repair_audit=str(AUDIT),startup_verification_pending=True)
    save(AUDIT/'RESUMED.json',record);save(ROOT/'CURRENT_ALLOCATION.json',record)
    print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['snapshot','regress','migrate','resume'])
    globals()[parser.parse_args().action]()
