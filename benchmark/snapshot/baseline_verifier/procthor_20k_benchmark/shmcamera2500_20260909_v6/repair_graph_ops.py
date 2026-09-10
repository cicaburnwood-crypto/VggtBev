"""Audited repair of one known pre-route sampling error; no blind retry."""
import argparse
import json
from pathlib import Path
import shutil

import repair_start_ops as base

HERE=base.HERE
ROOT=base.ROOT
AUDIT=ROOT/'operations/empty_graph_repair_20260910'
ACCEPT=ROOT.parent/'native_interface_acceptance_graphfix_20260910'
FAILED=ROOT/'groups/group_0081_procthor/attempt_000'
base.AUDIT=AUDIT
read,save,sha,utc=base.read,base.save,base.sha,base.utc


def snapshot():
    from protocol import verify_root
    verify_root(ROOT)
    error=read(FAILED/'INFRASTRUCTURE_ERROR.json')
    if error['routes_started'] or error['error']!="RuntimeError('reachable graph has no useful connected component')":
        raise RuntimeError('Not the audited pre-route graph failure')
    if any((FAILED/p).exists() for p in ('tasks.json','routes.json','results.json')) or list(FAILED.glob('route_*')):
        raise RuntimeError('Failed scene already has route evidence')
    base.snapshot()
    shutil.copy2(FAILED/'INFRASTRUCTURE_ERROR.json',AUDIT/'FAILED_SCENE_ERROR.json')
    shutil.copy2(ROOT/'logs/lane1.log',AUDIT/'lane1_before.log')
    save(AUDIT/'FAILURE_SCOPE.json',dict(failed=str(FAILED),routes_started=False,
        error=error,policy='Exact existing legacy_sampling_failure classifier only; no model outcome resampling'))


def regress():
    base.regress()


def migrate():
    from protocol import manifest,check_model_acceptance
    old=read(AUDIT/'RUN_PROTOCOL.json');new=manifest()
    changed=[k for k in new['code_sha256'] if new['code_sha256'][k]!=old['code_sha256'][k]]
    if new['config']!=old['config'] or changed!=['run_group.py']:
        raise RuntimeError('Unexpected scientific/source changes')
    tests=read(AUDIT/'CPU_REGRESSION.json')
    if tests['exit_code'] or tests['run_group_sha256']!=sha(HERE/'run_group.py'):
        raise RuntimeError('Missing current successful regression proof')
    rejected=read(ACCEPT/'empty_graph_seed_2026107117/REJECTED.json')
    if not rejected['replacement_allowed'] or rejected['stage']!='before_routes':
        raise RuntimeError('Missing live regression proof')
    check_model_acceptance()
    proof=read(HERE/'MODEL_INTERFACE_ACCEPTED.json')
    if not all('native_interface_acceptance_graphfix_20260910' in p for p in proof['evidence_paths']):
        raise RuntimeError('Need fresh seven-model interface evidence')
    before=read(AUDIT/'BEFORE.json')
    for name,digest in before['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Existing output changed: '+name)
    if read(ROOT/'RUN_PROTOCOL.json')!=old:raise RuntimeError('Concurrent migration')
    save(AUDIT/'SOURCE_MIGRATION.json',dict(utc=utc(),changed=changed,old=old,new=new,
        interface_proof=proof,live_rejection=rejected,preserved_files=len(before['files']),
        reason='Classify exact unstarted ProcTHOR graph infeasibility with existing scene-sampling rejection'))
    save(ROOT/'RUN_PROTOCOL.json',new)
    from summary import summarize
    summary=summarize(ROOT)
    if summary['completed_episodes']!=before['summary']['completed_episodes']:
        raise RuntimeError('Committed outcome count changed')
    save(AUDIT/'MIGRATED.json',dict(utc=utc(),summary=summary))
    print(json.dumps(dict(migrated=True,committed=summary['completed_episodes'])),flush=True)


def prepare_resume():
    from protocol import verify_root,check_model_acceptance
    verify_root(ROOT);check_model_acceptance()
    if not (AUDIT/'MIGRATED.json').exists() or (AUDIT/'STOPPED.json').exists():
        raise RuntimeError('Migration missing or resume preparation already performed')
    breaker=read(ROOT/'CIRCUIT_BREAKER.json')
    if breaker!=read(AUDIT/'CIRCUIT_BREAKER.json'):
        raise RuntimeError('Circuit breaker changed; re-audit required')
    if str(FAILED) not in breaker['reason'] or breaker['lane']!=1:
        raise RuntimeError('Different failure')
    # Only this benchmark may be absent. Never stop unrelated tasks here.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(str(HERE/name) in cmd for name in ('launch.sh','dispatch.py','run_group.py','shm_runtime.py','navigation_baseline_runtime.py')):
            raise RuntimeError('Own benchmark/acceptance processes remain: '+proc.name)
    recovery=base.run(['nvidia-smi','-i','0,1,2,3,4',
        '--query-gpu=index,uuid,gpu_recovery_action','--format=csv,noheader,nounits'])
    if len(recovery.splitlines())!=5 or any(line.split(',')[-1].strip().lower()!='none' for line in recovery.splitlines()):
        raise RuntimeError('GPU requires hardware recovery')
    apps=base.run(['nvidia-smi','-i','0,1,2,3,4',
        '--query-compute-apps=pid,process_name','--format=csv,noheader'])
    if apps:raise RuntimeError('Assigned GPUs occupied')
    save(AUDIT/'LATCH_CLEAR_PREFLIGHT.json',dict(utc=utc(),recovery=recovery,apps=apps,
        kernel_log_access='unavailable; verified software traceback and live GPU recovery state; NVML guards retained'))
    before=read(AUDIT/'BEFORE.json')
    for name,digest in before['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Output changed before resume: '+name)
    # Original failed attempt is retained. Patched worker replays it once and
    # emits qualified REJECTED before the dispatcher samples the next scene.
    (ROOT/'CIRCUIT_BREAKER.json').rename(AUDIT/'CLEARED_SOFTWARE_CIRCUIT_BREAKER.json')
    summary=before['summary']
    save(AUDIT/'STOPPED.json',dict(utc=utc(),reason='Already stopped by audited software latch; no signal needed',
        completed_episodes=summary['completed_episodes'],completed_groups=summary['complete_scene_groups'],
        completed_routes=summary['fully_compared_routes'],sha256=before['files']))
    print('Audited software latch cleared; use existing five-GPU guarded resume',flush=True)


def verify():
    from protocol import verify_root
    from summary import summarize
    verify_root(ROOT)
    before=read(AUDIT/'BEFORE.json')
    for name,digest in before['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Existing output changed: '+name)
    if (ROOT/'CIRCUIT_BREAKER.json').exists():raise RuntimeError('New circuit breaker')
    allocation=read(ROOT/'CURRENT_ALLOCATION.json')
    if allocation['operation']!=AUDIT.name:raise RuntimeError('Unexpected allocation')
    rejected=read(FAILED/'REJECTED.json')
    if not rejected['replacement_allowed'] or rejected['stage']!='before_routes':
        raise RuntimeError('Failed scene was not safely rejected')
    records=[]
    for lane in range(5):
        guard=read(ROOT/f'logs/guard{lane}.ready.json')
        for key in ('pid','owner_pid'):
            if not (Path('/proc')/str(guard[key])/'cmdline').exists():
                raise RuntimeError('Guard/owner no longer alive')
        current=read(ROOT/f'logs/current_lane{lane}.json')
        records.append(dict(lane=lane,guard=guard,current=current))
    summary=summarize(ROOT)
    if summary['completed_episodes']<=before['summary']['completed_episodes']:
        raise RuntimeError('No new committed outputs yet')
    proof=dict(utc=utc(),preserved_files=len(before['files']),
        before_episodes=before['summary']['completed_episodes'],
        current_episodes=summary['completed_episodes'],complete_groups=summary['complete_scene_groups'],
        fully_compared_routes=summary['fully_compared_routes'],allocation=allocation,lanes=records,
        formal_failed_scene_rejection=rejected,circuit_breaker=None)
    save(AUDIT/'RECOVERY_VERIFIED.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('allocation','lanes')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['snapshot','regress','migrate','prepare_resume','verify'])
    globals()[parser.parse_args().action]()
