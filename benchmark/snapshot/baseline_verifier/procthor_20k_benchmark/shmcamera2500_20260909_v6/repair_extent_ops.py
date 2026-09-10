"""Audited prediction-rejection classification; preserve all scored routes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import repair_start_ops as b
import repair_roundoff_ops as previous

HERE,ROOT=b.HERE,b.ROOT
AUDIT=ROOT/'operations/shm_extent_repair_20260910'
ACCEPT=ROOT.parent/'native_interface_acceptance_extent_20260910'
FAILED=ROOT/'groups/group_0104_mp3d/attempt_000'
read,save,sha,utc=b.read,b.save,b.sha,b.utc
previous.AUDIT=AUDIT
previous.FAILED=FAILED
preserved=previous.preserved
verify=previous.verify


def regress():
    tests=['test_start_sampling','test_exact_executor','test_virtual_episode',
           'test_native_execution','test_adapter_v5','test_shm_contract','test_protocol',
           'test_prediction_rejection']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=HERE,
                          capture_output=True,text=True,timeout=90)
    proof=dict(utc=utc(),exit_code=result.returncode,tests=tests,output=result.stdout+result.stderr,
        changed_code_sha256={k:sha(HERE/k) for k in ('episode.py','shm_contract.py')})
    save(AUDIT/'CPU_REGRESSION.json',proof)
    print(proof['output'],flush=True)
    if result.returncode:raise RuntimeError('Regression failed')


def migrate():
    from protocol import manifest,check_model_acceptance
    old=read(AUDIT/'RUN_PROTOCOL.json');new=manifest()
    changed=[k for k in new['code_sha256'] if new['code_sha256'][k]!=old['code_sha256'][k]]
    if new['config']!=old['config'] or changed!=['episode.py','shm_contract.py']:
        raise RuntimeError('Unexpected protocol/source changes: '+repr(changed))
    tests=read(AUDIT/'CPU_REGRESSION.json')
    if tests['exit_code'] or any(sha(HERE/k)!=v for k,v in tests['changed_code_sha256'].items()):
        raise RuntimeError('Missing current CPU proof')
    check_model_acceptance();interface=read(HERE/'MODEL_INTERFACE_ACCEPTED.json')
    if not all(str(ACCEPT) in p for p in interface['evidence_paths']):
        raise RuntimeError('Need fresh seven-model interface proof')
    replay=read(ACCEPT/'frozen_mp3d_scale_replay/REPRODUCED.json')
    if replay['original_error']!='Invalid SHM metric extent' or replay['prediction']['path_metric_m']:
        raise RuntimeError('Missing actual frozen-scene model rejection replay')
    # Neither the executor nor any of its already-qualified sources changed.
    for k,v in read(HERE/'EXECUTOR_SIMULATOR_ACCEPTED.json')['code_sha256'].items():
        if sha(HERE/k)!=v:raise RuntimeError('Executor qualification stale: '+k)
    before=preserved()
    if read(ROOT/'RUN_PROTOCOL.json')!=old:raise RuntimeError('Concurrent migration')
    save(AUDIT/'SOURCE_MIGRATION.json',dict(utc=utc(),changed=changed,old=old,new=new,
        interface_proof=interface,live_replay=replay,preserved_files=len(before['files']),
        reason='Only numeric out-of-bounds/nonfinite predicted extent becomes existing no-path outcome; no scale clamp, GT fallback, resampling or hidden retries',
        persistent_rejection='existing six fresh views then scored planner_failure; structural/transport/GPU errors remain fatal'))
    save(ROOT/'RUN_PROTOCOL.json',new)
    from summary import summarize
    summary=summarize(ROOT)
    if summary['completed_episodes']!=before['summary']['completed_episodes']:
        raise RuntimeError('Committed count changed')
    save(AUDIT/'MIGRATED.json',dict(utc=utc(),summary=summary))
    print(json.dumps(dict(migrated=True,committed=summary['completed_episodes'])),flush=True)


def prepare_resume():
    from protocol import verify_root,check_model_acceptance
    verify_root(ROOT);check_model_acceptance()
    if not (AUDIT/'MIGRATED.json').exists() or (AUDIT/'STOPPED.json').exists():
        raise RuntimeError('Migration missing or already prepared')
    breaker=read(ROOT/'CIRCUIT_BREAKER.json')
    if breaker!=read(AUDIT/'CIRCUIT_BREAKER.json') or str(FAILED) not in breaker['reason'] or breaker['lane']!=4:
        raise RuntimeError('Different failure; re-audit')
    error=read(FAILED/'INFRASTRUCTURE_ERROR.json')
    if error['error']!="RuntimeError('Invalid SHM metric extent')" or not error['routes_started']:
        raise RuntimeError('Not the diagnosed failure')
    names=('launch.sh','dispatch.py','run_group.py','shm_runtime.py','navigation_baseline_runtime.py',
           'extent_acceptance_launch.sh','extent_acceptance_runner.py','interface_acceptance.py')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(str(HERE/name) in cmd for name in names):
            raise RuntimeError('Own process still active: '+proc.name)
    recovery=b.run(['nvidia-smi','-i','0,1,2,3,4','--query-gpu=index,uuid,gpu_recovery_action','--format=csv,noheader,nounits'])
    if len(recovery.splitlines())!=5 or any(l.split(',')[-1].strip().lower()!='none' for l in recovery.splitlines()):
        raise RuntimeError('GPU hardware recovery needed')
    apps=b.run(['nvidia-smi','-i','0,1,2,3,4','--query-compute-apps=pid,process_name','--format=csv,noheader'])
    if apps:raise RuntimeError('Assigned GPUs occupied')
    before=preserved()
    save(AUDIT/'LATCH_CLEAR_PREFLIGHT.json',dict(utc=utc(),recovery=recovery,apps=apps,
        kernel_log_access='unavailable; exact model rejection reproduced, GPU recovery verified, NVML guard retained'))
    (ROOT/'CIRCUIT_BREAKER.json').rename(AUDIT/'CLEARED_SOFTWARE_CIRCUIT_BREAKER.json')
    summary=before['summary']
    save(AUDIT/'STOPPED.json',dict(utc=utc(),reason='Verified software latch; no signals required',
        completed_episodes=summary['completed_episodes'],completed_groups=summary['complete_scene_groups'],
        completed_routes=summary['fully_compared_routes'],sha256=before['files']))
    print('Audited software latch archived; ready for five-GPU guarded resume',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['regress','migrate','prepare_resume','verify'])
    globals()[p.parse_args().action]()
