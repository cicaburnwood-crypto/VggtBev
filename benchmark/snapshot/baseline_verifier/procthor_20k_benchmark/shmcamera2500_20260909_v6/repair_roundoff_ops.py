"""Audited, numerical-only executor correction and frozen-task recovery."""
import argparse
import importlib.util
import json
from pathlib import Path
import repair_start_ops as b

HERE,ROOT=b.HERE,b.ROOT
AUDIT=ROOT/'operations/executor_roundoff_repair_20260910'
ACCEPT=ROOT.parent/'native_interface_acceptance_roundoff_20260910'
FAILED=ROOT/'groups/group_0084_hm3d/attempt_000'
read,save,sha,utc=b.read,b.save,b.sha,b.utc


def reproduce():
    import exact_executor as new
    spec=importlib.util.spec_from_file_location('old_exact_executor',AUDIT/'exact_executor.py')
    import sys
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    path=[[0.,.02000007920647981]]
    xyz=(76.39907467126628,-49.29987478570277,-2.1130002583387655)
    error=None
    try:list(old.path_steps(path,old.Pose(*xyz),horizon_m=.5,allow_reverse=True))
    except ValueError as caught:error=str(caught)
    if error!='Translation rate limit exceeded':raise RuntimeError('Old source did not reproduce')
    steps=list(new.path_steps(path,new.Pose(*xyz),horizon_m=.5,allow_reverse=True))
    count=0;max_time_delta=0.;max_endpoint_delta=0.
    for f in FAILED.glob('route_*/*/events.jsonl'):
        for line in f.read_text().splitlines():
            row=json.loads(line);native=row.get('native_path');pose=row.get('observation_pose')
            if not native or pose is None:continue
            args=(pose['x'],pose['z'],pose['yaw_rad'])
            kwargs=dict(horizon_m=row.get('execution_horizon_m',.5),allow_reverse=True,
                        native_headings_left_rad=row.get('path_headings_left_rad'))
            a=list(old.path_steps(native,old.Pose(*args),**kwargs))
            z=list(new.path_steps(native,new.Pose(*args),**kwargs))
            count+=1
            max_time_delta=max(max_time_delta,abs(sum(s.dt_s for s in a)-sum(s.dt_s for s in z)))
            if a and z:max_endpoint_delta=max(max_endpoint_delta,float(new.np.linalg.norm(a[-1].end.xz-z[-1].end.xz)))
    if max_time_delta>1e-8 or max_endpoint_delta>1e-8:raise RuntimeError('Material semantic change')
    proof=dict(utc=utc(),old_error=error,reproduction_path=path,reproduction_pose=xyz,
        original_trigger_unavailable=True,new_max_speed_m_s=max(abs(s.speed_m_s) for s in steps),
        saved_event_replays=count,max_nominal_time_delta_seconds=max_time_delta,
        max_endpoint_delta_m=max_endpoint_delta,executor_sha256=sha(HERE/'exact_executor.py'))
    save(AUDIT/'NUMERICAL_REPRODUCTION.json',proof);print(json.dumps(proof),flush=True)


def preserved():
    before=read(AUDIT/'BEFORE.json')
    for name,digest in before['files'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Existing output changed: '+name)
    return before


def qualify_executor():
    tests=read(AUDIT/'CPU_REGRESSION.json')
    if tests['exit_code'] or tests['executor_sha256']!=sha(HERE/'exact_executor.py'):
        raise RuntimeError('Missing current CPU regression')
    evidence=[ROOT.parent/f'executor_acceptance_roundoff_{source}_20260910/trials'
              for source in ('procthor','hm3d')]
    rows=[]
    for folder in evidence:
        proof=read(folder/'PASS.json')
        if len(proof['routes'])!=5:raise RuntimeError('Need five actual camera routes per backend')
        rows.extend(proof['routes'])
    for r in rows:
        if not (r['status']=='complete' and r['endpoint_error_m']<1e-7 and
                r['cross_track_max_m']<1e-7 and r['rgb_changed'] and
                r['max_speed_m_s']<=1.+1e-8 and r['max_yaw_deg_s']<=90.+1e-6 and
                abs(r['camera_height_min_m']-.5)<1e-5 and abs(r['camera_height_max_m']-.5)<1e-5):
            raise RuntimeError('Actual executor acceptance failed')
    proof=dict(utc=utc(),cpu_pass=True,live_routes=len(rows),live_completed=len(rows),
        endpoint_error_max_m=max(r['endpoint_error_m'] for r in rows),
        cross_track_error_max_m=max(r['cross_track_max_m'] for r in rows),
        max_speed_m_s=max(r['max_speed_m_s'] for r in rows),
        max_yaw_deg_s=max(r['max_yaw_deg_s'] for r in rows),
        rgb_frames=sum(r['rgb_frames'] for r in rows),
        measured_wall_seconds=sum(r['wall_seconds'] for r in rows),
        nominal_motion_seconds=sum(r['motion_seconds'] for r in rows),
        sources=['procthor','hm3d'],evidence=list(map(str,evidence)),
        scope='Numerical executor correction; GT command-route qualification, not model scores',
        code_sha256={k:sha(HERE/k) for k in read(AUDIT/'EXECUTOR_SIMULATOR_ACCEPTED.json')['code_sha256']})
    save(AUDIT/'EXECUTOR_REQUALIFIED.json',proof)
    save(HERE/'EXECUTOR_SIMULATOR_ACCEPTED.json',proof)
    print(json.dumps(proof),flush=True)


def migrate():
    from protocol import manifest,check_model_acceptance
    old=read(AUDIT/'RUN_PROTOCOL.json');new=manifest()
    changed=[k for k in new['code_sha256'] if new['code_sha256'][k]!=old['code_sha256'][k]]
    if new['config']!=old['config'] or changed!=['exact_executor.py']:
        raise RuntimeError('Unexpected scientific/source changes')
    check_model_acceptance()
    interface=read(HERE/'MODEL_INTERFACE_ACCEPTED.json')
    if not all(str(ACCEPT) in p for p in interface['evidence_paths']):
        raise RuntimeError('Need fresh seven-model live interface evidence')
    executor=read(AUDIT/'EXECUTOR_REQUALIFIED.json')
    if any(sha(HERE/k)!=v for k,v in executor['code_sha256'].items()):
        raise RuntimeError('Executor proof stale')
    before=preserved()
    if read(ROOT/'RUN_PROTOCOL.json')!=old:raise RuntimeError('Concurrent migration')
    save(AUDIT/'SOURCE_MIGRATION.json',dict(utc=utc(),changed=changed,old=old,new=new,
        interface_proof=interface,executor_proof=executor,preserved_files=len(before['files']),
        reason='Even segment tick subdivision avoids tiny floating-point tail; represented distance sets duration; speed guards unchanged',
        inference_outputs_native_vertices_collision_and_scoring='unchanged',
        original_trigger_not_saved='Exception occurred before event commit; deterministic valid-path reproduction verified instead'))
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
        raise RuntimeError('Migration missing or resume already prepared')
    breaker=read(ROOT/'CIRCUIT_BREAKER.json')
    if breaker!=read(AUDIT/'CIRCUIT_BREAKER.json') or str(FAILED) not in breaker['reason'] or breaker['lane']!=4:
        raise RuntimeError('Different failure; re-audit needed')
    error=read(FAILED/'INFRASTRUCTURE_ERROR.json')
    if error['error']!="ValueError('Translation rate limit exceeded')" or not error['routes_started']:
        raise RuntimeError('Not the diagnosed execution failure')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(str(HERE/name) in cmd for name in ('launch.sh','dispatch.py','run_group.py','shm_runtime.py','navigation_baseline_runtime.py','executor_scene_smoke.py')):
            raise RuntimeError('Own benchmark/acceptance process remains: '+proc.name)
    recovery=b.run(['nvidia-smi','-i','0,1,2,3,4','--query-gpu=index,uuid,gpu_recovery_action','--format=csv,noheader,nounits'])
    if len(recovery.splitlines())!=5 or any(l.split(',')[-1].strip().lower()!='none' for l in recovery.splitlines()):
        raise RuntimeError('GPU requires hardware recovery')
    apps=b.run(['nvidia-smi','-i','0,1,2,3,4','--query-compute-apps=pid,process_name','--format=csv,noheader'])
    if apps:raise RuntimeError('Assigned GPUs occupied')
    before=preserved()
    save(AUDIT/'LATCH_CLEAR_PREFLIGHT.json',dict(utc=utc(),recovery=recovery,apps=apps,
        kernel_log_access='unavailable; software reproduction and GPU recovery state checked, NVML guards retained'))
    (ROOT/'CIRCUIT_BREAKER.json').rename(AUDIT/'CLEARED_SOFTWARE_CIRCUIT_BREAKER.json')
    summary=before['summary']
    save(AUDIT/'STOPPED.json',dict(utc=utc(),reason='Already software-latched; no signals required',
        completed_episodes=summary['completed_episodes'],completed_groups=summary['complete_scene_groups'],
        completed_routes=summary['fully_compared_routes'],sha256=before['files']))
    print('Audited software latch archived; ready for guarded five-GPU resume',flush=True)


def verify():
    from protocol import verify_root
    from summary import summarize
    verify_root(ROOT);before=preserved()
    if (ROOT/'CIRCUIT_BREAKER.json').exists():raise RuntimeError('New circuit breaker')
    allocation=read(ROOT/'CURRENT_ALLOCATION.json')
    if allocation['operation']!=AUDIT.name:raise RuntimeError('Unexpected allocation')
    lanes=[]
    for lane in range(5):
        guard=read(ROOT/f'logs/guard{lane}.ready.json')
        for key in ('pid','owner_pid'):
            if not (Path('/proc')/str(guard[key])/'cmdline').exists():raise RuntimeError('Guard/owner not alive')
        lanes.append(dict(lane=lane,guard=guard,current=read(ROOT/f'logs/current_lane{lane}.json')))
    summary=summarize(ROOT)
    if summary['completed_episodes']<=before['summary']['completed_episodes']:
        raise RuntimeError('No new committed output yet')
    if (FAILED/'REJECTED.json').exists():raise RuntimeError('Frozen task improperly rejected')
    proof=dict(utc=utc(),preserved_files=len(before['files']),before_episodes=before['summary']['completed_episodes'],
        current_episodes=summary['completed_episodes'],complete_groups=summary['complete_scene_groups'],
        fully_compared_routes=summary['fully_compared_routes'],lanes=lanes,allocation=allocation,circuit_breaker=None)
    save(AUDIT/'RECOVERY_VERIFIED.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('lanes','allocation')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['reproduce','qualify_executor','migrate','prepare_resume','verify'])
    globals()[p.parse_args().action]()
