"""Read-only data audit plus a saved verification artifact after recovery."""
import hashlib
import json
from pathlib import Path
import subprocess
from repair_start_ops import ROOT, AUDIT, read, save, sha, utc
from protocol import digest, validate_result
from summary import summarize

before=read(AUDIT/'BEFORE.json')
unchanged=0;changed=[]
for name,old in before['files'].items():
    file=ROOT/name
    if sha(file)==old:
        unchanged+=1;continue
    if name!='groups/group_0051_mp3d/attempt_001/tasks.json':
        raise RuntimeError('Unexpected modification: '+name)
    audit=read(file.parent/'START_BODY_REPAIR.json')
    original_bytes=(json.dumps(audit['original_tasks'],indent=2)+'\n').encode('utf-8')
    if hashlib.sha256(original_bytes).hexdigest()!=old:
        raise RuntimeError('Archived original tasks do not match pre-repair byte hash')
    new=read(file)
    if new!=audit['updated_tasks'] or digest(new)!=audit['updated_tasks_sha256']:
        raise RuntimeError('Repaired task journal differs')
    actual=[i for i,(a,b) in enumerate(zip(audit['original_tasks'],new)) if a!=b]
    if actual!=[4] or audit['preexisting_model_evidence']:
        raise RuntimeError('Unexpected repaired route/evidence')
    changed.append(dict(file=name,replaced_indices=actual,original_bytes_preserved_in_journal=True))

attempt=ROOT/'groups/group_0051_mp3d/attempt_001'
tasks=read(attempt/'tasks.json')
repaired_results=[]
for p in (attempt/'route_04').glob('*/result.json'):
    row=read(p)
    if 'protocol_id' not in row:continue
    validate_result(row,p.parent.name,tasks[4])
    repaired_results.append(dict(method=p.parent.name,status=row['status'],success=row['success']))
state=summarize(ROOT)
allocation=read(ROOT/'CURRENT_ALLOCATION.json')
workers=[]
raw=subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,rss,args'],text=True)
for line in raw.splitlines():
    fields=line.split()
    if len(fields)>7 and fields[6].endswith('/run_group.py') and str(ROOT) in line:
        workers.append(line)
guards=[]
for lane in allocation['lanes']:
    n=lane['lane'];ready=read(ROOT/f'logs/guard{n}.ready.json')
    mapping=read(ROOT/f'logs/gpu_mapping_lane{n}.json')
    guards.append(dict(lane=n,ready=ready,mapping=mapping,
        guard_alive=Path(f"/proc/{ready['pid']}").exists(),
        owner_alive=Path(f"/proc/{ready['owner_pid']}").exists()))
report=dict(utc=utc(),original_committed=before['summary']['completed_episodes'],
    unchanged_files=unchanged,audited_changed_task_files=changed,
    repaired_route_model_results=repaired_results,workers=workers,guards=guards,
    allocation=allocation,circuit_breaker=(ROOT/'CIRCUIT_BREAKER.json').exists(),summary=state)
save(AUDIT/'RECOVERY_VERIFIED.json',report)
print(json.dumps(dict(utc=report['utc'],original_committed=report['original_committed'],
    unchanged_files=unchanged,audited_changed_task_files=changed,
    repaired_route_model_results=repaired_results,worker_count=len(workers),
    all_guards_alive=all(g['guard_alive'] and g['owner_alive'] for g in guards),
    circuit_breaker=report['circuit_breaker'],episodes=state['completed_episodes'],
    groups=state['complete_scene_groups'],routes=state['fully_compared_routes'])))
