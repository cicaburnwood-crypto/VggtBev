"""One-shot eligible-GPU inventory and detached, equal deterministic shards."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from protocol import check_model_acceptance, prepare_root, manifest


def main():
    here=Path(__file__).resolve().parent
    root=Path('/home/liudiwen/VGGT/baseline_verifier/procthor_20k_runs/shmcamera2500_20260909_v6')
    check_model_acceptance()
    if root.exists(): raise RuntimeError('Formal output already exists; inspect, never overwrite or re-shard automatically')
    samples=[]
    for _ in range(10):
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True)
        current=[]
        for line in raw.strip().splitlines():
            i,u,used,total,util=[x.strip() for x in line.split(',')]
            current.append(dict(index=int(i),uuid=u,vram_percent=100*float(used)/float(total),util=float(util)))
        samples.append(current);time.sleep(1)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
        '--format=csv,noheader,nounits'],text=True)
    busy={line.split(',')[0].strip() for line in apps.splitlines()}
    eligible=[r for r in samples[-1] if r['uuid'] not in busy and all(
        any(s['uuid']==r['uuid'] and s['vram_percent']<30 and s['util']<20 for s in window)
        for window in samples)]
    if not eligible: raise RuntimeError('No idle GPU; no job launched')
    prepare_root(root)
    record=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        samples=samples,foreign_compute_apps=apps,lanes=[],protocol=manifest())
    for lane,gpu in enumerate(eligible):
        name=f'shm_v6_gpu{gpu["index"]}_20260909'
        command=f'env REALTIME_LANES={len(eligible)} REALTIME_OUTPUT={root} bash {here}/launch.sh {gpu["index"]} {lane}'
        subprocess.run(['tmux','new-session','-d','-s',name,command],check=True)
        record['lanes'].append(dict(lane=lane,gpu_index=gpu['index'],uuid=gpu['uuid'],tmux=name,command=command,
            assigned_groups=len(range(lane,500,len(eligible)))))
        (root/'LAUNCH_SESSION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(output=str(root),lanes=record['lanes']),indent=2),flush=True)


if __name__=='__main__':main()
