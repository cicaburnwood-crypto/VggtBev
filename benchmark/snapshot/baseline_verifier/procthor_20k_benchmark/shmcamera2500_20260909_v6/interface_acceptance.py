"""Small live interface qualification, deliberately NOT benchmark results.

Two fresh rendered observations per method and scene backend. Exercise released
commands / external path ticks with real elapsed time, then reset at the identical
start for the next method. Does not select checkpoints or require model success.
The complete episode scheduling has separate CPU integration tests.
"""
import argparse
from collections import deque
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen


def validate_platform(payload):
    from robot_contract import SPEC, SAFETY_MARGIN_M
    platform=payload.get('robot_platform',{})
    for name,expected in (('max_v_m_s',SPEC.speed_m_s),
                          ('max_w_rad_s',SPEC.yaw_rate_rad_s),
                          ('nomad_spacing_m',.05)):
        value=platform.get(name)
        if value is None or not math.isclose(float(value),expected,rel_tol=0,abs_tol=1e-9):
            raise RuntimeError('Runtime platform/calibration mismatch: '+name)
    if (platform.get('body_dimensions_m') != [SPEC.width_m,SPEC.length_m,SPEC.height_m] or
            platform.get('planner_safety_margin_m') != SAFETY_MARGIN_M):
        raise RuntimeError('Runtime body/safety margin differs from current protocol')


def probe_scene(scene, task, goal_images, output):
    import numpy as np
    from PIL import Image
    import procthor_benchmark_worker as w
    from episode import Inference, METHODS, native_image_history
    from exact_executor import Pose, path_steps
    from robot_contract import SPEC, collider_for_scene
    from execution_policy import policy_for, native_velocity
    from native_motion import TwistStep, sweep_twist
    records=[]
    with urlopen(os.environ['BASELINE_URL']+'/health',timeout=10) as reply:
        health=json.load(reply)
    if health.get('runtime_contract')!='realtime-native-platform-v6':
        raise RuntimeError('Stale native runtime adapter')
    validate_platform(health)
    w.atomic_json(output/'native_runtime_health.json',health)
    for method in METHODS:
        folder=output/'interfaces'/method;folder.mkdir(parents=True,exist_ok=True)
        infer=Inference(method,os.environ['SINGLE_URL'],os.environ['BASELINE_URL'],str(folder))
        infer.reset(); frames=deque(maxlen=100)
        pose=Pose(task['start_world'][0],task['start_world'][2],math.radians(task['start_yaw_degrees']),scene.floor_y)
        collider=collider_for_scene(scene,w)
        if collider.contact_at(pose): raise RuntimeError('Interface route starts inside robot collision geometry')
        service=w.OracleTargetService.from_task(task)
        policy=policy_for(method); predictions=[]
        for index in range(10 if method=='our_model' else 2):
            xyz=[pose.x,pose.base_y,pose.z]
            rgb=scene.rgb(xyz,math.degrees(pose.yaw_rad))
            if np.asarray(rgb).shape != (480,640,3): raise RuntimeError('Unexpected actual RGB shape')
            Image.fromarray(rgb).save(folder/f'observation_{index}.jpg',quality=95)
            frames.append((time.monotonic(),w.encode_jpeg(rgb)))
            target=service.query(xyz,math.degrees(pose.yaw_rad))
            goal=task['subgoals_world'][service.current_index]
            dx,dz=goal[0]-pose.x,goal[2]-pose.z
            expected=[math.cos(pose.yaw_rad)*dx-math.sin(pose.yaw_rad)*dz,
                      math.sin(pose.yaw_rad)*dx+math.cos(pose.yaw_rad)*dz]
            if not np.allclose(target,expected,atol=1e-7): raise RuntimeError('Current subgoal coordinate mismatch')
            before=time.monotonic()
            response=infer.predict(native_image_history(frames,method),target,goal_images[service.current_index])
            if method!='our_model' and not response.get('planning_failure'):
                validate_platform(response)
            if method=='our_model' and response.get('history_frame_count')!=index+1:
                raise RuntimeError('SHM cache not accumulating within a route')
            record=dict(index=index,observation_pose=asdict(pose),current_target_metric_m=target,
                current_subgoal_index=service.current_index,expected_target_metric_m=expected,
                request_wall_seconds=time.monotonic()-before,response=response,executed_seconds=0.,contact=False)
            path=response.get('path_metric_m') or []
            if path and (np.asarray(path).ndim!=2 or np.asarray(path).shape[1]!=2 or not np.isfinite(path).all()):
                raise RuntimeError('Malformed native path: '+method)
            if policy['kind']=='native_velocity':
                v,omega=native_velocity(response)
                steps=(TwistStep(pose,v,omega,SPEC.tick_s) for _ in range(25))
            elif path:
                headings=response.get('path_headings_left_rad')
                if method in {'limo_tel','limo_aug'} and headings is None:
                    raise RuntimeError('LiMo native SE2 heading not provided')
                steps=path_steps(path,pose,SPEC,horizon_m=response.get('execution_horizon_m',.5),
                                 allow_reverse=True,native_headings_left_rad=headings)
            else:
                steps=iter(())
            # Limited interface exercise, not a shortened formal episode.
            for step in steps:
                contact=sweep_twist(collider,step) if policy['kind']=='native_velocity' else collider.sweep(step)
                fraction=1. if contact is None else contact
                duration=step.dt_s*fraction
                time.sleep(duration);pose=step.at(fraction);record['executed_seconds']+=duration
                if contact is not None:
                    record['contact']=True;break
                if record['executed_seconds']>=.5-1e-8: break
            record['end_pose']=asdict(pose);predictions.append(record)
            w.atomic_json(folder/'interface.json',dict(method=method,executor_policy=policy,predictions=predictions))
        infer.reset()
        records.append(dict(method=method,executor_policy=policy,predictions=predictions))
    from protocol import manifest
    w.atomic_json(output/'INTERFACES_CHECKED.json',dict(**manifest(),checks=records,
        scope='live RGB/current subgoal/native output/bounded simulator motion; not full-route performance'))


def main():
    from protocol import manifest, METHODS
    p=argparse.ArgumentParser();p.add_argument('--gpu-index',type=int,required=True);args=p.parse_args()
    root=Path(os.environ['REALTIME_OUTPUT']);here=Path(__file__).resolve().parent
    evidence=[]
    for source in ('procthor','hm3d'):
        for attempt in range(20):
            out=root/f'{source}_{attempt:02d}'
            python=os.environ['WORKER_PYTHON'] if source=='procthor' else os.environ['HABITAT_PYTHON']
            subprocess.run([python,str(here/'run_group.py'),'--source',source,
                '--seed',str(2026090917+attempt),'--output',str(out),
                '--gpu-index',str(args.gpu_index),'--interface-check'],check=True)
            if (out/'REJECTED.json').exists(): continue
            file=out/'INTERFACES_CHECKED.json';row=json.loads(file.read_text())
            if row['code_sha256']!=manifest()['code_sha256']: raise RuntimeError('Probe source changed')
            if {r['method'] for r in row['checks']}!=set(METHODS): raise RuntimeError('Missing model probe')
            if any(len(r['predictions'])!=(10 if r['method']=='our_model' else 2) for r in row['checks']):
                raise RuntimeError('Incomplete prediction probe')
            evidence.append(str(file));break
        else: raise RuntimeError('No valid interface scene: '+source)
    proof=dict(**manifest(),checked_methods=METHODS,evidence_paths=evidence,
        accepted_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        scope='Ten-frame SHM plus two probes per other model on ProcTHOR and Habitat; not full-route performance')
    tmp=here/'MODEL_INTERFACE_ACCEPTED.tmp';tmp.write_text(json.dumps(proof,indent=2)+'\n')
    tmp.replace(here/'MODEL_INTERFACE_ACCEPTED.json')
    (root/'ACCEPTED.json').write_text(json.dumps(proof,indent=2)+'\n')


if __name__=='__main__': main()
