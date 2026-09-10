"""Native controller cmd_vel, with real asynchronous observation/inference.

The simulator plant integrates bounded velocities; it does not steer toward a
path or a goal. Rendering/inference latency is wall time under the OLD command.
No new prediction is applied retroactively to its own inference interval.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
import procthor_benchmark_worker as w
from episode import Inference, native_image_history, HISTORY_SPEC, world_point
from motion import State
from exact_executor import Pose
from robot_contract import SPEC, collider_for_scene
from native_motion import TwistStep, sweep_twist
from execution_policy import MODE, NativeCommand, policy_for
from virtual_episode import advance_target


def run_episode(scene,task,method,single_url,baseline_url,out,goal_images,warmup=True):
    policy=policy_for(method)
    if policy['kind']!='native_velocity': raise ValueError('Native executor method required')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    infer=Inference(method,single_url,baseline_url,str(out))
    state=State(task['start_world'][0],task['start_world'][2],math.radians(task['start_yaw_degrees']))
    collider=collider_for_scene(scene,w)
    if collider.contact_at(Pose(state.x,state.z,state.yaw_rad,scene.floor_y)):
        raise RuntimeError('Invalid benchmark start: robot body collides')
    service=w.OracleTargetService.from_task(task)
    warm_started=time.monotonic(); infer.reset()
    first=w.encode_jpeg(scene.rgb(task['start_world'],task['start_yaw_degrees']))
    if warmup: infer.predict([first],service.query(task['start_world'],task['start_yaw_degrees']),goal_images[0])
    infer.reset(); warm_s=time.monotonic()-warm_started
    command=NativeCommand(method); frames=deque(maxlen=100)
    start=physical_t=time.monotonic(); gt=float(task['gt_shortest_path_m'])
    deadline=max(300.,20.*gt+60.); distance_limit=3.*gt+5.
    travelled=render_s=request_s=inference_s=planner_s=motion_s=0.
    translating_s=rotating_s=idle_s=rotation_rad=0.
    replans=image_count=stale_results=0; latencies=[]
    next_render=next_request=next_status=start
    image_time=-math.inf; last_requested=-math.inf; image_state=state; image_subgoal=-1
    pending=None; request_subgoal=-1; pool=ThreadPoolExecutor(max_workers=1)
    status,reason='timeout','common wall-clock deadline'
    collision_world=None
    trace=[dict(t=0.,**asdict(state))]

    def advance_until(until):
        nonlocal physical_t,state,travelled,motion_s,translating_s,rotating_s,idle_s
        nonlocal rotation_rad,status,reason,collision_world,next_request,next_render
        until=min(until,start+deadline)
        while physical_t<until-1e-10:
            if advance_target(service,scene,state):
                status,reason='success',None;command.reset();return False
            old_subgoal=service.current_index
            v,yaw_rate=command.at(physical_t)
            dt=min(SPEC.tick_s,until-physical_t,command.next_boundary()-physical_t)
            if dt<=1e-10:
                command.at(physical_t+1e-9);continue
            step=TwistStep(Pose(state.x,state.z,state.yaw_rad,scene.floor_y),v,yaw_rate,dt)
            contact=sweep_twist(collider,step)
            fraction=1. if contact is None else contact
            duration=dt*fraction; pose=step.at(fraction)
            travelled+=step.distance_m*fraction; physical_t+=duration
            motion_s+=duration
            if abs(v)>1e-10: translating_s+=duration
            elif abs(yaw_rate)>1e-10: rotating_s+=duration
            else: idle_s+=duration
            rotation_rad+=abs(yaw_rate)*duration
            state=State(pose.x,pose.z,pose.yaw_rad,v,yaw_rate)
            trace.append(dict(t=physical_t-start,**asdict(state)))
            if contact is not None:
                status,reason='collision','contact during current native-command cuboid arc'
                collision_world=world_point(state,scene.floor_y);command.reset();return False
            if travelled>=distance_limit:
                status,reason='distance_budget','common executed distance budget';command.reset();return False
            if advance_target(service,scene,state):
                status,reason='success',None;command.reset();return False
            if service.current_index!=old_subgoal:
                command.reset(); next_request=physical_t; next_render=physical_t
        return True

    with (out/'events.jsonl').open('a',buffering=1) as events:
        try:
            while time.monotonic()-start<deadline:
                if not advance_until(time.monotonic()): break
                if time.monotonic()>=next_render:
                    before=time.monotonic(); image_state=state; image_time=physical_t
                    image_subgoal=service.current_index
                    rgb=scene.rgb(world_point(state,scene.floor_y),math.degrees(state.yaw_rad))
                    frames.append((image_time,w.encode_jpeg(rgb))); image_count+=1
                    render_s+=time.monotonic()-before
                    next_render=before+(1/9 if method=='nomad' else .1)
                    if image_count==1 or image_count%10==0:
                        Image.fromarray(rgb).save(out/f'rgb_{image_count:06d}.jpg',quality=90)
                if pending is not None and pending.done():
                    # Integrate render/wait under the old command BEFORE receive.
                    if not advance_until(time.monotonic()): break
                    response=pending.result(); pending=None
                    cost=time.monotonic()-request_started
                    replans+=1; request_s+=cost;latencies.append(cost)
                    inference_s+=float(response.get('inference_seconds',0.) or 0.)
                    planner_s+=float(response.get('planner_seconds',0.) or 0.)
                    stale=request_subgoal!=service.current_index
                    events.write(json.dumps(dict(t=time.monotonic()-start,
                        observation_t=request_time-start,observation_pose=asdict(request_state),
                        target_metric_m=request_target,upper_subgoal_index=request_subgoal,
                        native_path=response.get('path_metric_m'),
                        control_velocity=response.get('control_velocity'),
                        controller=policy['controller'],stale_subgoal_discarded=stale,
                        planning_failure=response.get('planning_failure'),request_wall_seconds=cost))+'\n')
                    if not advance_until(time.monotonic()): break
                    if request_subgoal==service.current_index:
                        # No path availability/quality test here: the native
                        # controller owns movement, even if its plotted path is empty.
                        command.accept(response,physical_t)
                    else: stale_results+=1
                    next_request=max(next_request,request_started+1/policy['inference_hz'])
                if (pending is None and frames and time.monotonic()>=next_request and
                    image_time>last_requested and image_subgoal==service.current_index):
                    # Never pair a new subgoal with a pre-switch camera image.
                    request_state=image_state; request_time=image_time
                    request_subgoal=service.current_index
                    request_target=service.query(world_point(request_state,scene.floor_y),math.degrees(request_state.yaw_rad))
                    request_started=time.monotonic();last_requested=image_time
                    pending=pool.submit(infer.predict,native_image_history(frames,method),request_target,goal_images[request_subgoal])
                if time.monotonic()>=next_status:
                    w.atomic_json(out/'current.json',dict(method=method,elapsed_s=time.monotonic()-start,
                        travelled_m=travelled,replans=replans,subgoal=service.current_index,
                        execution_mode=MODE,control_mode='native_velocity'))
                    next_status=time.monotonic()+1.
                time.sleep(.005)
            # Any final elapsed interval belongs to the old command as well.
            if status=='timeout': advance_until(min(time.monotonic(),start+deadline))
        finally:
            ended=time.monotonic();command.reset()
            pool.shutdown(wait=True)  # avoid resetting a model under an active request
            drain_s=time.monotonic()-ended
    terminal=w.distance_xz(world_point(state,scene.floor_y),task['goal_world'])
    before=time.monotonic()
    Image.fromarray(scene.rgb(world_point(state,scene.floor_y),math.degrees(state.yaw_rad))).save(out/'rgb_terminal.jpg',quality=95)
    evidence_s=time.monotonic()-before; wall=ended-start; success=status=='success'
    robot_meta=SPEC.metadata()
    robot_meta['execution']='native_cmd_vel_analytic_arc'
    result=dict(method=method,success=success,status=status,failure_reason=reason,
        execution_mode=MODE,control_mode='native_velocity',executor_policy=policy,robot=robot_meta,
        total_wall_seconds=wall,wall_seconds=wall,warmup_seconds=warm_s,inference_drain_seconds=drain_s,
        final_evidence_render_seconds=evidence_s,resource_wall_seconds=wall+warm_s+drain_s+evidence_s,
        model_request_wall_seconds=request_s,model_inference_seconds=inference_s,planner_seconds=planner_s,
        render_seconds=render_s,nominal_motion_seconds=translating_s+rotating_s,
        translating_seconds=translating_s,rotating_seconds=rotating_s,idle_plant_seconds=idle_s,
        absolute_rotation_rad=rotation_rad,execution_horizon_m=None,nominal_speed_m_s=SPEC.speed_m_s,
        replans=replans,effective_replanning_hz=replans/max(wall,1e-9),stale_subgoal_predictions=stale_results,
        inference_latency_mean_s=float(np.mean(latencies)) if latencies else None,
        inference_latency_p95_s=float(np.quantile(latencies,.95)) if latencies else None,
        rgb_frames=image_count,executed_path_length_m=travelled,gt_shortest_path_m=gt,
        terminal_distance_m=terminal,raw_executed_path_ratio=travelled/gt if success else None,
        completion_adjusted_path_ratio=(travelled+terminal)/gt if success else None,
        spl=gt/max(gt,travelled) if success else 0.,subgoals_completed=service.completed_subgoals,
        subgoals_total=len(service.subgoals_world),collision_world=collision_world,
        future_gt_collision_veto=False,native_path_tracking_error_max_m=None,
        deadline_seconds=deadline,executed_distance_budget_m=distance_limit,
        inference_motion_overlap=True,native_waypoint_timeout_s=policy['waypoint_timeout_s'],
        model_history_frames=HISTORY_SPEC.get(method,(1,10.))[0],
        camera_heading_rule='native angular command; bounded analytic cmd_vel plant')
    np.savez_compressed(out/'trajectory.npz',**{k:[p[k] for p in trace] for k in ['t','x','z','yaw_rad','v','w']})
    w.atomic_json(out/'result.json',result)
    return result
