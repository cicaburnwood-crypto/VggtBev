"""Closed-loop native planar paths with the bounded cuboid camera executor.

Inference is synchronous at a stationary exposure. Paths are re-anchored only
at that exposure, so no asynchronous stale-pose connection or tracking drift is
possible. Only executed 2cm increments are tested for collision, never the
future plan. GT target acceptance is evaluator/upper-guide-only.
"""
from collections import deque
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
from exact_executor import Pose, path_steps
from robot_contract import SPEC, collider_for_scene
from recovery_views import RecoveryViews

SPEED_M_S=SPEC.speed_m_s
HORIZON_M=.5
STEP_M=SPEC.speed_m_s*SPEC.tick_s

MODE='wall_clock_bounded_bidirectional_camera_v6'


def advance_target(service, scene, state):
    """Same 20cm acceptance for all, with an evaluator-only wall barrier check.

    This tests only current-goal proximity, never future model waypoints.
    It does not select/repair a local path or steer the camera.
    """
    while not service.complete:
        here=world_point(state,scene.floor_y)
        goal=service.subgoals_world[service.current_index]
        if w.distance_xz(here,goal)>.2: return False
        if not scene.geometry.check_path([here,goal]).safe: return False
        service.current_index+=1
        service.switch_count+=1
    return True


def run_episode(scene, task, method, single_url, baseline_url, out, goal_images, warmup=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    infer=Inference(method,single_url,baseline_url,str(out))
    state=State(x=task['start_world'][0],z=task['start_world'][2],
                yaw_rad=math.radians(task['start_yaw_degrees']))
    collider=collider_for_scene(scene,w)
    initial_pose=Pose(state.x,state.z,state.yaw_rad,scene.floor_y)
    if collider.contact_at(initial_pose):
        raise RuntimeError('Invalid benchmark start: configured robot body already collides')
    service=w.OracleTargetService.from_task(task)
    warm_started=time.monotonic();infer.reset()
    first=w.encode_jpeg(scene.rgb(task['start_world'],task['start_yaw_degrees']))
    if warmup:
        infer.predict([first],service.query(task['start_world'],task['start_yaw_degrees']),goal_images[0])
    infer.reset();warm_s=time.monotonic()-warm_started
    service=w.OracleTargetService.from_task(task)
    frames=deque(maxlen=100)
    start=time.monotonic();gt=float(task['gt_shortest_path_m'])
    deadline=max(300.,20.*gt+60.);distance_limit=3.*gt+5.
    travelled=render_s=request_s=inference_s=planner_s=moving_s=0.
    translating_s=rotating_s=absolute_rotation_rad=0.
    replans=image_count=0;latencies=[];collision_world=None
    trace=[dict(t=0.,motion_t=0.,**asdict(state))]
    status,reason='timeout','common wall-clock deadline'
    last_render=-math.inf;last_evidence=-math.inf
    recovery=RecoveryViews()
    recovery_turns=0
    reverse_distance_m=0.

    def render(force=False):
        nonlocal render_s,image_count,last_render,last_evidence
        now=time.monotonic()
        camera_hz=9. if method=='nomad' else 10.
        if not force and now-last_render<1./camera_hz: return
        before=time.monotonic();exposure=before
        rgb=scene.rgb(world_point(state,scene.floor_y),math.degrees(state.yaw_rad))
        frames.append((exposure,w.encode_jpeg(rgb)))
        image_count+=1;last_render=exposure
        if image_count==1 or now-last_evidence>=1.:
            Image.fromarray(rgb).save(out/f'rgb_{image_count:06d}.jpg',quality=90)
            last_evidence=now
        render_s+=time.monotonic()-before

    with (out/'events.jsonl').open('a',buffering=1) as events:
        while time.monotonic()-start<deadline and travelled<distance_limit:
            if advance_target(service,scene,state):
                status,reason='success',None;break
            render(force=True)
            exposure_state=state;subgoal=service.current_index
            target=service.query(world_point(state,scene.floor_y),math.degrees(state.yaw_rad))
            request_started=time.monotonic()
            response=infer.predict(native_image_history(frames,method),target,goal_images[subgoal])
            cost=time.monotonic()-request_started
            replans+=1;request_s+=cost;latencies.append(cost)
            inference_s+=float(response.get('inference_seconds',0.) or 0.)
            planner_s+=float(response.get('planner_seconds',0.) or 0.)
            path=response.get('path_metric_m') or []
            event=dict(t=time.monotonic()-start,observation_t=frames[-1][0]-start,
                observation_pose=asdict(exposure_state),target_metric_m=target,
                upper_subgoal_index=subgoal,native_path=path,
                original_execution_preference=response.get('execution_preference'),
                ignored_velocity_output=response.get('control_velocity'),
                planning_failure=response.get('planning_failure'),
                metric_extent_m=response.get('metric_extent_m'),
                request_wall_seconds=cost,
                execution_horizon_m=response.get('execution_horizon_m',HORIZON_M),
                navigation_details=response.get('navigation_details'),
                path_headings_left_rad=response.get('path_headings_left_rad'))
            if time.monotonic()-start>=deadline: break
            pose=Pose(exposure_state.x,exposure_state.z,exposure_state.yaw_rad,scene.floor_y)
            horizon=float(response.get('execution_horizon_m',HORIZON_M))
            if not 0<horizon<=HORIZON_M+1e-9:
                raise RuntimeError('Invalid external execution horizon')
            if path:
                headings=response.get('path_headings_left_rad')
                if method in {'limo_tel','limo_aug'} and headings is None:
                    raise RuntimeError('LiMo native SE(2) heading missing; stale/wrong adapter')
                steps=list(path_steps(path,Pose(exposure_state.x,exposure_state.z,
                    exposure_state.yaw_rad,scene.floor_y),SPEC,horizon_m=horizon,
                    allow_reverse=True,native_headings_left_rad=headings))
            else:
                steps=[]
            if not steps:
                turn=recovery.next_turn(state.yaw_rad,target)
                if turn is None:
                    event['action']='recovery_exhausted'
                    events.write(json.dumps(event)+'\n')
                    status='planner_failure'
                    reason=str(response.get('planning_failure') or 'no executable native path after six fresh views')
                    break
                recovery_turns+=1
                event.update(action='fresh_view_after_failed_plan',recovery_attempt=recovery.attempts,
                             recovery_yaw_rad=turn)
                steps=list(path_steps([[0.,0.]],pose,SPEC,final_yaw_rad=turn,allow_reverse=True))
            else:
                recovery.reset()
                event['action']='execute_native_path'
            events.write(json.dumps(event)+'\n')
            terminal=False
            for step in steps:
                if time.monotonic()-start>=deadline: terminal=True;break
                if travelled+step.distance_m>distance_limit:
                    status,reason='distance_budget','common executed distance budget';terminal=True;break
                # GT sees only the CURRENT <=20ms move or rotation. Continuous
                # cuboid sweep prevents thin-wall tunnelling and corner clipping.
                contact=collider.sweep(step)
                fraction=1. if contact is None else contact
                dt=step.dt_s*fraction
                time.sleep(dt)
                moving_s+=dt
                if step.phase=='rotate': rotating_s+=dt
                else: translating_s+=dt
                absolute_rotation_rad+=abs(step.yaw_delta_rad)*fraction
                pose=step.at(fraction)
                state=State(pose.x,pose.z,pose.yaw_rad,
                    v=step.speed_m_s if fraction>0 else 0.,
                    w=step.yaw_rate_rad_s if fraction>0 else 0.)
                travelled+=step.distance_m*fraction
                if step.speed_m_s<0: reverse_distance_m+=step.distance_m*fraction
                trace.append(dict(t=time.monotonic()-start,motion_t=moving_s,**asdict(state),
                    phase=step.phase,native_vertex_index=step.native_vertex_index))
                if contact is not None:
                    collision_world=world_point(state,scene.floor_y)
                    status,reason='collision','contact during current oriented-cuboid sweep'
                    terminal=True;break
                if advance_target(service,scene,state):
                    status,reason='success',None;terminal=True;break
                render()
                # A newly reached common subgoal invalidates the old objective.
                # Re-observe and replan at the same physical pose, for every model.
                if service.current_index!=subgoal: break
            state=State(x=state.x,z=state.z,yaw_rad=state.yaw_rad,v=0.,w=0.)
            trace.append(dict(t=time.monotonic()-start,motion_t=moving_s,**asdict(state)))
            w.atomic_json(out/'current.json',dict(method=method,elapsed_s=time.monotonic()-start,
                travelled_m=travelled,replans=replans,subgoal=service.current_index,execution_mode=MODE))
            if terminal: break
        else:
            if travelled>=distance_limit: status,reason='distance_budget','common executed distance budget'
    ended=time.monotonic();terminal=w.distance_xz(world_point(state,scene.floor_y),task['goal_world'])
    before=time.monotonic()
    Image.fromarray(scene.rgb(world_point(state,scene.floor_y),math.degrees(state.yaw_rad))).save(out/'rgb_terminal.jpg',quality=95)
    evidence_s=time.monotonic()-before
    success=status=='success';wall=ended-start
    result=dict(method=method,success=success,status=status,failure_reason=reason,
        execution_mode=MODE,control_mode='common_exact_bounded_polyline_camera',
        robot=SPEC.metadata(),
        total_wall_seconds=wall,wall_seconds=wall,warmup_seconds=warm_s,
        inference_drain_seconds=0.,final_evidence_render_seconds=evidence_s,
        resource_wall_seconds=wall+warm_s+evidence_s,model_request_wall_seconds=request_s,
        model_inference_seconds=inference_s,planner_seconds=planner_s,render_seconds=render_s,
        nominal_motion_seconds=moving_s,nominal_speed_m_s=SPEED_M_S,
        translating_seconds=translating_s,rotating_seconds=rotating_s,
        absolute_rotation_rad=absolute_rotation_rad,
        execution_horizon_m=HORIZON_M,collision_step_max_m=STEP_M,
        reverse_distance_m=reverse_distance_m,recovery_observation_turns=recovery_turns,
        replans=replans,effective_replanning_hz=replans/max(wall,1e-9),
        inference_latency_mean_s=float(np.mean(latencies)) if latencies else None,
        inference_latency_p95_s=float(np.quantile(latencies,.95)) if latencies else None,
        rgb_frames=image_count,executed_path_length_m=travelled,gt_shortest_path_m=gt,
        terminal_distance_m=terminal,terminal_distance_type='euclidean; goal acceptance also requires collision-free <=20cm connector',
        raw_executed_path_ratio=travelled/gt if success else None,
        completion_adjusted_path_ratio=(travelled+terminal)/gt if success else None,
        spl=gt/max(gt,travelled) if success else 0.,
        subgoals_completed=service.completed_subgoals,subgoals_total=len(service.subgoals_world),
        collision_world=collision_world,future_gt_collision_veto=False,
        native_path_tracking_error_max_m=None,deadline_seconds=deadline,
        executed_distance_budget_m=distance_limit,
        camera_heading_rule='native SE2 selects forward/reverse tangent; XY-only minimum-turn gear sequence; bounded90deg/s',
        inference_motion_overlap=False,model_history_frames=HISTORY_SPEC.get(method,(1,10.))[0])
    np.savez_compressed(out/'trajectory.npz',**{k:[p[k] for p in trace] for k in ['t','motion_t','x','z','yaw_rad','v','w']})
    w.atomic_json(out/'result.json',result)
    return result
