"""Real wall-clock closed-loop evaluator. No future-path GT collision veto."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError

import numpy as np
import procthor_benchmark_worker as w
import planner_backends as pb
from motion import Limits, State, NativePathTracker, integrate_step

METHODS = ['our_model', 'limo_tel', 'limo_aug', 'omnivla', 'mbra_logonav', 'nomad', 'genie_samtp']
RATES = dict(our_model=10., limo_tel=10., limo_aug=10., omnivla=3., mbra_logonav=5., nomad=4., genie_samtp=5.)
HISTORY_SPEC = {'mbra_logonav': (6, 5.), 'nomad': (4, 9.)}
CONTROL_WATCHDOG_S = 2.


def native_image_history(frames, method):
    """Sample context by exposure time, independently of inference latency.

    MBRA declares input_fps5. NoMaD appends camera callbacks: the official
    frontcamera is9Hz although its inference loop is4Hz. Missing startup
    history is padded with the initial exposure, never with future frames.
    """
    if not frames:
        return []
    count, hz = HISTORY_SPEC.get(method, (1, 10.))
    values = list(frames)
    latest = values[-1][0]
    result = []
    for index in range(count):
        desired = latest-(count-1-index)/hz
        eligible = [item for item in values if item[0] <= desired+1e-7]
        result.append((eligible[-1] if eligible else values[0])[1])
    return result


def world_point(state, floor):
    return [float(state.x), float(floor), float(state.z)]


def anchor_path(path, state):
    """Common ideal wheel-odometry transform; no simulator pose query."""
    c, s = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
    return [[state.x+c*float(p[0])+s*float(p[1]),
             state.z-s*float(p[0])+c*float(p[1])] for p in path]


def calibration():
    f = w.CAMERA_WIDTH/2 / math.tan(math.radians(w.HORIZONTAL_FOV_DEGREES)/2)
    return dict(intrinsics=[[f,0,(w.CAMERA_WIDTH-1)/2],[0,f,(w.CAMERA_HEIGHT-1)/2],[0,0,1]],
                T_ground_camera=[[1,0,0,0],[0,0,1,0],[0,-1,0,w.CAMERA_HEIGHT_M],[0,0,0,1]], ground_z_m=0.)


class Inference:
    def __init__(self, method, single_url, baseline_url, segment):
        self.method, self.single_url, self.baseline_url, self.segment = method, single_url, baseline_url, segment
        self.backend = pb.BITStarBackend() if method == 'our_model' else None
        self.frame = 0

    def reset(self):
        url = self.single_url if self.method == 'our_model' else self.baseline_url
        w.post_json(url, '/reset', dict(segment_id=self.segment), 60.)
        self.frame = 0
        if self.backend is not None: self.backend.reset()

    def predict(self, images, point_goal, goal_image):
        before = time.monotonic()
        if self.method == 'our_model':
            self.frame += 1
            response = w.post_json(self.single_url, '/predict', dict(segment_id=self.segment,
                frame_seq=self.frame, image_png_base64=images[-1], threshold=.5,
                physical_camera_height_m=w.CAMERA_HEIGHT_M), 300.)
            inference_s = time.monotonic()-before
            from shm_contract import planner_payload, InvalidMetricExtent
            try:
                fused_payload = planner_payload(response)
            except InvalidMetricExtent as exc:
                # Reject this prediction without changing its scale or using GT.
                # Existing six-view no-path recovery / scored planner failure
                # handles it, including warmup. Schema, transport, CUDA, fusion
                # and other runtime errors remain fatal, never silently retried.
                evidence = Path(self.segment)/'prediction_rejections.jsonl'
                evidence.parent.mkdir(parents=True, exist_ok=True)
                diagnostics = {k: response.get(k) for k in (
                    'frame_seq', 'shm_metric_extent_m', 'history_frame_count',
                    'history_frame_seqs', 'model_scale_token_bev_per_vggt',
                    'camera_metric_scale_m_per_vggt', 'bev_metric_scale_m_per_bev',
                    'physical_camera_height_m', 'vggt_camera_height',
                    'ground_inlier_fraction', 'ground_fallback_used')}
                with evidence.open('a') as stream:
                    stream.write(json.dumps(dict(error=str(exc), **diagnostics))+'\n')
                return dict(path_metric_m=[], execution_preference='native_path',
                    planning_failure=dict(code='invalid_predicted_metric_extent', message=str(exc),
                                          diagnostics=diagnostics),
                    metric_extent_m=exc.extent, execution_horizon_m=.5,
                    inference_seconds=float(response.get('inference_seconds', inference_s)),
                    planner_seconds=0., request_wall_seconds=time.monotonic()-before,
                    **{k:response[k] for k in ('planner_bev_source','history_frame_count',
                        'history_frame_seqs','shm_historical_owner_pixels','shm_total_supported_pixels')})
            sem, prob, conf, extent = w.decode_bev_response(fused_payload)
            shm_meta={k:response[k] for k in ('planner_bev_source','history_frame_count',
                'history_frame_seqs','shm_historical_owner_pixels','shm_total_supported_pixels')}
            history_file=Path(self.segment)/'shm_history.jsonl'
            history_file.parent.mkdir(parents=True,exist_ok=True)
            with history_file.open('a') as stream:
                stream.write(json.dumps(dict(frame_seq=self.frame,metric_extent_m=extent,**shm_meta))+'\n')
            from predicted_navigation import (plan_reliable_bitstar,
                risk_limited_execution_horizon, NavigationConfig)
            config=NavigationConfig()
            started = time.monotonic()
            try:
                plan,problem=plan_reliable_bitstar(self.backend,semantic=sem,
                    occupancy_probability=prob,navigation_confidence=conf,extent_m=extent,
                    target_metric_m=point_goal,inflation_radius_m=w.TOTAL_INFLATION_M,config=config)
                path = plan.path_metric_m
                horizon,unknown_arc=risk_limited_execution_horizon(path,sem,extent,config)
                details=plan.backend_details
                error = None
            except pb.PlannerFailure as exc:
                path = []; error = dict(code=exc.code, message=str(exc))
                horizon,unknown_arc,details=.5,0.,{}
            response = dict(path_metric_m=path, execution_preference='native_path',
                **shm_meta,
                inference_seconds=float(response.get('inference_seconds', inference_s)),
                planner_seconds=time.monotonic()-started, planning_failure=error,
                metric_extent_m=extent,execution_horizon_m=horizon,
                unknown_execution_arc_m=unknown_arc,navigation_details=details)
        else:
            payload = dict(method=self.method, target_metric_m=point_goal,
                images_base64=list(images), gt_trajectory_history_metric_m=[])
            if self.method == 'nomad': payload['goal_image_base64'] = goal_image
            if self.method == 'genie_samtp': payload['camera_calibration'] = calibration()
            try:
                response = w.post_json(self.baseline_url, '/predict', payload, 900.)
            except HTTPError as exc:
                # A native planner returning no path is a model outcome, not a
                # crashed scene/GPU. Do not swallow OOM, transport, asset or
                # unclassified backend failures as ordinary model failures.
                body = exc.read().decode('utf-8', errors='replace')
                try:
                    error = json.loads(body)
                except ValueError:
                    raise RuntimeError('baseline runtime HTTP failure: '+body[:1000]) from exc
                message = str(error.get('error', ''))
                if self.method == 'genie_samtp' and message.startswith('GENIE native planner failed:'):
                    response = dict(path_metric_m=[], execution_preference='native_path_1m_receding_horizon',
                        planning_failure=dict(code='native_no_path', message=message))
                else:
                    raise RuntimeError('baseline runtime failure: '+body[:1000]) from exc
        response['request_wall_seconds'] = time.monotonic()-before
        return response


def run_episode(scene, task, method, single_url, baseline_url, out, goal_images, warmup=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    infer = Inference(method, single_url, baseline_url, str(out))
    initial = State(x=task['start_world'][0], z=task['start_world'][2],
                    yaw_rad=math.radians(task['start_yaw_degrees']))
    state = initial
    limits = Limits()
    service = w.OracleTargetService.from_task(task)
    warm_started = time.monotonic()
    infer.reset()
    # Exclude cold weight load, but record it. Discard warmup prediction/state.
    initial_image = w.encode_jpeg(scene.rgb(task['start_world'], task['start_yaw_degrees']))
    if warmup:
        infer.predict([initial_image], service.query(task['start_world'], task['start_yaw_degrees']), goal_images[0])
    infer.reset()
    warm_s = time.monotonic()-warm_started
    service = w.OracleTargetService.from_task(task)
    tracker = NativePathTracker()
    frames = deque(maxlen=100)
    pool = ThreadPoolExecutor(max_workers=1)
    pending = None
    path_active = False
    velocity = None
    last_accept = -math.inf
    replans = image_count = 0
    inference_s = request_s = planner_s = render_s = travelled = 0.
    start = last_physics = time.monotonic()
    next_render = next_request = next_status = start
    image_state = state
    image_time = start
    last_requested_image_time = -math.inf
    deadline = max(120., 8.*float(task['gt_shortest_path_m'])+30.)
    trace = [dict(t=0., **asdict(state))]
    decisions = []
    status, reason = 'timeout', 'common wall-clock episode deadline'
    control_mode = None
    collision = None
    path_tracking_errors = []

    def advance_until(target_time):
        """Integrate elapsed time with ONLY commands already delivered then.

        This must run after a blocking render and before accepting a completed
        inference response. Otherwise the new command would be retroactively
        applied to time spent rendering or waiting for its own inference.
        """
        nonlocal state, last_physics, travelled, collision, path_active, status, reason
        while last_physics < target_time-1e-8:
            dt = min(.05, target_time-last_physics)
            valid = last_physics-last_accept <= CONTROL_WATCHDOG_S
            if velocity is not None and valid:
                command_v, command_w = velocity
            elif path_active and valid:
                cmd = tracker.command(state)
                command_v, command_w = cmd.v, cmd.w
                if math.isfinite(cmd.cross_track_m):
                    path_tracking_errors.append(cmd.cross_track_m)
                if cmd.done: path_active = False
            else:
                command_v = command_w = 0.
            moved = integrate_step(state, command_v, command_w, dt, limits)
            collision = scene.geometry.check_path([
                world_point(state, scene.floor_y), world_point(moved.state, scene.floor_y)])
            if not collision.safe:
                status, reason = 'collision', 'contact during executed differential-drive step'
                return False
            state = moved.state
            travelled += moved.distance_m
            last_physics += dt
            trace.append(dict(t=last_physics-start, **asdict(state)))
            service.advance_if_reached(world_point(state, scene.floor_y))
            if service.complete:
                status, reason = 'success', None
                return False
        return True

    with (out/'events.jsonl').open('a', buffering=1) as events:
        try:
            while True:
                now = time.monotonic()
                if now-start >= deadline: break
                # Advance the actual robot through all elapsed time, even during
                # inference. Every check is only a currently executed <=50ms step.
                if not advance_until(now): break
                if now >= next_render:
                    before = time.monotonic()
                    rgb = scene.rgb(world_point(state, scene.floor_y), math.degrees(state.yaw_rad))
                    encoded = w.encode_jpeg(rgb)
                    render_s += time.monotonic()-before
                    image_state = state; image_time = last_physics
                    frames.append((image_time, encoded)); image_count += 1
                    next_render = before+(1./9. if method == 'nomad' else .1)
                    # Persist evidence at 1Hz, without storing gigabytes of video.
                    if image_count == 1 or image_count % 10 == 0:
                        from PIL import Image
                        Image.fromarray(rgb).save(out/f'rgb_{image_count:06d}.jpg', quality=90)
                if pending is not None and pending.done():
                    # Rendering may have blocked while the robot kept moving.
                    # Catch up under the OLD command before installing this one.
                    if not advance_until(time.monotonic()): break
                    response = pending.result(); pending = None
                    replans += 1
                    inference_s += float(response.get('inference_seconds', 0.))
                    request_s += float(response.get('request_wall_seconds', 0.))
                    planner_s += float(response.get('planner_seconds', 0.))
                    pref = response.get('execution_preference', 'native_path')
                    native_path = response.get('path_metric_m') or []
                    control = response.get('control_velocity') or {}
                    event = dict(t=time.monotonic()-start, observation_t=request_time-start,
                        target_metric_m=request_target, execution_preference=pref,
                        native_path=native_path, control=control,
                        planning_failure=response.get('planning_failure'),
                        metric_extent_m=response.get('metric_extent_m'),
                        inference_seconds=response.get('inference_seconds'),
                        planner_seconds=response.get('planner_seconds'))
                    events.write(json.dumps(event)+'\n'); decisions.append(event)
                    # JSON/event I/O also has wall cost. The new command starts
                    # at THIS physical timestamp, never earlier in that cost.
                    if not advance_until(time.monotonic()): break
                    # A result for an already completed upper waypoint is stale,
                    # equally for all systems. Never overwrite the current goal.
                    if request_subgoal == service.current_index:
                        if pref == 'control_velocity':
                            velocity = (float(control.get('forward_m_s',0.)),
                                        -float(control.get('angular_left_rad_s',0.)))
                            path_active = False; control_mode = 'native_velocity'
                        elif len(native_path) >= 2:
                            if pref == 'native_path_1m_receding_horizon':
                                native_path, _ = w.executable_path_prefix(native_path, 1.)
                            anchored = anchor_path(native_path, request_state)
                            tracker.set_path(anchored)
                            # The robot can move while inference processes the
                            # older exposure. Initialize progress from common
                            # wheel odometry within a physically reachable prefix,
                            # not a global-nearest jump across self-intersections.
                            tracker.initialize_progress(state, max_initial_arc_m=
                                limits.max_speed_m_s*max(0., last_physics-request_time)+.5)
                            path_active = True; velocity = None
                            control_mode = 'shared_differential_drive_path_tracker'
                        else:
                            velocity = None; path_active = False
                            status, reason = 'planner_failure', str(response.get('planning_failure') or 'native output has no executable path/control')
                            break
                        last_accept = last_physics
                    next_request = max(next_request, request_started+1./RATES[method])
                if pending is None and frames and time.monotonic() >= next_request and image_time > last_requested_image_time:
                    # PointGoal and RGB refer to the SAME exposure pose.
                    request_state, request_time = image_state, image_time
                    request_subgoal = service.current_index
                    request_target = service.query(world_point(request_state, scene.floor_y), math.degrees(request_state.yaw_rad))
                    request_started = time.monotonic()
                    last_requested_image_time = image_time
                    pending = pool.submit(infer.predict, native_image_history(frames, method), request_target, goal_images[request_subgoal])
                if now >= next_status:
                    w.atomic_json(out/'current.json', dict(method=method, elapsed_s=now-start,
                        travelled_m=travelled, replans=replans, subgoal=service.current_index))
                    next_status = now+1.
                time.sleep(.005)
        finally:
            ended = time.monotonic()
            # Drain in-flight HTTP before resetting model for another episode;
            # include drain as resource overhead, NOT successful task time.
            pool.shutdown(wait=True)
            drain_s = time.monotonic()-ended
    terminal = w.distance_xz(world_point(state, scene.floor_y), task['goal_world'])
    # Final evidence is rendered from the actual terminal plant pose, not the
    # last inference exposure (which can precede it by several control ticks).
    final_render_started = time.monotonic()
    terminal_rgb = scene.rgb(world_point(state, scene.floor_y), math.degrees(state.yaw_rad))
    from PIL import Image
    Image.fromarray(terminal_rgb).save(out/'rgb_terminal.jpg', quality=95)
    final_render_s = time.monotonic()-final_render_started
    gt = float(task['gt_shortest_path_m'])
    result = dict(method=method, success=status=='success', status=status, failure_reason=reason,
        total_wall_seconds=ended-start, wall_seconds=ended-start,
        warmup_seconds=warm_s, inference_drain_seconds=drain_s,
        final_evidence_render_seconds=final_render_s,
        resource_wall_seconds=warm_s+(ended-start)+drain_s+final_render_s,
        model_inference_seconds=inference_s, model_request_wall_seconds=request_s,
        planner_seconds=planner_s, render_seconds=render_s,
        replans=replans, effective_replanning_hz=replans/max(ended-start,1e-9),
        rgb_frames=image_count, executed_path_length_m=travelled,
        gt_shortest_path_m=gt, terminal_distance_m=terminal,
        raw_executed_path_ratio=travelled/gt if status=='success' else None,
        completion_adjusted_path_ratio=(travelled+terminal)/gt if status=='success' else None,
        spl=(gt/max(gt,travelled)) if status=='success' else 0.,
        subgoals_completed=service.completed_subgoals, subgoals_total=len(service.subgoals_world),
        collision_world=None if collision is None or collision.safe else collision.collision_world,
        robot_limits=asdict(limits), control_mode=control_mode,
        shared_control_watchdog_seconds=CONTROL_WATCHDOG_S,
        model_history_frames=HISTORY_SPEC.get(method, (1, 10.))[0],
        model_history_camera_hz=HISTORY_SPEC.get(method, (1, 10.))[1],
        native_path_tracking_error_mean_m=float(np.mean(path_tracking_errors)) if path_tracking_errors else None,
        native_path_tracking_error_max_m=max(path_tracking_errors, default=None),
        execution_mode='wall_clock_differential_drive_rgb_online_v1',
        future_gt_collision_veto=False, deadline_seconds=deadline)
    np.savez_compressed(out/'trajectory.npz', t=[p['t'] for p in trace],
        x=[p['x'] for p in trace], z=[p['z'] for p in trace],
        yaw_rad=[p['yaw_rad'] for p in trace], v=[p['v'] for p in trace], w=[p['w'] for p in trace])
    w.atomic_json(out/'result.json', result)
    return result
