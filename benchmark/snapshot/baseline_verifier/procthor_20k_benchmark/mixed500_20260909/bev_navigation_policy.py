"""Opt-in RGB/PointGoal navigation improvements; no model or GT map changes.

The policy receives predicted rasters and the common local metric PointGoal.
Simulator geometry is accessed only by the existing evaluator AFTER a command
has been chosen. Original full-native-path evaluation remains untouched.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage

import planner_backends as pb


@dataclass(frozen=True)
class NavigationConfig:
    execution_horizon_m: float = 0.50
    unknown_execution_budget_m: float = 0.20
    unknown_penalty: float = 2.0
    unknown_depth_penalty: float = 2.0
    unknown_depth_scale_m: float = 1.0
    unknown_depth_cap: float = 2.0
    target_alignment_degrees: float = 35.0
    maximum_turn_degrees: float = 60.0
    initial_budget_s: float = 0.55
    extended_budget_s: float = 1.65
    minimum_progress_m: float = 0.06
    maximum_recovery_views: int = 6
    maximum_decisions: int = 300

    def __post_init__(self):
        for name in ('execution_horizon_m', 'unknown_execution_budget_m',
                     'unknown_depth_scale_m', 'initial_budget_s', 'extended_budget_s'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('unknown_penalty', 'unknown_depth_penalty', 'unknown_depth_cap'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(f'{name} must be nonnegative and finite')


def uncertainty_fields(semantic, occupancy, confidence, cell_size_m, config):
    """Finite-risk unknown, not observed free; never mutate model rasters."""
    semantic=np.asarray(semantic,dtype=np.uint8)
    unknown=(semantic!=pb.FREE_VALUE)&(semantic!=pb.OCCUPIED_VALUE)
    planner_semantic=semantic.copy()
    planner_semantic[unknown]=pb.FREE_VALUE  # traversability view ONLY
    p=np.clip(np.asarray(occupancy,dtype=np.float32),0.,1.).copy()
    c=np.clip(np.asarray(confidence,dtype=np.float32),0.,1.).copy()
    # An unsupported decoder value is not evidence of observed free space.
    p[unknown]=np.maximum(p[unknown],.5)
    c[unknown]=0.
    if unknown.all():
        depth=np.full(unknown.shape,config.unknown_depth_scale_m*config.unknown_depth_cap)
    else:
        depth=ndimage.distance_transform_edt(unknown)*cell_size_m
    surcharge=unknown*(config.unknown_penalty+config.unknown_depth_penalty*np.minimum(
        depth/config.unknown_depth_scale_m,config.unknown_depth_cap))
    return planner_semantic,p,c,unknown,surcharge


def risk_limited_execution_horizon(path, semantic, extent_m, config):
    """Limit UNKNOWN arc per observation, while preserving native vertices.

    Sample each edge at <= half a cell for the risk integral. Returned value
    is an arc horizon; the existing executor only interpolates that boundary.
    """
    points=[[0.,0.]]+[list(map(float,p)) for p in path]
    cell=extent_m/len(semantic)
    total=unknown_arc=0.
    for a,b in zip(points,points[1:]):
        a=np.asarray(a);b=np.asarray(b);length=float(np.linalg.norm(b-a))
        if length<=1e-12:continue
        n=max(1,int(math.ceil(length/(cell/2))))
        edge_step=length/n
        for i in range(n):
            point=a+(b-a)*((i+.5)/n)
            row=int(math.floor((extent_m/2-point[1])/cell))
            col=int(math.floor((point[0]+extent_m/2)/cell))
            unknown=not (0<=row<len(semantic) and 0<=col<len(semantic)) or \
                semantic[row,col] not in (pb.FREE_VALUE,pb.OCCUPIED_VALUE)
            remaining=config.execution_horizon_m-total
            if unknown:remaining=min(remaining,config.unknown_execution_budget_m-unknown_arc)
            step=max(0.,min(edge_step,remaining))
            total+=step
            if unknown:unknown_arc+=step
            if step<edge_step-1e-12 or total>=config.execution_horizon_m-1e-12 or \
               unknown_arc>=config.unknown_execution_budget_m-1e-12:
                return max(total,1e-9),unknown_arc
    return max(total,1e-9),unknown_arc


def alignment_turn(target: Sequence[float], config: NavigationConfig) -> float:
    bearing = math.degrees(math.atan2(float(target[0]), float(target[1])))
    if abs(bearing) <= config.target_alignment_degrees:
        return 0.0
    return float(np.clip(bearing, -config.maximum_turn_degrees, config.maximum_turn_degrees))


def reachable_component(problem: pb.GridProblem) -> np.ndarray:
    # 4-connectivity is equivalent to connectivity of an 8-neighbour grid that
    # disallows diagonal corner cutting, as the planner does.
    seed = np.zeros_like(problem.blocked, dtype=bool)
    seed[problem.start] = True
    return ndimage.binary_propagation(seed, mask=~problem.blocked,
                                     structure=ndimage.generate_binary_structure(2, 1))


def plan_reliable_bitstar(backend, *, semantic, occupancy_probability,
                         navigation_confidence, extent_m, target_metric_m,
                         inflation_radius_m, config=NavigationConfig()):
    """Same BIT*, with connectivity-aware targets and bounded budget escalation.

    Unknown is traversable with a finite uncertainty penalty. Known occupied
    and its inflation remain hard barriers. Never use A*/GT fallback.
    """
    target = np.asarray(target_metric_m, dtype=np.float64)
    if target.shape != (2,) or not np.isfinite(target).all() or np.linalg.norm(target) <= 1e-9:
        raise pb.PlannerFailure('invalid_target', 'invalid local metric PointGoal')
    horizon = extent_m * .45
    desired = target * min(1., horizon / np.linalg.norm(target))
    planner_semantic,p,c,unknown,surcharge=uncertainty_fields(
        semantic,occupancy_probability,navigation_confidence,extent_m/len(semantic),config)
    common = dict(semantic=planner_semantic, occupancy_probability=p,
                  navigation_confidence=c, extent_m=extent_m,
                  inflation_radius_m=inflation_radius_m)
    repaired = False
    repair_cause = None
    try:
        problem = pb.build_problem(**common, target_metric_m=desired)
    except pb.PlannerFailure as error:
        if error.code != 'target_blocked':
            raise
        repaired = True
        repair_cause = 'target_blocked'
    if not repaired and not reachable_component(problem)[problem.goal]:
        repaired = True
        repair_cause = 'target_free_but_disconnected'
    if repaired:
        desired, selection = pb._select_engineered_subgoal(
            **common, requested_target_metric_m=target)
        # Do not march along a boundary if there is no meaningful progress
        # toward the requested goal. Acquire a different view instead.
        improvement = float(np.linalg.norm(target) - np.linalg.norm(target - desired))
        if improvement < min(config.minimum_progress_m, .20 * np.linalg.norm(target)):
            raise pb.PlannerFailure('no_progress', 'reachable subgoal offers no target progress',
                                    improvement_m=improvement, repair_cause=repair_cause)
        problem = pb.build_problem(**common, target_metric_m=desired)
    else:
        selection = {}
    if not reachable_component(problem)[problem.goal]:
        raise pb.PlannerFailure('disconnected_prediction', 'selected target is disconnected')
    problem.weight[~problem.blocked]+=surcharge[~problem.blocked]
    failures = []
    for budget in (config.initial_budget_s, config.extended_budget_s):
        try:
            result = backend.plan(problem, budget_s=budget)
            break
        except pb.PlannerFailure as error:
            failures.append({'code': error.code, 'message': str(error), 'budget_s': budget})
            if error.code != 'no_path':
                raise
    else:
        raise pb.PlannerFailure('search_budget_exhausted',
                                'connected prediction, BIT* exhausted both budgets',
                                attempts=failures)

    # The old executor implicitly inserts the metric origin before the first
    # free anchor. Audit that short connection against predicted OCCUPIED,
    # including inflation. Unknown at the camera-origin apex is not relabelled.
    anchor = np.asarray(result.path_metric_m[0])
    occupied = np.asarray(semantic) == pb.OCCUPIED_VALUE
    occupied = ndimage.binary_dilation(occupied, structure=pb._disk(
        int(math.ceil(inflation_radius_m / problem.cell_size_m))))
    n = max(2, int(math.ceil(np.linalg.norm(anchor) / (problem.cell_size_m / 2))) + 1)
    for point in np.linspace(np.zeros(2), anchor, n):
        row = int(math.floor((extent_m / 2 - point[1]) / problem.cell_size_m))
        col = int(math.floor((point[0] + extent_m / 2) / problem.cell_size_m))
        if not (0 <= row < len(occupied) and 0 <= col < len(occupied)) or occupied[row, col]:
            raise pb.PlannerFailure('origin_connector_blocked',
                                    'origin-to-anchor intersects predicted inflated occupancy')
    result.backend_details.update(
        policy='rgb_pointgoal_unknown_risk_bitstar_v2',
        requested_target_metric_m=target.tolist(), planned_subgoal_metric_m=desired.tolist(),
        planned_goal_repaired=repaired, repair_cause=repair_cause,
        predicted_connectivity_checked=True, search_budget_s=budget,
        extended_budget_used=bool(failures), cross_backend_fallback=False,
        unknown_traversable=True,unknown_fraction=float(np.mean(unknown)),
        unknown_cost='unknown_penalty*U + unknown_depth_penalty*U*min(d/unknown_depth_scale_m,unknown_depth_cap)',
        uncertainty_cost_config={name:getattr(config,name) for name in (
            'unknown_penalty','unknown_depth_penalty','unknown_depth_scale_m','unknown_depth_cap')},
        unknown_occupancy_prior_floor=.5,unknown_confidence=0.,
        selection=selection,
    )
    return result, problem


def run_optimized_bev_episode(scene, task, model_url, attempt_id, *,
                              config=NavigationConfig(), diagnostics_dir=None):
    import procthor_benchmark_worker as w

    target_service = w.OracleTargetService.from_task(task)
    motion = w.Motion(list(task['start_world']), float(task['start_yaw_degrees']))
    segment = f'{attempt_id}-optimized-v2-unknown-risk'
    w.post_json(model_url, '/reset', {'segment_id': segment}, 30.)
    backend = pb.BITStarBackend()
    started = time.monotonic()
    simulated = length = inference_s = request_s = planning_s = 0.
    replans = rotations = budget_extensions = repairs = 0
    frame = 0
    events, segment_events, extents = [], [], []
    path = [list(motion.position)]
    status, reason = 'timeout', 'unchanged simulation time limit'
    scale = {}; last_response = None; last_plan = None; last_rgb = None
    recovery_index = 0
    recovery_base_yaw = 0.
    # Absolute offsets around the failing pose's heading; no simulator obstacle
    # information is consulted to choose these views.
    recovery_offsets = [40., -40., 80., -80., 120., -120.]
    skip_alignment_once = False
    for decision in range(config.maximum_decisions):
        if simulated > w.episode_timeout(task['path_kind'], task['gt_shortest_path_m']):
            break
        previous = target_service.completed_subgoals
        target_service.advance_if_reached(motion.position)
        for completed in range(previous + 1, target_service.completed_subgoals + 1):
            segment_events.append(dict(subgoal_number=completed,
                simulated_seconds=simulated, executed_path_length_m=length))
        if target_service.complete:
            status, reason = 'success', None
            break
        point_goal = target_service.query(motion.position, motion.yaw_degrees)
        turn = alignment_turn(point_goal, config)
        if skip_alignment_once:
            turn = 0.
            skip_alignment_once = False
        if turn:
            advance = w.apply_geometric_path(scene, motion, [[0., 0.]], maximum_arc_m=.01,
                                             forced_yaw_delta_degrees=turn)
            rotations += 1
            events.append(dict(event='target_alignment', turn_degrees=turn,
                               target_metric_m=point_goal, safe=advance.safe))
            simulated += abs(math.radians(turn)) / w.MOTION_LIMITS.maximum_angular_speed_rad_s
            if not advance.safe:
                status, reason = 'collision', 'physical footprint collision during in-place turn'
                break
            # The next decision obtains a NEW RGB before any translation.
            continue

        last_rgb = scene.rgb(motion.position, motion.yaw_degrees)
        frame += 1
        before = time.monotonic()
        try:
            response = w.post_json(model_url, '/predict', dict(segment_id=segment,
                frame_seq=frame, image_png_base64=w.encode_jpeg(last_rgb),
                physical_camera_height_m=w.CAMERA_HEIGHT_M, threshold=.5), 300.)
        except Exception as error:
            status, reason = 'inference_failure', f'{type(error).__name__}: {error}'
            break
        request_s += time.monotonic() - before
        last_response = response
        inference_s += float(response.get('inference_seconds', 0.))
        semantic, occupancy, confidence, extent = w.decode_bev_response(response)
        extents.append(extent)
        scale = {k:response.get(k) for k in ('model_scale_token_bev_per_vggt',
            'camera_metric_scale_m_per_vggt','bev_metric_scale_m_per_bev',
            'vggt_camera_height','ground_inlier_fraction','ground_fallback_used')}
        before = time.monotonic()
        try:
            result, problem = plan_reliable_bitstar(backend, semantic=semantic,
                occupancy_probability=occupancy, navigation_confidence=confidence,
                extent_m=extent, target_metric_m=point_goal,
                inflation_radius_m=w.TOTAL_INFLATION_M, config=config)
        except pb.PlannerFailure as error:
            planning_s += time.monotonic() - before
            events.append(dict(event='planning_failure', code=error.code, message=str(error),
                               details=error.details, target_metric_m=point_goal, frame=frame))
            if recovery_index >= config.maximum_recovery_views:
                status, reason = 'planner_failure', f'{error.code}: {error}'
                break
            if recovery_index == 0:
                recovery_base_yaw = motion.yaw_degrees
            desired_yaw = recovery_base_yaw + recovery_offsets[recovery_index % len(recovery_offsets)]
            delta = (desired_yaw - motion.yaw_degrees + 180) % 360 - 180
            recovery_index += 1
            advance = w.apply_geometric_path(scene, motion, [[0.,0.]], maximum_arc_m=.01,
                                             forced_yaw_delta_degrees=delta)
            rotations += 1
            simulated += abs(math.radians(delta)) / w.MOTION_LIMITS.maximum_angular_speed_rad_s
            events.append(dict(event='recovery_observation',turn_degrees=delta,safe=advance.safe))
            if not advance.safe:
                status, reason = 'collision', 'physical footprint collision during recovery turn'
                break
            skip_alignment_once = True
            continue
        planning_s += time.monotonic() - before
        recovery_index = 0
        last_plan = result.path_metric_m
        repairs += int(result.backend_details['planned_goal_repaired'])
        budget_extensions += int(result.backend_details['extended_budget_used'])
        execution_horizon,unknown_arc=risk_limited_execution_horizon(
            result.path_metric_m,semantic,extent,config)
        advance = w.apply_geometric_path(scene, motion, result.path_metric_m,
                                         maximum_arc_m=execution_horizon)
        replans += 1
        events.append(dict(event='execution',frame=frame,target_metric_m=point_goal,
            pose_before_world=path[-1],executed_prefix_world=advance.world_path,
            planned_path_metric_m=last_plan, backend=result.backend_details,
            execution_horizon_m=execution_horizon,unknown_execution_arc_m=unknown_arc,
            safe=advance.safe, collision_world=advance.collision.collision_world,
            extent_m=extent))
        if not advance.safe:
            status, reason = 'collision', f'executed prefix intersects GT geometry at {advance.collision.collision_world}'
            break
        length += advance.travelled_m
        path.extend(advance.world_path[1:])
        simulated += max(1./w.BEV_INFERENCE_HZ,
                         advance.travelled_m/w.MOTION_LIMITS.maximum_speed_m_s)
    else:
        status, reason = 'decision_limit', 'bounded policy decision limit'

    terminal = w.distance_xz(motion.position,task['goal_world'])
    gt_length = task['gt_shortest_path_m']
    result = dict(status=status, success=status=='success', reason=reason,
        model='single_baseline',backend='bitstar',policy='rgb_pointgoal_unknown_risk_bitstar_v2',
        policy_config=asdict(config),path_kind=task['path_kind'],gt_shortest_path_m=gt_length,
        executed_path_length_m=length,executed_world_path=path,terminal_distance_m=terminal,
        executed_path_ratio=max(1.,length/gt_length) if status=='success' else None,
        completion_adjusted_path_ratio=max(1.,(length+terminal)/gt_length),
        simulated_seconds=simulated,wall_seconds=time.monotonic()-started,
        model_inference_seconds=inference_s,model_request_wall_seconds=request_s,
        planner_seconds=planning_s,replans=replans,observation_count=frame,
        rotation_count=rotations,target_repair_count=repairs,
        extended_budget_count=budget_extensions,subgoals_total=len(task['subgoals_world']),
        subgoals_completed=target_service.completed_subgoals,
        target_service_refreshes=target_service.refresh_count,
        segment_completion_events=segment_events,last_scale_diagnostics=scale,
        metric_extent_mean_m=float(np.mean(extents)) if extents else None,
        predicted_extrinsic_used_for_target_tracking=False,
        planner_total_inflation_m=w.TOTAL_INFLATION_M,
        geometry_collision_robot_radius_m=w.ROBOT_COLLISION_RADIUS_M,
        execution_mode='receding_native_prefix_0p5m_unknown_budget_0p2m_v2',events=events)
    if diagnostics_dir:
        root = Path(diagnostics_dir); root.mkdir(parents=True,exist_ok=True)
        (root/'events.json').write_text(json.dumps(events,indent=2))
        if last_response is not None:
            sem,occ,conf,extent = w.decode_bev_response(last_response)
            np.savez_compressed(root/'last_prediction.npz',semantic=sem,
                occupancy=occ,confidence=conf,extent_m=extent,
                planned_path=np.asarray(last_plan if last_plan is not None else []))
            from PIL import Image
            Image.fromarray(last_rgb).save(root/'last_rgb.jpg',quality=95)
    return result
