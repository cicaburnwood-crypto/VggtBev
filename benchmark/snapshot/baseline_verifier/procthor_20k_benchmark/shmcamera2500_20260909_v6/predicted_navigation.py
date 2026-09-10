"""Opt-in RGB/PointGoal navigation improvements; no model or GT map changes.

The policy receives predicted rasters and the common local metric PointGoal.
Simulator geometry is accessed only by the existing evaluator AFTER a command
has been chosen. Original full-native-path evaluation remains untouched.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
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


def predicted_segment_clear(problem, a, b):
    """Continuous segment / predicted inflated cell AABBs; no GT access."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    half=problem.extent_m/2
    if np.any(np.abs(a)>=half) or np.any(np.abs(b)>=half): return False
    rows,cols=np.nonzero(problem.blocked)
    cell=problem.cell_size_m
    centers=np.column_stack((-half+(cols+.5)*cell,half-(rows+.5)*cell))
    low,high=np.minimum(a,b)-cell/2,np.maximum(a,b)+cell/2
    selected=np.all((centers>=low)&(centers<=high),axis=1)
    centers=centers[selected]
    if not len(centers): return True
    enter=np.zeros(len(centers));leave=np.ones(len(centers));valid=np.ones(len(centers),bool)
    for axis in range(2):
        d=b[axis]-a[axis];lo=centers[:,axis]-cell/2;hi=centers[:,axis]+cell/2
        if abs(d)<1e-12:valid&=(a[axis]>=lo-1e-12)&(a[axis]<=hi+1e-12)
        else:
            t0,t1=(lo-a[axis])/d,(hi-a[axis])/d
            enter=np.maximum(enter,np.minimum(t0,t1));leave=np.minimum(leave,np.maximum(t0,t1))
    return not bool(np.any(valid&(enter<=leave+1e-12)))


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
    # BIT* already searches continuous coordinates. Supply the true metric
    # origin as its initial state, not an arbitrary centre of one of four
    # equidistant pixels. Never relocate the camera or shift the raster/path.
    if not predicted_segment_clear(problem,[0.,0.],[0.,0.]):
        raise pb.PlannerFailure('origin_connector_blocked','metric origin intersects predicted inflation')
    solver_problem=replace(problem,start=((problem.size-1)/2.,(problem.size-1)/2.))
    failures = []
    for budget in (config.initial_budget_s, config.extended_budget_s):
        try:
            result = backend.plan(solver_problem, budget_s=budget)
            if any(not predicted_segment_clear(problem,a,b) for a,b in
                   zip(result.path_metric_m,result.path_metric_m[1:])):
                raise pb.PlannerFailure('no_path','BIT* candidate touches predicted inflated occupancy')
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
        origin_state='exact_metric_origin_in_continuous_BITstar',
        predicted_continuous_path_checked=True,
    )
    return result, problem
