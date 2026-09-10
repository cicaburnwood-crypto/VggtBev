"""Bounded, exact native-polyline execution. No map, goal or model dependency.

Coordinates: x=right, y=up, z=forward; yaw=0 faces +z, positive yaw faces +x.
This is an ideal rate-limited executor, NOT a dynamic/differential-drive model.
To preserve a polyline exactly at finite angular speed, translation stops at
vertices while heading changes. There is no acceleration constraint or shortcut.
"""
from dataclasses import asdict, dataclass
import math
from typing import Optional

import numpy as np


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class RobotSpec:
    speed_m_s: float = 1.0
    yaw_rate_deg_s: float = 90.0
    camera_height_m: float = 0.50
    width_m: float = 0.20
    length_m: float = 0.20
    height_m: float = 0.50
    tick_s: float = 0.02
    collision_tolerance_m: float = 0.0001

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Robot limits and dimensions must be finite and positive')

    @property
    def yaw_rate_rad_s(self):
        return math.radians(self.yaw_rate_deg_s)

    @property
    def circumradius_m(self):
        return math.hypot(self.width_m, self.length_m) / 2

    def metadata(self):
        return dict(**asdict(self), yaw_rate_rad_s=self.yaw_rate_rad_s,
                    execution='exact_polyline_bidirectional_v6', reverse_allowed=True,
                    acceleration_constraint=None, lateral_translation=False,
                    camera_mount='base center + height; optical axis horizontal',
                    collision_body='upright oriented cuboid, no safety inflation')


@dataclass(frozen=True)
class Pose:
    x: float = 0.
    z: float = 0.
    yaw_rad: float = 0.
    base_y: float = 0.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Nonfinite pose')

    @property
    def xz(self):
        return np.array([self.x, self.z], dtype=float)

    def camera_xyz(self, spec=RobotSpec()):
        return [self.x, self.base_y + spec.camera_height_m, self.z]


@dataclass(frozen=True)
class MotionStep:
    start: Pose
    end: Pose
    dt_s: float
    distance_m: float
    yaw_delta_rad: float
    phase: str
    native_vertex_index: int
    drive_sign: int = 1

    @property
    def speed_m_s(self):
        return self.drive_sign * self.distance_m / self.dt_s

    @property
    def yaw_rate_rad_s(self):
        return self.yaw_delta_rad / self.dt_s

    def at(self, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError('fraction must be in [0,1]')
        return Pose(self.start.x + fraction * (self.end.x-self.start.x),
                    self.start.z + fraction * (self.end.z-self.start.z),
                    wrap(self.start.yaw_rad + fraction*self.yaw_delta_rad),
                    self.start.base_y)


def validate_step(step, spec=RobotSpec()):
    """Also reject malformed external steps before any collision query/motion."""
    if not all(math.isfinite(v) for v in (step.dt_s,step.distance_m,step.yaw_delta_rad)):
        raise ValueError('Nonfinite motion tick')
    if not 0 < step.dt_s <= spec.tick_s + 1e-10:
        raise ValueError('Invalid tick duration')
    if step.start.base_y != step.end.base_y:
        raise ValueError('Executor is same-floor only')
    actual = float(np.linalg.norm(step.end.xz-step.start.xz))
    if not math.isclose(actual, step.distance_m, abs_tol=1e-9):
        raise ValueError('Step distance mismatch')
    if abs(wrap(step.end.yaw_rad-step.start.yaw_rad-step.yaw_delta_rad)) > 1e-9:
        raise ValueError('Step yaw mismatch')
    if step.drive_sign not in (-1, 1):
        raise ValueError('Invalid drive direction')
    if actual / step.dt_s > spec.speed_m_s + 1e-8:
        raise ValueError('Translation rate limit exceeded')
    if abs(step.yaw_rate_rad_s) > spec.yaw_rate_rad_s + 1e-8:
        raise ValueError('Angular rate limit exceeded')
    if actual > 1e-10 and abs(step.yaw_delta_rad) > 1e-10:
        raise ValueError('Exact polyline tick cannot translate and rotate together')
    if actual > 1e-10:
        direction = (step.end.xz-step.start.xz) / actual
        forward = np.array([math.sin(step.start.yaw_rad), math.cos(step.start.yaw_rad)])
        if np.linalg.norm(direction-step.drive_sign*forward) > 1e-7:
            raise ValueError('Lateral translation is not permitted')


def local_to_world(point, pose):
    c, s = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    return np.array([pose.x+c*point[0]+s*point[1],
                     pose.z-s*point[0]+c*point[1]])


def rotation_steps(start, target_yaw, spec=RobotSpec(), vertex_index=-1):
    if not math.isfinite(target_yaw):
        raise ValueError('Nonfinite heading')
    delta = wrap(target_yaw-start.yaw_rad)
    total, elapsed = abs(delta)/spec.yaw_rate_rad_s, 0.
    pose = start
    while elapsed < total - 1e-12:
        dt = min(spec.tick_s, total-elapsed)
        elapsed += dt
        yaw = wrap(start.yaw_rad + delta * elapsed/total)
        end = Pose(start.x, start.z, yaw, start.base_y)
        step = MotionStep(pose, end, dt, 0., math.copysign(spec.yaw_rate_rad_s*dt, delta),
                          'rotate', vertex_index)
        validate_step(step, spec)
        yield step
        pose = end


def segment_orientations(path, initial_yaw, *, allow_reverse=False,
                         native_headings_left_rad=None):
    """Choose gear, never change a waypoint or consult a goal/map.

    SE(2) heading selects the tangent branch closest to the native heading.
    Arbitrary learned XY/yaw pairs need not be nonholonomically consistent:
    retain the native yaw as reference, project to a legal tangent, and expose
    that distinction. XY-only paths minimize total in-place turning by a
    two-state dynamic program (forward/backward), with forward tie breaking.
    """
    points=np.asarray(path,dtype=float)
    delta=np.diff(np.vstack(([0.,0.],points)),axis=0)
    lengths=np.linalg.norm(delta,axis=1)
    indices=np.flatnonzero(lengths>1e-12)
    angles=np.arctan2(delta[indices,0],delta[indices,1])+initial_yaw
    if native_headings_left_rad is not None:
        refs=np.asarray(native_headings_left_rad,dtype=float)
        if refs.shape!=(len(points),) or not np.isfinite(refs).all():
            raise ValueError('Malformed native SE(2) headings')
    else: refs=None
    if not len(indices): return {}
    options=np.column_stack((angles,angles+math.pi))
    if not allow_reverse:
        selected=np.zeros(len(indices),dtype=int)
    elif refs is not None:
        selected=[]
        for i,index in enumerate(indices):
            # Average native start/end orientation on the circle, not XY jitter.
            previous=0. if index==0 else refs[index-1]
            reference=initial_yaw-(previous+wrap(refs[index]-previous)/2)
            errors=[abs(wrap(h-reference)) for h in options[i]]
            selected.append(0 if errors[0]<=errors[1] else 1)
    else:
        costs=np.array([abs(wrap(h-initial_yaw)) for h in options[0]])
        parents=[]
        for i in range(1,len(indices)):
            transitions=np.array([[costs[a]+abs(wrap(options[i,b]-options[i-1,a]))
                                   for b in range(2)] for a in range(2)])
            parents.append(np.argmin(transitions,axis=0))
            costs=np.min(transitions,axis=0)
        selected=[int(np.argmin(costs))]
        for parent in reversed(parents): selected.append(int(parent[selected[-1]]))
        selected.reverse()
    return {int(index):(wrap(float(options[i,gear])),1 if gear==0 else -1)
            for i,(index,gear) in enumerate(zip(indices,selected))}


def path_steps(native_path, initial=Pose(), spec=RobotSpec(), *,
               horizon_m: Optional[float] = None, final_yaw_rad=None,
               allow_reverse=False, native_headings_left_rad=None):
    """Lazy exact motion ticks, anchored at the stationary RGB exposure pose.

    The origin-to-first-waypoint connector is explicitly traversed, never
    teleported. All original vertices (including duplicates) retain their index.
    No nearest-path projection, resampling, goal snapping, recovery or GT input.
    Heading follows the path tangent (or its reverse). Optional final heading is relative to the
    exposure pose and is applied only if the ENTIRE supplied path was traversed.
    """
    path = np.asarray(native_path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or not len(path) or not np.isfinite(path).all():
        raise ValueError('invalid_native_path')
    if horizon_m is not None and (not math.isfinite(horizon_m) or horizon_m <= 0):
        raise ValueError('Execution horizon must be finite and positive')
    budget = math.inf if horizon_m is None else horizon_m
    orientations=segment_orientations(path,initial.yaw_rad,allow_reverse=allow_reverse,
                                      native_headings_left_rad=native_headings_left_rad)
    pose, total_arc = initial, 0.
    completed = True
    for index, point in enumerate(path):
        target = local_to_world(point, initial)
        delta = target-pose.xz
        length = float(np.linalg.norm(delta))
        if length < 1e-12:
            continue
        if total_arc >= budget - 1e-12:
            completed = False
            break
        heading,drive_sign=orientations[index]
        for step in rotation_steps(pose, heading, spec, index):
            yield step
            pose = step.end
        origin = pose.xz
        distance = min(length, budget-total_arc)
        direction = delta/length
        # Split a segment evenly instead of leaving a near-zero final tick.
        # Subtracting world coordinates in that tiny tail amplified roundoff
        # into a spurious rate violation. Native vertices/horizon are unchanged.
        ticks = max(1, math.ceil(distance/(spec.speed_m_s*spec.tick_s)))
        ds = distance/ticks
        for tick in range(1, ticks+1):
            moved = distance*(tick/ticks)
            destination = origin+direction*moved
            if tick == ticks and distance == length:
                destination = target  # arithmetic endpoint, not tolerance snapping
            end = Pose(float(destination[0]), float(destination[1]), wrap(heading), pose.base_y)
            actual = float(np.linalg.norm(end.xz-pose.xz))
            # Time the represented displacement, never relax the speed guard.
            dt = max(ds, actual)/spec.speed_m_s
            step = MotionStep(pose, end, dt, actual, 0.,
                              'translate', index, drive_sign)
            validate_step(step, spec)
            yield step
            pose = end
        total_arc += distance
        if distance < length - 1e-12:
            completed = False
            break
    if completed and final_yaw_rad is not None:
        yield from rotation_steps(pose, initial.yaw_rad+final_yaw_rad, spec, len(path))


def execute_steps(steps, *, initial, collider, render, clock, spec=RobotSpec(),
                  render_hz=10., on_step=None):
    """Standalone acceptance-test driver with actual paced movement and RGB.

    Collision receives only the tick being executed. Its return is a continuous
    contact fraction, not a future-path veto. Rendering/computation pauses the
    ideal robot and adds to wall time (never retroactively changes its pose).
    A fake clock is permitted ONLY for tests, never reported as measured runtime.
    """
    started = clock.monotonic()
    pose, motion_s, distance, turned = initial, 0., 0., 0.
    trace = []
    render(pose)
    last_render = clock.monotonic()
    status = 'complete'
    if collider.contact_at(pose):
        return dict(status='invalid_initial_collision', pose=pose, trace=[],
                    distance_m=0., rotation_rad=0., motion_s=0., wall_s=clock.monotonic()-started)
    for step in steps:
        validate_step(step, spec)
        if np.linalg.norm(pose.xz-step.start.xz) > 1e-8 or abs(wrap(pose.yaw_rad-step.start.yaw_rad)) > 1e-8:
            raise ValueError('Disconnected path/pose: refusing a teleport')
        contact = collider.sweep(step)
        fraction = 1. if contact is None else contact
        clock.sleep(step.dt_s*fraction)
        pose = step.at(fraction)
        motion_s += step.dt_s*fraction
        distance += step.distance_m*fraction
        turned += abs(step.yaw_delta_rad)*fraction
        trace.append(dict(t=clock.monotonic()-started, motion_t=motion_s, **asdict(pose),
                          phase=step.phase, vertex=step.native_vertex_index,
                          ds=step.distance_m*fraction, dyaw=step.yaw_delta_rad*fraction,
                          dt=step.dt_s*fraction))
        if on_step is not None:
            on_step(pose)
        if contact is not None:
            status = 'collision'
            break
        if clock.monotonic()-last_render >= 1/render_hz:
            render(pose)
            last_render = clock.monotonic()
    render(pose)
    return dict(status=status, pose=pose, trace=trace, distance_m=distance,
                rotation_rad=turned, motion_s=motion_s, wall_s=clock.monotonic()-started)
