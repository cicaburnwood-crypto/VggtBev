"""Shared, non-holonomic benchmark plant and native-waypoint follower.

Only robot state and model-produced waypoints/controls enter this module. It has
no map, collision, GT goal, visibility, oracle, or replanning interface. Collision
testing belongs to the caller and may test only the *executed* step returned by
``integrate_step``. All original waypoint vertices are retained by the follower.

Coordinates: x/right, z/forward; yaw=0 faces +z and yaw=pi/2 faces +x.
Limits are identical for every competitor, including native velocity policies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class Limits:
    max_speed_m_s: float = 1.0
    max_accel_m_s2: float = 1.0
    max_yaw_rate_rad_s: float = 1.0
    max_yaw_accel_rad_s2: float = 1.0
    max_dt_s: float = 0.05

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_dt_s > 0.05:
            raise ValueError("Physical integration must use dt <= 0.05 s")


@dataclass(frozen=True)
class State:
    x: float
    z: float
    yaw_rad: float
    v: float = 0.0
    w: float = 0.0

    @property
    def position_xz(self) -> np.ndarray:
        return np.asarray([self.x, self.z], dtype=np.float64)

    @property
    def yaw_degrees(self) -> float:
        return math.degrees(self.yaw_rad)

    @classmethod
    def from_degrees(cls, x: float, z: float, yaw_degrees: float,
                     v: float = 0.0, w: float = 0.0) -> "State":
        return cls(x, z, math.radians(yaw_degrees), v, w)


@dataclass(frozen=True)
class Step:
    state: State
    distance_m: float
    dt_s: float
    applied_v_midpoint: float
    applied_w_midpoint: float


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _constant_twist(x: float, z: float, yaw: float, v: float, w: float,
                    dt: float) -> Tuple[float, float, float]:
    """Exact constant-twist arc, stable even when yaw rate approaches zero."""
    half_turn = 0.5 * w * dt
    sinc = 1.0 if abs(half_turn) < 1e-12 else math.sin(half_turn) / half_turn
    distance = v * dt * sinc
    heading_midpoint = yaw + half_turn
    return (x + distance * math.sin(heading_midpoint),
            z + distance * math.cos(heading_midpoint),
            wrap_angle(yaw + 2.0 * half_turn))


def integrate_step(state: State, requested_v: float, requested_w: float,
                   dt: float = 0.05, limits: Optional[Limits] = None) -> Step:
    """Advance one physical tick; do not pass a simulator or future path.

    Endpoint velocities obey both speed and acceleration limits. During the
    tick, a midpoint constant twist is integrated as an exact circular arc. This
    is a second-order approximation of varying v/w, *not* a claim of an analytic
    continuous-acceleration solution. Position is never snapped to a waypoint.
    """
    limits = limits or Limits()
    values = (state.x, state.z, state.yaw_rad, state.v, state.w,
              requested_v, requested_w, dt)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Motion state, controls, and dt must be finite")
    if dt <= 0 or dt > limits.max_dt_s + 1e-12:
        raise ValueError(f"dt must be in (0, {limits.max_dt_s}]")
    if abs(state.v) > limits.max_speed_m_s + 1e-9:
        raise ValueError("Initial speed exceeds the common robot limit")
    if abs(state.w) > limits.max_yaw_rate_rad_s + 1e-9:
        raise ValueError("Initial yaw rate exceeds the common robot limit")
    target_v = _clamp(requested_v, -limits.max_speed_m_s, limits.max_speed_m_s)
    target_w = _clamp(requested_w, -limits.max_yaw_rate_rad_s,
                      limits.max_yaw_rate_rad_s)
    next_v = state.v + _clamp(target_v - state.v,
                              -limits.max_accel_m_s2 * dt,
                              limits.max_accel_m_s2 * dt)
    next_w = state.w + _clamp(target_w - state.w,
                              -limits.max_yaw_accel_rad_s2 * dt,
                              limits.max_yaw_accel_rad_s2 * dt)
    mid_v, mid_w = 0.5 * (state.v + next_v), 0.5 * (state.w + next_w)
    x, z, yaw = _constant_twist(state.x, state.z, state.yaw_rad, mid_v, mid_w, dt)
    # Signed velocity may cross zero while reversing: count both traversals.
    if state.v * next_v < 0:
        distance = dt * (state.v ** 2 + next_v ** 2) / (2 * abs(next_v - state.v))
    else:
        distance = abs(mid_v) * dt
    return Step(State(x, z, yaw, next_v, next_w), distance, dt, mid_v, mid_w)


@dataclass(frozen=True)
class Command:
    v: float
    w: float
    done: bool = False
    remaining_m: float = math.inf
    progress_m: float = 0.0
    cross_track_m: float = math.inf
    target_xz: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class TrackerConfig:
    lookahead_m: float = 0.25
    lookahead_time_s: float = 0.15
    heading_gain_s: float = 2.0
    heading_rate_damping: float = 1.0
    endpoint_gain_s: float = 1.0
    goal_tolerance_m: float = 0.04
    stopped_speed_m_s: float = 0.03
    stopped_yaw_rate_rad_s: float = 0.03
    projection_window_m: float = 1.0
    sharp_corner_rad: float = math.radians(35.0)
    corner_approach_m: float = 0.08
    corner_speed_m_s: float = 0.08
    allow_reverse: bool = True

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name != "allow_reverse" and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")


class NativePathTracker:
    """Common continuous unicycle follower for methods outputting waypoints.

    The original polyline is neither resampled, simplified, nor collision
    repaired. Arc-length interpolation selects a moving lookahead point on that
    polyline. Monotone local projection avoids jumping to the far end of a
    self-intersection. Continuous heading feedback and endpoint braking replace
    alternating 'rotate then move' phases. A direction latch avoids repeatedly
    swapping forward/reverse near 90 degrees. This is an execution adapter, not
    a replacement local planner. Native v/w policies use integrate_step directly.
    """

    def __init__(self, limits: Optional[Limits] = None,
                 config: Optional[TrackerConfig] = None) -> None:
        self.limits = limits or Limits()
        self.config = config or TrackerConfig()
        self.path = np.empty((0, 2), dtype=np.float64)
        self.cumulative = np.empty(0, dtype=np.float64)
        self.progress_m = 0.0
        self._direction: Optional[float] = None
        self._arrived = False
        self._corner_lengths = np.empty(0, dtype=np.float64)

    def set_path(self, world_xz: Sequence[Sequence[float]]) -> None:
        path = np.asarray(world_xz, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 1:
            raise ValueError("A native path must contain one or more (x,z) vertices")
        if not np.isfinite(path).all():
            raise ValueError("Native path contains non-finite vertices")
        # Preserve duplicate vertices too: no changes to native planner output.
        self.path = path.copy()
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.cumulative = np.r_[0.0, np.cumsum(lengths)]
        # A differential-drive plant cannot traverse a discontinuous path tangent
        # at cruise speed. Brake before sharp *native* vertices, without rounding
        # or changing the polyline. Dense smooth paths do not acquire new corners.
        nonzero = np.flatnonzero(lengths > 1e-12)
        corners = []
        for before, after in zip(nonzero[:-1], nonzero[1:]):
            vin = (path[before + 1] - path[before]) / lengths[before]
            vout = (path[after + 1] - path[after]) / lengths[after]
            angle = math.acos(_clamp(float(np.dot(vin, vout)), -1.0, 1.0))
            if angle >= self.config.sharp_corner_rad:
                corners.append(self.cumulative[after])
        self._corner_lengths = np.asarray(corners, dtype=np.float64)
        self.progress_m = 0.0
        self._direction = None
        self._arrived = False

    def _point(self, arc_length: float) -> np.ndarray:
        arc_length = _clamp(arc_length, 0.0, float(self.cumulative[-1]))
        idx = min(len(self.path) - 1,
                  int(np.searchsorted(self.cumulative, arc_length, side="right")))
        if idx == 0:
            return self.path[0].copy()
        length = self.cumulative[idx] - self.cumulative[idx - 1]
        if length <= 1e-12:
            return self.path[idx].copy()
        alpha = (arc_length - self.cumulative[idx - 1]) / length
        return self.path[idx - 1] + alpha * (self.path[idx] - self.path[idx - 1])

    def initialize_progress(self, state: State, max_initial_arc_m: float) -> float:
        """Reconcile a newly delivered plan with motion since its exposure.

        ``max_initial_arc_m`` must be an odometry/latency-derived upper bound,
        e.g. maximum speed times observation age plus a fixed small margin. Only
        that prefix is searched, so a delayed result cannot jump arbitrarily far
        through a self-crossing route. Ordinary updates retain the configured
        small monotone projection window. Neither GT nor a map is consulted.
        """
        if not math.isfinite(max_initial_arc_m) or max_initial_arc_m < 0:
            raise ValueError("Initial projection bound must be finite and nonnegative")
        if not len(self.path):
            return 0.0
        self._project(state.position_xz,
                      max_forward_m=max(0., max_initial_arc_m - self.progress_m))
        return self.progress_m

    def _project(self, position: np.ndarray,
                 max_forward_m: Optional[float] = None) -> float:
        best_distance = float(np.linalg.norm(position - self._point(self.progress_m)))
        best_s = self.progress_m
        window = self.config.projection_window_m if max_forward_m is None else max_forward_m
        max_s = min(float(self.cumulative[-1]), self.progress_m + window)
        first = max(0, int(np.searchsorted(self.cumulative, self.progress_m,
                                           side="right")) - 1)
        last = min(len(self.path) - 1,
                   int(np.searchsorted(self.cumulative, max_s, side="right")))
        for i in range(first, last):
            length = float(self.cumulative[i + 1] - self.cumulative[i])
            if length < 1e-12:
                continue
            delta = self.path[i + 1] - self.path[i]
            lo = max(0.0, (self.progress_m - self.cumulative[i]) / length)
            hi = min(1.0, (max_s - self.cumulative[i]) / length)
            alpha = _clamp(float(np.dot(position - self.path[i], delta)) / length ** 2,
                           lo, hi)
            distance = float(np.linalg.norm(position - (self.path[i] + alpha * delta)))
            if distance < best_distance - 1e-12:
                best_distance = distance
                best_s = float(self.cumulative[i] + alpha * length)
        self.progress_m = best_s
        return best_distance

    def command(self, state: State) -> Command:
        if not len(self.path):
            return Command(0.0, 0.0, False)
        position = state.position_xz
        cross_track = self._project(position)
        endpoint_distance = float(np.linalg.norm(self.path[-1] - position))
        remaining = max(float(self.cumulative[-1]) - self.progress_m, endpoint_distance)
        if remaining <= self.config.goal_tolerance_m:
            self._arrived = True
        if self._arrived:
            return Command(0.0, 0.0,
                           abs(state.v) <= self.config.stopped_speed_m_s and
                           abs(state.w) <= self.config.stopped_yaw_rate_rad_s,
                           remaining, self.progress_m, cross_track,
                           tuple(float(x) for x in self.path[-1]))
        lookahead = self.config.lookahead_m + self.config.lookahead_time_s * abs(state.v)
        lookahead_s = self.progress_m + lookahead
        corner_speed = self.limits.max_speed_m_s
        future_corners = self._corner_lengths[self._corner_lengths > self.progress_m + 1e-5]
        if len(future_corners):
            distance_to_corner = float(future_corners[0] - self.progress_m)
            corner_speed = math.sqrt(self.config.corner_speed_m_s ** 2 +
                                     2 * self.limits.max_accel_m_s2 *
                                     max(0.0, distance_to_corner - self.config.corner_approach_m))
            if distance_to_corner > self.config.corner_approach_m:
                lookahead_s = min(lookahead_s, float(future_corners[0]))
        target = self._point(lookahead_s)
        delta = target - position
        bearing = math.atan2(float(delta[0]), float(delta[1]))
        angle = wrap_angle(bearing - state.yaw_rad)
        if self._direction is None:
            self._direction = (-1.0 if self.config.allow_reverse and
                                abs(angle) > math.pi / 2 else 1.0)
        if self._direction < 0:
            angle = wrap_angle(angle + math.pi)
        # A heading-proportional unicycle servo; no discontinuous motion phases.
        desired_w = _clamp(self.config.heading_gain_s * angle -
                           self.config.heading_rate_damping * state.w,
                           -self.limits.max_yaw_rate_rad_s,
                           self.limits.max_yaw_rate_rad_s)
        braking_speed = math.sqrt(2 * self.limits.max_accel_m_s2 * remaining)
        endpoint_speed = self.config.endpoint_gain_s * remaining
        pursuit_curvature = 2 * abs(math.sin(angle)) / max(float(np.linalg.norm(delta)), 1e-6)
        curvature_speed = self.limits.max_yaw_rate_rad_s / max(pursuit_curvature, 1e-6)
        desired_v = self._direction * min(self.limits.max_speed_m_s, corner_speed, braking_speed,
                                           endpoint_speed, curvature_speed) * max(0.0, math.cos(angle))
        return Command(desired_v, desired_w, False, remaining, self.progress_m,
                       cross_track, tuple(float(x) for x in target))
