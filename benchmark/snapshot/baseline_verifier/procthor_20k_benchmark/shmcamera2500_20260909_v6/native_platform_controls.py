"""Published waypoint controllers with explicit deployment-platform limits.

These functions do not generate or repair paths. The PD equations/waypoint
indices are from the released deployments. v5 explicitly adds coupled actuator
saturation for NoMaD; this is an adapter, not an unmodified official controller.
Metric calibration is separate from physical actuator limits.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


@dataclass(frozen=True)
class RobotPlatform:
    max_v_m_s: float = 1.0
    max_w_rad_s: float = math.pi / 2
    nomad_spacing_m: float = 0.05

    def __post_init__(self) -> None:
        for value in (self.max_v_m_s, self.max_w_rad_s,self.nomad_spacing_m):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("robot velocity limits must be finite and positive")

    @classmethod
    def from_environment(cls) -> "RobotPlatform":
        return cls(
            float(os.environ.get("NAV_ROBOT_MAX_V_M_S", "1.0")),
            float(os.environ.get("NAV_ROBOT_MAX_W_RAD_S", str(math.pi / 2))),
            float(os.environ.get("NAV_NOMAD_METRIC_SPACING_M", "0.05")),
        )


def curvature_limit(linear, angular, platform):
    """Actuator saturation only: preserve the native v/w ratio, including reverse.

    No target, GT, map, or path input; never accelerates an issued command.
    This deployment adaptation is explicitly versioned, not a new planner.
    """
    if not math.isfinite(linear) or not math.isfinite(angular):
        raise ValueError('Nonfinite command')
    scale=max(1.,abs(linear)/platform.max_v_m_s,abs(angular)/platform.max_w_rad_s)
    return linear/scale,angular/scale


def footprint_pixels(width_m, length_m, safety_margin_m, source_shape,
                     source_resolution_m, planner_size):
    """Native square-footprint parameter, rounded outward after grid resize."""
    cell_m=min(source_shape)*source_resolution_m/planner_size
    half_m=max(width_m,length_m)/2+safety_margin_m
    if cell_m<=0 or half_m<=0:raise ValueError('Invalid metric planner footprint')
    return 2*math.ceil(half_m/cell_m)


def omni_control(
    waypoint: list[float], platform: RobotPlatform
) -> tuple[float, float]:
    """Official index-4 PD and curvature-preserving limits at deployment3Hz.

Input is [forward_m, left_m, heading_cos, heading_sin]. Official MAX_V=0.3,
    MAX_W=0.3 are platform parameters. The official preliminary clips0.5/1.0
    are raised only if smaller than the requested platform maximum. With
    platform0.3/0.3 this exactly reproduces the released controller.
    No multiplication/renormalization of model output occurs.
"""
    dx, dy, hx, hy = map(float, waypoint)
    if not all(math.isfinite(v) for v in (dx, dy, hx, hy)):
        raise ValueError("non-finite OmniVLA waypoint")
    dt, epsilon = 1.0 / 3.0, 1e-8
    if abs(dx) < epsilon and abs(dy) < epsilon:
        linear, angular = 0.0, math.atan2(hy, hx) / dt
    elif abs(dx) < epsilon:
        linear, angular = 0.0, math.copysign(math.pi / (2.0 * dt), dy)
    else:
        linear = dx / dt
        # This is deliberately atan(dy/dx), matching official deployment.
        # atan2 would change the branch for a backward-predicted waypoint.
        angular = math.atan(dy / dx) / dt
    linear = _clip(linear, 0.0, max(0.5, platform.max_v_m_s))
    preliminary_w = max(1.0, platform.max_w_rad_s)
    angular = _clip(angular, -preliminary_w, preliminary_w)
    max_v, max_w = platform.max_v_m_s, platform.max_w_rad_s
    if abs(linear) <= max_v and abs(angular) <= max_w:
        return linear, angular
    if abs(linear) > max_v and abs(angular) <= 0.001:
        return math.copysign(max_v, linear), 0.0
    radius = linear / angular
    if abs(radius) >= max_v / max_w:
        return math.copysign(max_v, linear), math.copysign(max_v / abs(radius), angular)
    return math.copysign(max_w * abs(radius), linear), math.copysign(max_w, angular)


def nomad_metric_spacing(platform: RobotPlatform) -> float:
    """Frozen deployment action calibration, independent of actuator speed cap.

    Retain the previously used 0.2m/s / 4Hz reference (=0.05m). Raising the
    physical speed ceiling to 1m/s must not silently enlarge paths fivefold.
    A different calibration requires an explicit parameter/protocol version.
    """
    return platform.nomad_spacing_m


def nomad_control(
    waypoint: list[float], platform: RobotPlatform
) -> tuple[float, float]:
    """Released NoMaD index-2 PD followed by the v5 coupled actuator limiter."""
    dx, dy = map(float, waypoint[:2])
    if not all(math.isfinite(v) for v in (dx, dy)):
        raise ValueError("non-finite NoMaD waypoint")
    dt, epsilon = 1.0 / 4.0, 1e-8
    if abs(dx) < epsilon:
        linear = 0.0
        angular = math.copysign(math.pi / (2 * dt), dy) if abs(dy) else 0.0
    else:
        linear, angular = dx / dt, math.atan(dy / dx) / dt
    return curvature_limit(max(0.,linear),angular,platform)
