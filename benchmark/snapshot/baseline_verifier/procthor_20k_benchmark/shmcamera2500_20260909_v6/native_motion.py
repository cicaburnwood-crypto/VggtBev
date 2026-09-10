"""Ideal cmd_vel plant, NOT a path follower or replacement native controller.

Accepts already-produced bounded native v/w; analytically integrates the arc.
No goal, path, braking law, lookahead or GT steering input exists here.
"""
from dataclasses import dataclass
import math
from exact_executor import Pose, RobotSpec, wrap


@dataclass(frozen=True)
class TwistStep:
    start: Pose
    v: float
    w: float  # world yaw: positive right; convert native positive-left ONCE
    dt_s: float

    def at(self, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError('Invalid step fraction')
        t = self.dt_s*fraction
        a = self.w*t
        # Stable sinc form of the exact circular-arc integral, also for w=0.
        sinc = math.sin(a/2)/(a/2) if abs(a)>1e-12 else 1.
        distance = self.v*t*sinc
        heading = self.start.yaw_rad+a/2
        return Pose(self.start.x+distance*math.sin(heading),
                    self.start.z+distance*math.cos(heading),
                    wrap(self.start.yaw_rad+a), self.start.base_y)

    @property
    def end(self): return self.at(1.)

    @property
    def distance_m(self): return abs(self.v)*self.dt_s

    @property
    def yaw_delta_rad(self): return self.w*self.dt_s


def validate_twist(step, spec=RobotSpec()):
    if not all(math.isfinite(x) for x in (step.v,step.w,step.dt_s)):
        raise ValueError('Nonfinite native command')
    if not 0 < step.dt_s <= spec.tick_s+1e-10:
        raise ValueError('Native integration tick exceeds common bound')
    if abs(step.v)>spec.speed_m_s+1e-8 or abs(step.w)>spec.yaw_rate_rad_s+1e-8:
        raise ValueError('Native controller exceeds configured robot limits; fix adapter, not path')


def sweep_twist(world, step):
    """Evaluator-only continuous current arc with <=0.1mm contact uncertainty.

    Every body point moves no more than (|v|+r|w|)*dt. Midpoint-box
    expansion therefore encloses the entire interval, not only sampled poses.
    Returned lo is a conservative contact bracket; no future command is read.
    """
    validate_twist(step,world.spec)
    world.sweep_calls += 1
    radius = world.spec.circumradius_m
    boxes = world._boxes_near(step.start,radius+step.distance_m,step.end)
    if not len(boxes): return None
    if world._touch(step.start,boxes): return 0.
    movement = step.distance_m+radius*abs(step.yaw_delta_rad)
    def earliest(lo,hi):
        mid = (lo+hi)/2
        bound = movement*(hi-lo)/2
        if not world._touch(step.at(mid),boxes,padding=bound): return None
        if world._touch(step.at(lo),boxes): return lo
        if 2*bound <= world.spec.collision_tolerance_m: return lo
        left = earliest(lo,mid)
        return earliest(mid,hi) if left is None else left
    return earliest(0.,1.)
