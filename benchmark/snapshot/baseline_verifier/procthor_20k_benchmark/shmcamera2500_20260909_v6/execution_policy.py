"""Explicit routing: released velocity controller versus external executor.

No inference response is allowed to silently change this choice or fall back
to geometric path execution. NoMaD retains its released 9Hz/1s command lifecycle.
"""
import math
from robot_contract import SPEC

MODE = 'wall_clock_native_or_external_adapter_v6'
NATIVE = {
    'omnivla': dict(controller='official_omnivla_waypoint4_pd', inference_hz=3.,
                    publish_hz=None, waypoint_timeout_s=None),
    'mbra_logonav': dict(controller='released_mbra_pg_pd_curvature_limiter_v5', inference_hz=5.,
                        publish_hz=None, waypoint_timeout_s=None),
    'nomad': dict(controller='nomad_waypoint2_pd_calibrated_curvature_limiter_v5', inference_hz=4.,
                  publish_hz=9., waypoint_timeout_s=1.),
}
EXTERNAL = {'our_model','limo_tel','limo_aug','genie_samtp'}


def policy_for(method):
    if method in NATIVE:
        return dict(kind='native_velocity',**NATIVE[method])
    if method in EXTERNAL:
        return dict(kind='external_exact_polyline',controller='bounded_exact_executor',
                    execution_prefix_m=.5,reverse_allowed=True,
                    heading_policy='native_SE2_branch_else_minimum_total_turn',
                    max_failed_plan_observation_turns=6)
    raise ValueError('No execution policy for method: '+str(method))


def native_velocity(response):
    if (response.get('execution_preference')!='control_velocity' or
        response.get('native_low_level_controller_released') is not True or
        response.get('control_velocity_is_placeholder') is not False):
        raise RuntimeError('Required native controller missing/placeholder; no path fallback allowed')
    control = response.get('control_velocity')
    if not isinstance(control,dict): raise RuntimeError('Missing native cmd_vel')
    v = float(control['forward_m_s']); left = float(control['angular_left_rad_s'])
    lateral = float(control.get('right_m_s',-float(control.get('left_m_s',0.))))
    if not all(math.isfinite(x) for x in (v,left,lateral)) or abs(lateral)>1e-9:
        raise RuntimeError('Invalid native planar velocity')
    # Released MBRA returns float32 tensors: angular caps can round slightly high.
    # Accept numerical roundoff only, then enforce the exact physical bound.
    if abs(v)>SPEC.speed_m_s+1e-6 or abs(left)>SPEC.yaw_rate_rad_s+1e-6:
        raise RuntimeError('Native controller platform limits do not match benchmark')
    v=max(-SPEC.speed_m_s,min(SPEC.speed_m_s,v))
    left=max(-SPEC.yaw_rate_rad_s,min(SPEC.yaw_rate_rad_s,left))
    return v,-left


class NativeCommand:
    def __init__(self,method):
        self.policy=policy_for(method)
        if self.policy['kind']!='native_velocity': raise ValueError('Not a native controller')
        self.received=-math.inf
        self.candidate=self.held=(0.,0.)
        self.next_publish=-math.inf

    def accept(self,response,now):
        self.candidate=native_velocity(response); self.received=now
        if self.policy['publish_hz'] is None: self.held=self.candidate

    def reset(self):
        # A common guide switch/episode end invalidates the old goal, not a
        # GT-based path repair. Pending old-goal predictions are discarded too.
        self.received=-math.inf; self.candidate=self.held=(0.,0.)

    def at(self,now):
        hz=self.policy['publish_hz']
        if hz is not None and now>=self.next_publish-1e-10:
            timeout=self.policy['waypoint_timeout_s']
            self.held=self.candidate if now-self.received<timeout else (0.,0.)
            self.next_publish=now+1/hz
        return self.held

    def next_boundary(self):
        return self.next_publish if self.policy['publish_hz'] else math.inf
