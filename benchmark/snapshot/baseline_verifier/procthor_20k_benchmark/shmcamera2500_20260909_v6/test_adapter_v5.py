"""Regression coverage for observed real-path failures and prediction-only planning."""
import itertools
import math
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from exact_executor import Pose, RobotSpec, MotionStep, path_steps, segment_orientations, validate_step, wrap
from body_collision import BoxWorld
from native_platform_controls import RobotPlatform, nomad_metric_spacing, nomad_control, curvature_limit
from execution_policy import native_velocity
from recovery_views import RecoveryViews
import planner_backends as pb
from predicted_navigation import (NavigationConfig, plan_reliable_bitstar,
    risk_limited_execution_horizon, predicted_segment_clear)


class AdapterMotionTests(unittest.TestCase):
    def test_centimetre_reverse_does_not_turn_360(self):
        path=[[0,0],[0,-.01],[0,.5]]
        old=list(path_steps(path,allow_reverse=False))
        new=list(path_steps(path,allow_reverse=True,native_headings_left_rad=[0,0,0]))
        self.assertAlmostEqual(sum(s.dt_s for s in old),4.52)
        self.assertAlmostEqual(sum(s.dt_s for s in new),.52)
        self.assertTrue(any(s.speed_m_s<0 for s in new))
        self.assertTrue(all(s.yaw_delta_rad==0 for s in new))
        np.testing.assert_allclose(new[-1].end.xz,[0,.5])
        for s in new: validate_step(s)

    def test_recorded_first_limo_points_preserved(self):
        path=[[0,0],[.003526400774717331,-.01039067655801773],
              [.004072149284183979,.007025100290775299],[.005036286078393459,.03789029270410538],
              [.006876840256154537,.0833631306886673],[.00550451036542654,.1241895854473114]]
        old=list(path_steps(path))
        new=list(path_steps(path,allow_reverse=True))
        self.assertLess(sum(abs(s.yaw_delta_rad) for s in new),sum(abs(s.yaw_delta_rad) for s in old)/2)
        self.assertAlmostEqual(sum(s.distance_m for s in new),sum(s.distance_m for s in old))
        for p in path[1:]:
            self.assertTrue(any(np.linalg.norm(s.end.xz-p)<1e-10 for s in new))

    def test_reverse_stops_at_actual_rear_face(self):
        world=BoxWorld([[[-1,0,-.501],[1,1,-.5]]])
        for step in path_steps([[0,-1]],allow_reverse=True):
            contact=world.sweep(step)
            if contact is not None:
                self.assertAlmostEqual(step.at(contact).z,-.40,9)
                break
        else: self.fail('rear obstacle not detected')

    def test_native_heading_selects_gear(self):
        fwd=list(path_steps([[0,1]],allow_reverse=True,native_headings_left_rad=[0]))
        rev=list(path_steps([[0,-1]],allow_reverse=True,native_headings_left_rad=[0]))
        self.assertTrue(all(s.speed_m_s>=0 for s in fwd))
        self.assertTrue(all(s.speed_m_s<=0 for s in rev))
        with self.assertRaises(ValueError):
            list(path_steps([[0,1]],allow_reverse=True,native_headings_left_rad=[0,0]))

    def test_minimum_turn_gear_dynamic_program(self):
        path=np.array([[.2,.5],[-.1,.49],[.4,.2]])
        chosen=segment_orientations(path,0,allow_reverse=True)
        result=list(path_steps(path,allow_reverse=True))
        angles=np.arctan2(np.diff(np.vstack(([0,0],path)),axis=0)[:,0],
                          np.diff(np.vstack(([0,0],path)),axis=0)[:,1])
        costs=[]
        for bits in itertools.product((0,1),repeat=3):
            h=angles+np.array(bits)*math.pi
            costs.append(sum(abs(wrap(b-a)) for a,b in zip(np.r_[0,h[:-1]],h)))
        self.assertAlmostEqual(sum(abs(s.yaw_delta_rad) for s in result),min(costs),10)

    def test_no_lateral_escape(self):
        with self.assertRaises(ValueError):
            validate_step(MotionStep(Pose(),Pose(.01,0),.02,.01,0,'translate',0,-1))

    def test_new_90_degree_limit_applied(self):
        self.assertAlmostEqual(RobotPlatform().max_w_rad_s, math.pi/2)
        steps=list(path_steps([[1,0]],allow_reverse=True))
        self.assertAlmostEqual(sum(s.dt_s for s in steps if s.phase=='rotate'),1.)
        self.assertAlmostEqual(max(abs(s.yaw_rate_rad_s) for s in steps),math.pi/2)
        with self.assertRaises(RuntimeError):
            native_velocity(dict(execution_preference='control_velocity',native_low_level_controller_released=True,
                control_velocity_is_placeholder=False,control_velocity=dict(forward_m_s=0,angular_left_rad_s=1.7)))


class NativeCalibrationTests(unittest.TestCase):
    def test_genie_footprint_uses_metric_size_after_resize(self):
        from native_platform_controls import footprint_pixels
        self.assertEqual(footprint_pixels(.2,.2,.05,(134,134),.03,240),18)
        self.assertEqual(footprint_pixels(.3,.3,.10,(134,134),.03,240),30)

    def test_live_platform_guard_rejects_stale_parameters(self):
        from interface_acceptance import validate_platform
        good=dict(max_v_m_s=1.,max_w_rad_s=math.pi/2,nomad_spacing_m=.05,
                  body_dimensions_m=[.2,.2,.5],planner_safety_margin_m=.05)
        validate_platform(dict(robot_platform=good))
        for key,value in (('max_w_rad_s',math.pi/6),('nomad_spacing_m',.25),('max_v_m_s',.2)):
            with self.assertRaises(RuntimeError):
                validate_platform(dict(robot_platform={**good,key:value}))

    def test_speed_ceiling_does_not_change_nomad_scale(self):
        self.assertEqual(nomad_metric_spacing(RobotPlatform(.2,.4)),.05)
        self.assertEqual(nomad_metric_spacing(RobotPlatform(1.,math.pi/2)),.05)
        self.assertEqual(nomad_metric_spacing(RobotPlatform(1.,math.pi/2,.07)),.07)

    def test_coupled_limits_keep_curvature_and_reverse(self):
        robot=RobotPlatform(1.,math.pi/2)
        for v,w in ((3.,4.),(-2.,5.),(.2,2.),(1.,0.),(0.,3.)):
            a,b=curvature_limit(v,w,robot)
            self.assertLessEqual(abs(a),1.)
            self.assertLessEqual(abs(b),math.pi/2)
            self.assertAlmostEqual(a*w,b*v)
            self.assertLessEqual(abs(a),abs(v))

    def test_nomad_does_not_enforce_one_metre_per_second(self):
        v,w=nomad_control([.06,0.],RobotPlatform(1.,math.pi/2))
        self.assertAlmostEqual(v,.24);self.assertEqual(w,0)

    def test_recovery_finite_and_resets(self):
        r=RecoveryViews();yaw=0
        for _ in range(6):
            delta=r.next_turn(yaw,[0.,2.]);self.assertIsNotNone(delta);yaw+=delta
        self.assertIsNone(r.next_turn(yaw,[0.,2.]))
        r.reset();self.assertIsNotNone(r.next_turn(yaw,[0.,2.]))


class PredictionNavigationTests(unittest.TestCase):
    def problem(self,sem=None,target=(0,1),backend=None):
        sem=np.full((64,64),255,np.uint8) if sem is None else sem
        return plan_reliable_bitstar(backend or pb.BITStarBackend(),semantic=sem,
            occupancy_probability=(sem==0).astype(float),navigation_confidence=np.ones_like(sem,float),
            extent_m=4.,target_metric_m=target,inflation_radius_m=.05)

    def test_even_grid_solver_starts_at_exact_metric_origin(self):
        result,p=self.problem()
        np.testing.assert_allclose(result.path_metric_m[0],[0.,0.],atol=1e-12)
        self.assertEqual(result.backend_details['origin_state'],'exact_metric_origin_in_continuous_BITstar')
        self.assertTrue(all(predicted_segment_clear(p,a,b) for a,b in zip(result.path_metric_m,result.path_metric_m[1:])))

    def test_blocked_target_uses_predicted_reachable_subgoal(self):
        sem=np.full((64,64),255,np.uint8);sem[15:19,30:35]=0
        result,p=self.problem(sem=sem,target=(0,1.0))
        self.assertTrue(result.backend_details['planned_goal_repaired'])
        self.assertLess(np.linalg.norm(np.array([0,1])-result.path_metric_m[-1]),1)

    def test_out_of_bounds_target_is_local_subgoal_not_immediate_failure(self):
        result,_=self.problem(target=(0,6))
        self.assertLessEqual(np.max(np.abs(result.path_metric_m[-1])),1.81)

    def test_true_origin_blocked_never_cleared(self):
        sem=np.full((64,64),255,np.uint8);sem[31:33,31:33]=0
        with self.assertRaises(pb.PlannerFailure) as caught:self.problem(sem=sem)
        self.assertIn(caught.exception.code,('origin_connector_blocked','start_not_free'))

    def test_connected_search_gets_bounded_extra_budget(self):
        class Flaky:
            def __init__(self):self.calls=[];self.real=pb.BITStarBackend()
            def plan(self,p,budget_s):
                self.calls.append(budget_s)
                if len(self.calls)==1:raise pb.PlannerFailure('no_path','budget')
                return self.real.plan(p,budget_s=budget_s)
        backend=Flaky();result,_=self.problem(backend=backend)
        self.assertEqual(backend.calls,[.55,1.65]);self.assertTrue(result.backend_details['extended_budget_used'])

    def test_unknown_execution_budget_not_half_metre_blind(self):
        sem=np.full((64,64),112,np.uint8)
        horizon,unknown=risk_limited_execution_horizon([[0,0],[0,1]],sem,4.,NavigationConfig())
        self.assertAlmostEqual(horizon,.2);self.assertAlmostEqual(unknown,.2)

    def test_exact_segment_check_detects_corner_contact(self):
        sem=np.full((64,64),255,np.uint8)
        p=pb.build_problem(semantic=sem,occupancy_probability=np.zeros((64,64)),
            navigation_confidence=np.ones((64,64)),extent_m=4,target_metric_m=[0,1],inflation_radius_m=0)
        p.blocked[31,32]=True
        self.assertFalse(predicted_segment_clear(p,[0,0],[.1,.1]))


if __name__=='__main__':unittest.main(verbosity=2)
