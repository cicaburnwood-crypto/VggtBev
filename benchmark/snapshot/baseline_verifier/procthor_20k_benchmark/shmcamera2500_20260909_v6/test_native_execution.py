"""Native routing, circular-arc plant and actual asynchronous loop, no GPU."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
Image.init()
from body_collision import BoxWorld
from exact_executor import Pose, RobotSpec
from execution_policy import policy_for, NativeCommand, native_velocity, MODE
from native_motion import TwistStep, sweep_twist, validate_twist
from native_platform_controls import RobotPlatform, omni_control, nomad_control
from test_exact_executor import Clock
from test_virtual_episode import Targets


def response(v=.4, left=0., path=None):
    return dict(path_metric_m=path or [],control_velocity=dict(forward_m_s=v,angular_left_rad_s=left),
        native_low_level_controller_released=True,control_velocity_is_placeholder=False,
        execution_preference='control_velocity',inference_seconds=.1)


class NativeMotionTests(unittest.TestCase):
    def test_routing(self):
        for m in ('omnivla','mbra_logonav','nomad'):
            self.assertEqual(policy_for(m)['kind'],'native_velocity')
        for m in ('our_model','limo_tel','limo_aug','genie_samtp'):
            self.assertEqual(policy_for(m)['kind'],'external_exact_polyline')
        with self.assertRaises(ValueError): policy_for('unknown')

    def test_left_sign_and_no_rescale_or_minimum_speed(self):
        self.assertEqual(native_velocity(response(.03,.1)),(.03,-.1))
        self.assertEqual(native_velocity(response(-.4,.1)),(-.4,-.1))
        self.assertEqual(native_velocity(response(0,0)),(0.,0.))

    def test_missing_native_cannot_fall_back_to_valid_path(self):
        r=response(path=[[0,0],[0,2]]);r['control_velocity_is_placeholder']=True
        with self.assertRaises(RuntimeError):native_velocity(r)
        with self.assertRaises(RuntimeError):native_velocity(response(1.01))
        with self.assertRaises(RuntimeError):native_velocity(response(.5,1.6))

    def test_float32_native_limit_roundoff_is_not_false_failure(self):
        rounded=float(np.float32(math.pi/2))
        v,w=native_velocity(response(1.,rounded))
        self.assertEqual(v,1.)
        self.assertEqual(w,-math.pi/2)

    def test_native_controller_caps_keep_own_reduction(self):
        robot=RobotPlatform(1.,math.pi/6)
        for fn,point in ((omni_control,[1.,1.,1.,0.]),(nomad_control,[1.,1.])):
            v,w=fn(point,robot)
            self.assertLessEqual(abs(v),1.)
            self.assertLessEqual(abs(w),math.pi/6+1e-10)
        self.assertLess(omni_control([1.,1.,1.,0.],robot)[0],1.)

    def test_analytic_arc_not_stop_turn_or_straight_chord(self):
        p=Pose(); total=0.
        for _ in range(150):
            step=TwistStep(p,1.,math.pi/6,.02);validate_twist(step)
            p=step.end;total+=step.distance_m
        radius=6/math.pi
        np.testing.assert_allclose(p.xz,[radius,radius],atol=1e-10)
        self.assertAlmostEqual(p.yaw_rad,math.pi/2,10)
        self.assertAlmostEqual(total,3.)

    def test_reverse_and_turn_in_place(self):
        self.assertAlmostEqual(TwistStep(Pose(),-.4,0.,.02).end.z,-.008)
        step=TwistStep(Pose(),0.,math.pi/6,.02)
        np.testing.assert_array_equal(step.end.xz,[0,0])
        self.assertAlmostEqual(step.end.yaw_rad,math.pi/300)

    def test_bound_violation_rejected(self):
        for step in (TwistStep(Pose(),1.01,0,.02),TwistStep(Pose(),.5,1.6,.02),TwistStep(Pose(),0,0,.021)):
            with self.assertRaises(ValueError):validate_twist(step)

    def test_continuous_thin_wall_and_clear_arc(self):
        world=BoxWorld([[[.6,0,-2],[.601,1,2]]])
        step=TwistStep(Pose(.49,0,math.pi/2),1.,0.,.02)
        fraction=sweep_twist(world,step)
        self.assertIsNotNone(fraction)
        self.assertLessEqual(abs(step.at(fraction).x-.50),RobotSpec().collision_tolerance_m)
        self.assertIsNone(sweep_twist(BoxWorld(),TwistStep(Pose(),1.,.5,.02)))

    def test_arc_collision_detects_intermediate_rotation(self):
        # Exact square expands in x as it turns, although center is stationary.
        world=BoxWorld([[[.1006,0,-.101],[.101,1,.101]]])
        step=TwistStep(Pose(),0.,math.pi/6,.02)
        self.assertFalse(world.contact_at(step.start))
        self.assertIsNotNone(sweep_twist(world,step))

    def test_nomad_native_9hz_publish_and_1s_waypoint_timeout(self):
        command=NativeCommand('nomad')
        self.assertEqual(command.at(0.),(0.,0.))
        command.accept(response(.4),.01)
        self.assertEqual(command.at(.02),(0.,0.))
        self.assertEqual(command.at(1/9),(.4,-0.))
        self.assertEqual(command.at(1.),(.4,-0.))
        self.assertEqual(command.at(1+1/9),(0.,0.))

    def test_other_controllers_not_forced_through_nomad_timeout(self):
        for method in ('omnivla','mbra_logonav'):
            command=NativeCommand(method);command.accept(response(.2),0.)
            self.assertEqual(command.at(1.1),(.2,-0.))
            command.reset();self.assertEqual(command.at(1.2),(0.,0.))


class NativeLoopTests(unittest.TestCase):
    def run_fake(self,method='omnivla',latency=.1,goal=(0,0,.8),boxes=(),subgoals=None):
        clock=Clock(); renders=[]; queries=[]; world=BoxWorld(boxes)
        w=types.ModuleType('procthor_benchmark_worker');w.OracleTargetService=Targets
        w.encode_jpeg=lambda img:int(img[0,0,0])
        w.distance_xz=lambda a,b:math.hypot(a[0]-b[0],a[2]-b[2])
        w.atomic_json=lambda file,data:Path(file).write_text(json.dumps(data))
        e=types.ModuleType('episode');e.HISTORY_SPEC={};e.Inference=None
        e.native_image_history=lambda frames,method:[frames[-1][1]]
        e.world_point=lambda state,floor:[state.x,floor,state.z]
        # Reuse the real common target evaluator; avoid importing GPU modules.
        from test_virtual_episode import load
        target_module=load(clock)
        spec=importlib.util.spec_from_file_location('_native_loop_test',Path(__file__).with_name('native_velocity_episode.py'))
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'procthor_benchmark_worker':w,'episode':e,'virtual_episode':target_module}):
            spec.loader.exec_module(module)
        class Scene:
            floor_y=0.
            geometry=types.SimpleNamespace(check_path=lambda points:types.SimpleNamespace(safe=True))
            def rgb(self,point,yaw):
                renders.append((clock.now,list(point),yaw));clock.sleep(.005)
                return np.full((8,8,3),len(renders)%255,dtype=np.uint8)
        class Inference:
            def __init__(self,*args): pass
            def reset(self): pass
            def predict(self,images,target,goal_image):
                queries.append(dict(t=clock.now,target=target,image=images[-1],goal_image=goal_image))
                # Intentionally wrong geometric path: native velocity must win.
                return response(.4,0.,path=[[10,0],[20,0]])
        class Future:
            def __init__(self,row):self.row=row;self.ready=clock.now+latency
            def done(self):return clock.now>=self.ready
            def result(self):return self.row
        class Pool:
            def __init__(self,**kwargs):pass
            def submit(self,fn,*args):return Future(fn(*args))
            def shutdown(self,wait=True):pass
        module.time=clock;module.Inference=Inference;module.ThreadPoolExecutor=Pool
        module.collider_for_scene=lambda scene,worker:world
        task=dict(start_world=[0,0,0],start_yaw_degrees=0.,goal_world=list(goal),gt_shortest_path_m=1.)
        if subgoals:task['subgoals_world']=subgoals
        with tempfile.TemporaryDirectory() as tmp:
            row=module.run_episode(Scene(),task,method,'','',tmp,list(range(len(subgoals or [goal]))),warmup=False)
            trajectory=dict(np.load(Path(tmp)/'trajectory.npz'))
            events=[json.loads(x) for x in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        return row,trajectory,events,queries,renders

    def test_actual_native_loop_uses_velocity_not_path(self):
        row,traj,events,*_=self.run_fake()
        self.assertTrue(row['success']);self.assertEqual(row['control_mode'],'native_velocity')
        self.assertTrue(np.allclose(traj['x'],0.))
        self.assertAlmostEqual(np.max(traj['v']),.4)
        self.assertIsNone(row['execution_horizon_m'])
        self.assertTrue(row['inference_motion_overlap'])

    def test_no_motion_before_first_command_and_motion_during_next_inference(self):
        row,traj,events,queries,_=self.run_fake(latency=.4)
        first=events[0]['t']
        self.assertTrue(np.allclose(traj['z'][traj['t']<first-.02],0.))
        second=events[1]
        during=(traj['t']>second['observation_t']+.02)&(traj['t']<second['t']-.02)
        self.assertGreater(np.ptp(traj['z'][during]),.05)

    def test_native_collision_at_executed_face_not_whole_path(self):
        row,traj,*_=self.run_fake(goal=(0,0,2),boxes=[[[-2,0,.6],[2,1,.601]]])
        self.assertEqual(row['status'],'collision')
        self.assertAlmostEqual(row['executed_path_length_m'],.50,delta=.00011)

    def test_nomad_native_loop_without_polyline(self):
        row,traj,*_=self.run_fake(method='nomad',latency=.12)
        self.assertTrue(row['success']);self.assertEqual(row['native_waypoint_timeout_s'],1.)

    def test_switch_discards_old_subgoal_and_fetches_new_rgb(self):
        row,traj,events,queries,renders=self.run_fake(latency=.5,goal=(0,0,1.4),subgoals=[[0,0,.7],[0,0,1.4]])
        self.assertTrue(row['success'])
        self.assertEqual(set(q['goal_image'] for q in queries),{0,1})
        self.assertGreater(row['stale_subgoal_predictions'],0)
        new=[q for q in queries if q['goal_image']==1][0]
        self.assertGreater(renders[new['image']-1][1][2],.49)


if __name__ == '__main__':unittest.main()
