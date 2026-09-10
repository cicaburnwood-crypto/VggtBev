"""CPU-only integration tests of the actual new benchmark episode loop."""
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
Image.init()  # Load plugins before patch.dict restores the temporary import table.
from body_collision import BoxWorld
from test_exact_executor import Clock


class Targets:
    def __init__(self,task):
        self.subgoals_world=task.get('subgoals_world',[task['goal_world']])
        self.current_index=self.switch_count=0
    @classmethod
    def from_task(cls,task): return cls(task)
    @property
    def complete(self): return self.current_index>=len(self.subgoals_world)
    @property
    def completed_subgoals(self): return self.current_index
    def query(self,point,yaw):
        dx,_,dz=np.asarray(self.subgoals_world[self.current_index])-point
        c,s=math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
        return [c*dx-s*dz,s*dx+c*dz]


def load(clock):
    w=types.ModuleType('procthor_benchmark_worker')
    w.OracleTargetService=Targets
    w.encode_jpeg=lambda image:int(image[0,0,0])
    w.distance_xz=lambda a,b:math.hypot(a[0]-b[0],a[2]-b[2])
    w.atomic_json=lambda path,data:Path(path).write_text(json.dumps(data))
    e=types.ModuleType('episode')
    e.Inference=None
    e.HISTORY_SPEC={}
    e.native_image_history=lambda frames,method:[frames[-1][1]]
    e.world_point=lambda state,floor:[state.x,floor,state.z]
    file=Path(__file__).with_name('virtual_episode.py')
    spec=importlib.util.spec_from_file_location('_bounded_episode_test',file)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'procthor_benchmark_worker':w,'episode':e}):
        spec.loader.exec_module(module)
    module.time=clock
    return module


class EpisodeIntegrationTests(unittest.TestCase):
    def run_fake(self,goal=(2,0,0),boxes=(),latency=.07,subgoals=None,no_path=False,warmup=False):
        clock=Clock();module=load(clock);renders=[];queries=[]
        world=BoxWorld(boxes)
        class Scene:
            floor_y=0.
            geometry=types.SimpleNamespace(check_path=lambda p:types.SimpleNamespace(safe=True))
            def rgb(self,point,yaw):
                renders.append((clock.now,list(point),yaw))
                clock.sleep(.005)
                return np.full((8,8,3),len(renders)%256,dtype=np.uint8)
        class Inference:
            def __init__(self,*args):pass
            def reset(self):pass
            def predict(self,images,target,goal_image):
                queries.append((clock.now,images,target,renders[-1]))
                clock.sleep(latency)
                if no_path:
                    return dict(path_metric_m=[],execution_preference='native_path',
                        planning_failure=dict(code='invalid_predicted_metric_extent'))
                return dict(path_metric_m=[[0,0],target],inference_seconds=latency,
                            path_headings_left_rad=[0.,-math.atan2(target[0],target[1])],
                            execution_preference='native_velocity',
                            control_velocity=dict(forward_m_s=999,angular_left_rad_s=999))
        module.Inference=Inference
        module.collider_for_scene=lambda scene,worker:world
        task=dict(start_world=[0,0,0],start_yaw_degrees=0,goal_world=list(goal),
                  gt_shortest_path_m=float(np.linalg.norm(np.asarray(goal)[[0,2]])))
        if subgoals:task['subgoals_world']=subgoals
        with tempfile.TemporaryDirectory() as temp:
            result=module.run_episode(Scene(),task,'limo_tel','single','baseline',temp,
                                      ['image']*len(subgoals or [goal]),warmup=warmup)
            trace=dict(np.load(Path(temp)/'trajectory.npz'))
            events=[json.loads(x) for x in (Path(temp)/'events.jsonl').read_text().splitlines()]
        return result,trace,events,renders,queries,world

    def test_turn_and_translation_in_actual_loop(self):
        result,trace,events,renders,queries,world=self.run_fake()
        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['rotating_seconds'],1.,8)
        self.assertAlmostEqual(result['absolute_rotation_rad'],math.pi/2,8)
        self.assertLessEqual(np.max(trace['v']),1.)
        self.assertLessEqual(np.max(np.abs(trace['w'])),math.radians(90)+1e-9)
        self.assertEqual(result['robot']['width_m'],.2)

    def test_persistent_invalid_prediction_is_scored_failure_not_infrastructure(self):
        result,trace,events,*_=self.run_fake(no_path=True,warmup=True)
        self.assertFalse(result['success'])
        self.assertEqual(result['status'],'planner_failure')
        self.assertEqual(result['recovery_observation_turns'],6)
        self.assertEqual(result['replans'],7)
        self.assertEqual(result['executed_path_length_m'],0.)
        self.assertTrue(np.all(trace['v']==0))
        self.assertIn('invalid_predicted_metric_extent',result['failure_reason'])

    def test_no_use_of_bogus_velocity_output(self):
        result,trace,*_=self.run_fake()
        self.assertTrue(result['success'])
        self.assertTrue(np.all(np.abs(trace['z'])<1e-9))
        self.assertLessEqual(np.max(trace['v']),1.)

    def test_replan_uses_new_exposure_and_local_goal(self):
        result,trace,events,renders,queries,world=self.run_fake()
        self.assertGreater(len(events),2)
        self.assertTrue(all(b['observation_t']>a['observation_t'] for a,b in zip(events,events[1:])))
        self.assertAlmostEqual(events[1]['observation_pose']['x'],.5,8)
        self.assertAlmostEqual(events[1]['target_metric_m'][0],0.,8)
        self.assertAlmostEqual(events[1]['target_metric_m'][1],1.5,8)

    def test_collision_at_body_face_not_goal_or_future_path(self):
        result,trace,events,renders,queries,world=self.run_fake(
            boxes=[[[.6,0,-2],[.601,1,2]]])
        self.assertEqual(result['status'],'collision')
        self.assertAlmostEqual(result['executed_path_length_m'],.50,8)
        self.assertFalse(result['future_gt_collision_veto'])

    def test_inference_wait_adds_wall_time_without_motion(self):
        result,trace,events,renders,queries,world=self.run_fake(latency=.25)
        self.assertGreaterEqual(result['total_wall_seconds'],
            result['nominal_motion_seconds']+result['model_request_wall_seconds'])
        self.assertFalse(result['inference_motion_overlap'])
        for event in events:
            during=(trace['t']>event['observation_t']+.02)&(trace['t']<event['t']-.001)
            self.assertFalse(np.any(during))

    def test_public_subgoal_switch_causes_fresh_replan(self):
        result,trace,events,*_=self.run_fake(goal=(1,0,1),subgoals=[[0,0,1],[1,0,1]])
        self.assertTrue(result['success'])
        self.assertEqual(result['subgoals_completed'],2)
        self.assertEqual(set(e['upper_subgoal_index'] for e in events),{0,1})


if __name__=='__main__':unittest.main(verbosity=2)
