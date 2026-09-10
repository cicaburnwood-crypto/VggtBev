"""CPU-only regression for the sampler, importing no simulator/model modules."""
import ast
import json
import math
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from protocol import CONFIG, stamp, digest
from exact_executor import Pose
from body_collision import BoxWorld


def functions():
    path = Path(__file__).with_name('run_group.py')
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {'five_tasks', 'resume_tasks'}]
    def save(file, data):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(data))
    worker = NS(atomic_json=save, reconstruct=lambda parent, goal: parent+[goal],
                segment_gt_route=lambda graph, route, **kw: [route[-1]])
    scope = dict(CONFIG=CONFIG, stamp=stamp, digest=digest, math=math, random=random,
                 json=json, time=time, w=worker, SceneSamplingError=RuntimeError)
    exec(compile(ast.Module(body=nodes, type_ignores=[]),str(path),'exec'),scope)
    return scope


def scene():
    points = [[0,0,0],[0,0,12],[2,0,0],[2,0,12]]
    return NS(floor_y=0., graph=NS(points=points, components=lambda:[[0,1,2,3]],
        dijkstra=lambda start: ([12.]*4,[start])),
        validate_task_geometry=Mock(), geometry=NS(check_path=lambda p:NS(safe=True)))


def task(i, x):
    return dict(route_index=i, start_world=[x,0,0], start_yaw_degrees=0.,
                start_index=i*2, goal_index=i*2+1, seed=1)


class StartSamplingTests(unittest.TestCase):
    def setUp(self):
        self.scope=functions()

    def test_full_cell_edge_can_collide_while_cell_center_is_clear(self):
        # Exact offending MP3D pose and obstacle cell, independent of a GPU.
        p=Pose(3.2694220542907715,.21476411819458008,1.2325160767074372,-3.0083818435668945)
        box=[[[3.1319220542907713,-3.0083818335668946,.2772641181945805],
              [3.156922054290771,-2.5083818435668945,.30226411819458054]]]
        world=BoxWorld(box)
        center=((box[0][0][0]+box[0][1][0])/2,(box[0][0][2]+box[0][1][2])/2)
        self.assertGreater(math.hypot(p.x-center[0],p.z-center[1]),world.spec.circumradius_m)
        self.assertTrue(world.contact_at(p))

    def test_sampling_uses_exact_body_and_rejects_before_geometry(self):
        rng=Mock();rng.choice.side_effect=[0,1,2,3];rng.uniform.return_value=0
        s=scene(); collider=Mock();collider.contact_at.side_effect=[True,False]
        with patch('robot_contract.collider_for_scene',return_value=collider):
            result=self.scope['five_tasks'](s,rng,0,count=1)
        self.assertEqual(result[0]['start_index'],2)
        self.assertEqual(collider.contact_at.call_count,2)
        self.assertEqual(s.validate_task_geometry.call_count,1)

    def test_legacy_replay_preserves_rng_consumption(self):
        rng=Mock();rng.choice.side_effect=[0,1];rng.uniform.return_value=0
        with patch('robot_contract.collider_for_scene') as make:
            result=self.scope['five_tasks'](scene(),rng,0,count=1,check_start_body=False)
        make.assert_not_called()
        self.assertEqual(result[0]['start_index'],0)

    def setup_repair(self, directory):
        args=NS(output=Path(directory), seed=1, source='mp3d')
        original=[task(0,0),task(1,5)]
        (args.output/'tasks.json').write_text(json.dumps(original))
        self.scope['five_tasks']=Mock(return_value=[task(0,2)])
        return args,original,NS(contact_at=lambda p:p.x==0)

    def test_unstarted_route_repaired_only_once_and_other_outcomes_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            args,old,collider=self.setup_repair(directory)
            evidence=args.output/'route_01/limo_tel/result.json'
            evidence.parent.mkdir(parents=True);evidence.write_text('{"success":false}')
            with patch('robot_contract.collider_for_scene',return_value=collider):
                updated=self.scope['resume_tasks'](scene(),old,args,0,'asset')
                replayed=self.scope['resume_tasks'](scene(),old,args,0,'asset')
            self.assertEqual(updated,replayed)
            self.assertEqual(updated[1],old[1])
            self.assertEqual(evidence.read_text(),'{"success":false}')
            self.scope['five_tasks'].assert_called_once()
            audit=json.loads((args.output/'START_BODY_REPAIR.json').read_text())
            self.assertEqual(audit['original_tasks'],old)
            self.assertEqual(audit['replaced_route_indices'],[0])

    def test_any_model_evidence_prevents_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            args,old,collider=self.setup_repair(directory)
            evidence=args.output/'route_00/limo_tel/trajectory.npz'
            evidence.parent.mkdir(parents=True);evidence.write_bytes(b'keep')
            with patch('robot_contract.collider_for_scene',return_value=collider):
                with self.assertRaisesRegex(RuntimeError,'model evidence'):
                    self.scope['resume_tasks'](scene(),old,args,0,'asset')
            self.scope['five_tasks'].assert_not_called()
            self.assertEqual(json.loads((args.output/'tasks.json').read_text()),old)
            self.assertEqual(evidence.read_bytes(),b'keep')

    def test_journal_recovers_interrupted_task_write_without_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            args,old,collider=self.setup_repair(directory)
            with patch('robot_contract.collider_for_scene',return_value=collider):
                updated=self.scope['resume_tasks'](scene(),old,args,0,'asset')
                (args.output/'tasks.json').write_text(json.dumps(old))
                self.assertEqual(self.scope['resume_tasks'](scene(),old,args,0,'asset'),updated)
            self.scope['five_tasks'].assert_called_once()

    def test_tampered_tasks_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            args,old,collider=self.setup_repair(directory)
            (args.output/'tasks.json').write_text('[]')
            with self.assertRaisesRegex(RuntimeError,'differ'):
                self.scope['resume_tasks'](scene(),old,args,0,'asset')


class SceneInitializationTests(unittest.TestCase):
    def setUp(self):
        # Run with the existing mixed-adapter dependency on PYTHONPATH.
        from sampling_errors import SceneSamplingError
        self.sampling_error=SceneSamplingError
        path=Path(__file__).with_name('run_group.py')
        node=next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='create_procthor_scene')
        self.worker=NS(Scene=Mock())
        scope=dict(w=self.worker,SceneSamplingError=SceneSamplingError)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
        self.create=scope['create_procthor_scene']

    def test_good_scene_returned_without_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            args=NS(output=Path(directory))
            value=object();self.worker.Scene.return_value=value
            self.assertIs(self.create({},args,42),value)
            self.worker.Scene.assert_called_once_with({},args,42)

    def test_exact_empty_graph_rejected_before_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            self.worker.Scene.side_effect=RuntimeError('reachable graph has no useful connected component')
            with self.assertRaisesRegex(self.sampling_error,'ProcTHOR scene 42'):
                self.create({},NS(output=Path(directory)),42)

    def test_unrelated_runtime_error_stays_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            for text in ('CUDA out of memory','GPU binding mismatch','GetReachablePositions failed',
                         'reachable graph has no useful connected component: unexpected variant'):
                error=RuntimeError(text);self.worker.Scene.side_effect=error
                with self.assertRaises(RuntimeError) as caught:
                    self.create({},NS(output=Path(directory)),42)
                self.assertIs(caught.exception,error)

    def test_any_frozen_task_or_model_evidence_prevents_rejection(self):
        for name in ('tasks.json','routes.json','results.json','route_00'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);p=root/name
                if name.startswith('route_'):p.mkdir()
                else:p.write_text('preserve')
                error=RuntimeError('reachable graph has no useful connected component')
                self.worker.Scene.side_effect=error
                with self.assertRaises(RuntimeError) as caught:
                    self.create({},NS(output=root),42)
                self.assertIs(caught.exception,error)
                self.assertTrue(p.exists())


if __name__=='__main__': unittest.main()
