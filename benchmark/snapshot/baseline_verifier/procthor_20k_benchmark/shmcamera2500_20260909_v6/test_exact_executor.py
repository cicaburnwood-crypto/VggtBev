import math
import unittest
from types import SimpleNamespace
import numpy as np

from exact_executor import (RobotSpec, Pose, MotionStep, path_steps, rotation_steps,
                            execute_steps, validate_step, wrap, local_to_world)
from body_collision import BoxWorld, RasterBodyWorld
from robot_contract import configure_worker


class Clock:
    def __init__(self): self.now=0.
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now+=seconds


class ExactExecutorTests(unittest.TestCase):
    def test_requested_contract(self):
        spec=RobotSpec()
        self.assertEqual((spec.speed_m_s,spec.yaw_rate_deg_s),(1.,90.))
        self.assertEqual((spec.width_m,spec.length_m,spec.height_m),(.2,.2,.5))
        self.assertEqual(Pose(2,3,0,4).camera_xyz(spec),[2,4.5,3])

    def test_straight_exact_time_endpoint(self):
        steps=list(path_steps([[0,0],[0,3.123]]))
        self.assertAlmostEqual(sum(s.dt_s for s in steps),3.123,10)
        self.assertAlmostEqual(steps[-1].end.z,3.123,12)
        self.assertTrue(all(s.yaw_delta_rad==0 for s in steps))

    def test_right_angle_no_corner_cutting_and_turn_takes_three_seconds(self):
        steps=list(path_steps([[0,0],[0,1],[1,1]]))
        turns=[s for s in steps if s.phase=='rotate']
        self.assertAlmostEqual(sum(s.dt_s for s in turns),1.,10)
        self.assertAlmostEqual(sum(s.dt_s for s in steps),3.,10)
        self.assertTrue(all(np.array_equal(s.start.xz,[0,1]) for s in turns))
        self.assertTrue(all(abs(s.end.x)<1e-10 or abs(s.end.z-1)<1e-10 for s in steps))
        np.testing.assert_allclose(steps[-1].end.xz,[1,1],atol=1e-12)

    def test_goal_behind_turns_180_without_lateral_motion(self):
        steps=list(path_steps([[0,-1]]))
        self.assertAlmostEqual(sum(s.dt_s for s in steps),3.,10)
        for s in steps: validate_step(s)
        self.assertAlmostEqual(steps[-1].end.z,-1.)

    def test_wrap_179_to_minus179_is_two_degrees(self):
        steps=list(rotation_steps(Pose(yaw_rad=math.radians(179)),math.radians(-179)))
        self.assertAlmostEqual(sum(s.dt_s for s in steps),2/90,10)

    def test_dense_and_sparse_collinear_paths_same_length_time(self):
        sparse=list(path_steps([[0,2]]))
        dense=list(path_steps(np.column_stack([np.zeros(201),np.linspace(0,2,201)])))
        self.assertAlmostEqual(sum(s.dt_s for s in sparse),sum(s.dt_s for s in dense),10)

    def test_connector_is_executed_not_teleported(self):
        steps=list(path_steps([[1,1],[1,2]]))
        self.assertEqual(steps[0].start,Pose())
        self.assertAlmostEqual(sum(s.distance_m for s in steps),math.sqrt(2)+1,10)

    def test_repeated_vertices_do_not_change_motion(self):
        steps=list(path_steps([[0,0],[0,0],[0,1],[0,1]]))
        self.assertAlmostEqual(sum(s.dt_s for s in steps),1.,10)
        self.assertTrue(all(s.native_vertex_index==2 for s in steps))

    def test_self_intersection_does_not_skip_loops(self):
        path=[[0,1],[1,1],[1,0],[0,0],[0,2]]
        steps=list(path_steps(path))
        self.assertAlmostEqual(sum(s.distance_m for s in steps),6.,10)
        self.assertEqual(set(s.native_vertex_index for s in steps),set(range(5)))

    def test_horizon_does_not_turn_toward_unexecuted_future(self):
        steps=list(path_steps([[0,1],[1,1]],horizon_m=.5))
        self.assertAlmostEqual(sum(s.distance_m for s in steps),.5,10)
        self.assertTrue(all(s.phase=='translate' for s in steps))

    def test_heading_only_command_and_final_heading(self):
        steps=list(path_steps([[0,0]],final_yaw_rad=math.pi/2))
        self.assertAlmostEqual(sum(s.dt_s for s in steps),1.,10)
        self.assertTrue(all(s.distance_m==0 for s in steps))

    def test_no_final_heading_after_truncated_path(self):
        steps=list(path_steps([[0,2]],horizon_m=.5,final_yaw_rad=math.pi))
        self.assertAlmostEqual(sum(s.dt_s for s in steps),.5,10)

    def test_transform_matches_camera_right_forward_convention(self):
        pose=Pose(10,20,math.pi/2)
        np.testing.assert_allclose(local_to_world([1,2],pose),[12,19],atol=1e-12)
        steps=list(path_steps([[0,2]],pose))
        np.testing.assert_allclose(steps[-1].end.xz,[12,20],atol=1e-12)

    def test_bad_inputs_fail_closed(self):
        for path in ([],[[float('nan'),0]],[[0,0,1]]):
            with self.assertRaises(ValueError): list(path_steps(path))
        for horizon in (0,-1,float('nan')):
            with self.assertRaises(ValueError): list(path_steps([[0,1]],horizon_m=horizon))

    def test_world_coordinate_roundoff_does_not_create_micro_tail_overspeed(self):
        initial=Pose(76.39907467126628,-49.29987478570277,-2.1130002583387655)
        path=[[0.,0.02000007920647981]]
        steps=list(path_steps(path,initial,horizon_m=.5,allow_reverse=True))
        moving=[s for s in steps if s.phase=='translate']
        self.assertEqual(len(moving),2)
        self.assertGreater(min(s.dt_s for s in moving),.009)
        np.testing.assert_array_equal(steps[-1].end.xz,local_to_world(path[-1],initial))
        for step in steps: validate_step(step)

    def test_tiny_tick_remainders_do_not_change_vertices_or_speed_limit(self):
        rng=np.random.default_rng(20260910)
        for _ in range(200):
            initial=Pose(*rng.uniform(-100,100,2),rng.uniform(-math.pi,math.pi))
            length=.02+rng.uniform(1e-12,1e-7)
            sign=rng.choice([-1.,1.])
            steps=list(path_steps([[0.,sign*length]],initial,allow_reverse=True))
            self.assertAlmostEqual(sum(s.distance_m for s in steps),length,11)
            for step in steps:
                self.assertLessEqual(abs(step.speed_m_s),1.+1e-12)
                self.assertLessEqual(step.dt_s,.02+1e-10)
                validate_step(step)

    def test_actual_overspeed_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'Translation rate limit exceeded'):
            validate_step(MotionStep(Pose(),Pose(z=.021),.02,.021,0.,'translate',0))

    def test_random_polylines_analytic_time_and_speed(self):
        rng=np.random.default_rng(917)
        for _ in range(20):
            path=np.cumsum(rng.uniform(-.5,.5,(8,2)),axis=0)
            steps=list(path_steps(path))
            expected_distance=np.linalg.norm(np.diff(np.vstack([[0,0],path]),axis=0),axis=1).sum()
            angles=np.arctan2(np.diff(np.vstack([[0,0],path]),axis=0)[:,0],
                              np.diff(np.vstack([[0,0],path]),axis=0)[:,1])
            expected_turn=sum(abs(wrap(b-a)) for a,b in zip(np.r_[0,angles[:-1]],angles))
            self.assertAlmostEqual(sum(s.dt_s for s in steps),expected_distance+expected_turn/math.radians(90),8)
            np.testing.assert_allclose(steps[-1].end.xz,path[-1],atol=1e-10)
            for s in steps: validate_step(s)


class CollisionTests(unittest.TestCase):
    def test_wall_stops_at_front_face_not_center(self):
        world=BoxWorld([[[.5,0,-5],[.501,1,5]]])
        clock=Clock()
        result=execute_steps(path_steps([[2,0]]),initial=Pose(),collider=world,
                             render=lambda p: None,clock=clock)
        self.assertEqual(result['status'],'collision')
        self.assertAlmostEqual(result['pose'].x,.4,9)

    def test_continuous_sweep_thin_wall_between_endpoints(self):
        # Endpoints safe; a thin obstacle is crossed in between. Test with a
        # tiny body to ensure endpoint-only collision would genuinely miss it.
        spec=RobotSpec(width_m=.001,length_m=.001)
        world=BoxWorld([[[-1,0,.011],[1,1,.0111]]],spec)
        step=next(path_steps([[0,1]],spec=spec))
        self.assertFalse(world.contact_at(step.start))
        self.assertFalse(world.contact_at(step.end))
        self.assertAlmostEqual(world.sweep(step),.0105/.02,9)

    def test_actual_square_fits_22cm_gap_not_circumscribed_circle(self):
        world=BoxWorld([[[-2,0,-1],[-.11,1,3]],[[.11,0,-1],[2,1,3]]])
        steps=list(path_steps([[0,2]]))
        self.assertTrue(all(world.sweep(s) is None for s in steps))

    def test_rotation_corner_collision_even_if_start_and_end_free(self):
        # Obstacle outside an axis-aligned square, inside its swept corner.
        world=BoxWorld([[[.13,0,-.015],[.14,.4,.015]]])
        self.assertFalse(world.contact_at(Pose()))
        self.assertFalse(world.contact_at(Pose(yaw_rad=math.pi/2)))
        contacts=[world.sweep(s) for s in rotation_steps(Pose(),math.pi/2)]
        self.assertTrue(any(x is not None for x in contacts))

    def test_ceiling_above_body_is_clear(self):
        world=BoxWorld([[[-2,.51,-2],[2,1,2]]])
        self.assertFalse(world.contact_at(Pose()))

    def test_low_hanging_obstacle_hits_body(self):
        world=BoxWorld([[[-.1,.49,-.1],[.1,.8,.1]]])
        self.assertTrue(world.contact_at(Pose()))

    def test_floor_support_ignored_but_one_mm_bump_is_not(self):
        self.assertFalse(BoxWorld([[[-2,-1,-2],[2,0,2]]]).contact_at(Pose()))
        self.assertTrue(BoxWorld([[[-.1,0,-.1],[.1,.001,.1]]]).contact_at(Pose()))

    def test_raised_floor_and_camera_height(self):
        world=BoxWorld([[[-1,2.6,-1],[1,3,1]]])
        self.assertFalse(world.contact_at(Pose(base_y=2)))
        self.assertEqual(Pose(base_y=2).camera_xyz(),[0,2.5,0])

    def test_old_height_cache_rejected(self):
        for lower,upper in ((.03,1.4),(0,1.4),(.03,.5)):
            with self.assertRaises(ValueError):
                RasterBodyWorld(np.full((10,10),255),[0,0,0],.01,floor_y=0,
                    obstacle_min_height_m=lower,obstacle_max_height_m=upper)

    def test_raster_metric_alignment_and_outside_bounds(self):
        truth=np.full((101,101),255,dtype=np.uint8)
        truth[50,70]=0  # center x=.2,z=0, voxel extends .195--.205
        world=RasterBodyWorld(truth,[-.5,0,-.5],.01,floor_y=0,
                    obstacle_min_height_m=0,obstacle_max_height_m=.5)
        self.assertFalse(world.contact_at(Pose()))
        self.assertTrue(world.contact_at(Pose(x=.10)))
        self.assertTrue(world.contact_at(Pose(z=.45)))
        with self.assertRaises(ValueError): world.contact_at(Pose(base_y=1))

    def test_only_current_tick_is_checked_future_wall_not_rejected(self):
        world=BoxWorld([[[-1,0,10],[1,2,10.1]]])
        clock=Clock()
        result=execute_steps(path_steps([[0,20]],horizon_m=.5),initial=Pose(),
                             collider=world,render=lambda p: None,clock=clock)
        self.assertEqual(result['status'],'complete')
        self.assertAlmostEqual(result['distance_m'],.5)
        self.assertEqual(world.sweep_calls,25)


class IntegrationTests(unittest.TestCase):
    def test_rendering_rotation_and_all_costs_count(self):
        clock=Clock();frames=[]
        def render(pose):
            frames.append(pose)
            clock.sleep(.003)
        result=execute_steps(path_steps([[0,1],[1,1]]),initial=Pose(),collider=BoxWorld(),
                             render=render,clock=clock)
        self.assertAlmostEqual(result['motion_s'],3.,9)
        self.assertAlmostEqual(result['wall_s'],3.+len(frames)*.003,9)
        self.assertTrue(any(0<p.yaw_rad<math.pi/2 and abs(p.z-1)<1e-8 for p in frames))

    def test_disconnected_plan_rejected_without_teleport(self):
        with self.assertRaises(ValueError):
            execute_steps(path_steps([[0,1]],Pose(x=2)),initial=Pose(),
                          collider=BoxWorld(),render=lambda p:None,clock=Clock())

    def test_zero_path_stays_at_origin(self):
        result=execute_steps(path_steps([[0,0]]),initial=Pose(),collider=BoxWorld(),
                             render=lambda p:None,clock=Clock())
        self.assertEqual(result['pose'],Pose())
        self.assertEqual(result['distance_m'],0)

    def test_worker_old_default_radius_is_replaced(self):
        class Geometry:
            def check_path(self,points,*,robot_radius_m=.07071,**kwargs):
                return robot_radius_m
        worker=SimpleNamespace(GroundTruthGeometry=Geometry,SAFETY_MARGIN_M=.1)
        configure_worker(worker)
        self.assertAlmostEqual(Geometry().check_path([]),math.sqrt(2)*.10)
        self.assertEqual(worker.SAFETY_MARGIN_M,.05)
        self.assertAlmostEqual(worker.TOTAL_INFLATION_M,.05+math.sqrt(2)*.10)
        self.assertEqual(worker.CAMERA_HEIGHT_M,.5)
        self.assertEqual(worker.GEOMETRY_OBSTACLE_MAX_HEIGHT_M,.5)


if __name__=='__main__': unittest.main(verbosity=2)
