"""Habitat RGB scene adapter; complete geometry stays evaluator-only."""
from __future__ import annotations
import math
import time
from pathlib import Path
import numpy as np
import quaternion
import procthor_benchmark_worker as w
import collect_random_sessions as collector
from web_navigation import render_full_obstacle_map
from sampling_errors import SceneSamplingError


class HabitatScene:
    def __init__(self, asset, args, seed):
        started = time.monotonic()
        self.sim = None
        try:
            self.sim = collector.make_simulator(asset, camera_width=w.CAMERA_WIDTH,
                camera_height=w.CAMERA_HEIGHT, horizontal_fov_degrees=w.HORIZONTAL_FOV_DEGREES,
                sensor_height_m=w.CAMERA_HEIGHT_M, gpu_device_id=0)
            collector.prepare_navmesh(self.sim, asset, args.geometry_cache_root / 'navmesh')
            pf = self.sim.pathfinder
            pf.seed(seed)
            origin = np.asarray(pf.get_random_navigable_point())
            if not np.isfinite(origin).all():
                raise SceneSamplingError('no navigable floor')
            self.floor_y = float(origin[1])
            lower = np.asarray(pf.get_bounds()[0], dtype=float)
            nav = pf.get_topdown_view(w.GEOMETRY_VOXEL_SIZE_M, self.floor_y)
            truth, valid, stats = render_full_obstacle_map(self.sim,
                floor_height=self.floor_y, meters_per_pixel=w.GEOMETRY_VOXEL_SIZE_M,
                rows=nav.shape[0], columns=nav.shape[1], lower_bound=lower,
                obstacle_min_height=w.GEOMETRY_OBSTACLE_MIN_HEIGHT_M,
                obstacle_max_height=w.GEOMETRY_OBSTACLE_MAX_HEIGHT_M, navigable_map=nav)
            # Habitat is right-handed; the existing THOR evaluator uses +Z
            # forward with +X camera-right. Reflect world Z, never mirror RGB.
            benchmark_lower=lower.copy()
            benchmark_lower[2]=-(lower[2]+(truth.shape[0]-1)*w.GEOMETRY_VOXEL_SIZE_M)
            self.geometry = w.GroundTruthGeometry.create(np.flipud(truth), benchmark_lower,
                                                        w.GEOMETRY_VOXEL_SIZE_M)
            reachable = []
            coarse = pf.get_topdown_view(w.GRID_SIZE_M, self.floor_y)
            for r, c in np.argwhere(coarse):
                point = [lower[0]+int(c)*w.GRID_SIZE_M, self.floor_y,
                         lower[2]+int(r)*w.GRID_SIZE_M]
                snapped = np.asarray(pf.snap_point(point))
                if not np.isfinite(snapped).all() or abs(snapped[1]-self.floor_y) > .05:
                    continue
                if np.linalg.norm(snapped[[0,2]]-np.asarray(point)[[0,2]]) > .03:
                    continue
                point[2] = -point[2]
                if self.geometry.check_path([point]).safe:
                    reachable.append(dict(x=point[0], y=point[1], z=point[2]))
            try:
                self.graph = w.ReachableGraph.build(reachable)
            except RuntimeError as error:
                if str(error) == 'reachable graph has no useful connected component':
                    raise SceneSamplingError(str(error)) from error
                raise
            self.graph.edges = [[(j,d) for j,d in edges if self.geometry.check_path(
                [self.graph.points[i],self.graph.points[j]]).safe]
                for i,edges in enumerate(self.graph.edges)]
            if max(map(len,self.graph.components())) < 20:
                raise SceneSamplingError('no collision-safe same-floor component')
            self.load_seconds = time.monotonic()-started
            self.geometry_statistics = stats
        except BaseException:
            self.stop()
            raise

    def rgb(self, position, yaw_degrees):
        state = self.sim.get_agent(0).get_state()
        state.position = np.asarray([position[0], self.floor_y, -position[2]],dtype=np.float32)
        # Shared yaw 0 = +Z becomes Habitat -Z; positive yaw turns camera right.
        state.rotation = quaternion.from_rotation_vector([0, -math.radians(yaw_degrees), 0])
        self.sim.get_agent(0).set_state(state)
        return np.asarray(self.sim.get_sensor_observations()['camera_sensor'])[...,:3].copy()

    def validate_task_geometry(self, task):
        if not self.geometry.check_path(task['gt_path_world']).safe:
            raise ValueError('GT route intersects complete geometry')
        last = task['start_world']
        for goal in task['subgoals_world']:
            if not self.geometry.check_path([last,goal]).safe:
                raise ValueError('GT subgoal chord intersects complete geometry')
            last = goal

    def stop(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
