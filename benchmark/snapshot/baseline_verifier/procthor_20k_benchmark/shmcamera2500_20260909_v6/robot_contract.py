"""One explicit embodiment contract applied BEFORE creating a simulator.

This only changes the imported worker in the current process. Legacy collectors,
checkpoints, old benchmark scripts and old results remain untouched.
"""
from exact_executor import RobotSpec

SPEC = RobotSpec()
SCHEMA = 'bounded_camera_cuboid_20x20x50cm_v6'
SAFETY_MARGIN_M = 0.05


def configure_worker(worker):
    worker.CAMERA_HEIGHT_M = SPEC.camera_height_m
    worker.SAFETY_MARGIN_M = SAFETY_MARGIN_M
    worker.ROBOT_SIDE_M = SPEC.width_m
    worker.ROBOT_COLLISION_RADIUS_M = SPEC.circumradius_m
    worker.TOTAL_INFLATION_M = worker.SAFETY_MARGIN_M + SPEC.circumradius_m
    # Exclude the mathematical support plane only, not 3cm of real obstacles.
    worker.GEOMETRY_OBSTACLE_MIN_HEIGHT_M = 1e-6
    worker.GEOMETRY_OBSTACLE_MAX_HEIGHT_M = SPEC.height_m
    worker.GEOMETRY_CACHE_SCHEMA = SCHEMA
    # Existing function defaults captured the old 10cm radius at import time.
    # Change the runtime wrapper explicitly; changing the constant alone is NOT enough.
    cls = worker.GroundTruthGeometry
    if not getattr(cls, '_bounded_body_contract', False):
        original = cls.check_path
        def conservative_upper_guide(self, points, *, robot_radius_m=None, **kwargs):
            return original(self, points, robot_radius_m=(SPEC.circumradius_m
                            if robot_radius_m is None else robot_radius_m), **kwargs)
        cls.check_path = conservative_upper_guide
        cls._bounded_body_contract = True
    return SPEC


def collider_for_scene(scene, worker):
    from body_collision import RasterBodyWorld
    if abs(worker.CAMERA_HEIGHT_M-SPEC.camera_height_m) > 1e-9:
        raise RuntimeError('Camera configuration not applied before scene initialization')
    g = scene.geometry
    return RasterBodyWorld(g.truth, g.lower_bound, g.voxel_size_m,
        floor_y=scene.floor_y, obstacle_min_height_m=worker.GEOMETRY_OBSTACLE_MIN_HEIGHT_M,
        obstacle_max_height_m=worker.GEOMETRY_OBSTACLE_MAX_HEIGHT_M, spec=SPEC)
