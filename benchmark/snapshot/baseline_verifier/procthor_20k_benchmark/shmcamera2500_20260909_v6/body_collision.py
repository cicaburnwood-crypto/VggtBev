"""Evaluator-only upright cuboid collision, including rotation in place.

Translation uses continuous swept separating-axis tests against axis-aligned
solid boxes. Rotation uses conservative interval subdivision with a documented
<=0.1mm boundary uncertainty, NOT endpoint-only sampling. No safety margin is
added to the robot. Raster input must represent the actual 0--0.50m body band;
old 0.03--1.40m BEV caches are explicitly refused.
"""
import math
import numpy as np
from exact_executor import RobotSpec, Pose, MotionStep, validate_step


class BoxWorld:
    """Input boxes have shape [N,2,3], ordered min/max world x,y,z (metres)."""
    def __init__(self, boxes=(), spec=RobotSpec()):
        self.spec = spec
        self.boxes = np.asarray(boxes, dtype=float).reshape(-1, 2, 3)
        if not np.isfinite(self.boxes).all() or np.any(self.boxes[:,1] <= self.boxes[:,0]):
            raise ValueError('Invalid collision boxes')
        self.sweep_calls = 0

    def _boxes_near(self, pose, radius, end=None):
        end = end or pose
        b = self.boxes
        if not len(b): return b
        low = np.minimum(pose.xz, end.xz)-radius
        high = np.maximum(pose.xz, end.xz)+radius
        # The supporting floor at y=base_y is not itself an obstacle. Any
        # positive overlap above that plane counts, including millimetre bumps.
        active = ((b[:,1,1] > pose.base_y+1e-9) &
                  (b[:,0,1] <= pose.base_y+self.spec.height_m+1e-12) &
                  (b[:,1,0] >= low[0]) & (b[:,0,0] <= high[0]) &
                  (b[:,1,2] >= low[1]) & (b[:,0,2] <= high[1]))
        return b[active]

    def _axes(self, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[1.,0.],[0.,1.],[c,-s],[s,c]])

    def _intervals(self, pose, boxes, padding=0.):
        axes = self._axes(pose.yaw_rad)
        center = (boxes[:,0][:,[0,2]]+boxes[:,1][:,[0,2]])/2
        half = (boxes[:,1][:,[0,2]]-boxes[:,0][:,[0,2]])/2
        right, forward = axes[2:]
        body = ((self.spec.width_m/2+padding)*np.abs(axes @ right) +
                (self.spec.length_m/2+padding)*np.abs(axes @ forward))
        radii = half @ np.abs(axes).T + body
        separation = (pose.xz-center) @ axes.T
        return axes, separation, radii

    def _touch(self, pose, boxes, padding=0.):
        if not len(boxes): return False
        _, d, r = self._intervals(pose, boxes, padding)
        return bool(np.any(np.all(np.abs(d) <= r+1e-12, axis=1)))

    def contact_at(self, pose):
        return self._touch(pose, self._boxes_near(pose, self.spec.circumradius_m))

    def sweep(self, step):
        """First contact fraction of THIS <=20ms tick, or None when clear."""
        validate_step(step, self.spec)
        self.sweep_calls += 1
        boxes = self._boxes_near(step.start, self.spec.circumradius_m, step.end)
        if not len(boxes): return None
        if self._touch(step.start, boxes): return 0.
        if step.distance_m > 1e-12:
            axes, d, r = self._intervals(step.start, boxes)
            velocity = (step.end.xz-step.start.xz) @ axes.T
            enter, leave = np.zeros(len(boxes)), np.ones(len(boxes))
            possible = np.ones(len(boxes), dtype=bool)
            for axis, rate in enumerate(velocity):
                if abs(rate) < 1e-15:
                    possible &= np.abs(d[:,axis]) <= r[:,axis]+1e-12
                else:
                    a = (-r[:,axis]-d[:,axis])/rate
                    b = (r[:,axis]-d[:,axis])/rate
                    enter = np.maximum(enter, np.minimum(a,b))
                    leave = np.minimum(leave, np.maximum(a,b))
            possible &= (enter <= leave+1e-12) & (leave >= 0) & (enter <= 1)
            return float(np.clip(np.min(enter[possible]),0,1)) if np.any(possible) else None

        radius = self.spec.circumradius_m
        # A midpoint box expanded by r*dtheta/2 encloses every orientation
        # within this interval (Euclidean displacement bound for all body points).
        def earliest(lo, hi):
            mid = (lo+hi)/2
            bound = radius * abs(step.yaw_delta_rad) * (hi-lo)/2
            if not self._touch(step.at(mid), boxes, padding=bound):
                return None
            if self._touch(step.at(lo), boxes): return lo
            if 2*bound <= self.spec.collision_tolerance_m:
                return lo  # conservative contact bracket, at most tolerance early
            left = earliest(lo, mid)
            return earliest(mid, hi) if left is None else left

        return earliest(0., 1.)


class RasterBodyWorld(BoxWorld):
    """Exact square-vs-cell evaluation of a height-projected voxel geometry.

    Raster lower_bound denotes CELL CENTERS, matching the benchmark voxelizer;
    array row zero is highest world Z. All nonwhite cells and outside-map space
    are blocked. Height projection is valid because the body is upright,
    fixed-height and moves only on one floor; it is NOT a camera visibility map.
    """
    def __init__(self, truth, lower_bound, voxel_size_m, *, floor_y,
                 obstacle_min_height_m, obstacle_max_height_m, spec=RobotSpec()):
        super().__init__((), spec)
        self.truth = np.asarray(truth)
        self.lower = np.asarray(lower_bound, dtype=float)
        self.voxel = float(voxel_size_m)
        self.floor_y = float(floor_y)
        if (self.truth.ndim != 2 or not self.truth.size or self.lower.shape != (3,) or
            not np.isfinite(self.lower).all() or not math.isfinite(self.voxel) or self.voxel <= 0):
            raise ValueError('Invalid metric raster')
        if not 0 <= obstacle_min_height_m <= 1e-5:
            raise ValueError('Collision band must include the body down to its floor plane')
        if abs(obstacle_max_height_m-spec.height_m) > 1e-8:
            raise ValueError('Collision cache height does not match robot body; rebuild it')
        self.height_band = (float(obstacle_min_height_m), float(obstacle_max_height_m))

    def _boxes_near(self, pose, radius, end=None):
        if abs(pose.base_y-self.floor_y) > 1e-8:
            raise ValueError('Raster collision floor mismatch')
        end = end or pose
        low = np.minimum(pose.xz,end.xz)-radius
        high = np.maximum(pose.xz,end.xz)+radius
        h,w = self.truth.shape
        v = self.voxel
        # Boundaries lie half a cell outside the first/last cell center.
        xmin,zmin = self.lower[[0,2]]-v/2
        xmax,zmax = xmin+w*v,zmin+h*v
        c0,c1 = max(0,int(math.floor((low[0]-xmin)/v))), min(w,int(math.floor((high[0]-xmin)/v))+1)
        q0,q1 = max(0,int(math.floor((low[1]-zmin)/v))), min(h,int(math.floor((high[1]-zmin)/v))+1)
        boxes = []
        if c0 < c1 and q0 < q1:
            rows,cols = np.nonzero(self.truth[h-q1:h-q0,c0:c1] != 255)
            xs = xmin+(cols+c0)*v
            zs = zmin+(q1-1-rows)*v
            a = np.column_stack([xs,np.full(len(xs),self.floor_y+1e-8),zs])
            b = a+np.array([v,self.spec.height_m-1e-8,v])
            boxes.extend(np.stack([a,b],axis=1))
        # Only local strips are needed to represent the unbounded exterior.
        margin = radius+v
        ly,uy = self.floor_y+1e-8,self.floor_y+self.spec.height_m
        if low[0] <= xmin:
            boxes.append([[low[0]-margin,ly,low[1]-margin],[xmin,uy,high[1]+margin]])
        if high[0] >= xmax:
            boxes.append([[xmax,ly,low[1]-margin],[high[0]+margin,uy,high[1]+margin]])
        if low[1] <= zmin:
            boxes.append([[low[0]-margin,ly,low[1]-margin],[high[0]+margin,uy,zmin]])
        if high[1] >= zmax:
            boxes.append([[low[0]-margin,ly,zmax],[high[0]+margin,uy,high[1]+margin]])
        return np.asarray(boxes,dtype=float).reshape(-1,2,3)
