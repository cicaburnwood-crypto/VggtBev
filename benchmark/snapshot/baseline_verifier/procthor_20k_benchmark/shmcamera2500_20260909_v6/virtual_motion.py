"""Exact polyline execution: no controller, shortcut, acceleration or GT input."""
import math
import numpy as np

SPEED_M_S = 1.0
HORIZON_M = 0.5
STEP_M = 0.02


def path_steps(path, horizon_m=HORIZON_M, step_m=STEP_M):
    """Yield local positions, tangent yaw and arc increment; retain all corners.

    A predicted first waypoint is connected to the observation origin, never
    teleported to. Only duplicate points are removed. Camera yaw is the common
    path tangent (instant ideal rotation); predicted angular velocities and
    SE(2) waypoint headings are not executed in this planar-planner experiment.
    """
    p=np.asarray(path,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or not len(p) or not np.isfinite(p).all():
        raise ValueError('invalid_native_path')
    if horizon_m<=0 or step_m<=0: raise ValueError('invalid execution distance')
    previous=np.zeros(2); arc=0.
    for target in p:
        delta=target-previous; length=float(np.linalg.norm(delta))
        if length<1e-10: continue
        direction=delta/length; yaw=math.atan2(direction[0],direction[1])
        traversed=0.
        while traversed<length-1e-10 and arc<horizon_m-1e-10:
            ds=min(step_m,length-traversed,horizon_m-arc)
            traversed+=ds; arc+=ds
            yield (previous+direction*traversed).tolist(),yaw,ds
        previous=target
        if arc>=horizon_m-1e-10: return


def local_to_world(point, origin, yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return [origin[0]+c*point[0]+s*point[1],
            origin[1]-s*point[0]+c*point[1]]
