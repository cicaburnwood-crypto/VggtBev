"""Five GT-route executor acceptance trials; no models, no benchmark scores."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image

from exact_executor import Pose, path_steps, execute_steps, wrap
from robot_contract import configure_worker, collider_for_scene, SPEC, SCHEMA


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--scene-index',type=int,default=7390)
    p.add_argument('--source',choices=['procthor','hm3d','hssd','mp3d'],default='procthor')
    p.add_argument('--asset-id')
    p.add_argument('--seed',type=int,default=2026090900)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(Path(__file__).resolve().parent),
                  os.environ['BEV_ORIGINAL_BENCHMARK_ROOT'],os.environ['DATABUILDER_ROOT'],
                  os.environ['BEV_ORIGINAL_BENCHMARK_ROOT']+'/mixed500_20260909']
    import procthor_benchmark_worker as w
    configure_worker(w)
    options=SimpleNamespace(gpu_index=args.gpu,geometry_cache_root=args.output/'geometry_cache')
    options.geometry_cache_root.mkdir()
    scene=None
    try:
        if args.source=='procthor':
            import procthor_collection_core as core
            from procthor_gpu import configure,isolated_cloud_controller,assert_process_gpu_binding
            runtime,_=core.locate_ai2thor_runtime()
            uuid=core.nvidia_gpu_inventory()[args.gpu]
            binding=configure(runtime,args.gpu,uuid)
            os.environ['REALTIME_UNITY_LOG_DIR']=str(args.output/'unity')
            core.isolated_cloud_controller=isolated_cloud_controller
            core.assert_process_gpu_binding=assert_process_gpu_binding
            house=w.load_houses(Path(os.environ['DATASET_DIR']),{args.scene_index})[args.scene_index]
            options.procthor_runtime_root=runtime;options.resolved_nvidia_gpu_uuid=uuid
            scene=w.Scene(house,options,args.scene_index)
        else:
            from collect_random_sessions import discover_scenes
            from habitat_adapter import HabitatScene
            assets=discover_scenes(Path(os.environ['SCENES_ROOT'])/
                dict(hm3d='hm3d',hssd='hssd-hab',mp3d='mp3d')[args.source])
            asset=next(a for a in assets if a.scene_id==args.asset_id)
            binding=dict(gpu_index=args.gpu,uuid=os.environ['CUDA_VISIBLE_DEVICES'],
                         renderer='Habitat EGL logical GPU0 with UUID mask and external guard')
            scene=HabitatScene(asset,options,args.seed)
        w.atomic_json(args.output/'binding.json',binding)
        collider=collider_for_scene(scene,w)
        # GT-only acceptance route generation: conservative circle clearance
        # ensures an oriented square can rotate at every route vertex.
        radius=SPEC.circumradius_m+scene.geometry.voxel_size_m*math.sqrt(2)/2
        valid=[scene.geometry.check_path([p],robot_radius_m=radius).safe for p in scene.graph.points]
        scene.graph.edges=[[(j,d) for j,d in edges if valid[i] and valid[j] and
            scene.geometry.check_path([scene.graph.points[i],scene.graph.points[j]],robot_radius_m=radius).safe]
            for i,edges in enumerate(scene.graph.edges)]
        component=max(scene.graph.components(),key=len)
        rng=random.Random(2026090918)
        tasks=[]
        for _ in range(200):
            start=rng.choice(component)
            distances,parent=scene.graph.dijkstra(start)
            eligible=[g for g in component if 2<=distances[g]<=6]
            if not eligible: continue
            goal=rng.choice(eligible)
            route=w.reconstruct(parent,goal)
            points=np.asarray([scene.graph.points[i] for i in route])
            tangent=math.atan2(points[1,0]-points[0,0],points[1,2]-points[0,2])
            yaw=tangent+[0.,math.pi/2,-math.pi/2,math.pi,0.][len(tasks)]
            tasks.append(dict(world=points.tolist(),yaw_rad=yaw,length_m=float(distances[goal])))
            if len(tasks)==5: break
        if len(tasks)!=5: raise RuntimeError('GT setup cannot provide five valid acceptance routes')
        w.atomic_json(args.output/'tasks.json',tasks)
        np.savez_compressed(args.output/'geometry.npz',truth=scene.geometry.truth,
            lower_bound=scene.geometry.lower_bound,voxel_size_m=scene.geometry.voxel_size_m,
            floor_y=scene.floor_y,obstacle_min_height_m=w.GEOMETRY_OBSTACLE_MIN_HEIGHT_M,
            obstacle_max_height_m=w.GEOMETRY_OBSTACLE_MAX_HEIGHT_M)
        rows=[]
        for i,task in enumerate(tasks):
            out=args.output/f'route_{i:02d}';out.mkdir()
            points=np.asarray(task['world'])
            pose=Pose(float(points[0,0]),float(points[0,2]),task['yaw_rad'],scene.floor_y)
            delta=points[:,[0,2]]-pose.xz
            c,s=math.cos(pose.yaw_rad),math.sin(pose.yaw_rad)
            local=np.column_stack([c*delta[:,0]-s*delta[:,1],s*delta[:,0]+c*delta[:,1]])
            frame_count=0;first=None;changed=False;heights=[];last_rgb=None
            def render(pose):
                nonlocal frame_count,first,changed,last_rgb
                rgb=scene.rgb([pose.x,pose.base_y,pose.z],math.degrees(pose.yaw_rad))
                if args.source=='procthor':
                    cameras=scene.controller.last_event.metadata.get('thirdPartyCameras') or []
                    if not cameras: raise RuntimeError('Missing actual renderer camera-pose metadata')
                    actual=cameras[0]['position']
                    actual_yaw=math.radians(float(cameras[0]['rotation']['y']))
                else:
                    import quaternion
                    sensor=scene.sim.get_agent(0).get_state().sensor_states['camera_sensor']
                    actual=dict(x=float(sensor.position[0]),y=float(sensor.position[1]),z=-float(sensor.position[2]))
                    forward=quaternion.rotate_vectors(sensor.rotation,np.array([0.,0.,-1.]))
                    actual_yaw=math.atan2(float(forward[0]),-float(forward[2]))
                height=float(actual['y'])-scene.floor_y
                heights.append(height)
                if abs(height-.5)>1e-5: raise RuntimeError(f'Actual camera height is {height}, not .5')
                if abs(float(actual['x'])-pose.x)>1e-5 or abs(float(actual['z'])-pose.z)>1e-5:
                    raise RuntimeError('Renderer camera did not follow commanded translation')
                if abs(wrap(actual_yaw-pose.yaw_rad))>1e-5:
                    raise RuntimeError('Renderer yaw did not follow bounded executor heading')
                if first is None: first=rgb.copy()
                else: changed=changed or not np.array_equal(first,rgb)
                if frame_count%25==0: Image.fromarray(rgb).save(out/f'rgb_{frame_count:04d}.jpg',quality=95)
                frame_count+=1;last_rgb=rgb
            w.atomic_json(args.output/'current.json',dict(route=i,stage='executing_GT_route',robot=SPEC.metadata()))
            result=execute_steps(path_steps(local,pose,SPEC,allow_reverse=True),initial=pose,
                                 collider=collider,render=render,clock=time,spec=SPEC)
            endpoint=float(np.linalg.norm(result['pose'].xz-points[-1,[0,2]]))
            path_errors=[]
            for row in result['trace']:
                index=row['vertex'];point=np.array([row['x'],row['z']])
                a=points[max(0,index-1),[0,2]];b=points[index,[0,2]]
                ab=b-a;den=float(ab@ab)
                t=0. if den<1e-18 else np.clip(float((point-a)@ab)/den,0,1)
                path_errors.append(float(np.linalg.norm(point-(a+t*ab))))
            metrics=dict(status=result['status'],endpoint_error_m=endpoint,
                cross_track_max_m=max(path_errors,default=0),travelled_m=result['distance_m'],
                gt_command_length_m=task['length_m'],motion_seconds=result['motion_s'],
                wall_seconds=result['wall_s'],absolute_rotation_rad=result['rotation_rad'],
                max_speed_m_s=max((r['ds']/r['dt'] for r in result['trace'] if r['dt']>0),default=0),
                max_yaw_deg_s=max((abs(math.degrees(r['dyaw']/r['dt'])) for r in result['trace'] if r['dt']>0),default=0),
                rgb_frames=frame_count,rgb_changed=changed,camera_height_min_m=min(heights),
                camera_height_max_m=max(heights),robot=SPEC.metadata())
            w.atomic_json(out/'result.json',metrics)
            Image.fromarray(last_rgb).save(out/'rgb_terminal.jpg',quality=95)
            np.savez_compressed(out/'trajectory.npz',**{k:np.asarray([r[k] for r in result['trace']])
                for k in result['trace'][0]})
            rows.append(metrics)
            print(json.dumps(dict(route=i,**metrics)),flush=True)
            if not (result['status']=='complete' and endpoint<1e-7 and max(path_errors,default=0)<1e-7 and changed):
                raise RuntimeError('GT executor route did not meet acceptance thresholds')
        w.atomic_json(args.output/'PASS.json',dict(schema=SCHEMA,scene_index=args.scene_index,
            source=args.source,asset_id=args.asset_id or str(args.scene_index),
            robot=SPEC.metadata(),routes=rows,model_inference=False,counts_as_benchmark=False,
            geometry_statistics=scene.geometry_statistics,binding=binding))
        print('EXECUTOR_SIMULATOR_PASS',flush=True)
    except BaseException as error:
        w.atomic_json(args.output/'FAILED.json',dict(error=repr(error),robot=SPEC.metadata()))
        raise
    finally:
        if scene is not None: scene.stop()


if __name__=='__main__': main()
