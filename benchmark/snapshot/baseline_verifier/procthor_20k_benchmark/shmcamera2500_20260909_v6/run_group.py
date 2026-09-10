"""One initialization, five frozen long routes, seven native online systems."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import numpy as np

sys.path.insert(0, os.environ['MIXED_ADAPTER_ROOT'])
sys.path.insert(0, os.environ['BEV_ORIGINAL_BENCHMARK_ROOT'])
sys.path.insert(0, os.environ['DATABUILDER_ROOT'])
# Keep THIS frozen episode/controller module ahead of old benchmark directories.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import procthor_benchmark_worker as w
from robot_contract import configure_worker, SPEC, SCHEMA
configure_worker(w)
from episode import METHODS
from hybrid_episode import run_episode, MODE, HORIZON_M
from protocol import CONFIG, stamp, require_stamp, digest, check_runtime
from sampling_errors import SceneSamplingError

check_runtime(w, SPEC, MODE, HORIZON_M, METHODS)


def create_procthor_scene(house, args, index):
    """An infeasible, unstarted scene is a sampling rejection, not a GPU fault."""
    from sampling_errors import legacy_sampling_failure
    try:
        # Scene.__init__ already closes its controller on initialization errors.
        return w.Scene(house, args, index)
    except RuntimeError as error:
        has_task = any((args.output/name).exists() for name in ('tasks.json', 'routes.json'))
        has_results = (args.output/'results.json').exists() or bool(list(args.output.glob('route_*')))
        # Exact existing classification only: renderer/model/driver errors and
        # scenes with any frozen/evaluated route remain fatal and preserved.
        if legacy_sampling_failure(repr(error), has_task, has_results):
            raise SceneSamplingError(f'ProcTHOR scene {index}: {error}') from error
        raise


def five_tasks(scene, rng, index, *, check_start_body=True, count=None, excluded_pairs=()):
    """Uniform random start and eligible endpoint, not longest-only sampling."""
    component = max(scene.graph.components(), key=len)
    tasks, pairs = [], set(excluded_pairs)
    sampling = CONFIG['sampling']
    count = sampling['routes_per_group'] if count is None else count
    from robot_contract import collider_for_scene
    from exact_executor import Pose
    collider = collider_for_scene(scene, w) if check_start_body else None
    minimum, maximum = sampling['route_length_m']
    for attempt in range(sampling['max_path_attempts_per_scene']):
        start = rng.choice(component)
        distances, parent = scene.graph.dijkstra(start)
        eligible = [g for g in component if minimum <= distances[g] <= maximum and (start,g) not in pairs]
        if not eligible: continue
        goal = rng.choice(eligible)
        route = w.reconstruct(parent, goal)
        subgoals = w.segment_gt_route(scene.graph, route, maximum_arc_m=sampling['upper_subgoal_arc_m'])
        origin, heading = scene.graph.points[start], scene.graph.points[route[1]]
        jitter = sampling['initial_yaw_jitter_deg']
        yaw = (math.degrees(math.atan2(heading[0]-origin[0], heading[2]-origin[2]))+rng.uniform(-jitter,jitter)) % 360
        task = dict(**stamp(), schema=CONFIG['task_schema'], scene_index=index,
            route_index=len(tasks), path_kind='long', start_index=start, goal_index=goal,
            start_world=origin, start_yaw_degrees=yaw, goal_world=scene.graph.points[goal],
            gt_shortest_path_m=distances[goal], gt_path_world=[scene.graph.points[i] for i in route],
            subgoals_world=[scene.graph.points[i] for i in subgoals], subgoal_count=len(subgoals),
            upper_guide='frozen same-floor GT graph route, <=2m visible chords',
            oracle_permissions=dict(current_local_metric_point_goal=True, native_nomad_goal_image=True,
                gt_bev=False, gt_depth=False, future_route=False, predicted_extrinsic_global_tracking=False))
        try:
            # EDT checks obstacle CELL CENTERS. Execution collides with the
            # complete cell area, so use its identical oriented-body test here.
            if collider is not None and collider.contact_at(
                    Pose(origin[0], origin[2], math.radians(yaw), scene.floor_y)):
                continue
            scene.validate_task_geometry(task)
            # The common guide itself must not cut through geometry.
            if not scene.geometry.check_path([origin]+task['subgoals_world']).safe: continue
        except (RuntimeError, ValueError):
            continue
        tasks.append(task); pairs.add((start,goal))
        if len(tasks) == count: return tasks
    raise SceneSamplingError(f'cannot sample {count} distinct collision-safe 10–25m routes in 2000 attempts')


def resume_tasks(scene, regenerated, args, index, asset_id):
    """Repair only pre-execution invalid starts, retaining every scored outcome.

    Original seeds/routes stay reproducible. A separate deterministic repair RNG
    neither changes method order nor other routes. The journal is written before
    tasks.json so an interrupted repair can be replayed without data loss.
    """
    from exact_executor import Pose
    from robot_contract import collider_for_scene
    file = args.output/'tasks.json'
    journal = args.output/'START_BODY_REPAIR.json'
    frozen = json.loads(file.read_text())
    regenerated = json.loads(json.dumps(regenerated))
    if journal.exists():
        audit = json.loads(journal.read_text())
        if audit['original_tasks'] != regenerated:
            raise RuntimeError('Start-repair journal does not match original seeded routes')
        updated = audit['updated_tasks']
        if digest(updated) != audit['updated_tasks_sha256'] or frozen not in (regenerated, updated):
            raise RuntimeError('Start-repair tasks/journal mismatch')
        changed = [i for i in range(len(frozen)) if regenerated[i] != updated[i]]
        if changed != audit['replaced_route_indices']:
            raise RuntimeError('Start repair modified an unaudited route')
        if frozen != updated:
            w.atomic_json(file, updated)
        frozen = updated
    elif frozen != regenerated:
        raise RuntimeError('Recovery tasks differ from frozen original routes')
    collider = collider_for_scene(scene, w)
    invalid = [i for i, task in enumerate(frozen) if collider.contact_at(Pose(
        task['start_world'][0], task['start_world'][2],
        math.radians(task['start_yaw_degrees']), scene.floor_y))]
    if not invalid:
        return frozen
    if journal.exists():
        raise RuntimeError('An already repaired route still has an invalid body start')
    # Never resample after any model was scored or started moving on this route.
    for i in invalid:
        route_dir = args.output/f'route_{i:02d}'
        if (list(route_dir.glob('*/result.json')) or
                list(route_dir.glob('*/trajectory*')) or
                any(p.is_file() for p in route_dir.glob('*/*'))):
            raise RuntimeError('Invalid start already has model evidence; manual audit required')
    updated = json.loads(json.dumps(frozen))
    repair_seed = args.seed + 0x5354415254
    repair_rng = random.Random(repair_seed)
    pairs = {(t['start_index'], t['goal_index']) for t in frozen}
    for i in invalid:
        replacement = five_tasks(scene, repair_rng, index, count=1, excluded_pairs=pairs)[0]
        replacement.update(route_index=i, source=args.source, asset_id=asset_id, seed=args.seed)
        updated[i] = replacement
        pairs.add((replacement['start_index'], replacement['goal_index']))
    audit = dict(reason='EDT cell-center clearance missed oriented body contact with full obstacle cell',
        repair_seed=repair_seed, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        original_tasks=frozen, updated_tasks=updated,
        original_tasks_sha256=digest(frozen), updated_tasks_sha256=digest(updated),
        replaced_route_indices=invalid, preexisting_model_evidence=False,
        policy='Only pre-execution invalid starts; same replacement for all seven models; no scored results removed')
    w.atomic_json(journal, audit)
    w.atomic_json(file, updated)
    return updated


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', required=True, choices=['hm3d','hssd','mp3d','procthor'])
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--gpu-index', required=True, type=int)
    p.add_argument('--resume', action='store_true', help='Explicit recovery of the same frozen tasks; never rerun completed episodes')
    p.add_argument('--geometry-only', action='store_true')
    p.add_argument('--interface-check', action='store_true', help='Two real predictions and bounded motion per model; not evaluation episodes')
    p.add_argument('--smoke', action='store_true', help='one route per method, never mark a full group COMPLETE')
    p.add_argument('--smoke-methods', default=','.join(METHODS))
    args=p.parse_args()
    recovery_id=str(time.time_ns())
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('terminated')))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        old_error=args.output/'INFRASTRUCTURE_ERROR.json'
        if old_error.exists():
            old_error.rename(args.output/f'INTERRUPTION_{recovery_id}.json')
    args.geometry_cache_root = Path(os.environ['REALTIME_OUTPUT'])/'geometry_cache'/args.source
    args.geometry_cache_root.mkdir(parents=True, exist_ok=True)
    started=time.monotonic(); rng=random.Random(args.seed); scene=None; routes_started=False
    try:
        if args.source=='procthor':
            import procthor_collection_core as core
            from procthor_gpu import configure, isolated_cloud_controller, assert_process_gpu_binding
            from procthor_collection_core import locate_ai2thor_runtime, nvidia_gpu_inventory
            args.procthor_runtime_root,_=locate_ai2thor_runtime()
            args.resolved_nvidia_gpu_uuid=nvidia_gpu_inventory()[args.gpu_index]
            binding=configure(args.procthor_runtime_root,args.gpu_index,args.resolved_nvidia_gpu_uuid)
            os.environ['REALTIME_UNITY_LOG_DIR']=str(args.output/'unity_logs'/str(time.time_ns()))
            # Scope override to this worker interpreter, not the shared collector.
            core.isolated_cloud_controller=isolated_cloud_controller
            core.assert_process_gpu_binding=assert_process_gpu_binding
            w.atomic_json(args.output/f'vulkan_binding_{time.time_ns()}.json',binding)
            index=rng.randrange(10000)
            house=w.load_houses(Path(os.environ['DATASET_DIR']), {index})[index]
            scene=create_procthor_scene(house,args,index); asset_id=str(index)
        else:
            from habitat_adapter import HabitatScene
            from collect_random_sessions import discover_scenes
            assets=discover_scenes(Path(os.environ['SCENES_ROOT'])/dict(hm3d='hm3d',hssd='hssd-hab',mp3d='mp3d')[args.source])
            if not assets: raise RuntimeError('no scene assets: '+args.source)
            index=rng.randrange(len(assets)); asset=assets[index]
            scene=HabitatScene(asset,args,args.seed); asset_id=asset.scene_id
        scene_meta=dict(**stamp(), source=args.source, asset_id=asset_id, embodiment=SPEC.metadata(),
            collision_geometry_schema=SCHEMA,
            seed=args.seed, scene_initializations=1, load_seconds=scene.load_seconds)
        if args.resume and (args.output/'scene.json').exists():
            previous=json.loads((args.output/'scene.json').read_text())
            require_stamp(previous)
            if any(previous[k]!=scene_meta[k] for k in ('source','asset_id','seed')):
                raise RuntimeError('Recovery scene differs from original asset/seed')
            w.atomic_json(args.output/f'resume_scene_{time.time_ns()}.json',scene_meta)
        else:
            w.atomic_json(args.output/'scene.json',scene_meta)
        resuming_tasks = args.resume and (args.output/'tasks.json').exists()
        sampler_file = args.output/'TASK_SAMPLER.json'
        # Reproduce old RNG consumption for old frozen groups. New groups use
        # the corrected sampler. Repairs use an independent RNG above.
        body_checked = not resuming_tasks or sampler_file.exists()
        if sampler_file.exists() and json.loads(sampler_file.read_text()) != {'revision':'exact_body_start_v1'}:
            raise RuntimeError('Unrecognized task sampler revision')
        tasks=five_tasks(scene,rng,index,check_start_body=body_checked)
        for task in tasks: task.update(source=args.source, asset_id=asset_id, seed=args.seed)
        if resuming_tasks:
            saved=np.load(args.output/'geometry.npz')
            if not (np.array_equal(saved['truth'],scene.geometry.truth) and
                    np.array_equal(saved['lower_bound'],scene.geometry.lower_bound) and
                    float(saved['voxel_size_m'])==scene.geometry.voxel_size_m):
                raise RuntimeError('Recovery geometry differs from frozen original geometry')
            tasks=resume_tasks(scene,tasks,args,index,asset_id)
        else:
            w.atomic_json(sampler_file, {'revision':'exact_body_start_v1'})
            w.atomic_json(args.output/'tasks.json', tasks)
            np.savez_compressed(args.output/'geometry.npz', truth=scene.geometry.truth,
                lower_bound=scene.geometry.lower_bound, voxel_size_m=scene.geometry.voxel_size_m,
                obstacle_min_height_m=w.GEOMETRY_OBSTACLE_MIN_HEIGHT_M,
                obstacle_max_height_m=w.GEOMETRY_OBSTACLE_MAX_HEIGHT_M,
                floor_y=scene.floor_y, embodiment_json=json.dumps(SPEC.metadata()))
        if args.geometry_only:
            w.atomic_json(args.output/'GEOMETRY_OK.json', dict(route_count=5, scene_initializations=1,
                lengths_m=[t['gt_shortest_path_m'] for t in tasks])); return
        # Pre-render immutable NoMaD endpoint images once, never in timed motion.
        goals=[]; before=time.monotonic()
        for task in tasks:
            goals.append([w.encode_jpeg(scene.rgb(point,w.subgoal_image_yaw_degrees(task,i)))
                for i,point in enumerate(task['subgoals_world'])])
        goal_image_seconds=time.monotonic()-before
        if args.interface_check:
            from interface_acceptance import probe_scene
            probe_scene(scene,tasks[0],goals[0],args.output)
            return
        results=[dict(task=task, systems={}) for task in tasks]
        methods=args.smoke_methods.split(',') if args.smoke else list(METHODS)
        # Rotate method order reproducibly across scenes; all share same five routes.
        rng.shuffle(methods)
        routes_started=True
        for method in methods:
            warmed=False
            for i,task in enumerate(tasks[:1] if args.smoke else tasks):
                episode_dir=args.output/f'route_{i:02d}'/method
                if args.resume:
                    from resume_support import completed_or_archive
                    existing=completed_or_archive(episode_dir,method,task)
                    if existing is not None:
                        results[i]['systems'][method]=existing
                        continue
                w.atomic_json(args.output/'current.json', dict(method=method, route=i,
                    elapsed_s=time.monotonic()-started, load_seconds=scene.load_seconds))
                results[i]['systems'][method]=run_episode(scene,task,method,
                    os.environ['SINGLE_URL'],os.environ['BASELINE_URL'],
                    episode_dir,goals[i],warmup=(not warmed))
                results[i]['systems'][method].update(**stamp(), task_sha256=digest(task))
                w.atomic_json(episode_dir/'result.json', results[i]['systems'][method])
                warmed=True
                w.atomic_json(args.output/'results.json', results)
        final=dict(**stamp(), complete=True,route_count=1 if args.smoke else 5,
            scene_initializations=1,load_seconds=scene.load_seconds,
            goal_image_seconds=goal_image_seconds,group_wall_seconds=time.monotonic()-started,
            methods=methods,results=results[:1] if args.smoke else results)
        w.atomic_json(args.output/('SMOKE_COMPLETE.json' if args.smoke else 'COMPLETE.json'),final)
    except SceneSamplingError as error:
        if routes_started: raise
        w.atomic_json(args.output/'REJECTED.json',dict(reason=str(error), stage='before_routes',
            replacement_allowed=True, counts_as_model_failure=False))
    except BaseException as error:
        w.atomic_json(args.output/'INFRASTRUCTURE_ERROR.json',dict(error=repr(error), routes_started=routes_started,
            recovery=args.resume, recovery_id=recovery_id))
        raise
    finally:
        if scene is not None: scene.stop()


if __name__=='__main__': main()
