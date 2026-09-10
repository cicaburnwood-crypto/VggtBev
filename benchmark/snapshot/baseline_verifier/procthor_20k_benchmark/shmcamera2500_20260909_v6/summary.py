"""One-shot new-protocol statistics; incompatible old results are never pooled."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from protocol import CONFIG, METHODS, stamp, verify_root, require_stamp, validate_result


def stats(rows):
    successes = [r for r in rows if r['success']]
    def avg(key, population):
        values = [r[key] for r in population if r.get(key) is not None]
        return statistics.fmean(values) if values else None
    return dict(episodes=len(rows), successes=len(successes),
        success_rate=len(successes)/len(rows) if rows else None,
        total_episode_wall_seconds=sum(r['total_wall_seconds'] for r in rows),
        total_resource_seconds=sum(r['resource_wall_seconds'] for r in rows),
        successful_wall_seconds_mean=avg('total_wall_seconds', successes),
        all_episode_wall_seconds_mean=avg('total_wall_seconds', rows),
        inference_hz_mean=avg('effective_replanning_hz', rows),
        raw_length_ratio_own_successes_mean=avg('raw_executed_path_ratio', successes),
        spl_mean=avg('spl', rows),
        time_components_seconds={k:sum(r.get(k, 0.) for r in rows) for k in
            ('warmup_seconds', 'translating_seconds', 'rotating_seconds',
             'model_request_wall_seconds', 'model_inference_seconds',
             'planner_seconds', 'render_seconds')},
        failures=dict(Counter(r['status'] for r in rows if not r['success'])))


def summarize(root):
    root = Path(root)
    groups_expected = sum(CONFIG['sampling']['groups_per_source'].values())
    targets = dict(groups=groups_expected,
        routes=groups_expected*CONFIG['sampling']['routes_per_group'],
        episodes=groups_expected*CONFIG['sampling']['routes_per_group']*len(METHODS))
    if not root.exists():
        return dict(**stamp(), state='not_started', output_exists=False, targets=targets,
                    completed_episodes=0, eta_seconds=None,
                    note='No new formal result directory; acceptance tests are not benchmark data.')
    verify_root(root)
    records = {m:[] for m in METHODS}
    sources = defaultdict(lambda: defaultdict(list))
    routes = defaultdict(dict)
    initializations = 0
    init_s = 0.
    scene_files = list(root.glob('groups/group_*/attempt_*/scene.json'))
    resumed = list(root.glob('groups/group_*/attempt_*/resume_scene_*.json'))
    for file in scene_files+resumed:
        row = json.loads(file.read_text()); require_stamp(row)
        initializations += 1; init_s += row['load_seconds']
    tasks_cache = {}
    pending = 0
    for file in root.glob('groups/group_*/attempt_*/route_*/*/result.json'):
        row = json.loads(file.read_text())
        # Core writes before the group stamps: not yet a committed formal row.
        if 'protocol_id' not in row and row.get('execution_mode') in {
                CONFIG['execution_mode'], 'wall_clock_bounded_camera_native_polyline_v2'}:
            pending += 1
            continue
        attempt = file.parents[2]
        if attempt not in tasks_cache:
            tasks_cache[attempt] = json.loads((attempt/'tasks.json').read_text())
        task = tasks_cache[attempt][int(file.parents[1].name.split('_')[-1])]
        require_stamp(task)
        method = file.parent.name
        validate_result(row, method, task)
        if not (file.parent/'trajectory.npz').is_file():
            raise RuntimeError(f'Missing trajectory: {file}')
        records[method].append(row)
        sources[task['source']][method].append(row)
        routes[str(file.parents[1])][method] = row
    complete = [r for r in routes.values() if set(r) == set(METHODS)]
    eligible = [r for r in complete if any(x['success'] for x in r.values())]
    common = [r for r in complete if all(x['success'] for x in r.values())]
    finished_groups = list(root.glob('groups/group_*/COMPLETE.json'))
    for file in finished_groups:
        require_stamp(json.loads(file.read_text()))
    count = sum(map(len, records.values()))
    # Process liveness is deliberately not inferred from files alone.
    return dict(**stamp(), state='results_present' if count else 'prepared_no_results',
        process_liveness='must_be_checked_separately', targets=targets,
        completed_episodes=count, complete_scene_groups=len(finished_groups),
        fully_compared_routes=len(complete), any_success_routes=len(eligible),
        all_failed_routes=len(complete)-len(eligible), all_seven_success_routes=len(common),
        scene_initializations=initializations, initialization_seconds=init_s,
        recovery_initializations=len(resumed), pending_result_commits=pending,
        interrupted_episode_archives=len(list(root.glob('groups/group_*/attempt_*/interrupted_episodes/*'))),
        model_stats={m:stats(v) for m,v in records.items()},
        paired_fully_compared_stats={m:stats([r[m] for r in complete]) for m in METHODS},
        conditional_any_success_stats={m:stats([r[m] for r in eligible]) for m in METHODS},
        paired_all_success_stats={m:stats([r[m] for r in common]) for m in METHODS},
        source_stats={s:{m:stats(v) for m,v in models.items()} for s,models in sources.items()},
        eta_seconds=None,
        warnings=['Raw length ratios use a conservative discrete GT graph and 20cm acceptance; not clamped.',
                  'OmniVLA/MBRA/NoMaD use native cmd_vel; the other four use the external path executor.',
                  'NoMaD has ImageGoal input; do not call all seven equal-input PointGoal models.',
                  'Inference/planner time are components of request time; do not double count them.',
                  'A speed ranking must use paired successes; fast failures are not fast successful navigation.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
