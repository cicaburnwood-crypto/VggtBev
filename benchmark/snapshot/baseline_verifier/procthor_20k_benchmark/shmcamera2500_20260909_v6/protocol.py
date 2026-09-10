"""CPU-only protocol fingerprints and fail-closed run/resume boundaries."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/'protocol.json').read_text())
METHODS = CONFIG['methods']
CODE_FILES = ('exact_executor.py', 'body_collision.py', 'robot_contract.py',
    'virtual_episode.py', 'episode.py', 'navigation_baseline_runtime.py',
    'hybrid_episode.py', 'native_velocity_episode.py', 'native_motion.py', 'execution_policy.py',
    'native_platform_controls.py', 'run_group.py', 'dispatch.py', 'procthor_gpu.py',
    'resume_support.py', 'protocol.py', 'summary.py', 'launch.sh', 'interface_acceptance.py',
    'shm_runtime.py', 'shm_contract.py', 'predicted_navigation.py', 'recovery_views.py',
    'planner_backends.py', 'shm_support.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def stamp():
    return dict(protocol_id=CONFIG['protocol_id'], protocol_sha256=digest(CONFIG))


def require_stamp(row):
    if any(row.get(k) != v for k, v in stamp().items()):
        raise RuntimeError('Incompatible or missing new benchmark protocol fingerprint')


def manifest():
    return dict(**stamp(), config=CONFIG,
        code_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in CODE_FILES})


def prepare_root(root):
    root = Path(root)
    file = root/'RUN_PROTOCOL.json'
    expected = manifest()
    root.mkdir(parents=True, exist_ok=True)
    with (root/'RUN_PROTOCOL.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if file.exists():
            if json.loads(file.read_text()) != expected:
                raise RuntimeError('Output root uses different protocol/code; never pool or resume it')
            return expected
        if any(p.name != 'RUN_PROTOCOL.lock' for p in root.iterdir()):
            raise RuntimeError('Nonempty output root has no new protocol manifest')
        tmp = root/'RUN_PROTOCOL.tmp'
        tmp.write_text(json.dumps(expected, indent=2)+'\n')
        tmp.replace(file)
    return expected


def verify_root(root):
    data = json.loads((Path(root)/'RUN_PROTOCOL.json').read_text())
    if data != manifest():
        raise RuntimeError('Run protocol/source mismatch; statistics and resume refused')
    return data


def check_runtime(worker, spec, mode, horizon, methods):
    assert list(methods) == METHODS, 'Incorrect competitor list'
    assert mode == CONFIG['execution_mode']
    assert horizon == CONFIG['execution']['external_replan_prefix_m']
    from execution_policy import NATIVE, EXTERNAL
    assert set(NATIVE)==set(CONFIG['execution']['native_methods'])
    assert EXTERNAL==set(CONFIG['execution']['external_methods'])
    assert {m:p['inference_hz'] for m,p in NATIVE.items()}==CONFIG['execution']['native_inference_hz']
    assert NATIVE['nomad']['publish_hz']==CONFIG['execution']['nomad_publish_hz']
    assert NATIVE['nomad']['waypoint_timeout_s']==CONFIG['execution']['nomad_waypoint_timeout_s']
    for key, value in CONFIG['robot'].items():
        assert getattr(spec, key) == value, 'Robot parameter mismatch: '+key
    for name, value in (('CAMERA_WIDTH',640), ('CAMERA_HEIGHT',480),
                        ('CAMERA_HEIGHT_M',.5), ('HORIZONTAL_FOV_DEGREES',90.),
                        ('SAFETY_MARGIN_M',CONFIG['execution']['planner_safety_margin_m']), ('GEOMETRY_OBSTACLE_MIN_HEIGHT_M',1e-6),
                        ('GEOMETRY_OBSTACLE_MAX_HEIGHT_M',.5)):
        assert getattr(worker, name) == value, 'Worker parameter mismatch: '+name
    assert worker.GEOMETRY_CACHE_SCHEMA == CONFIG['geometry_schema']


def validate_result(row, method, task):
    require_stamp(row)
    if (row.get('method') != method or method not in METHODS or
            row.get('execution_mode') != CONFIG['execution_mode'] or
            row.get('task_sha256') != digest(task)):
        raise RuntimeError('Completed result does not match frozen task/method/executor')
    for key, value in CONFIG['robot'].items():
        if row.get('robot', {}).get(key) != value:
            raise RuntimeError('Completed result embodiment differs: '+key)
    if not isinstance(row.get('success'), bool):
        raise RuntimeError('Missing success outcome')
    from execution_policy import policy_for
    if row.get('executor_policy')!=policy_for(method):
        raise RuntimeError('Method used an incompatible executor')


def check_model_acceptance():
    """Executor acceptance is not proof that all seven model adapters are sound.

    A small pre-formal check must document interfaces, not require every model
    to succeed on a selected route. Never select checkpoints by formal outcomes.
    """
    file = HERE/'MODEL_INTERFACE_ACCEPTED.json'
    if not file.exists():
        raise RuntimeError('New protocol is designed; seven-model interface smoke check is still pending')
    proof = json.loads(file.read_text())
    require_stamp(proof)
    if proof.get('code_sha256') != manifest()['code_sha256']:
        raise RuntimeError('Model interface proof predates current source')
    if set(proof.get('checked_methods', [])) != set(METHODS):
        raise RuntimeError('Not all seven native path interfaces were checked')
    if not proof.get('evidence_paths') or not all(Path(p).is_file() for p in proof['evidence_paths']):
        raise RuntimeError('Model interface evidence is missing')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-root', type=Path)
    p.add_argument('--check-launch', action='store_true')
    args = p.parse_args()
    if args.check_launch:
        check_model_acceptance()
    print(json.dumps(prepare_root(args.prepare_root) if args.prepare_root else manifest(), indent=2))
