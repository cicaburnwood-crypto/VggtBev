"""Use a native executor where released; otherwise use the accepted follower."""
from pathlib import Path
import procthor_benchmark_worker as w
from execution_policy import MODE, policy_for
from virtual_episode import run_episode as external_episode, HORIZON_M
from native_velocity_episode import run_episode as native_episode


def run_episode(scene,task,method,single_url,baseline_url,out,goal_images,warmup=True):
    policy=policy_for(method)
    implementation=native_episode if policy['kind']=='native_velocity' else external_episode
    row=implementation(scene,task,method,single_url,baseline_url,out,goal_images,warmup=warmup)
    row['execution_mode']=MODE
    row['executor_policy']=policy
    if method=='our_model':
        from protocol import CONFIG
        row['our_model_configuration']=CONFIG['our_model']
    # No artificial zero tracking-error claim for a native velocity controller.
    if policy['kind']=='native_velocity':
        row['native_path_tracking_error_max_m']=None
    w.atomic_json(Path(out)/'result.json',row)
    return row
