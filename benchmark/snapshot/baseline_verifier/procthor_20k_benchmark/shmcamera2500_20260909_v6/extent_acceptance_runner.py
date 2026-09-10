"""Replay the frozen MP3D start, then qualify unchanged seven-model interfaces."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
FORMAL=HERE.parents[1]/'procthor_20k_runs/shmcamera2500_20260909_v6'
AUDIT=FORMAL/'operations/shm_extent_repair_20260910'


def replay(gpu):
    import run_group as rg
    from habitat_adapter import HabitatScene
    from collect_random_sessions import discover_scenes
    from episode import Inference
    from PIL import Image
    import numpy as np
    import procthor_benchmark_worker as w
    failed=FORMAL/'groups/group_0104_mp3d/attempt_000'
    task=json.loads((failed/'tasks.json').read_text())[0]
    out=Path(os.environ['REALTIME_OUTPUT'])/'frozen_mp3d_scale_replay'
    out.mkdir(parents=True,exist_ok=False)
    assets=discover_scenes(Path(os.environ['SCENES_ROOT'])/'mp3d')
    asset=next(a for a in assets if a.scene_id==task['asset_id'])
    args=SimpleNamespace(output=out,gpu_index=gpu,geometry_cache_root=out/'geometry_cache')
    args.geometry_cache_root.mkdir()
    scene=HabitatScene(asset,args,task['seed'])
    try:
        saved=np.load(failed/'geometry.npz')
        assert np.array_equal(saved['truth'],scene.geometry.truth)
        assert float(saved['floor_y'])==scene.floor_y
        rgb=scene.rgb(task['start_world'],task['start_yaw_degrees'])
        Image.fromarray(rgb).save(out/'frozen_start.png')
        encoded=w.encode_jpeg(rgb)
        infer=Inference('our_model',os.environ['SINGLE_URL'],os.environ['BASELINE_URL'],str(out))
        infer.reset()
        payload=dict(segment_id=str(out),frame_seq=1,image_png_base64=encoded,
                     threshold=.5,physical_camera_height_m=.5)
        response=w.post_json(os.environ['SINGLE_URL'],'/predict',payload,300.)
        w.atomic_json(out/'raw_response.json',response)
        oldspec=importlib.util.spec_from_file_location('old_shm_contract',AUDIT/'shm_contract.py')
        old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old)
        try:old.planner_payload(response)
        except RuntimeError as error:
            assert str(error)=='Invalid SHM metric extent'
        else:raise RuntimeError('Frozen RGB no longer reproduces original extent rejection; re-audit')
        # Feed this exact actual model response through the corrected boundary.
        from unittest.mock import patch
        target=w.OracleTargetService.from_task(task).query(task['start_world'],task['start_yaw_degrees'])
        with patch.object(w,'post_json',return_value=response):
            prediction=infer.predict([encoded],target,None)
        assert prediction['path_metric_m']==[]
        assert prediction['planning_failure']['code']=='invalid_predicted_metric_extent'
        proof=dict(task=task,source='exact frozen scene/pose RGB, actual full SHM model',
            original_error='Invalid SHM metric extent',prediction=prediction,
            raw_response=str(out/'raw_response.json'),formal_results_modified=False)
        w.atomic_json(out/'REPRODUCED.json',proof)
        print('LIVE_SCALE_REJECTION_REPRODUCED',json.dumps(prediction),flush=True)
        infer.reset()
    finally:scene.stop()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-index',type=int,required=True)
    p.add_argument('--replay-only',action='store_true');args=p.parse_args()
    if args.replay_only:replay(args.gpu_index)
    else:
        subprocess.run([os.environ['HABITAT_PYTHON'],str(HERE/'extent_acceptance_runner.py'),
                        '--gpu-index',str(args.gpu_index),'--replay-only'],check=True)
        subprocess.run([sys.executable,str(HERE/'interface_acceptance.py'),
                        '--gpu-index',str(args.gpu_index)],check=True)
