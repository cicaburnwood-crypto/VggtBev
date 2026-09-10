"""Scoped ProcTHOR isolation regression; no model evaluation is counted."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--scene-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(Path(__file__).resolve().parent),
                  os.environ['BEV_ORIGINAL_BENCHMARK_ROOT'],os.environ['DATABUILDER_ROOT']]
    import procthor_benchmark_worker as w
    import procthor_collection_core as core
    from procthor_gpu import configure,isolated_cloud_controller,assert_process_gpu_binding
    from dispatch import NvmlEvents,descendants,save
    from types import SimpleNamespace
    runtime,_=core.locate_ai2thor_runtime();uuid=core.nvidia_gpu_inventory()[args.gpu]
    binding=configure(runtime,args.gpu,uuid)
    save(args.output/'binding.json',binding)
    os.environ['REALTIME_UNITY_LOG_DIR']=str(args.output/'unity')
    core.isolated_cloud_controller=isolated_cloud_controller
    core.assert_process_gpu_binding=assert_process_gpu_binding
    index=int(json.loads(args.scene_file.read_text())['asset_id'])
    house=w.load_houses(Path(os.environ['DATASET_DIR']),{index})[index]
    options=SimpleNamespace(gpu_index=args.gpu,procthor_runtime_root=runtime,
        resolved_nvidia_gpu_uuid=uuid,geometry_cache_root=args.output/'geometry_cache')
    scene=None;events=NvmlEvents();samples=[]
    try:
        scene=w.Scene(house,options,index)
        started=time.monotonic();frames=0;last_check=-100.;first=None;different=False
        while time.monotonic()-started<45:
            rgb=scene.rgb(scene.graph.points[0],float(frames%360));frames+=1
            if first is None:first=rgb.copy()
            else:different=different or not np.array_equal(rgb,first)
            if time.monotonic()-last_check>=5:
                rows=[r for r in events.processes() if r['pid'] in descendants(os.getpid())]
                if not rows or any(r['uuid'].lower()!=uuid.lower() for r in rows):
                    raise RuntimeError(f'Wrong GPU / absent context: {rows}')
                samples.append(dict(t=time.monotonic()-started,rows=rows));last_check=time.monotonic()
                event=events.wait()
                if event is not None:raise RuntimeError(f'Xid during smoke: {event}')
                save(args.output/'progress.json',dict(frames=frames,samples=samples))
            time.sleep(.04)
        if not different or float(np.std(rgb))<1:raise RuntimeError('RGB did not vary or was empty')
        from PIL import Image
        Image.fromarray(first).save(args.output/'first.png');Image.fromarray(rgb).save(args.output/'last.png')
        save(args.output/'PASS.json',dict(frames=frames,moving_rgb_seconds=time.monotonic()-started,
            binding=binding,samples=samples,scene_index=index,rgb_changed=different))
        print('PASS',flush=True)
    finally:
        if scene is not None:scene.stop()
        events.close()

if __name__=='__main__':main()
