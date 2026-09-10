"""Replay saved RGB through SHM; diagnostic only, no navigation/GT input."""
import base64
import io
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),os.environ['BEV_ORIGINAL_BENCHMARK_ROOT'],os.environ['DATABUILDER_ROOT']]
import procthor_benchmark_worker as w
from robot_contract import configure_worker
from shm_contract import planner_payload
from predicted_navigation import predicted_segment_clear, uncertainty_fields, NavigationConfig
import planner_backends as pb
configure_worker(w)


def inspect(payload):
    sem,p,c,extent=w.decode_bev_response(payload)
    size=len(sem); cell=extent/size
    rr,cc=np.nonzero(sem==0)
    distance=np.hypot(-extent/2+(cc+.5)*cell,extent/2-(rr+.5)*cell)
    planner_semantic,prob,conf,_,_=uncertainty_fields(sem,p,c,cell,NavigationConfig())
    # Build the identical blocked raster, without requiring a reachable goal.
    from scipy import ndimage
    occupied=sem==0
    blocked=ndimage.binary_dilation(occupied,structure=pb._disk(int(np.ceil(w.TOTAL_INFLATION_M/cell))))
    from types import SimpleNamespace
    origin_clear=predicted_segment_clear(SimpleNamespace(blocked=blocked,extent_m=extent,cell_size_m=cell),[0.,0.],[0.,0.])
    return (sem,p,c),dict(extent_m=extent,cell_m=cell,
        closest_occupied_cell_center_m=float(distance.min()) if len(distance) else None,
        occupied_cell_count=int(occupied.sum()),center_four=sem[size//2-1:size//2+1,size//2-1:size//2+1].tolist(),
        origin_clear_at_declared_inflation=origin_clear,inflation_m=w.TOTAL_INFLATION_M)


def main():
    source=Path(os.environ['BEV_REPLAY_SOURCE']);output=Path(os.environ['REALTIME_OUTPUT'])
    records=[]
    for folder in sorted(source.glob('*_*/interfaces/our_model')):
        name=folder.parent.parent.name;segment=str(output/name)
        w.post_json(os.environ['SINGLE_URL'],'/reset',dict(segment_id=segment),60.)
        out=output/name;out.mkdir()
        rows=[]
        for i,file in enumerate(sorted(folder.glob('observation_*.jpg'),key=lambda p:int(p.stem.split('_')[-1]))):
            response=w.post_json(os.environ['SINGLE_URL'],'/predict',dict(segment_id=segment,
                frame_seq=i+1,image_png_base64=base64.b64encode(file.read_bytes()).decode(),
                threshold=.5,physical_camera_height_m=.5),300.)
            single,single_meta=inspect(response)
            shm,shm_meta=inspect(planner_payload(response))
            meta={k:v for k,v in response.items() if not k.endswith('_base64')}
            row=dict(index=i,rgb=str(file),single=single_meta,shm=shm_meta,runtime_metadata=meta)
            rows.append(row)
            np.savez_compressed(out/f'frame_{i:02d}.npz',single=single[0],shm=shm[0],
                single_probability=single[1],shm_probability=shm[1],extent_m=shm_meta['extent_m'])
            w.atomic_json(out/'audit.json',rows)
            print(json.dumps({k:v for k,v in row.items() if k!='runtime_metadata'}),flush=True)
        records.append(dict(scene=name,frames=rows))
    w.atomic_json(output/'AUDIT_COMPLETE.json',dict(scope='saved_RGB_model_replay_only',scenes=records))


if __name__=='__main__':main()
