"""Counterfactual executor audit of recorded XY plans, not navigation results.

No simulator/model rerun and no success-rate estimate. v4 did not record LiMo
SE(2) headings, so this replay can only audit the XY-only gear selector.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from exact_executor import RobotSpec, path_steps


def replay(path, rate, reverse):
    ticks=list(path_steps(path, spec=replace(RobotSpec(),yaw_rate_deg_s=rate),
                          allow_reverse=reverse, horizon_m=.5))
    return dict(seconds=sum(t.dt_s for t in ticks),
                rotation_seconds=sum(t.dt_s for t in ticks if t.phase=='rotate'),
                arc_m=sum(t.distance_m for t in ticks),
                final_xz=ticks[-1].end.xz.tolist() if ticks else [0.,0.],
                reverse_m=sum(t.distance_m for t in ticks if t.speed_m_s<0))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=[]
    for file in sorted(args.evidence.rglob('events.jsonl')):
        if file.parent.name not in {'our_model','limo_tel','limo_aug','genie_samtp'}:continue
        events=[json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        valid=[(i,r) for i,r in enumerate(events) if len(r.get('native_path') or [])>=2]
        if not valid:continue
        for pick in sorted({0,len(valid)//2,len(valid)-1}):
            index,row=valid[pick];path=row['native_path']
            variants={name:replay(path,rate,reverse) for name,rate,reverse in
                [('old_30',30,False),('rate_only_90',90,False),('gear_adapter_90',90,True)]}
            ref=variants['old_30']
            for value in variants.values():
                np.testing.assert_allclose(value['final_xz'],ref['final_xz'],atol=1e-10)
                assert abs(value['arc_m']-ref['arc_m'])<1e-10
            rows.append(dict(file=str(file),event_index=index,method=file.parent.name,variants=variants))
    summary={}
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method]
        summary[method]=dict(recorded_prefixes=len(group),**{
            name:dict(mean_motion_seconds=float(np.mean([r['variants'][name]['seconds'] for r in group])),
                      mean_rotation_seconds=float(np.mean([r['variants'][name]['rotation_seconds'] for r in group])))
            for name in ('old_30','rate_only_90','gear_adapter_90')})
    result=dict(scope='OFFLINE_REPLAY_ONLY; no model/simulator success-rate claim; same stored .5m XY prefixes; native SE2 headings unavailable',
                preserved_xy_endpoints_and_arc=True,summary=summary,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
