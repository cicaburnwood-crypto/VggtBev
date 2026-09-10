"""Live reproduction of the exact empty graph, then unchanged seven-model probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser();parser.add_argument('--gpu-index',type=int,required=True)
args=parser.parse_args()
here=Path(__file__).resolve().parent
root=Path(os.environ['REALTIME_OUTPUT'])
out=root/'empty_graph_seed_2026107117'
subprocess.run([sys.executable,str(here/'run_group.py'),'--source','procthor',
    '--seed','2026107117','--output',str(out),'--gpu-index',str(args.gpu_index),
    '--geometry-only'],check=True)
row=json.loads((out/'REJECTED.json').read_text())
assert row['stage']=='before_routes' and row['replacement_allowed']
assert not row['counts_as_model_failure']
assert 'reachable graph has no useful connected component' in row['reason']
assert not (out/'tasks.json').exists() and not list(out.glob('route_*'))
assert not (out/'INFRASTRUCTURE_ERROR.json').exists()
print('LIVE_EMPTY_GRAPH_REJECTION_PASSED',json.dumps(row),flush=True)
subprocess.run([sys.executable,str(here/'interface_acceptance.py'),
    '--gpu-index',str(args.gpu_index)],check=True)
