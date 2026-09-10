#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
base="$here/snapshot/baseline_verifier/procthor_20k_benchmark"
code="$base/shmcamera2500_20260909_v6"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export MIXED_ADAPTER_ROOT="$base/mixed500_20260909"
export BEV_ORIGINAL_BENCHMARK_ROOT="$base"
export DATABUILDER_ROOT="$here/snapshot/databuilder"
export PYTHONPATH="$code:$MIXED_ADAPTER_ROOT:$base:$DATABUILDER_ROOT"
exec "${PYTHON:-python3}" -m unittest discover -s "$code" -p 'test_*.py' -v
