#!/usr/bin/env bash
set -Eeuo pipefail
source /home/liudiwen/VGGT/baseline_verifier/procthor_20k_benchmark/profile_5090.env
here=$(cd "$(dirname "$0")" && pwd)
export REALTIME_OUTPUT=/home/liudiwen/VGGT/baseline_verifier/procthor_20k_runs/virtualcamera_isolation_smoke_20260909
export BEV_ORIGINAL_BENCHMARK_ROOT="$BENCHMARK_ROOT" DATABUILDER_ROOT DATASET_DIR
export PYTHONPATH="$here:$BENCHMARK_ROOT:$DATABUILDER_ROOT"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=GPU-73b19d68-1d1d-7fe0-2dab-92e5e59cd8df
mkdir -p "$REALTIME_OUTPUT/logs" "$REALTIME_OUTPUT/locks"
exec >>"$REALTIME_OUTPUT/smoke.log" 2>&1
"$WORKER_PYTHON" "$here/dispatch.py" preflight --gpu 4 --uuid "$CUDA_VISIBLE_DEVICES"
worker=''; guard=''
cleanup() {
 trap - EXIT TERM INT
 if [[ -n "$worker" ]]; then kill -TERM -- "-$worker" 2>/dev/null || true; wait "$worker" || true; fi
 if [[ -n "$guard" ]]; then kill -TERM "$guard" 2>/dev/null || true; wait "$guard" || true; fi
}
trap cleanup EXIT
trap 'exit 130' TERM INT
"$WORKER_PYTHON" "$here/dispatch.py" guard --owner-pid "$$" --uuid "$CUDA_VISIBLE_DEVICES" --lane 4 >>"$REALTIME_OUTPUT/guard.log" 2>&1 &
guard=$!
setsid "$WORKER_PYTHON" "$here/gpu_smoke.py" --gpu 4 \
 --scene-file /home/liudiwen/VGGT/baseline_verifier/procthor_20k_runs/realtime2500_20260909_r1/groups/group_0010_procthor/attempt_000/scene.json \
 --output "$REALTIME_OUTPUT/render_smoke" &
worker=$!
wait "$worker"
