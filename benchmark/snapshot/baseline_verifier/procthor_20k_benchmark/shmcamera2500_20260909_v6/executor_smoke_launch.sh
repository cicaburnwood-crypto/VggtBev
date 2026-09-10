#!/usr/bin/env bash
# Detached, single-GPU, model-free acceptance. Stops/relinquishes GPU on exit.
set -Eeuo pipefail
source /home/liudiwen/VGGT/baseline_verifier/procthor_20k_benchmark/profile_5090.env
here=$(cd "$(dirname "$0")" && pwd)
gpu="${1:?physical GPU index}"
export REALTIME_OUTPUT="${2:?new acceptance output directory}"
source_name="${3:-procthor}"
asset_id="${4:-}"
seed="${5:-2026090900}"
[[ ! -e "$REALTIME_OUTPUT" ]] || { printf 'Output exists; refusing overwrite.\n'; exit 1; }
export BEV_ORIGINAL_BENCHMARK_ROOT="$BENCHMARK_ROOT" DATABUILDER_ROOT DATASET_DIR
export PYTHONPATH="$here:$BENCHMARK_ROOT:$DATABUILDER_ROOT"
export SCENES_ROOT=/home/liudiwen/VGGT/scenes
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
mkdir -p "$REALTIME_OUTPUT/logs" "$REALTIME_OUTPUT/locks"
exec >>"$REALTIME_OUTPUT/executor.log" 2>&1
"$WORKER_PYTHON" -m unittest discover -s "$here" -p test_exact_executor.py -v
"$WORKER_PYTHON" "$here/dispatch.py" preflight --gpu "$gpu" --uuid "$CUDA_VISIBLE_DEVICES"
worker=''; guard=''
cleanup() {
 trap - EXIT TERM INT
 if [[ -n "$worker" ]]; then kill -TERM -- "-$worker" 2>/dev/null || true; wait "$worker" || true; fi
 if [[ -n "$guard" ]]; then kill -TERM "$guard" 2>/dev/null || true; wait "$guard" || true; fi
}
trap cleanup EXIT
trap 'exit 130' TERM INT
"$WORKER_PYTHON" "$here/dispatch.py" guard --owner-pid "$$" --uuid "$CUDA_VISIBLE_DEVICES" --lane "$gpu" >>"$REALTIME_OUTPUT/guard.log" 2>&1 &
guard=$!
for _ in $(seq 1 20); do
  kill -0 "$guard"
  [[ -f "$REALTIME_OUTPUT/logs/guard${gpu}.ready.json" ]] && break
  sleep 1
done
[[ -f "$REALTIME_OUTPUT/logs/guard${gpu}.ready.json" ]]
scene_python="$WORKER_PYTHON"
if [[ "$source_name" != procthor ]]; then
  scene_python=/home/liudiwen/.conda/envs/vggnav-habitat/bin/python
  export MAGNUM_CUDA_DEVICE=0
fi
setsid "$scene_python" "$here/executor_scene_smoke.py" --gpu "$gpu" \
 --source "$source_name" --asset-id "$asset_id" --seed "$seed" --output "$REALTIME_OUTPUT/trials" &
worker=$!
wait "$worker"
