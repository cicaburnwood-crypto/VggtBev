#!/usr/bin/env bash
# Single scoped diagnostic GPU; no formal work or automatic retry.
set -Eeuo pipefail
source /home/liudiwen/VGGT/baseline_verifier/procthor_20k_benchmark/profile_5090.env
here=$(cd "$(dirname "$0")" && pwd)
gpu="${1:?GPU}"; export REALTIME_OUTPUT="${2:?new output}"
export BEV_REPLAY_SOURCE="${3:?completed interface evidence}"
[[ ! -e "$REALTIME_OUTPUT" ]]
mkdir -p "$REALTIME_OUTPUT/logs" "$REALTIME_OUTPUT/locks"
exec >>"$REALTIME_OUTPUT/logs/audit.log" 2>&1
export BEV_ORIGINAL_BENCHMARK_ROOT="$BENCHMARK_ROOT" DATABUILDER_ROOT
export PYTHONPATH="$here:$BENCHMARK_ROOT:$DATABUILDER_ROOT"
export BEV_MODEL_ROOT=/home/liudiwen/VGGT/baseline_verifier
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
export NVIDIA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" MAGNUM_CUDA_DEVICE=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONNOUSERSITE=1
export VERIFIER_GPU_INFERENCE_LOCK="$REALTIME_OUTPUT/locks/inference.lock"
export SINGLE_URL=http://127.0.0.1:29790
if ss -ltn | awk '{print $4}' | grep -Eq ':29790$'; then exit 1; fi
"$WORKER_PYTHON" "$here/dispatch.py" preflight --gpu "$gpu" --uuid "$CUDA_VISIBLE_DEVICES"
guard='';runtime='';worker=''
stop_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
cleanup() {
 trap - EXIT TERM INT HUP
 stop_group "$worker";stop_group "$runtime"
 if [[ -n "$guard" ]]; then kill -TERM "$guard" 2>/dev/null || true; wait "$guard" || true; fi
}
trap cleanup EXIT
trap 'exit 130' TERM INT HUP
"$WORKER_PYTHON" "$here/dispatch.py" guard --owner-pid "$$" --uuid "$CUDA_VISIBLE_DEVICES" --lane 0 &
guard=$!
for _ in $(seq 1 20); do
 kill -0 "$guard"
 [[ -f "$REALTIME_OUTPUT/logs/guard0.ready.json" ]] && break
 sleep 1
done
[[ -f "$REALTIME_OUTPUT/logs/guard0.ready.json" ]]
setsid "$SINGLE_PYTHON" "$here/shm_runtime.py" --checkpoint "$SINGLE_CHECKPOINT" \
 --backbone-source "$BACKBONE_SOURCE" --backbone-checkpoint "$BACKBONE_CHECKPOINT" \
 --device cuda:0 --host 127.0.0.1 --port 29790 --max-history 10 &
runtime=$!
for _ in $(seq 1 180); do
 if curl -fsS --max-time 2 "$SINGLE_URL/health" >/dev/null; then break; fi
 kill -0 "$runtime";kill -0 "$guard";sleep 1
done
curl -fsS --max-time 5 "$SINGLE_URL/health" >/dev/null
setsid "$WORKER_PYTHON" "$here/audit_saved_bevs.py" &
worker=$!
wait "$worker"
