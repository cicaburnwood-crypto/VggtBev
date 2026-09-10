#!/usr/bin/env bash
# Each invocation owns one GPU and its model services for a deterministic shard.
set -Eeuo pipefail
source /home/liudiwen/VGGT/baseline_verifier/procthor_20k_benchmark/profile_5090.env
gpu="${1:?physical GPU index required}"; lane="${2:?lane index required}"
here=$(cd "$(dirname "$0")" && pwd)
# Independent executor tests come first. Do not accidentally launch the old
# embodiment or consume formal GPUs before the simulator acceptance is signed off.
[[ -f "$here/EXECUTOR_SIMULATOR_ACCEPTED.json" ]] || {
  printf 'Executor simulator acceptance is missing; refusing formal experiment.\n'; exit 1;
}
"$WORKER_PYTHON" - "$here" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
proof=json.loads((root/'EXECUTOR_SIMULATOR_ACCEPTED.json').read_text())
for name,digest in proof['code_sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise SystemExit('Executor changed after acceptance; revalidate before running: '+name)
PY
export ROOT="$here" BENCHMARK_ROOT WORKER_PYTHON DATASET_DIR DATABUILDER_ROOT
export BEV_ORIGINAL_BENCHMARK_ROOT="$BENCHMARK_ROOT"
export MIXED_ADAPTER_ROOT="$BENCHMARK_ROOT/mixed500_20260909"
export HABITAT_PYTHON=/home/liudiwen/.conda/envs/vggnav-habitat/bin/python
export SCENES_ROOT=/home/liudiwen/VGGT/scenes
export REALTIME_OUTPUT="${REALTIME_OUTPUT:-/home/liudiwen/VGGT/baseline_verifier/procthor_20k_runs/shmcamera2500_20260909_v6}"
export BEV_MODEL_ROOT=/home/liudiwen/VGGT/baseline_verifier
SINGLE_RUNTIME="$here/shm_runtime.py"
export MIXED_OUTPUT="$REALTIME_OUTPUT"
export BASELINE_RUNTIME="$here/navigation_baseline_runtime.py"
# No formal run before seven native model interfaces have been checked with
# this executor. This CPU-only check happens before GPU allocation or loading.
if [[ "${BENCHMARK_INTERFACE_CHECK:-0}" == 1 ]]; then
  [[ "$(basename "$REALTIME_OUTPUT")" == native_interface_acceptance_* ]] || {
    printf 'Interface probes require a separate acceptance output root\n'; exit 1;
  }
  "$WORKER_PYTHON" "$here/protocol.py" --prepare-root "$REALTIME_OUTPUT"
else
  "$WORKER_PYTHON" "$here/protocol.py" --check-launch --prepare-root "$REALTIME_OUTPUT"
fi
mkdir -p "$REALTIME_OUTPUT/logs" "$REALTIME_OUTPUT/locks"
exec >>"$REALTIME_OUTPUT/logs/lane${lane}.log" 2>&1
# Avoid six simultaneous driver/model initialization bursts, then recheck idle.
sleep "$((lane * 8))"
# Refuse duplicate owners without changing any existing process/lock.
exec 9>"$REALTIME_OUTPUT/locks/lane${lane}.owner.lock"
flock -n 9 || { printf 'Lane %s already has an owner\n' "$lane"; exit 1; }
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
exec 8>"$REALTIME_OUTPUT/locks/gpu${gpu}.owner.lock"
flock -n 8 || { printf 'GPU %s already has a benchmark owner\n' "$gpu"; exit 1; }
"$WORKER_PYTHON" "$here/dispatch.py" preflight --gpu "$gpu" --uuid "$uuid"
export CUDA_VISIBLE_DEVICES="$uuid" NVIDIA_VISIBLE_DEVICES="$uuid" MAGNUM_CUDA_DEVICE=0 CUDA_MODULE_LOADING=LAZY
unset DISPLAY WAYLAND_DISPLAY
export PYTHONPATH="$here:$MIXED_ADAPTER_ROOT:$BENCHMARK_ROOT:$DATABUILDER_ROOT"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONNOUSERSITE=1
export TORCH_HOME="$BASELINE_ASSETS/torch_cache"
export VERIFIER_GPU_INFERENCE_LOCK="$REALTIME_OUTPUT/locks/lane${lane}.inference.lock"
export VGGT_RENDER_STARTUP_LOCK_ROOT="$REALTIME_OUTPUT/locks/render"
port=$((29600+lane*10))
export SINGLE_URL="http://127.0.0.1:$port" BASELINE_URL="http://127.0.0.1:$((port+1))"
for p in "$port" "$((port+1))"; do
  if ss -ltn | awk '{print $4}' | grep -Eq ":${p}$"; then
    printf 'Refusing occupied port %s\n' "$p"; exit 1
  fi
done
single=''; external=''; worker=''; watcher=''
stop_owned_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 35); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  trap - EXIT INT TERM HUP ERR
  stop_owned_group "$worker"
  stop_owned_group "$single"
  stop_owned_group "$external"
  if [[ -n "$watcher" ]]; then kill -TERM "$watcher" 2>/dev/null || true; wait "$watcher" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
on_error() {
  local status=$?
  trap - ERR
  "$WORKER_PYTHON" "$here/dispatch.py" trip --lane "$lane" \
    --reason "Launcher/service/worker failure status=$status on GPU=$gpu lane=$lane" || true
  exit "$status"
}
trap on_error ERR
"$WORKER_PYTHON" "$here/dispatch.py" guard --owner-pid "$$" --uuid "$uuid" --lane "$lane" \
 >>"$REALTIME_OUTPUT/logs/guard${lane}.log" 2>&1 &
watcher=$!
for _ in $(seq 1 30); do
  kill -0 "$watcher"
  if [[ -f "$REALTIME_OUTPUT/logs/guard${lane}.ready.json" ]] && \
     "$WORKER_PYTHON" -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["pid"]!=int(sys.argv[2]))' \
       "$REALTIME_OUTPUT/logs/guard${lane}.ready.json" "$watcher"; then break; fi
  sleep 1
done
"$WORKER_PYTHON" -c 'import json,sys;assert json.load(open(sys.argv[1]))["pid"]==int(sys.argv[2])' \
 "$REALTIME_OUTPUT/logs/guard${lane}.ready.json" "$watcher"
setsid "$SINGLE_PYTHON" "$SINGLE_RUNTIME" --checkpoint "$SINGLE_CHECKPOINT" \
 --backbone-source "$BACKBONE_SOURCE" --backbone-checkpoint "$BACKBONE_CHECKPOINT" \
 --device cuda:0 --host 127.0.0.1 --port "$port" --max-history 10 --gate-overwrite-inset-pixels 2 \
 >>"$REALTIME_OUTPUT/logs/single${lane}.log" 2>&1 &
single=$!
setsid "$BASELINE_PYTHON" "$BASELINE_RUNTIME" --assets-root "$BASELINE_ASSETS" --device cuda:0 \
 --host 127.0.0.1 --port "$((port+1))" --require-all-assets --robot-max-v 1 --robot-max-w 1.5707963267948966 --nomad-metric-spacing 0.05 \
 >>"$REALTIME_OUTPUT/logs/external${lane}.log" 2>&1 &
external=$!
for url in "$SINGLE_URL" "$BASELINE_URL"; do
  for _ in $(seq 1 300); do
    if curl -fsS --max-time 2 "$url/health" >/dev/null; then break; fi
    kill -0 "$single"; kill -0 "$external"; kill -0 "$watcher"; sleep 1
  done
  curl -fsS --max-time 5 "$url/health"
done
printf 'gpu=%s uuid=%s model_pids=%s,%s guard_pid=%s\n' "$gpu" "$uuid" "$single" "$external" "$watcher"
resume_args=()
if [[ "${REALTIME_RESUME:-0}" == 1 ]]; then resume_args+=(--resume); fi
if [[ "${BENCHMARK_INTERFACE_CHECK:-0}" == 1 ]]; then
  setsid "$WORKER_PYTHON" "$here/graphfix_acceptance_runner.py" --gpu-index "$gpu" &
else
  setsid "$WORKER_PYTHON" "$here/dispatch.py" run "$lane" "$gpu" \
   --lanes "${REALTIME_LANES:-6}" --limit-groups "${REALTIME_SMOKE_GROUPS:-0}" "${resume_args[@]}" &
fi
worker=$!
while kill -0 "$worker" 2>/dev/null; do
  kill -0 "$single"; kill -0 "$external"; kill -0 "$watcher"; sleep 1
done
wait "$worker"
