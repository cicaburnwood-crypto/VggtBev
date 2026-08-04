#!/usr/bin/env bash
set -euo pipefail

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${T9_VIDEO_CLEAN_ENV:-0}" != "1" ]]; then
  clean_environment=(
    "T9_VIDEO_CLEAN_ENV=1"
    "PATH=/usr/local/bin:/usr/bin:/bin"
    "VIDEO_UI_PORT=${VIDEO_UI_PORT:-8897}"
    "VIDEO_MODEL_PORT=${VIDEO_MODEL_PORT:-8898}"
    "GPU_INDEX=${GPU_INDEX:-0}"
    "NO_BROWSER=${NO_BROWSER:-0}"
    "LANG=${LANG:-C.UTF-8}"
  )
  for variable in DISPLAY WAYLAND_DISPLAY XAUTHORITY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS
  do
    if [[ -n "${!variable:-}" ]]; then
      clean_environment+=("$variable=${!variable}")
    fi
  done
  exec /usr/bin/env -i "${clean_environment[@]}" /bin/bash "$bundle_root/start_video_visualizer.sh"
fi

ui_port="${VIDEO_UI_PORT:-8897}"
model_port="${VIDEO_MODEL_PORT:-8898}"
gpu_index="${GPU_INDEX:-0}"
python="$bundle_root/runtime/ml-gpu/bin/python"
runtime_checkpoint="$bundle_root/checkpoints/p2b_nll_v5_step14320.pt"
backbone_checkpoint="$bundle_root/checkpoints/vggt_omega_1b_512_model.pt"
backbone_source="$bundle_root/app/vendor/backbone"
runtime_server="$bundle_root/app/simulator_ui/p2b_runtime_server.py"
video_server="$bundle_root/app/simulator_ui/video_visualizer.py"
video_html="$bundle_root/app/simulator_ui/video_ui/index.html"
data_dir="/media/user/T91/frodobots-mini-/data"
cache_dir="$bundle_root/video_cache"
log_dir="$bundle_root/logs"
run_dir="$bundle_root/run"
cache_runtime_dir="$bundle_root/runtime/cache"
temporary_dir="$bundle_root/runtime/tmp"
pid_file="$run_dir/video_launcher.pid"
primary_pid_file="$run_dir/launcher.pid"

mkdir -p "$cache_dir" "$log_dir" "$run_dir" "$cache_runtime_dir" "$temporary_dir"
cd "$bundle_root"

for required in \
  "$python" \
  "$runtime_checkpoint" \
  "$backbone_checkpoint" \
  "$backbone_source" \
  "$runtime_server" \
  "$video_server" \
  "$video_html" \
  "$data_dir" \
  /usr/bin/ffmpeg
do
  if [[ ! -e "$required" ]]; then
    echo "Missing required video runtime path: $required" >&2
    exit 1
  fi
done

if [[ -f "$primary_pid_file" ]]; then
  primary_pid="$(<"$primary_pid_file")"
  if [[ "$primary_pid" =~ ^[0-9]+$ ]] && kill -0 "$primary_pid" 2>/dev/null; then
    echo "Stop the Habitat visualizer before loading the video model runtime." >&2
    exit 1
  fi
fi
if [[ -f "$pid_file" ]]; then
  old_pid="$(<"$pid_file")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Video visualizer is already running as PID $old_pid" >&2
    exit 1
  fi
fi

port_is_open() {
  timeout 1 bash -c "</dev/tcp/127.0.0.1/$1" >/dev/null 2>&1
}
if port_is_open "$ui_port" || port_is_open "$model_port"; then
  echo "Video visualizer port is already in use" >&2
  exit 1
fi

echo "$$" > "$pid_file"
model_pid=""
ui_pid=""
cleanup() {
  trap - EXIT INT TERM HUP
  if [[ -n "$ui_pid" ]] && kill -0 "$ui_pid" 2>/dev/null; then kill -TERM "$ui_pid" 2>/dev/null || true; fi
  if [[ -n "$model_pid" ]] && kill -0 "$model_pid" 2>/dev/null; then kill -TERM "$model_pid" 2>/dev/null || true; fi
  if [[ -n "$ui_pid" ]]; then wait "$ui_pid" 2>/dev/null || true; fi
  if [[ -n "$model_pid" ]]; then wait "$model_pid" 2>/dev/null || true; fi
  rm -f "$pid_file"
}
trap cleanup EXIT INT TERM HUP

export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONPATH="$bundle_root/app/src:$bundle_root/app/simulator_ui"
export PATH="$bundle_root/runtime/ml-gpu/bin:/usr/local/bin:/usr/bin:/bin"
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$cache_runtime_dir"
export TORCH_HOME="$cache_runtime_dir/torch"
export HF_HOME="$cache_runtime_dir/huggingface"
export TRITON_CACHE_DIR="$cache_runtime_dir/triton"
export CUDA_CACHE_PATH="$cache_runtime_dir/cuda"
export MPLCONFIGDIR="$cache_runtime_dir/matplotlib"
export TMPDIR="$temporary_dir"

"$python" "$runtime_server" \
  --checkpoint "$runtime_checkpoint" \
  --backbone-source "$backbone_source" \
  --backbone-checkpoint "$backbone_checkpoint" \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port "$model_port" \
  --max-history 10 \
  > "$log_dir/video_model_runtime.log" 2>&1 &
model_pid=$!

for _ in $(seq 1 120); do
  if ! kill -0 "$model_pid" 2>/dev/null; then
    tail -n 100 "$log_dir/video_model_runtime.log" >&2 || true
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$model_port/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

"$python" "$video_server" \
  --data-dir "$data_dir" \
  --cache-dir "$cache_dir" \
  --html "$video_html" \
  --ffmpeg /usr/bin/ffmpeg \
  --model-server-url "http://127.0.0.1:$model_port" \
  --host 127.0.0.1 \
  --port "$ui_port" \
  > "$log_dir/video_visualizer.log" 2>&1 &
ui_pid=$!

for _ in $(seq 1 60); do
  if ! kill -0 "$ui_pid" 2>/dev/null; then
    tail -n 100 "$log_dir/video_visualizer.log" >&2 || true
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$ui_port/api/state" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "P2B video visualizer: http://127.0.0.1:$ui_port/"
if [[ "${NO_BROWSER:-0}" != "1" ]]; then
  xdg-open "http://127.0.0.1:$ui_port/" >/dev/null 2>&1 || true
fi
wait "$ui_pid"
