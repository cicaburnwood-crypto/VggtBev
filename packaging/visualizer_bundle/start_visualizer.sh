#!/usr/bin/env bash
set -euo pipefail

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Re-exec once with a minimal environment so a shell launched from an IDE,
# Conda, ROS, or another project cannot leak its paths into this bundle.
if [[ "${T9_BUNDLE_CLEAN_ENV:-0}" != "1" ]]; then
  clean_environment=(
    "T9_BUNDLE_CLEAN_ENV=1"
    "PATH=/usr/local/bin:/usr/bin:/bin"
    "UI_PORT=${UI_PORT:-8894}"
    "MODEL_PORT=${MODEL_PORT:-8896}"
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
  exec /usr/bin/env -i "${clean_environment[@]}" /bin/bash "$bundle_root/start_visualizer.sh"
fi

ui_port="${UI_PORT:-8894}"
model_port="${MODEL_PORT:-8896}"
gpu_index="${GPU_INDEX:-0}"
model_python="$bundle_root/runtime/ml-gpu/bin/python"
habitat_python="$bundle_root/runtime/habitat-sim/bin/python"
runtime_checkpoint="$bundle_root/checkpoints/p2b_nll_v5_step14320.pt"
backbone_checkpoint="$bundle_root/checkpoints/vggt_omega_1b_512_model.pt"
initial_scene="$bundle_root/scenes/00484-fc7RfUCN5mY/fc7RfUCN5mY.basis.glb"
scene_catalog="$bundle_root/scenes_10.json"
log_dir="$bundle_root/logs"
run_dir="$bundle_root/run"
launcher_pid_file="$run_dir/launcher.pid"
bundle_cache_dir="$bundle_root/runtime/cache"
bundle_tmp_dir="$bundle_root/runtime/tmp"

mkdir -p "$log_dir" "$run_dir" "$bundle_cache_dir" "$bundle_tmp_dir"
cd "$bundle_root"

require_bundle_path() {
  case "$1" in
    "$bundle_root"/*) ;;
    *)
      echo "Refusing path outside the T9 bundle: $1" >&2
      exit 1
      ;;
  esac
}

for required in \
  "$model_python" \
  "$habitat_python" \
  "$runtime_checkpoint" \
  "$backbone_checkpoint" \
  "$initial_scene" \
  "$scene_catalog"
do
  require_bundle_path "$required"
  if [[ ! -e "$required" ]]; then
    echo "Missing required local bundle file: $required" >&2
    exit 1
  fi
done

if [[ -f "$launcher_pid_file" ]]; then
  existing_pid="$(<"$launcher_pid_file")"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Visualizer is already running as PID $existing_pid" >&2
    echo "Open http://127.0.0.1:$ui_port/"
    exit 1
  fi
fi

port_is_open() {
  timeout 1 bash -c "</dev/tcp/127.0.0.1/$1" >/dev/null 2>&1
}

if port_is_open "$ui_port"; then
  echo "UI port $ui_port is already in use" >&2
  exit 1
fi
if port_is_open "$model_port"; then
  echo "Model port $model_port is already in use" >&2
  exit 1
fi

echo "$$" > "$launcher_pid_file"
model_pid=""
ui_pid=""

cleanup() {
  trap - EXIT INT TERM HUP
  if [[ -n "$ui_pid" ]] && kill -0 "$ui_pid" 2>/dev/null; then
    kill -TERM "$ui_pid" 2>/dev/null || true
  fi
  if [[ -n "$model_pid" ]] && kill -0 "$model_pid" 2>/dev/null; then
    kill -TERM "$model_pid" 2>/dev/null || true
  fi
  if [[ -n "$ui_pid" ]]; then wait "$ui_pid" 2>/dev/null || true; fi
  if [[ -n "$model_pid" ]]; then wait "$model_pid" 2>/dev/null || true; fi
  rm -f "$launcher_pid_file"
}
trap cleanup EXIT INT TERM HUP

export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONPATH="$bundle_root/app/src:$bundle_root/app/simulator_ui"
export PATH="$bundle_root/runtime/ml-gpu/bin:$bundle_root/runtime/habitat-sim/bin:/usr/local/bin:/usr/bin:/bin"
unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$bundle_cache_dir"
export TORCH_HOME="$bundle_cache_dir/torch"
export HF_HOME="$bundle_cache_dir/huggingface"
export TRITON_CACHE_DIR="$bundle_cache_dir/triton"
export CUDA_CACHE_PATH="$bundle_cache_dir/cuda"
export MPLCONFIGDIR="$bundle_cache_dir/matplotlib"
export TMPDIR="$bundle_tmp_dir"

"$model_python" "$bundle_root/app/simulator_ui/p2b_runtime_server.py" \
  --checkpoint "$runtime_checkpoint" \
  --backbone-source "$bundle_root/app/vendor/backbone" \
  --backbone-checkpoint "$backbone_checkpoint" \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port "$model_port" \
  --max-history 10 \
  > "$log_dir/model_runtime.log" 2>&1 &
model_pid=$!

echo "Loading local VGGT/P2B runtime on GPU $gpu_index…"
model_ready=0
for _ in $(seq 1 90); do
  if ! kill -0 "$model_pid" 2>/dev/null; then
    echo "Model runtime exited during startup:" >&2
    tail -n 80 "$log_dir/model_runtime.log" >&2 || true
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:$model_port/health" >/dev/null 2>&1; then
    model_ready=1
    break
  fi
  sleep 1
done
if [[ "$model_ready" -ne 1 ]]; then
  echo "Timed out waiting for the local model runtime" >&2
  exit 1
fi

browser_args=()
if [[ "${NO_BROWSER:-0}" == "1" ]]; then
  browser_args+=(--no-browser)
fi

export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"

"$habitat_python" "$bundle_root/app/simulator_ui/web_navigation.py" \
  "$initial_scene" \
  --scene-catalog "$scene_catalog" \
  --gpu-device-id 0 \
  --random-start \
  --random-yaw \
  --seed 20260804 \
  --host 127.0.0.1 \
  --port "$ui_port" \
  --model p2b \
  --model-server-url "http://127.0.0.1:$model_port" \
  --model-max-history 10 \
  --model-hz 1.0 \
  --camera-width 640 \
  --camera-height 480 \
  --sensor-height-m 0.35 \
  --horizontal-fov-degrees 90 \
  "${browser_args[@]}" &
ui_pid=$!

echo "Starting the first local Habitat scene…"
echo "The browser will open at http://127.0.0.1:$ui_port/"
echo "Use the 10-scene dropdown and WASD controls."
echo "Press Ctrl+C to stop both local processes."

wait "$ui_pid"
