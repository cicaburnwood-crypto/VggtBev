#!/usr/bin/env bash
set -euo pipefail

readonly GPU_INDEX="${1:-}"
readonly PORT="${2:-8892}"
readonly GPU0_OVERRIDE="${3:-}"

if [[ ! "${GPU_INDEX}" =~ ^[0-7]$ ]]; then
  echo "Usage: $0 <free-physical-gpu-0..7> [port=8892] [--allow-gpu0]" >&2
  exit 2
fi
if [[ "${GPU_INDEX}" == "0" && "${GPU0_OVERRIDE}" != "--allow-gpu0" ]]; then
  echo "GPU 0 requires the explicit --allow-gpu0 override." >&2
  exit 2
fi
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo "Port must be an integer in [1024, 65535]." >&2
  exit 2
fi

GPU_PIDS="$(
  nvidia-smi -i "${GPU_INDEX}" \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits
)" || {
  echo "Unable to verify physical GPU ${GPU_INDEX}; refusing to launch." >&2
  exit 3
}
if [[ -n "${GPU_PIDS//[[:space:]]/}" ]]; then
  echo "Refusing to share occupied physical GPU ${GPU_INDEX}; PIDs: ${GPU_PIDS}" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
exec .conda/bin/python simulator_ui/ranked_session_visualizer.py \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port "${PORT}"
