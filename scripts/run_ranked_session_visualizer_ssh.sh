#!/usr/bin/env bash
set -euo pipefail

readonly GPU_INDEX="${1:-}"
readonly LOCAL_PORT="${2:-8892}"
readonly REMOTE_PORT="${3:-8892}"
readonly GPU0_OVERRIDE="${4:-}"
readonly REMOTE_ALIAS="remote_gpu"
readonly REMOTE_ROOT="/home/user/VGGT/method1_train_p2b_nll"

if [[ ! "${GPU_INDEX}" =~ ^[0-7]$ ]]; then
  echo "Usage: $0 <free-physical-gpu-0..7> [local-port=8892] [remote-port=8892] [--allow-gpu0]" >&2
  exit 2
fi
if [[ "${GPU_INDEX}" == "0" && "${GPU0_OVERRIDE}" != "--allow-gpu0" ]]; then
  echo "GPU 0 requires the explicit --allow-gpu0 override." >&2
  exit 2
fi

echo "Opening a live SSH tunnel on http://127.0.0.1:${LOCAL_PORT}"
echo "The remote process reads only the selected session after a UI click."
echo "Press Ctrl-C to stop both the tunnel and the remote visualizer."

ssh -T \
  -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
  "${REMOTE_ALIAS}" \
  "cd '${REMOTE_ROOT}' && exec scripts/run_ranked_session_visualizer.sh '${GPU_INDEX}' '${REMOTE_PORT}' '${GPU0_OVERRIDE}'"
