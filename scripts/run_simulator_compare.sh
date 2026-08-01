#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${1:-5m}"
if [[ "$#" -gt 0 ]]; then
    shift
fi
case "$model" in
    3p5m|5m|6p5m) ;;
    *)
        echo "model must be one of: 3p5m, 5m, 6p5m" >&2
        exit 2
        ;;
esac

runtime_host="${VGGTBEV_RUNTIME_HOST:-127.0.0.1}"
runtime_port="${VGGTBEV_RUNTIME_PORT:-8765}"
runtime_url="http://${runtime_host}:${runtime_port}"
max_history="${VGGTBEV_MODEL_MAX_HISTORY:-34}"
camera_height_m="${VGGTBEV_CAMERA_HEIGHT_M:-0.35}"
artifact_dir="$project_root/artifacts/simulator_compare"
runtime_log="$artifact_dir/runtime_all_models.log"
mkdir -p "$artifact_dir"

runtime_pid=""
cleanup() {
    if [[ -n "$runtime_pid" ]] && kill -0 "$runtime_pid" 2>/dev/null; then
        kill "$runtime_pid"
        wait "$runtime_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"$project_root/.venv/bin/vggt-bev-runtime-server" \
    --all-models \
    --host "$runtime_host" \
    --port "$runtime_port" \
    --max-history "$max_history" \
    --camera-height-m "$camera_height_m" \
    >"$runtime_log" 2>&1 &
runtime_pid="$!"

ready=0
for _ in $(seq 1 120); do
    if ! kill -0 "$runtime_pid" 2>/dev/null; then
        echo "VGGTBEV runtime exited before becoming ready:" >&2
        tail -n 40 "$runtime_log" >&2
        exit 1
    fi
    if curl -fsS "$runtime_url/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done
if [[ "$ready" -ne 1 ]]; then
    echo "Timed out waiting for VGGTBEV runtime; see $runtime_log" >&2
    exit 1
fi

echo "Runtime ready: $runtime_url (models=3p5m,5m,6p5m; shared VGGT)"
echo "Runtime log: $runtime_log"
"$project_root/simulator_ui/run-in-env" \
    python "$project_root/simulator_ui/web_navigation.py" \
    --model "$model" \
    --all-model-comparison \
    --model-server-url "$runtime_url" \
    --model-max-history "$max_history" \
    "$@"
