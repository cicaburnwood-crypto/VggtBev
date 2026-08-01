#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hssd_root="${VGGTBEV_HSSD_ROOT:-/media/user/T9/scene_datasets/hssd-hab}"
scene_id="${VGGTBEV_HSSD_SCENE_ID:-102343992}"
scene="$hssd_root/scenes/${scene_id}.scene_instance.json"
scene_config="$hssd_root/hssd-hab.scene_dataset_config.json"
navmesh="${VGGTBEV_HSSD_NAVMESH:-/home/user/Project/VGGNAV/output/hssd_collector_smoke/_navmesh_cache/hssd/${scene_id}.navmesh}"
checkpoint="${VGGTBEV_6P5M_CHECKPOINT:-$project_root/models/hm3d_600x3_method2_dual_bev_6p5m.pt}"
runtime_host="${VGGTBEV_RUNTIME_HOST:-127.0.0.1}"
runtime_port="${VGGTBEV_RUNTIME_PORT:-8765}"
ui_host="${VGGTBEV_UI_HOST:-127.0.0.1}"
ui_port="${VGGTBEV_UI_PORT:-8000}"
max_history="${VGGTBEV_MODEL_MAX_HISTORY:-34}"
voxel_size="${VGGTBEV_VOXEL_SIZE:-0.1}"
seed="${VGGTBEV_SEED:-102343992}"
artifact_dir="$project_root/artifacts/hssd_6p5m_visualizer"
runtime_log="$artifact_dir/runtime.log"

for required in "$scene" "$scene_config" "$navmesh" "$checkpoint"; do
    if [[ ! -f "$required" ]]; then
        echo "Required file does not exist: $required" >&2
        exit 1
    fi
done
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
    --model 6p5m \
    --checkpoint "$checkpoint" \
    --host "$runtime_host" \
    --port "$runtime_port" \
    --max-history "$max_history" \
    >"$runtime_log" 2>&1 &
runtime_pid="$!"

runtime_url="http://${runtime_host}:${runtime_port}"
for _ in $(seq 1 180); do
    if ! kill -0 "$runtime_pid" 2>/dev/null; then
        echo "VGGTBEV runtime exited before becoming ready:" >&2
        tail -n 60 "$runtime_log" >&2
        exit 1
    fi
    if curl -fsS "$runtime_url/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
curl -fsS "$runtime_url/health" >/dev/null

echo "Runtime ready: $runtime_url (6.5 m checkpoint)"
echo "HSSD scene: $scene_id"
echo "Runtime log: $runtime_log"
"$project_root/simulator_ui/run-in-env" \
    python "$project_root/simulator_ui/web_navigation.py" \
    "$scene" \
    --scene-dataset-config "$scene_config" \
    --navmesh "$navmesh" \
    --model 6p5m \
    --model-server-url "$runtime_url" \
    --model-max-history "$max_history" \
    --voxel-size "$voxel_size" \
    --random-start \
    --random-yaw \
    --seed "$seed" \
    --host "$ui_host" \
    --port "$ui_port" \
    --no-browser
