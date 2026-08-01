#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_python="${P1B_MODEL_PYTHON:-/home/user/.local/share/micromamba/envs/ml-gpu/bin/python}"
scene="$project_root/simulator_assets/habitat-test-scenes/skokloster-castle.glb"
navmesh="$project_root/simulator_assets/habitat-test-scenes/skokloster-castle.navmesh"
checkpoint="${P1B_CHECKPOINT:-$project_root/checkpoints/visualizer/p1b_repair10e_best_observed_epoch3_step32572.pt}"
backbone_source="$project_root/vendor/backbone"
backbone_checkpoint="$project_root/checkpoints/model.pt"
runtime_port="${P1B_RUNTIME_PORT:-8875}"
ui_port="${P1B_UI_PORT:-8895}"
max_history="${P1B_MAX_HISTORY:-10}"
seed="${P1B_SEED:-$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')}"
artifact_dir="$project_root/artifacts/interactive_p1b"

read -r camera_height camera_fov < <(
    "$model_python" -c \
        "import random; r=random.Random($seed); print(f'{r.uniform(0.30,0.80):.4f} {r.uniform(60.0,120.0):.4f}')"
)

for required in \
    "$model_python" \
    "$scene" \
    "$navmesh" \
    "$checkpoint" \
    "$backbone_checkpoint"
do
    if [[ ! -e "$required" ]]; then
        echo "Required path is missing: $required" >&2
        exit 1
    fi
done

mkdir -p "$artifact_dir"
runtime_log="$artifact_dir/runtime.log"
ui_log="$artifact_dir/ui.log"
session_json="$artifact_dir/session.json"
runtime_pid=""

cleanup() {
    if [[ -n "$runtime_pid" ]] && kill -0 "$runtime_pid" 2>/dev/null; then
        kill "$runtime_pid"
        wait "$runtime_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"$model_python" "$project_root/simulator_ui/p1b_runtime_server.py" \
    --checkpoint "$checkpoint" \
    --backbone-source "$backbone_source" \
    --backbone-checkpoint "$backbone_checkpoint" \
    --device cuda:0 \
    --host 127.0.0.1 \
    --port "$runtime_port" \
    --max-history "$max_history" \
    >"$runtime_log" 2>&1 &
runtime_pid="$!"

runtime_url="http://127.0.0.1:${runtime_port}"
for _ in $(seq 1 180); do
    if ! kill -0 "$runtime_pid" 2>/dev/null; then
        echo "P1B runtime exited during startup:" >&2
        tail -n 80 "$runtime_log" >&2
        exit 1
    fi
    if curl -fsS "$runtime_url/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
read -r checkpoint_epoch checkpoint_step < <(
    curl -fsS "$runtime_url/health" | "$model_python" -c \
        "import json, sys; health=json.load(sys.stdin); print(health['checkpoint_epoch'], health['checkpoint_global_step'])"
)

"$model_python" -c \
    "import json; print(json.dumps({'scene_key':'habitat-test-scenes:skokloster-castle','dataset_status':'unbuilt_scene','full_manifest_scene_overlap':False,'full_manifest_sha256':'1e66a2b978e07f3e5fcfe0040955c50d3c47a36007058a589ef0ca86e8dcd017','seed':$seed,'camera_height_m':$camera_height,'horizontal_fov_degrees':$camera_fov,'max_history':$max_history,'checkpoint_epoch':$checkpoint_epoch,'checkpoint_global_step':$checkpoint_step,'trained_outputs':['single_bev','scale'],'disabled_outputs':['merged_bev'],'gt_fov_contract':'geometric cone intersect square BEV; no radial crop'},indent=2))" \
    >"$session_json"

echo "P1B runtime ready: $runtime_url"
echo "P1B UI: http://127.0.0.1:${ui_port}"
echo "Scene: habitat-test-scenes:skokloster-castle (absent from full training manifest)"
echo "Checkpoint: epoch ${checkpoint_epoch}, step ${checkpoint_step}; Single BEV + confidence + scale; merged disabled"
echo "Random camera: height=${camera_height}m, hfov=${camera_fov}deg, seed=${seed}"

"$project_root/simulator_ui/run-in-env" \
    python "$project_root/simulator_ui/web_navigation.py" \
    "$scene" \
    --navmesh "$navmesh" \
    --model p1b \
    --model-server-url "$runtime_url" \
    --model-max-history "$max_history" \
    --model-hz 0.5 \
    --voxel-size 0.1 \
    --random-start \
    --random-yaw \
    --seed "$seed" \
    --sensor-height-m "$camera_height" \
    --horizontal-fov-degrees "$camera_fov" \
    --host 127.0.0.1 \
    --port "$ui_port" \
    --no-browser \
    >"$ui_log" 2>&1
