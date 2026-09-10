#!/usr/bin/env bash
set -uo pipefail

root=/mnt/data/benyun/Leju-Kuavo5W/vbev
code="$root/m05_plus_train"
watchdog_dir="$code/watchdog"
log="$watchdog_dir/m05_plus_idle_watchdog.log"
state="$watchdog_dir/m05_plus_idle_watchdog.state"
lock="$watchdog_dir/m05_plus_idle_watchdog.lock"
resume_script="$code/scripts/resume_m05_plus_step260000_a100.sh"
resume_log="$code/logs/m05_plus_watchdog_resume_step260000_20260909_2100.log"
train_tmux=m05_plus_a100_resume_step260000_watchdog
interval_seconds=${M05_WATCHDOG_INTERVAL_SECONDS:-1500}
memory_idle_threshold_mib=${M05_WATCHDOG_MEMORY_IDLE_MIB:-1024}
utilization_idle_threshold=${M05_WATCHDOG_UTIL_IDLE_PERCENT:-5}

mkdir -p "$watchdog_dir"
exec 9>"$lock"
flock -n 9 || {
    printf '%s watchdog already active\n' "$(date -Is)" >>"$log"
    exit 9
}

start_epoch=$(date -d '2026-09-09 21:00:00 +0800' +%s)
printf '%s waiting_until=2026-09-09T21:00:00+08:00\n' "$(date -Is)" >>"$log"
while (( $(date +%s) < start_epoch )); do
    sleep "$((start_epoch - $(date +%s) + 1))"
done
idle_count=0

write_state() {
    local status="$1"
    local temporary="$state.tmp.$$"
    {
        printf 'idle_count=%s\n' "$idle_count"
        printf 'status=%s\n' "$status"
        printf 'updated_at=%s\n' "$(date -Is)"
        printf 'checkpoint_step=265000\n'
    } >"$temporary"
    mv "$temporary" "$state"
}

training_running() {
    pgrep -u "$(id -u)" -f '[v]ggt_bev_method1.cli_train_m05_plus' >/dev/null
}

all_gpus_idle() {
    local rows=0
    local index memory utilization
    local compute_pids
    compute_pids=$(nvidia-smi \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF && $1 ~ /^[0-9]+$/ {print $1}') || return 1
    [[ -z "$compute_pids" ]] || return 1

    while IFS=',' read -r index memory utilization; do
        index=${index//[[:space:]]/}
        memory=${memory//[[:space:]]/}
        utilization=${utilization//[[:space:]]/}
        [[ "$index" =~ ^[0-9]+$ ]] || return 1
        [[ "$memory" =~ ^[0-9]+$ ]] || return 1
        [[ "$utilization" =~ ^[0-9]+$ ]] || return 1
        (( memory <= memory_idle_threshold_mib )) || return 1
        (( utilization <= utilization_idle_threshold )) || return 1
        ((rows += 1))
    done < <(
        nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null
    )
    (( rows == 8 ))
}

gpu_snapshot() {
    nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | paste -sd ';' -
}

printf '%s watchdog start interval_seconds=%s required_consecutive_idle=2 memory_idle_mib=%s utilization_idle_percent=%s\n' \
    "$(date -Is)" "$interval_seconds" "$memory_idle_threshold_mib" \
    "$utilization_idle_threshold" >>"$log"

while :; do
    if training_running; then
        printf '%s M05+ training already running; watchdog exits\n' "$(date -Is)" >>"$log"
        write_state training_already_running
        exit 0
    fi

    snapshot=$(gpu_snapshot)
    if all_gpus_idle; then
        ((idle_count += 1))
        status="idle_${idle_count}_of_2"
    else
        idle_count=0
        status=busy
    fi
    printf '%s status=%s gpu_snapshot=%s\n' "$(date -Is)" "$status" "$snapshot" >>"$log"
    write_state "$status"

    if (( idle_count >= 2 )); then
        if training_running; then
            printf '%s M05+ appeared before launch; watchdog exits\n' "$(date -Is)" >>"$log"
            write_state training_already_running
            exit 0
        fi
        if ! all_gpus_idle; then
            idle_count=0
            printf '%s final idle recheck failed; counter reset\n' "$(date -Is)" >>"$log"
            write_state busy_on_final_recheck
        elif tmux has-session -t "$train_tmux" 2>/dev/null; then
            printf '%s resume tmux already exists; watchdog exits\n' "$(date -Is)" >>"$log"
            write_state resume_tmux_already_present
            exit 0
        else
            printf '%s two consecutive idle checks passed; launching resume\n' "$(date -Is)" >>"$log"
            tmux new-session -d -s "$train_tmux" \
                "cd '$code' && exec '$resume_script' >> '$resume_log' 2>&1"
            sleep 10
            if training_running || tmux has-session -t "$train_tmux" 2>/dev/null; then
                printf '%s resume launched successfully; watchdog exits\n' "$(date -Is)" >>"$log"
                write_state resume_launched
                exit 0
            fi
            idle_count=0
            printf '%s resume launch failed; counter reset for retry\n' "$(date -Is)" >>"$log"
            write_state resume_launch_failed
        fi
    fi

    sleep "$interval_seconds"
done
