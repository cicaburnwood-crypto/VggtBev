#!/usr/bin/env bash
set -euo pipefail

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pid_file="$bundle_root/run/launcher.pid"

if [[ ! -f "$pid_file" ]]; then
  echo "Visualizer is not running."
  exit 0
fi

launcher_pid="$(<"$pid_file")"
if [[ ! "$launcher_pid" =~ ^[0-9]+$ ]] || ! kill -0 "$launcher_pid" 2>/dev/null; then
  rm -f "$pid_file"
  echo "Removed a stale launcher PID file."
  exit 0
fi

kill -TERM "$launcher_pid"
echo "Stopped visualizer launcher PID $launcher_pid."
