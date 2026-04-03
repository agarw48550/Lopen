#!/usr/bin/env bash
# stop.sh — Stop all Lopen services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$(dirname "$SCRIPT_DIR")/.pids"

echo "==> Stopping Lopen services..."

for pidfile in "$PID_DIR"/*.pid; do
    [ -f "$pidfile" ] || continue
    svc=$(basename "$pidfile" .pid)
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
        echo "--> Stopping $svc (PID $pid)..."
        kill "$pid" && sleep 1
        kill -9 "$pid" 2>/dev/null || true
    else
        echo "--> $svc (PID $pid) not running"
    fi
    rm -f "$pidfile"
done

echo "==> All services stopped."
