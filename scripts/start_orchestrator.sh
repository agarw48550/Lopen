#!/usr/bin/env bash
# start_orchestrator.sh — Start the main Lopen orchestrator on port 8000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/logs"
PID_DIR="$REPO_ROOT/.pids"
VENV="$REPO_ROOT/.venv/bin/activate"

mkdir -p "$LOG_DIR" "$PID_DIR"
[ -f "$VENV" ] && source "$VENV"
cd "$REPO_ROOT"

echo "--> Starting orchestrator on port 8000..."
nohup python -m uvicorn orchestrator:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    >> "$LOG_DIR/orchestrator.log" 2>&1 &

echo $! > "$PID_DIR/orchestrator.pid"
echo "    Orchestrator started (PID $!). Log: $LOG_DIR/orchestrator.log"
sleep 2
curl -sf http://localhost:8000/health > /dev/null && echo "    Health check: OK" || echo "    Health check: PENDING"
