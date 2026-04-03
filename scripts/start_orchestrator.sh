#!/usr/bin/env bash
# start_orchestrator.sh — Start the main Lopen orchestrator on port 8000
#
# Usage:
#   bash scripts/start_orchestrator.sh          # normal mode
#   bash scripts/start_orchestrator.sh --debug  # debug mode (verbose logging)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/logs"
PID_DIR="$REPO_ROOT/.pids"
VENV="$REPO_ROOT/.venv/bin/activate"

mkdir -p "$LOG_DIR" "$PID_DIR"
[ -f "$VENV" ] && source "$VENV"
cd "$REPO_ROOT"

# Parse flags
DEBUG_MODE=false
for arg in "$@"; do
    [[ "$arg" == "--debug" ]] && DEBUG_MODE=true
done

LOG_LEVEL="info"
if [[ "$DEBUG_MODE" == "true" ]]; then
    export LOPEN_DEBUG=1
    export LOPEN_LOG_LEVEL=DEBUG
    LOG_LEVEL="debug"
    echo "--> Debug mode ON — verbose logging to $LOG_DIR/lopen_debug.log"
fi

echo "--> Starting orchestrator on port 8000 (log-level: ${LOG_LEVEL})..."
nohup python -m uvicorn orchestrator:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level "${LOG_LEVEL}" \
    >> "$LOG_DIR/orchestrator.log" 2>&1 &

echo $! > "$PID_DIR/orchestrator.pid"
echo "    Orchestrator started (PID $!). Log: $LOG_DIR/orchestrator.log"
sleep 2
curl -sf http://localhost:8000/health > /dev/null && echo "    Health check: OK" || echo "    Health check: PENDING"
