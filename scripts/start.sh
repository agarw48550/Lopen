#!/usr/bin/env bash
# start.sh — Start all Lopen services
#
# Usage:
#   bash scripts/start.sh          # normal start
#   bash scripts/start.sh --debug  # start with verbose debug logging

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_ROOT/.venv/bin/activate"
LOG_DIR="$REPO_ROOT/logs"
PID_DIR="$REPO_ROOT/.pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
fi

cd "$REPO_ROOT"

# Pass through any flags (e.g. --debug)
FLAGS="$@"

echo "==> Starting Lopen services..."

bash "$SCRIPT_DIR/start_orchestrator.sh" $FLAGS
bash "$SCRIPT_DIR/start_dashboard.sh"

echo "==> All services started."
echo "    Orchestrator: http://localhost:8000/health"
echo "    Dashboard:    http://localhost:8080/"
echo "    Logs:         $LOG_DIR/"
echo ""
echo "    Interactive CLI:  python cli.py"
echo "    Status:           bash scripts/status.sh"
echo "    Diagnostics:      bash scripts/diagnose.sh"
