#!/usr/bin/env bash
# start_dashboard.sh — Start the Lopen web dashboard on port 8080

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/logs"
PID_DIR="$REPO_ROOT/.pids"
VENV="$REPO_ROOT/.venv/bin/activate"

mkdir -p "$LOG_DIR" "$PID_DIR"
[ -f "$VENV" ] && source "$VENV"
cd "$REPO_ROOT"

echo "--> Starting web dashboard on port 8080..."
nohup python -c "
import uvicorn
from interfaces.web_dashboard.app import create_dashboard_app
app = create_dashboard_app()
uvicorn.run(app, host='0.0.0.0', port=8080, log_level='info')
" >> "$LOG_DIR/dashboard.log" 2>&1 &

echo $! > "$PID_DIR/dashboard.pid"
echo "    Dashboard started (PID $!). Log: $LOG_DIR/dashboard.log"
sleep 2
curl -sf http://localhost:8080/health > /dev/null && echo "    Health check: OK" || echo "    Health check: PENDING"
