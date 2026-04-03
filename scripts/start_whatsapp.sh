#!/usr/bin/env bash
# start_whatsapp.sh — Start the Lopen WhatsApp bridge

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/logs"
PID_DIR="$REPO_ROOT/.pids"
VENV="$REPO_ROOT/.venv/bin/activate"

mkdir -p "$LOG_DIR" "$PID_DIR"
[ -f "$VENV" ] && source "$VENV"
cd "$REPO_ROOT"

echo "--> Starting WhatsApp bridge..."
nohup python -c "
import asyncio
from interfaces.whatsapp_service.bridge import WhatsAppBridge
from interfaces.whatsapp_service.handler import WhatsAppHandler

async def run():
    bridge = WhatsAppBridge(headless=True)
    handler = WhatsAppHandler(bridge)
    await handler.start()

asyncio.run(run())
" >> "$LOG_DIR/whatsapp.log" 2>&1 &

echo $! > "$PID_DIR/whatsapp.pid"
echo "    WhatsApp bridge started (PID $!). Log: $LOG_DIR/whatsapp.log"
echo "    Note: check logs for QR code prompt on first run."
