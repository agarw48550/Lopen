#!/usr/bin/env bash
# start_voice.sh — Start the Lopen voice service

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/logs"
PID_DIR="$REPO_ROOT/.pids"
VENV="$REPO_ROOT/.venv/bin/activate"

mkdir -p "$LOG_DIR" "$PID_DIR"
[ -f "$VENV" ] && source "$VENV"
cd "$REPO_ROOT"

echo "--> Starting voice service..."
nohup python -c "
import asyncio
from interfaces.voice_service.voice_loop import VoiceLoop
from interfaces.voice_service.wake_word import WakeWordDetector
from interfaces.voice_service.asr_adapter import ASRAdapter
from interfaces.voice_service.tts_adapter import TTSAdapter

loop = asyncio.new_event_loop()
wwd = WakeWordDetector()
asr = ASRAdapter()
tts = TTSAdapter()
vl = VoiceLoop(wwd, asr, tts)
loop.run_until_complete(vl.start())
" >> "$LOG_DIR/voice.log" 2>&1 &

echo $! > "$PID_DIR/voice.pid"
echo "    Voice service started (PID $!). Log: $LOG_DIR/voice.log"
