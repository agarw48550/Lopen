#!/usr/bin/env bash
# diagnose.sh — Lopen self-diagnostics and system health check
# Run this script to quickly verify your Lopen installation and system state.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}✔${RESET}  $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail() { echo -e "  ${RED}✖${RESET}  $1"; }
info() { echo -e "  ${DIM}ℹ${RESET}  $1"; }

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo ""
echo -e "${CYAN}${BOLD}  Lopen Diagnostics${RESET}"
echo -e "${DIM}  ────────────────────────────────────────${RESET}"
echo -e "  ${DIM}$(date)${RESET}"
echo ""

# ---------------------------------------------------------------------------
# 1. Operating System
# ---------------------------------------------------------------------------
echo -e "${BOLD}  [1] Operating System${RESET}"
OS_NAME="$(uname -s)"
OS_VER="$(uname -r)"
echo -e "      OS      : ${OS_NAME} ${OS_VER}"
if [[ "$OS_NAME" == "Darwin" ]]; then
    MAC_VER="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
    ARCH="$(uname -m)"
    echo -e "      macOS   : ${MAC_VER}  (arch: ${ARCH})"
    if [[ "$ARCH" == "x86_64" ]]; then
        ok "Intel Mac detected — correct platform"
    else
        warn "ARM Mac detected — some llama.cpp builds may differ"
    fi
else
    info "Non-macOS system — some features (AppleScript, piper) may be unavailable"
fi
echo ""

# ---------------------------------------------------------------------------
# 2. Python
# ---------------------------------------------------------------------------
echo -e "${BOLD}  [2] Python${RESET}"
if command -v python3 &>/dev/null; then
    PY_VER="$(python3 --version 2>&1)"
    ok "${PY_VER}"
else
    fail "python3 not found — install Python 3.11+ from https://www.python.org/downloads/macos/ or run: xcode-select --install"
fi

# Check if venv is active
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    ok "Virtual environment active: ${VIRTUAL_ENV}"
elif [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    warn "Virtual environment exists but not active — run: source .venv/bin/activate"
else
    warn "No virtual environment found — run: bash scripts/setup_venv.sh"
fi

# Check key Python packages
# Format: "import_name|distribution_name"
PACKAGES=(
    "fastapi|fastapi"
    "uvicorn|uvicorn"
    "yaml|PyYAML"
    "psutil|psutil"
    "httpx|httpx"
    "numpy|numpy"
)
for spec in "${PACKAGES[@]}"; do
    import_name="${spec%%|*}"
    dist_name="${spec##*|}"
    if python3 -c "import ${import_name}" 2>/dev/null; then
        VER=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('${dist_name}'))" 2>/dev/null || echo "?")
        ok "${dist_name}==${VER}"
    else
        warn "${dist_name} not installed — run: pip install ${dist_name}"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 3. Memory
# ---------------------------------------------------------------------------
echo -e "${BOLD}  [3] Memory${RESET}"
if command -v python3 &>/dev/null && python3 -c "import psutil" 2>/dev/null; then
    RAM_TOTAL=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().total/1024**3:.1f}')")
    RAM_AVAIL=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().available/1024**3:.1f}')")
    RAM_PCT=$(python3 -c "import psutil; print(f'{psutil.virtual_memory().percent:.0f}')")
    echo -e "      Total   : ${RAM_TOTAL} GB"
    echo -e "      Free    : ${RAM_AVAIL} GB"
    if python3 -c "import sys; sys.exit(0 if float('${RAM_AVAIL}') > 1.5 else 1)" 2>/dev/null; then
        ok "Sufficient memory available for Lopen (need ≥1.5 GB free)"
    else
        warn "Low free memory (${RAM_AVAIL} GB) — close other apps before starting Lopen"
    fi
    if python3 -c "import sys; sys.exit(0 if float('${RAM_TOTAL}') >= 4.0 else 1)" 2>/dev/null; then
        ok "Total RAM ${RAM_TOTAL} GB — meets 4 GB requirement"
    else
        warn "Total RAM ${RAM_TOTAL} GB — below 4 GB target; may need to reduce model size"
    fi
else
    MEM_RAW="$(vm_stat 2>/dev/null | grep 'Pages free' | awk '{print $3}' | tr -d '.' || echo '?')"
    info "psutil not installed — install for detailed memory stats"
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Disk Space
# ---------------------------------------------------------------------------
echo -e "${BOLD}  [4] Disk Space${RESET}"
if command -v df &>/dev/null; then
    DISK_FREE=$(df -g "$REPO_ROOT" 2>/dev/null | awk 'NR==2{print $4}' || echo "?")
    if [[ "$DISK_FREE" != "?" ]]; then
        echo -e "      Free    : ${DISK_FREE} GB (at ${REPO_ROOT})"
        if [[ "$DISK_FREE" -ge 10 ]]; then
            ok "Sufficient disk space for models and logs"
        elif [[ "$DISK_FREE" -ge 5 ]]; then
            warn "Disk space is limited (${DISK_FREE} GB) — models need ~3-5 GB"
        else
            fail "Very low disk space (${DISK_FREE} GB) — free up space before downloading models"
        fi
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# 5. Models
# ---------------------------------------------------------------------------
echo -e "${BOLD}  [5] Models${RESET}"
MODEL_DIR="$REPO_ROOT/models"
LLM_DIR="$MODEL_DIR/llm"
ASR_DIR="$MODEL_DIR/asr"
TTS_DIR="$MODEL_DIR/tts"

# LLM
LLM_FILES=($(ls "$LLM_DIR"/*.gguf 2>/dev/null || true))
if [[ ${#LLM_FILES[@]} -gt 0 ]]; then
    for f in "${LLM_FILES[@]}"; do
        SIZE_MB=$(( $(wc -c < "$f") / 1024 / 1024 ))
        ok "LLM: $(basename $f)  (${SIZE_MB} MB)"
    done
else
    warn "No LLM model found in models/llm/"
    info "Download with: bash scripts/download_models.sh"
fi

# ASR
ASR_FILES=($(ls "$ASR_DIR"/*.bin 2>/dev/null || true))
if [[ ${#ASR_FILES[@]} -gt 0 ]]; then
    for f in "${ASR_FILES[@]}"; do
        SIZE_MB=$(( $(wc -c < "$f") / 1024 / 1024 ))
        ok "ASR: $(basename $f)  (${SIZE_MB} MB)"
    done
else
    warn "No ASR model found in models/asr/ — voice input will be unavailable"
fi

# TTS
TTS_FILES=($(ls "$TTS_DIR"/*.onnx 2>/dev/null || true))
if [[ ${#TTS_FILES[@]} -gt 0 ]]; then
    for f in "${TTS_FILES[@]}"; do
        SIZE_MB=$(( $(wc -c < "$f") / 1024 / 1024 ))
        ok "TTS: $(basename $f)  (${SIZE_MB} MB)"
    done
else
    warn "No TTS model found in models/tts/ — voice output will be unavailable"
fi
echo ""

# ---------------------------------------------------------------------------
# 6. System Binaries
# ---------------------------------------------------------------------------
echo -e "${BOLD}  [6] System Binaries${RESET}"
BINS=("git" "ffmpeg" "portaudio" "cmake")
for bin in "${BINS[@]}"; do
    if command -v "$bin" &>/dev/null; then
        ok "${bin} found: $(which ${bin})"
    else
        warn "${bin} not found"
    fi
done

for opt_bin in "piper" "whisper"; do
    if command -v "$opt_bin" &>/dev/null; then
        ok "${opt_bin} found: $(which ${opt_bin})"
    else
        info "${opt_bin} not found (optional — needed for voice)"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 7. Services
# ---------------------------------------------------------------------------
echo -e "${BOLD}  [7] Services${RESET}"
PID_DIR="$REPO_ROOT/.pids"
SERVICES=("orchestrator" "dashboard" "voice" "whatsapp")
ANY_RUNNING=false
for svc in "${SERVICES[@]}"; do
    pidfile="$PID_DIR/${svc}.pid"
    if [[ -f "$pidfile" ]]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            ok "${svc} running (PID ${pid})"
            ANY_RUNNING=true
        else
            warn "${svc} PID file exists but process is dead (stale)"
        fi
    else
        info "${svc} not started"
    fi
done

# HTTP checks
if command -v curl &>/dev/null; then
    for url in "http://localhost:8000/health" "http://localhost:8080/health"; do
        CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$url" 2>/dev/null || echo "ERR")
        if [[ "$CODE" == "200" ]]; then
            ok "${url}  → HTTP 200"
        else
            info "${url}  → ${CODE} (not running or unreachable)"
        fi
    done
fi
echo ""

# ---------------------------------------------------------------------------
# 8. Configuration
# ---------------------------------------------------------------------------
echo -e "${BOLD}  [8] Configuration${RESET}"
CONFIG_FILES=("config/settings.yaml" "config/models.yaml" "config/logging.yaml" "config/agents.yaml" ".env")
for cfg in "${CONFIG_FILES[@]}"; do
    if [[ -f "$REPO_ROOT/$cfg" ]]; then
        ok "${cfg}"
    else
        if [[ "$cfg" == ".env" ]]; then
            info ".env not found — copy .env.example to .env and fill in values"
        else
            fail "${cfg} missing — check your installation"
        fi
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 9. Logs
# ---------------------------------------------------------------------------
echo -e "${BOLD}  [9] Logs${RESET}"
LOG_DIR="$REPO_ROOT/logs"
if [[ -d "$LOG_DIR" ]]; then
    LOG_COUNT=$(ls "$LOG_DIR"/*.log 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$LOG_COUNT" -gt 0 ]]; then
        ok "${LOG_COUNT} log file(s) in ${LOG_DIR}/"
        # Show sizes
        ls -lh "$LOG_DIR"/*.log 2>/dev/null | awk '{print "         " $5 "  " $9}' | while read -r line; do
            echo -e "${DIM}${line}${RESET}"
        done
        # Warn if any log > 50 MB
        for f in "$LOG_DIR"/*.log; do
            SIZE_MB=$(( $(wc -c < "$f") / 1024 / 1024 ))
            if [[ "$SIZE_MB" -gt 50 ]]; then
                warn "$(basename $f) is ${SIZE_MB} MB — consider rotating (scripts/health_check.sh)"
            fi
        done
    else
        info "No log files yet (will be created on first run)"
    fi
else
    warn "logs/ directory missing — creating it now"
    mkdir -p "$LOG_DIR"
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo -e "${DIM}  ────────────────────────────────────────${RESET}"
echo -e "${BOLD}  Diagnostics complete.${RESET}"
echo ""
echo -e "  ${DIM}To start Lopen:       bash scripts/start.sh${RESET}"
echo -e "  ${DIM}To download models:   bash scripts/download_models.sh${RESET}"
echo -e "  ${DIM}To view status:       bash scripts/status.sh${RESET}"
echo -e "  ${DIM}Interactive CLI:      python cli.py${RESET}"
echo ""
