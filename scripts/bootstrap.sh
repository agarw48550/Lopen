#!/usr/bin/env bash
# bootstrap.sh — Install macOS system dependencies for Lopen without Homebrew
#
# This script guides you through a fully Homebrew-free setup:
#   - Python 3.11+ via python.org installer (if needed)
#   - cmake via pip  (for llama-cpp-python compilation)
#   - piper TTS binary via direct GitHub release download
#   - whisper.cpp ASR binary via direct GitHub release download
#   - ffmpeg via static pre-built binary (evermeet.cx)
#   - sounddevice (pip) replaces portaudio/pyaudio system dependency
#
# All binaries go to ~/.local/bin (added to PATH automatically by Lopen).

set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
DIM='\033[2m'; RESET='\033[0m'; BOLD='\033[1m'
ok()   { echo -e "  ${GREEN}✔${RESET}  $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
info() { echo -e "  ${DIM}ℹ${RESET}  $1"; }

LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"

echo ""
echo -e "${BOLD}${CYAN}  Lopen bootstrap — Homebrew-free system dependency install${RESET}"
echo ""

# ---------------------------------------------------------------------------
# Python 3.11+
# ---------------------------------------------------------------------------
echo -e "${BOLD}  [1] Python${RESET}"
if python3 -c "import sys; assert sys.version_info >= (3,9)" 2>/dev/null; then
    ok "Python $(python3 --version 2>&1 | awk '{print $2}') — OK"
else
    warn "Python 3.9+ not found."
    echo -e "  ${CYAN}  Download from: https://www.python.org/downloads/macos/${RESET}"
    echo -e "  ${CYAN}  Or run: xcode-select --install  (includes Python 3)${RESET}"
fi

# ---------------------------------------------------------------------------
# cmake via pip (no system install needed)
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}  [2] cmake (via pip)${RESET}"
if command -v cmake &>/dev/null; then
    ok "cmake already available: $(cmake --version | head -1)"
else
    info "Installing cmake via pip..."
    python3 -m pip install --quiet --upgrade cmake && ok "cmake installed via pip" \
        || warn "cmake pip install failed — try: pip install cmake"
fi

# ---------------------------------------------------------------------------
# piper TTS binary (pre-built release, Intel macOS)
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}  [3] piper TTS binary${RESET}"
PIPER_BIN="$LOCAL_BIN/piper"
if command -v piper &>/dev/null || [ -x "$PIPER_BIN" ]; then
    ok "piper found: $(command -v piper 2>/dev/null || echo "$PIPER_BIN")"
else
    warn "piper not found."
    ARCH="$(uname -m)"
    if [[ "$ARCH" == "x86_64" ]]; then
        PIPER_ASSET="piper_macos_x64.tar.gz"
    else
        PIPER_ASSET="piper_macos_aarch64.tar.gz"
    fi
    PIPER_URL="https://github.com/rhasspy/piper/releases/latest/download/${PIPER_ASSET}"
    info "To install manually:"
    echo -e "  ${CYAN}  curl -L '${PIPER_URL}' | tar -xz -C /tmp/piper_install${RESET}"
    echo -e "  ${CYAN}  cp /tmp/piper_install/piper/piper '${PIPER_BIN}'${RESET}"
    echo -e "  ${CYAN}  chmod +x '${PIPER_BIN}'${RESET}"
    info "Or set LOPEN_PIPER_BINARY=/path/to/piper in .env — macOS 'say' is the fallback TTS."
fi

# ---------------------------------------------------------------------------
# whisper.cpp ASR binary (pre-built release)
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}  [4] whisper.cpp ASR binary${RESET}"
WHISPER_BIN="$LOCAL_BIN/whisper"
if command -v whisper &>/dev/null || [ -x "$WHISPER_BIN" ]; then
    ok "whisper found: $(command -v whisper 2>/dev/null || echo "$WHISPER_BIN")"
else
    warn "whisper.cpp not found."
    info "Build from source (takes ~5 min, requires Xcode CLT):"
    echo -e "  ${CYAN}  git clone https://github.com/ggerganov/whisper.cpp /tmp/whisper.cpp${RESET}"
    echo -e "  ${CYAN}  cd /tmp/whisper.cpp && cmake -B build -DWHISPER_METAL=OFF && cmake --build build -j4${RESET}"
    echo -e "  ${CYAN}  cp /tmp/whisper.cpp/build/bin/whisper-cli '${WHISPER_BIN}'${RESET}"
    echo -e "  ${CYAN}  chmod +x '${WHISPER_BIN}'${RESET}"
    info "Or set LOPEN_WHISPER_BINARY in .env. Python-based faster-whisper is the fallback."
fi

# ---------------------------------------------------------------------------
# ffmpeg (optional — static binary, no Homebrew)
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}  [5] ffmpeg (optional)${RESET}"
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg found: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f1-3)"
else
    warn "ffmpeg not found — voice pipeline will use sounddevice (no ffmpeg needed)."
    info "To install a static build (Intel macOS, no Homebrew):"
    echo -e "  ${CYAN}  curl -L 'https://evermeet.cx/ffmpeg/ffmpeg-7.1.1.zip' -o /tmp/ffmpeg.zip${RESET}"
    echo -e "  ${CYAN}  unzip /tmp/ffmpeg.zip -d '$LOCAL_BIN' && chmod +x '$LOCAL_BIN/ffmpeg'${RESET}"
fi

# ---------------------------------------------------------------------------
# sounddevice (replaces portaudio pip package dependency)
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}  [6] sounddevice (Python audio, replaces portaudio)${RESET}"
if python3 -c "import sounddevice" 2>/dev/null; then
    ok "sounddevice installed"
else
    info "Installing sounddevice (bundles PortAudio — no system install needed)..."
    python3 -m pip install --quiet sounddevice && ok "sounddevice installed" \
        || warn "sounddevice install failed — run: pip install sounddevice"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}  Bootstrap complete.${RESET}"
echo -e "${DIM}  Add ~/.local/bin to your PATH if not already present:${RESET}"
echo -e "  ${CYAN}  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc && source ~/.zshrc${RESET}"
echo ""
echo -e "${DIM}  Next: bash scripts/setup_venv.sh && bash install.sh --no-models${RESET}"
echo ""
