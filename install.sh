#!/usr/bin/env bash
# install.sh — One-command Lopen installer for macOS (Intel or Apple Silicon)
#
# Usage:
#   bash install.sh                          # full install (interactive)
#   bash install.sh --no-models              # skip model downloads
#   bash install.sh --yes --no-models        # non-interactive quick install
#   bash install.sh --yes --with-llama       # non-interactive + llama-cpp-python
#   bash install.sh --debug                  # verbose output
#
# What this does:
#   1. Checks OS, Python 3.9+, and system tools (NO Homebrew required)
#   2. Checks/guides installation of optional system tools (cmake, ffmpeg,
#      portaudio) using direct download links or macOS-native alternatives
#   3. Creates a Python virtual environment in .venv/
#   4. Installs all Python requirements (pip only — no Homebrew)
#   5. Copies .env.example → .env (if not exists)
#   6. Creates required directories (logs/, models/, .pids/)
#   7. Optionally downloads AI models (Qwen3.5-0.8B + whisper-tiny + Piper TTS)
#   8. Runs self-diagnostics to verify the installation
#   9. Prints a quickstart guide
#
# Homebrew-free: This installer never requires or uses Homebrew.
# All dependencies are fetched via pip, direct binary downloads, or are
# already present on macOS 12+ / macOS 13+ (Ventura, Sonoma, Sequoia).

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours & helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'
STEP=0

step()  { STEP=$((STEP+1)); echo ""; echo -e "${BOLD}${CYAN}[${STEP}] $1${RESET}"; }
ok()    { echo -e "    ${GREEN}✔${RESET}  $1"; }
warn()  { echo -e "    ${YELLOW}⚠${RESET}  $1"; }
fail()  { echo -e "    ${RED}✖${RESET}  $1"; }
info()  { echo -e "    ${DIM}ℹ${RESET}  $1"; }
die()   { echo -e "\n${RED}ERROR: $1${RESET}\n"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
SKIP_MODELS=false
DEBUG=false
AUTO_YES=false
INSTALL_LLAMA="ask" # ask | yes | no
for arg in "$@"; do
    case "$arg" in
        --no-models) SKIP_MODELS=true ;;
        --debug)     DEBUG=true; set -x ;;
        --yes|-y)    AUTO_YES=true ;;
        --with-llama) INSTALL_LLAMA="yes" ;;
        --without-llama) INSTALL_LLAMA="no" ;;
        --help|-h)
            echo "Usage: bash install.sh [--no-models] [--yes|-y] [--with-llama|--without-llama] [--debug]"
            exit 0 ;;
    esac
done

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
clear
echo ""
echo -e "${CYAN}${BOLD}"
cat << 'EOF'
  _
 | |    ___  _ __   ___ _ __
 | |   / _ \| '_ \ / _ \ '_ \
 | |__| (_) | |_) |  __/ | | |
 |_____\___/| .__/ \___|_| |_|
            |_|
EOF
echo -e "${RESET}"
echo -e "${BOLD}  Lopen — Local-First Autonomous Assistant${RESET}"
echo -e "${DIM}  One-command installer for macOS${RESET}"
echo ""
echo -e "${DIM}  Target: 2017+ Intel MacBook Pro, ≤4 GB RAM${RESET}"
echo -e "${DIM}  Source: https://github.com/agarw48550/Lopen${RESET}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: OS check
# ---------------------------------------------------------------------------
step "Checking system"
OS="$(uname -s)"
ARCH="$(uname -m)"
echo -e "    OS   : ${OS} (${ARCH})"
if [[ "$OS" != "Darwin" ]]; then
    warn "Lopen is optimised for macOS. Continuing anyway (Linux should work too)."
fi
if [[ "$ARCH" == "x86_64" ]]; then
    ok "Intel Mac detected — this is the primary supported platform"
elif [[ "$ARCH" == "arm64" ]]; then
    ok "Apple Silicon detected — M-series Macs are supported (even better performance)"
else
    warn "Unknown architecture: ${ARCH}"
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' || echo "not found")
echo -e "    Python: ${PYTHON_VERSION}"
if python3 -c "import sys; assert sys.version_info >= (3,9)" 2>/dev/null; then
    ok "Python version OK"
else
    fail "Python 3.9+ required."
    echo ""
    echo -e "  ${YELLOW}Python 3.9 or newer is required. Install options (no Homebrew needed):${RESET}"
    echo -e "  ${CYAN}  • Download installer: https://www.python.org/downloads/macos/${RESET}"
    echo -e "  ${CYAN}  • Or install Xcode Command Line Tools (includes Python 3): xcode-select --install${RESET}"
    echo ""
    die "Please install Python 3.9+ and re-run this script."
fi

# ---------------------------------------------------------------------------
# Step 2: Check system tools (Homebrew-free)
# ---------------------------------------------------------------------------
step "Checking system tools"

# cmake — needed to compile llama-cpp-python
# Available via: pip install cmake  (no Homebrew required)
if command -v cmake &>/dev/null; then
    ok "cmake found: $(cmake --version | head -1)"
else
    warn "cmake not found — will install via pip (pip install cmake)"
    info "cmake is only needed to compile llama-cpp-python from source"
fi

# git — standard on macOS (Xcode CLT) or available at git-scm.com
if command -v git &>/dev/null; then
    ok "git found: $(git --version)"
else
    warn "git not found"
    info "Install Xcode Command Line Tools: xcode-select --install"
fi

# curl — built into macOS; used for model downloads
if command -v curl &>/dev/null; then
    ok "curl found"
else
    warn "curl not found — model downloads will try wget"
fi

# ffmpeg — optional; needed only for audio conversion in voice pipeline
# Homebrew-free options:
#   a) Pre-built static binary (recommended): https://evermeet.cx/ffmpeg/
#      curl -L https://evermeet.cx/ffmpeg/ffmpeg-7.1.1.zip -o /tmp/ffmpeg.zip
#      unzip /tmp/ffmpeg.zip -d /usr/local/bin && chmod +x /usr/local/bin/ffmpeg
#   b) python-ffmpeg-binary pip package (Python wrapper with bundled ffmpeg)
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg found: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f1-3)"
else
    warn "ffmpeg not found — voice pipeline will fall back to sounddevice/pyaudio"
    info "Optional — to install WITHOUT Homebrew, download a static build:"
    info "  curl -L https://evermeet.cx/ffmpeg/ffmpeg-7.1.1.zip -o /tmp/ffmpeg.zip"
    info "  sudo unzip -o /tmp/ffmpeg.zip -d /usr/local/bin && sudo chmod +x /usr/local/bin/ffmpeg"
fi

# portaudio — needed for pyaudio microphone input
# Homebrew-free options:
#   a) Use sounddevice instead (pip install sounddevice — bundles PortAudio)
#   b) macOS built-in CoreAudio (used by sounddevice automatically)
# Note: sounddevice is already in requirements.txt as a portaudio-free alternative
info "portaudio: using 'sounddevice' Python package (bundles PortAudio — no Homebrew needed)"

# ---------------------------------------------------------------------------
# Step 3: System dependencies summary
# ---------------------------------------------------------------------------
step "System dependency summary"
info "Lopen uses pip-only Python packages wherever possible."
info "The only optional native binaries are:"
info "  • piper     (TTS) — pre-built GitHub release: https://github.com/rhasspy/piper/releases"
info "  • whisper   (ASR) — pre-built from whisper.cpp: https://github.com/ggerganov/whisper.cpp/releases"
info "  • ffmpeg    (audio) — optional; static build: https://evermeet.cx/ffmpeg/"
info "See docs/INSTALL_NO_HOMEBREW.md for step-by-step manual install instructions."

# ---------------------------------------------------------------------------
# Step 4: Python virtual environment
# ---------------------------------------------------------------------------
step "Setting up Python virtual environment"
cd "$REPO_ROOT"
if [[ -d ".venv" ]]; then
    ok ".venv already exists"
else
    python3 -m venv .venv
    ok "Created .venv"
fi

# Activate venv
source .venv/bin/activate
ok "Virtual environment activated"

# Upgrade pip
pip install --quiet --upgrade pip setuptools wheel
ok "pip/setuptools upgraded"

# ---------------------------------------------------------------------------
# Step 5: Python packages
# ---------------------------------------------------------------------------
step "Installing Python dependencies"
pip install --quiet -r requirements.txt
ok "Core requirements installed"

# Optional: llama-cpp-python (recommended for local LLM)
echo ""
echo -e "  ${YELLOW}Optional:${RESET} Install llama-cpp-python for local LLM inference?"
echo -e "  ${DIM}  This enables Qwen3.5-0.8B and other GGUF models to run locally.${RESET}"
echo -e "  ${DIM}  Requires cmake (installed via pip if missing). Takes ~2-5 min.${RESET}"
if [[ "$INSTALL_LLAMA" == "ask" ]]; then
    if [[ "$AUTO_YES" == "true" ]]; then
        INSTALL_LLAMA="no"
        info "--yes set: skipping llama-cpp-python by default (use --with-llama to enable)"
    else
        read -r -p "  Install llama-cpp-python? [y/N] " REPLY
        if [[ "${REPLY,,}" == "y" ]]; then
            INSTALL_LLAMA="yes"
        else
            INSTALL_LLAMA="no"
        fi
    fi
fi

if [[ "$INSTALL_LLAMA" == "yes" ]]; then
    echo -e "  Ensuring cmake is available (pip install cmake)..."
    pip install --quiet cmake || warn "cmake pip install failed — trying system cmake"
    echo -e "  Installing llama-cpp-python (CPU-only build)..."
    # CMAKE_ARGS="-DGGML_METAL=OFF" disables Metal GPU acceleration.
    # Required for the 2017 Intel MacBook Pro (no compatible Metal GPU).
    # On Apple Silicon Macs, omit CMAKE_ARGS for Metal acceleration.
    CMAKE_ARGS="-DGGML_METAL=OFF" pip install --quiet "llama-cpp-python>=0.3.0" \
        && ok "llama-cpp-python installed" \
        || warn "llama-cpp-python install failed — run manually: CMAKE_ARGS='-DGGML_METAL=OFF' pip install llama-cpp-python"
else
    info "Skipped llama-cpp-python — Lopen will use mock LLM responses until installed"
fi

# ---------------------------------------------------------------------------
# Step 6: Configuration
# ---------------------------------------------------------------------------
step "Setting up configuration"
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    ok "Copied .env.example → .env"
    info "Edit .env to customise your settings"
else
    ok ".env already exists"
fi

# Ensure directories exist
mkdir -p logs models/llm models/asr models/tts .pids
ok "Required directories created (logs/, models/, .pids/)"

# Make scripts executable
chmod +x scripts/*.sh
ok "Scripts made executable"

# ---------------------------------------------------------------------------
# Step 7: Model download
# ---------------------------------------------------------------------------
if [[ "$SKIP_MODELS" == "true" ]]; then
    step "Skipping model download (--no-models flag)"
    info "Download models later with: bash scripts/download_models.sh"
else
    step "Downloading AI models"
    echo -e "  ${DIM}This will download:${RESET}"
    echo -e "  ${DIM}  • Qwen3.5-0.8B-Instruct Q4_K_M (~0.55 GB) — ultra-fast primary LLM${RESET}"
    echo -e "  ${DIM}  • whisper-tiny ASR model       (~39 MB)  — speech recognition${RESET}"
    echo -e "  ${DIM}  • Piper TTS model (ryan-high)  (~65 MB)  — natural male voice${RESET}"
    echo ""
    if [[ "$AUTO_YES" == "true" ]]; then
        REPLY="y"
    else
        read -r -p "  Download models now? [Y/n] " REPLY
    fi
    if [[ "${REPLY,,}" != "n" ]]; then
        bash scripts/download_models.sh \
            && ok "Models downloaded successfully" \
            || warn "Some models failed to download — run: bash scripts/download_models.sh"
    else
        info "Skipped — run: bash scripts/download_models.sh when ready"
    fi
fi

# ---------------------------------------------------------------------------
# Step 8: Self-diagnostics
# ---------------------------------------------------------------------------
step "Running self-diagnostics"
bash scripts/diagnose.sh 2>&1 | grep -E "(✔|⚠|✖|ℹ)" | head -30 || true
echo ""

# ---------------------------------------------------------------------------
# Done!
# ---------------------------------------------------------------------------
echo ""
echo -e "${DIM}  ────────────────────────────────────────────────────────${RESET}"
echo -e "${BOLD}${GREEN}  ✓ Lopen installation complete!${RESET}"
echo ""
echo -e "${BOLD}  Quickstart:${RESET}"
echo -e "  ${CYAN}  source .venv/bin/activate${RESET}"
echo -e "  ${CYAN}  bash scripts/start.sh${RESET}          ${DIM}# start orchestrator + dashboard${RESET}"
echo -e "  ${CYAN}  python cli.py${RESET}                  ${DIM}# interactive CLI${RESET}"
echo -e "  ${CYAN}  open http://localhost:8080${RESET}      ${DIM}# web dashboard${RESET}"
echo ""
echo -e "${BOLD}  Useful commands:${RESET}"
echo -e "  ${DIM}  bash scripts/status.sh${RESET}          ${DIM}# check service status${RESET}"
echo -e "  ${DIM}  bash scripts/diagnose.sh${RESET}        ${DIM}# full self-diagnostics${RESET}"
echo -e "  ${DIM}  bash scripts/stop.sh${RESET}            ${DIM}# stop all services${RESET}"
echo -e "  ${DIM}  bash scripts/health_check.sh${RESET}    ${DIM}# run health checks${RESET}"
echo ""
echo -e "  ${DIM}Docs: README.md  |  PLUGINS.md  |  docs/AI_ARCHITECTURE.md${RESET}"
echo ""
