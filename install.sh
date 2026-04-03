#!/usr/bin/env bash
# install.sh — One-command Lopen installer for macOS (Intel or Apple Silicon)
#
# Usage:
#   bash install.sh             # full install with model download prompt
#   bash install.sh --no-models # skip model downloads (install later)
#   bash install.sh --debug     # verbose output
#
# What this does:
#   1. Verifies Homebrew is present (or guides you to install it)
#   2. Installs system dependencies (cmake, ffmpeg, portaudio, python3)
#   3. Creates a Python virtual environment in .venv/
#   4. Installs all Python requirements
#   5. Copies .env.example → .env (if not exists)
#   6. Creates required directories (logs/, models/, .pids/)
#   7. Optionally downloads AI models (LLM, ASR, TTS)
#   8. Runs self-diagnostics to verify the installation
#   9. Prints a quickstart guide

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
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
SKIP_MODELS=false
DEBUG=false
for arg in "$@"; do
    case "$arg" in
        --no-models) SKIP_MODELS=true ;;
        --debug)     DEBUG=true; set -x ;;
        --help|-h)
            echo "Usage: bash install.sh [--no-models] [--debug]"
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
    die "Python 3.9+ required. Install: brew install python@3.11"
fi

# ---------------------------------------------------------------------------
# Step 2: Homebrew
# ---------------------------------------------------------------------------
step "Checking Homebrew"
if command -v brew &>/dev/null; then
    BREW_VER="$(brew --version | head -1)"
    ok "${BREW_VER}"
else
    fail "Homebrew not found"
    echo ""
    echo -e "  ${YELLOW}Homebrew is required for system dependencies.${RESET}"
    echo -e "  Install it with:"
    echo -e "  ${CYAN}  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${RESET}"
    echo ""
    die "Please install Homebrew and re-run this script."
fi

# ---------------------------------------------------------------------------
# Step 3: System dependencies
# ---------------------------------------------------------------------------
step "Installing system dependencies (via Homebrew)"
BREW_PKGS=("cmake" "ffmpeg" "portaudio" "git" "wget")
for pkg in "${BREW_PKGS[@]}"; do
    if brew list "$pkg" &>/dev/null; then
        ok "${pkg} already installed"
    else
        echo -e "    Installing ${pkg}..."
        brew install "$pkg" || warn "Failed to install ${pkg} — continuing"
    fi
done

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
echo -e "  ${DIM}  This enables Phi-3-mini and other GGUF models to run locally.${RESET}"
echo -e "  ${DIM}  Takes ~2-5 minutes to compile.${RESET}"
read -r -p "  Install llama-cpp-python? [y/N] " REPLY
if [[ "${REPLY,,}" == "y" ]]; then
    echo -e "  Installing llama-cpp-python (CPU-only build)..."
    # CMAKE_ARGS="-DGGML_METAL=OFF" disables GPU Metal acceleration.
    # This is intentional for the 2017 Intel MacBook Pro target which has no
    # compatible Metal GPU. If you have a newer Mac with Apple Silicon or a
    # dedicated GPU, remove the CMAKE_ARGS override to enable acceleration.
    CMAKE_ARGS="-DGGML_METAL=OFF" pip install --quiet "llama-cpp-python>=0.2.57" \
        && ok "llama-cpp-python installed" \
        || warn "llama-cpp-python install failed — run manually: pip install llama-cpp-python"
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
    echo -e "  ${DIM}  • Phi-3-mini Q4_K_M LLM       (~2.2 GB) — default local language model${RESET}"
    echo -e "  ${DIM}  • whisper-tiny ASR model       (~39 MB)  — speech recognition${RESET}"
    echo -e "  ${DIM}  • Piper TTS model (ryan-high)  (~65 MB)  — natural male voice${RESET}"
    echo ""
    read -r -p "  Download models now? [Y/n] " REPLY
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
