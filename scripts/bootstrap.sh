#!/usr/bin/env bash
# bootstrap.sh — Install macOS system dependencies for Lopen via Homebrew

set -euo pipefail

echo "==> Lopen bootstrap: installing system dependencies"

# Ensure Homebrew is available
if ! command -v brew &>/dev/null; then
    echo "Homebrew not found. Install from https://brew.sh and re-run."
    exit 1
fi

echo "--> Updating Homebrew..."
brew update

echo "--> Installing core dependencies..."
brew install \
    cmake \
    ffmpeg \
    portaudio \
    python@3.11 \
    git \
    wget \
    curl

echo "--> Installing optional: piper TTS + whisper.cpp (may take a moment)..."
# piper — pre-built releases available on GitHub
if ! command -v piper &>/dev/null; then
    echo "    piper binary not found in PATH."
    echo "    Download from: https://github.com/rhasspy/piper/releases"
    echo "    Place in /usr/local/bin/piper and chmod +x"
else
    echo "    piper found: $(which piper)"
fi

# whisper.cpp
if ! command -v whisper &>/dev/null; then
    echo "    whisper.cpp binary not found in PATH."
    echo "    Build from source: https://github.com/ggerganov/whisper.cpp"
    echo "    Or place pre-built binary at /usr/local/bin/whisper"
else
    echo "    whisper found: $(which whisper)"
fi

echo "==> Bootstrap complete. Run scripts/setup_venv.sh next."
