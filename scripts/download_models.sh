#!/usr/bin/env bash
# download_models.sh — Download AI models for Lopen
#
# Usage:
#   bash scripts/download_models.sh           # default stack (Phi-3-mini + whisper-tiny + piper-ryan)
#   bash scripts/download_models.sh --mistral # also download Mistral-7B Q4_K_M (for AirLLM engine)
#   bash scripts/download_models.sh --base    # use whisper-base instead of tiny (better accuracy)
#
# Memory budget (default stack): ~2.4 GB total
#   - Phi-3-mini Q4_K_M:   2.2 GB
#   - whisper-tiny:         39 MB
#   - piper-ryan-high:      65 MB
#   ──────────────────────────────
#   - Total:               2.3 GB  ✓ within 4 GB budget

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODELS_LLM="$REPO_ROOT/models/llm"
MODELS_ASR="$REPO_ROOT/models/asr"
MODELS_TTS="$REPO_ROOT/models/tts"

mkdir -p "$MODELS_LLM" "$MODELS_ASR" "$MODELS_TTS"

# Parse flags
DOWNLOAD_MISTRAL=false
USE_WHISPER_BASE=false
for arg in "$@"; do
    case "$arg" in
        --mistral) DOWNLOAD_MISTRAL=true ;;
        --base)    USE_WHISPER_BASE=true ;;
    esac
done

echo "==> Lopen model download"
echo "    Destination: $REPO_ROOT/models/"

# ---------------------------------------------------------------------------
# Helper: safe download with progress and error handling
# ---------------------------------------------------------------------------
_download() {
    local dest="$1"
    local url="$2"
    local label="$3"
    if [ -f "$dest" ]; then
        echo "    [skip] Already present: $dest"
        return 0
    fi
    echo "--> Downloading $label..."
    if command -v wget &>/dev/null; then
        wget -c --show-progress -O "$dest" "$url" || {
            echo "    [warn] wget failed. Manual download:"
            echo "           wget -O '$dest' '$url'"
            return 1
        }
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$dest" "$url" || {
            echo "    [warn] curl failed. Manual download:"
            echo "           curl -L -o '$dest' '$url'"
            return 1
        }
    else
        echo "    [error] Neither wget nor curl found. Install one and re-run."
        echo "           wget -O '$dest' '$url'"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# LLM: Phi-3-mini-4k-instruct Q4_K_M  (~2.2 GB)
# Best balance of quality and RAM for multi-agent use.
# Source: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
# ---------------------------------------------------------------------------
echo ""
echo "=== LLM model ==="
LLM_FILE="$MODELS_LLM/Phi-3-mini-4k-instruct-q4.gguf"
LLM_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
_download "$LLM_FILE" "$LLM_URL" "Phi-3-mini-4k-instruct Q4_K_M (~2.2 GB)" || true

# Create generic model.gguf symlink (used by config/settings.yaml)
GENERIC="$MODELS_LLM/model.gguf"
if [ -f "$LLM_FILE" ] && [ ! -e "$GENERIC" ]; then
    ln -sf "$(basename "$LLM_FILE")" "$GENERIC"
    echo "    Symlink: $GENERIC -> $(basename "$LLM_FILE")"
fi

# ---------------------------------------------------------------------------
# LLM (optional): Mistral-7B-Instruct-v0.2 Q4_K_M  (~4.1 GB)
# Use with AirLLM engine (engine: airllm in config/settings.yaml).
# Requires: disable reflection agent in config/agents.yaml
# Source: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# ---------------------------------------------------------------------------
if [ "$DOWNLOAD_MISTRAL" = true ]; then
    echo ""
    echo "=== LLM (Mistral-7B, AirLLM) ==="
    MISTRAL_FILE="$MODELS_LLM/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    MISTRAL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    _download "$MISTRAL_FILE" "$MISTRAL_URL" "Mistral-7B-Instruct-v0.2 Q4_K_M (~4.1 GB)" || true
    echo ""
    echo "    [note] To use Mistral-7B:"
    echo "    1. Set in config/settings.yaml:  llm.engine: airllm"
    echo "                                     llm.model_path: $MISTRAL_FILE"
    echo "    2. Set in config/agents.yaml:    enable_reflection: false"
    echo "                                     max_concurrent_agents: 1"
    echo "    Install AirLLM:                  pip install airllm"
fi

# ---------------------------------------------------------------------------
# ASR: whisper.cpp model  (tiny ~39 MB, base ~142 MB)
# Source: https://huggingface.co/ggerganov/whisper.cpp
# ---------------------------------------------------------------------------
echo ""
echo "=== ASR (Speech-to-Text) ==="
if [ "$USE_WHISPER_BASE" = true ]; then
    ASR_FILE="$MODELS_ASR/ggml-base.en.bin"
    ASR_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
    _download "$ASR_FILE" "$ASR_URL" "whisper-base (~142 MB, better accuracy)" || true
else
    ASR_FILE="$MODELS_ASR/ggml-tiny.en.bin"
    ASR_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
    _download "$ASR_FILE" "$ASR_URL" "whisper-tiny (~39 MB, fastest)" || true
fi

# ---------------------------------------------------------------------------
# TTS: Piper en_US-ryan-high  (natural male voice, ~65 MB)
# Source: https://huggingface.co/rhasspy/piper-voices
# ---------------------------------------------------------------------------
echo ""
echo "=== TTS (Text-to-Speech) ==="
TTS_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high"
_download "$MODELS_TTS/en_US-ryan-high.onnx" "$TTS_BASE/en_US-ryan-high.onnx" "Piper ryan-high ONNX model (~65 MB)" || true
_download "$MODELS_TTS/en_US-ryan-high.onnx.json" "$TTS_BASE/en_US-ryan-high.onnx.json" "Piper ryan-high config" || true

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "==> Download complete."
echo ""
echo "    Models directory:"
find "$REPO_ROOT/models" -type f ! -name ".gitkeep" | sort | while read -r f; do
    size=$(du -sh "$f" 2>/dev/null | cut -f1)
    echo "      $size  $f"
done
echo ""
echo "    Next steps:"
echo "      bash scripts/setup_venv.sh    # install Python dependencies"
echo "      bash scripts/start.sh         # start Lopen"
echo "      python -m pytest tests/ -q   # run all tests"

