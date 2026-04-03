#!/usr/bin/env bash
# download_models.sh — Download GGUF models, whisper, and piper binaries

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODELS_LLM="$REPO_ROOT/models/llm"
MODELS_ASR="$REPO_ROOT/models/asr"
MODELS_TTS="$REPO_ROOT/models/tts"

mkdir -p "$MODELS_LLM" "$MODELS_ASR" "$MODELS_TTS"

echo "==> Downloading Lopen models..."

# ---- LLM: Phi-3-mini Q4 ----
LLM_FILE="$MODELS_LLM/Phi-3-mini-4k-instruct-q4.gguf"
LLM_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
if [ ! -f "$LLM_FILE" ]; then
    echo "--> Downloading Phi-3-mini Q4 (~2.2 GB)..."
    wget -c --show-progress -O "$LLM_FILE" "$LLM_URL" || {
        echo "    Download failed. Try manually: wget -O '$LLM_FILE' '$LLM_URL'"
    }
else
    echo "--> LLM model already present: $LLM_FILE"
fi

# ---- ASR: whisper-tiny ----
ASR_FILE="$MODELS_ASR/ggml-tiny.en.bin"
ASR_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
if [ ! -f "$ASR_FILE" ]; then
    echo "--> Downloading whisper tiny (~39 MB)..."
    wget -c --show-progress -O "$ASR_FILE" "$ASR_URL" || {
        echo "    Download failed. Try: wget -O '$ASR_FILE' '$ASR_URL'"
    }
else
    echo "--> ASR model already present: $ASR_FILE"
fi

# ---- TTS: piper ryan-high ----
TTS_ONNX="$MODELS_TTS/en_US-ryan-high.onnx"
TTS_JSON="$MODELS_TTS/en_US-ryan-high.onnx.json"
TTS_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high"
if [ ! -f "$TTS_ONNX" ]; then
    echo "--> Downloading piper ryan-high voice (~65 MB)..."
    wget -c --show-progress -O "$TTS_ONNX" "$TTS_BASE/en_US-ryan-high.onnx" || true
    wget -c --show-progress -O "$TTS_JSON" "$TTS_BASE/en_US-ryan-high.onnx.json" || true
else
    echo "--> TTS model already present: $TTS_ONNX"
fi

echo ""
echo "==> Models download complete."
echo "    Update config/settings.yaml if you placed models in non-default paths."
