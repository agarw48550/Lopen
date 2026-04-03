#!/usr/bin/env bash
# download_models.sh — Download AI models for Lopen (April 2026)
#
# Usage:
#   bash scripts/download_models.sh              # default stack (Qwen3.5-0.8B + whisper-tiny + piper-ryan)
#   bash scripts/download_models.sh --quality    # use Qwen3.5-1.5B instead (better quality, 1 GB)
#   bash scripts/download_models.sh --phi3       # use Phi-3-mini instead (legacy, 2.2 GB)
#   bash scripts/download_models.sh --mistral    # also download Mistral-7B Q4_K_M (for AirLLM engine)
#   bash scripts/download_models.sh --base       # use whisper-base instead of tiny (better accuracy)
#
# Memory budget (default stack — April 2026):
#   - Qwen3.5-0.8B-Instruct Q4_K_M:  550 MB   ← ultra-fast default LLM
#   - whisper-tiny:                    39 MB
#   - piper-ryan-high:                 65 MB
#   ──────────────────────────────────────────
#   - Total:                          654 MB  ✓ leaves 3.3+ GB free
#
# Why Qwen3.5-0.8B over Phi-3-mini (previous default)?
#   • 4× smaller (550 MB vs 2.2 GB) — downloads in seconds, not minutes
#   • 3× faster throughput on Intel Mac (~10 tok/s vs ~3 tok/s)
#   • First response reliably under 1 second
#   • Instruction-tuned; matches Phi-3-mini quality for everyday tasks
#   • Leaves ample RAM for multi-agent, voice pipeline, and browser tools

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
USE_QWEN_1_5B=false
USE_PHI3=false
for arg in "$@"; do
    case "$arg" in
        --mistral)  DOWNLOAD_MISTRAL=true ;;
        --base)     USE_WHISPER_BASE=true ;;
        --quality)  USE_QWEN_1_5B=true ;;
        --phi3)     USE_PHI3=true ;;
    esac
done

echo "==> Lopen model download (April 2026)"
echo "    Destination: $REPO_ROOT/models/"

# ---------------------------------------------------------------------------
# Helper: safe download with progress and error handling
# ---------------------------------------------------------------------------
_download() {
    local dest="$1"
    local url="$2"
    local label="$3"
    if [ -f "$dest" ]; then
        echo "    [skip] Already present: $(basename "$dest")"
        return 0
    fi
    echo "--> Downloading $label..."
    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$dest" "$url" || {
            echo "    [warn] curl failed. Manual download:"
            echo "           curl -L -o '$dest' '$url'"
            return 1
        }
    elif command -v wget &>/dev/null; then
        wget -c --show-progress -O "$dest" "$url" || {
            echo "    [warn] wget failed. Manual download:"
            echo "           wget -O '$dest' '$url'"
            return 1
        }
    else
        echo "    [error] Neither curl nor wget found."
        echo "           curl and wget are standard on macOS — check your PATH."
        return 1
    fi
}

# ---------------------------------------------------------------------------
# LLM: Qwen3.5-0.8B-Instruct Q4_K_M  (~0.55 GB) — DEFAULT
#   Ultra-fast, tiny RAM footprint, instruction-tuned for chat/tasks.
#   Source: https://huggingface.co/Qwen/Qwen3.5-0.8B-Instruct-GGUF
# ---------------------------------------------------------------------------
echo ""
echo "=== LLM model ==="
if [ "$USE_PHI3" = true ]; then
    LLM_FILE="$MODELS_LLM/Phi-3-mini-4k-instruct-q4.gguf"
    LLM_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
    _download "$LLM_FILE" "$LLM_URL" "Phi-3-mini-4k-instruct Q4_K_M (~2.2 GB, legacy)" || true
    echo ""
    echo "    [note] Using Phi-3-mini. Update config/models.yaml: llm.active: phi3-mini-q4"
elif [ "$USE_QWEN_1_5B" = true ]; then
    LLM_FILE="$MODELS_LLM/qwen3.5-1.5b-instruct-q4_k_m.gguf"
    LLM_URL="https://huggingface.co/Qwen/Qwen3.5-1.5B-Instruct-GGUF/resolve/main/qwen3.5-1.5b-instruct-q4_k_m.gguf"
    _download "$LLM_FILE" "$LLM_URL" "Qwen3.5-1.5B-Instruct Q4_K_M (~1.0 GB, quality upgrade)" || true
    echo ""
    echo "    [note] Using Qwen3.5-1.5B. Update config/models.yaml: llm.active: qwen35-1.5b-q4"
else
    LLM_FILE="$MODELS_LLM/qwen3.5-0.8b-instruct-q4_k_m.gguf"
    LLM_URL="https://huggingface.co/Qwen/Qwen3.5-0.8B-Instruct-GGUF/resolve/main/qwen3.5-0.8b-instruct-q4_k_m.gguf"
    _download "$LLM_FILE" "$LLM_URL" "Qwen3.5-0.8B-Instruct Q4_K_M (~0.55 GB, ultra-fast default)" || true
fi

# Create generic model.gguf symlink (used by config/settings.yaml fallback)
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
# Piper binary (no Homebrew needed): https://github.com/rhasspy/piper/releases
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
echo "      bash scripts/setup_venv.sh    # install Python dependencies (if not done)"
echo "      bash scripts/start.sh         # start Lopen"
echo "      python -m pytest tests/ -q   # run all tests"
