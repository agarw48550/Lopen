# Lopen — Homebrew-Free Installation Guide (April 2026)

Lopen is fully installable on macOS without Homebrew. Every dependency can be
obtained via **pip**, a direct binary download, or macOS built-in tools.

---

## Prerequisites

| Tool | Built-in? | Install if missing |
|------|-----------|--------------------|
| Python 3.9+ | macOS 12.3+ includes Python 3 | [python.org/downloads/macos](https://www.python.org/downloads/macos/) or `xcode-select --install` |
| `curl` | ✅ Built into macOS | — |
| `git` | ✅ After `xcode-select --install` | `xcode-select --install` |
| `cmake` | ❌ | `pip install cmake` (inside venv — no system install needed) |
| `ffmpeg` | ❌ | Static build — see below |
| `portaudio` | ❌ | Not needed — replaced by `sounddevice` (pip) |

---

## Quick start (one-command, no Homebrew)

```bash
git clone https://github.com/agarw48550/Lopen
cd Lopen
bash install.sh          # never calls brew
bash install.sh --yes --no-models   # non-interactive quick path
```

The installer detects available tools and guides you through anything missing.

---

## Step-by-step manual install

### 1. Python 3.11

**Option A — python.org installer (recommended, no Homebrew):**
```bash
# Download the macOS Universal installer from:
# https://www.python.org/downloads/macos/
# (Click the most recent 3.11.x or 3.12.x "macOS installer")
# Run the .pkg — Python will be at /usr/local/bin/python3
```

**Option B — Xcode Command Line Tools (includes Python 3):**
```bash
xcode-select --install
# Provides python3 at /usr/bin/python3
```

### 2. cmake (via pip — zero system footprint)

cmake is only needed to compile `llama-cpp-python` from source.
Install it inside your virtual environment — no sudo, no system package:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install cmake
cmake --version  # verify
```

### 3. llama-cpp-python (local LLM inference)

```bash
# CPU-only build for 2017 Intel MacBook Pro
# (GGML_METAL=OFF avoids incompatible Metal build on Intel)
CMAKE_ARGS="-DGGML_METAL=OFF" pip install "llama-cpp-python>=0.3.0"
```

Apple Silicon users can omit `CMAKE_ARGS` to enable Metal GPU acceleration.

### 4. piper TTS binary (pre-built release — no Homebrew)

```bash
# Intel Mac (x86_64):
curl -L "https://github.com/rhasspy/piper/releases/latest/download/piper_macos_x64.tar.gz" \
  | tar -xz -C /tmp/piper_install

mkdir -p "$HOME/.local/bin"
cp /tmp/piper_install/piper/piper "$HOME/.local/bin/piper"
chmod +x "$HOME/.local/bin/piper"

# Add to PATH (once):
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

piper --version  # verify
```

> **Fallback:** If piper is not available, Lopen automatically uses macOS's
> built-in `say` command for TTS (no binary download needed).

### 5. whisper.cpp ASR binary (build from source — no Homebrew)

Requires Xcode Command Line Tools (free, ~1 GB):

```bash
xcode-select --install   # only needed once

git clone https://github.com/ggerganov/whisper.cpp /tmp/whisper.cpp
cd /tmp/whisper.cpp

# Build without Metal (Intel Mac):
cmake -B build -DWHISPER_METAL=OFF
cmake --build build --config Release -j"$(sysctl -n hw.logicalcpu)"

mkdir -p "$HOME/.local/bin"
cp build/bin/whisper-cli "$HOME/.local/bin/whisper"
chmod +x "$HOME/.local/bin/whisper"

whisper --help  # verify
```

> **Fallback:** If whisper is not available, Lopen uses Python's
> `faster-whisper` package (pure Python, `pip install faster-whisper`).

### 6. ffmpeg (optional — static build, no Homebrew)

Only needed for audio format conversion in the voice pipeline.
`sounddevice` (installed via pip) handles real-time mic input without ffmpeg.

```bash
# Intel Mac static build from evermeet.cx (reputable third-party mirror):
curl -L "https://evermeet.cx/ffmpeg/ffmpeg-7.1.1.zip" -o /tmp/ffmpeg.zip
unzip /tmp/ffmpeg.zip -d "$HOME/.local/bin"
chmod +x "$HOME/.local/bin/ffmpeg"
ffmpeg -version  # verify
```

### 7. sounddevice (replaces portaudio pip package)

```bash
pip install sounddevice   # bundles PortAudio — no system library needed
```

---

## Model downloads (Qwen3.5-0.8B, April 2026 default)

```bash
bash scripts/download_models.sh
```

This downloads:
- **Qwen3.5-0.8B-Instruct Q4_K_M** (~0.55 GB) — ultra-fast default LLM
- **whisper-tiny** (~39 MB) — speech recognition
- **Piper ryan-high ONNX** (~65 MB) — TTS model

For a quality upgrade (1.5B, ~1 GB):
```bash
bash scripts/download_models.sh --quality
```

---

## Verifying the installation

```bash
source .venv/bin/activate
bash scripts/diagnose.sh     # full self-diagnostics
python -m pytest tests/ -q  # run all tests
python cli.py --debug        # interactive CLI
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python3: command not found` | Install from python.org or run `xcode-select --install` |
| `pip install cmake` fails | Your Python version is too old — upgrade to 3.9+ |
| `llama-cpp-python` compile error | Ensure `cmake` is installed: `pip install cmake` |
| piper not found after install | Add `~/.local/bin` to PATH: `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc` |
| Model download too slow | Use `--no-models` flag and download later; models are on HuggingFace CDN |
| Whisper build fails | Ensure Xcode CLT installed: `xcode-select --install` |

---

## Why no Homebrew?

Homebrew can be unreliable on older Intel Macs (2017 MacBook Pro) running
macOS 12/13 because:
- Many formulae now require macOS 14 or Apple Silicon
- `brew update` can take minutes and fail silently on older systems
- Homebrew modifies system paths in ways that can conflict with Python venvs
- Direct downloads and pip packages are more reproducible across macOS versions

Lopen is designed to work with **pip only + pre-built binaries** for
maximum compatibility on your 2017 Intel hardware.
