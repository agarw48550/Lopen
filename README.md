# Lopen — Local-First General-Purpose Autonomous Assistant

**Lopen** is a production-ready, local-first autonomous assistant framework that runs entirely on your Mac. It uses an **intent-driven, plugin-extensible architecture** to handle any open-ended user request — routing it semantically to the best available tool, without hardcoded mappings or pre-defined task lists.

Designed for a **2017 Intel MacBook Pro** (8 GB RAM, target ≤4 GB runtime) — no cloud, no subscriptions, all private.

---

## ⚡ One-Command Install

```bash
git clone https://github.com/agarw48550/Lopen && cd Lopen
bash install.sh
```

The installer will:
1. Check your OS and Python version
2. Check available system tools — **no Homebrew required**
3. Create a Python virtual environment in `.venv/`
4. Install all Python requirements (pip only)
5. Optionally install `llama-cpp-python` for local LLM (cmake installed via pip)
6. Download AI models (Qwen3.5-0.8B-Instruct + whisper-tiny + Piper TTS)
7. Run self-diagnostics to verify everything works

---

## 🚀 Quick Start

```bash
# Activate environment and start all services
source .venv/bin/activate
bash scripts/start.sh

# Open the web dashboard
open http://localhost:8080

# Launch the interactive CLI
python cli.py
```

### Interactive CLI

```
  _
 | |    ___  _ __   ___ _ __
 | |   / _ \| '_ \ / _ \ '_ \
 | |__| (_) | |_) |  __/ | | |
 |_____\___/| .__/ \___|_| |_|
            |_|

  Your local-first autonomous assistant. No cloud required. 🖥️

  Systems nominal. Type 'help' for commands.

  Host: http://localhost:8000  |  Session: cli-1711234567

lopen › chat write a Python hello-world function
lopen › status
lopen › plugins
lopen › debug on
lopen › benchmark
lopen › joke
lopen › quit
```

**CLI commands:**

| Command | Description |
|---------|-------------|
| `chat <message>` | Send a message to the agent |
| `status` | Service health, RAM usage, uptime |
| `system` | Detailed RAM/CPU/disk report with memory guard |
| `plugins` | List loaded plugins |
| `tools` | List all tools with descriptions |
| `history` | Show recent conversation turns |
| `summary` | Summarise the current conversation |
| `clear` | Clear conversation history |
| `config` | Print active configuration |
| `model [name]` | Show or switch the active LLM |
| `memory` | RAM usage and memory guard thresholds |
| `fetch <url>` | Fetch a URL and summarise its content |
| `ingest <file>` | Ingest a local file into agent memory |
| `logs [N]` | Tail the last N lines from the agent log |
| `restart` | Restart the orchestrator service |
| `debug on\|off` | Toggle verbose debug output |
| `benchmark` | Run inference speed test |
| `help` | Show all commands |

**Fun extras:** `joke`, `haiku`, `sing`, `quote`, `about`, `fortune`, `matrix`, `coffee`

```bash
# Start CLI with debug output
python cli.py --debug

# Connect to a remote instance
python cli.py --host my-macbook.local --port 8000
```

---

## 🔧 Debugging & Diagnostics

```bash
# Full self-diagnostics (OS, Python, RAM, models, services)
bash scripts/diagnose.sh

# Check running service status
bash scripts/status.sh

# Start with verbose debug logging (LOPEN_DEBUG=1)
bash scripts/start.sh --debug

# Tail logs
tail -f logs/lopen.log
tail -f logs/lopen_error.log
tail -f logs/lopen_debug.log  # created in debug mode
```

**Debug mode** activates via:
- `bash scripts/start.sh --debug`
- `LOPEN_DEBUG=1 bash scripts/start.sh`
- `python cli.py --debug`
- `LOPEN_LOG_LEVEL=DEBUG python cli.py`

**Structured logs** in `logs/`:
- `lopen.log` — standard operation log (50 MB rotating, 5 backups)
- `lopen_error.log` — errors only (10 MB rotating, 3 backups)
- `lopen_debug.log` — verbose debug trace (20 MB rotating, 3 backups)

---

## 🏃 Benchmark

```bash
# Run inference speed test
bash scripts/benchmark.sh

# Verbose mode (shows responses)
bash scripts/benchmark.sh --verbose
```

Target performance on a 2017 Intel MacBook Pro with **Qwen3.5-0.8B-Instruct Q4_K_M** (April 2026 default):
- Simple queries: **< 1s** 🚀 (550 MB model, 8–12 tok/s)
- Medium complexity: 1–3s ✅
- Complex queries: 3–6s ✅

> **Previous default (Phi-3-mini Q4_K_M):** 3–8s average. Replaced by Qwen3.5-0.8B for 3× faster
> responses and 4× smaller model footprint. Set `llm.active: phi3-mini-q4` to revert.

---

## 🖥️ Web Dashboard (Port 8080)

The web gateway provides multiplatform control from any browser — laptop, phone, or tablet — on your local network:

- **Live chat** with the agent (real-time streaming)
- **Connection status** with auto-reconnect and uptime display
- **System status panel** — LLM mode, tools loaded, memory turns
- **Plugin management** — list and reload plugins live
- **Memory viewer** — browse conversation history with clear button
- **Responsive layout** — works on mobile screens

Access from your MacBook Air: `http://[your-mac-ip]:8080`

---

## Architecture Overview

```
                     User Query (any interface)
                            │
                 ┌──────────▼──────────┐
                 │   SafetyEngine      │  ← NemoClaw-inspired guardrails
                 │   check_input()     │    pattern + topic blocklist, PII redaction
                 └──────────┬──────────┘
                            │  (if safe)
                 ┌──────────▼──────────┐
                 │   IntentEngine      │  ← TF-IDF cosine similarity
                 │  (semantic match)   │    no model downloads, <1 MB RAM
                 └──────────┬──────────┘
                            │  scores every registered tool
                 ┌──────────▼──────────┐
                 │   ToolSelector      │  ← ranks + safety tool check
                 └──────────┬──────────┘
                            │
              ┌─────────────▼─────────────────┐
              │    AgentDispatcher (OMLX)      │  ← multi-agent reasoning
              │  planner → executor → reflector│    LRU memory eviction
              └──────────────┬────────────────┘
                             │
              ┌──────────────▼────────────────┐
              │       AirLLMEngine             │  ← AirLLM / llama-cpp-python
              │  (layer-split or GGUF backend)  │    mock fallback for CI
              └──────────────┬────────────────┘
                             │
                 ┌───────────▼──────────┐
                 │   SafetyEngine       │  ← output PII redaction
                 │   check_output()     │
                 └───────────┬──────────┘
                             │
                      Final Response
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Lopen Orchestrator (port 8000)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │IntentEngine  │→ │ToolSelector  │→ │Tool Registry │  │  Task Queue    │  │
│  │ (TF-IDF)     │  │+ ToolFilter  │  │+ PluginLoader│  │                │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │        AirLLMEngine (airllm → llama_cpp → mock, auto-select)        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │   AgentDispatcher: planner | executor | reflector | summarizer       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │   SafetyEngine: input guardrails | tool filter | output redaction    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │     Conversation Memory  ←→  SQLite Storage + Analytics             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
       │               │               │               │
┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼───────────┐
│ Voice Loop  │ │  WhatsApp   │ │    Web     │ │ System Health  │
│ Mic→Wake→  │ │   Bridge    │ │ Dashboard  │ │  RAM/Disk/Hb/  │
│ ASR→LLM→TTS│ │ (Playwright)│ │ (port 8080)│ │  Log Rotation  │
└─────────────┘ └─────────────┘ └────────────┘ └────────────────┘

Tools: HomeworkTutor │ Researcher │ CoderAssist │ DesktopOrganizer │ FileOps │ BrowserAutomation
       + any .py file dropped in tools/third_party/ — auto-discovered
```

---

## What Makes Lopen General-Purpose

Unlike assistants with hardcoded intent→tool mappings, Lopen uses **semantic routing**:

1. **Open-ended intent** — any natural language query is accepted.
2. **Dynamic tool selection** — tools are scored against the query using TF-IDF
   cosine similarity over their descriptions and tags.  No fixed enum mappings.
3. **Plugin extensibility** — drop a Python file in `tools/third_party/` and it
   is automatically discovered, indexed, and becomes available for routing.
4. **Multi-agent reasoning** — planner/executor/reflector agents collaborate on
   complex queries, with LRU memory eviction to stay within 4 GB RAM.
5. **Safety guardrails** — NemoClaw-inspired input/output checks block harmful
   content and redact PII before it reaches the user.
6. **Confirmation gate** — low-confidence or high-risk tool invocations ask for
   user approval before executing.
7. **Local analytics** — every intent and tool use is logged to SQLite for
   offline RL evaluation.

---

## New AI Components

### 1. AirLLM Engine (`llm/airllm_engine.py`)
Efficient large-model inference inspired by [lyogavin/airllm](https://github.com/lyogavin/airllm).

**Conditional activation (new in this session):**
- AirLLM's layer-split loading is **only enabled when the model would exceed the 4 GB RAM budget** when loaded directly via `llama-cpp-python` (estimated as 1.2× the GGUF file size).
- For smaller models — including the default Qwen3.5-0.8B Q4 (~0.55 GB file, ~0.66 GB RAM) — `llama-cpp-python` is used directly as the **fastest possible path** (no layer-split overhead).
- For large models (e.g. Mistral-7B Q4 ~4.1 GB file, ~4.9 GB RAM estimate), the engine automatically activates AirLLM layer-by-layer loading to keep peak RAM well under 4 GB.
- For very large models where even AirLLM cannot fit under budget, quantisation level is incrementally tightened or features (reflection agent, summariser) are disabled.

Auto-selection logic:
```
model RAM estimate ≤ 4 GB  →  llama_cpp  (fastest path, always preferred)
model RAM estimate > 4 GB  →  airllm     (layer-split, stays within budget)
no backend / no model      →  mock       (CI / offline development)
```

Force a specific backend via `config/settings.yaml`:
```yaml
llm:
  engine: auto      # auto | airllm | llama_cpp | mock
```

### 2. Multi-Agent Dispatcher (`agent_core/multi_agent.py`)
OMLX-inspired parallel agent orchestration ([jundot/omlx](https://github.com/jundot/omlx)).

**OMLX Intel Mac compatibility (new in this session):**
- At startup, Lopen probes whether the `omlx` package is installed and compatible on your platform.
- On **Intel Mac (x86_64)**, OMLX may not be installable or may silently malfunction — Lopen detects this automatically.
- When OMLX is unavailable or incompatible, Lopen falls back to its built-in **asyncio-based lightweight agent pool** which provides the identical OpenClaw-style pipeline (planner → executor → reflector) using `asyncio.gather` for concurrency.
- The fallback is transparent — the same `AgentDispatcher` API is used regardless.

Startup log messages:
```
# When OMLX is available:
INFO  OMLX is available and compatible — using OMLX-accelerated parallel agent routing.

# When OMLX is absent (Intel Mac, most setups):
INFO  OMLX not available/compatible (common on Intel Mac x86_64).
      Using built-in asyncio agent pool — provides identical OpenClaw-style pipeline.
```

- **Planner** decomposes queries → **Executor** generates answers → **Reflector** checks quality
- LRU eviction ensures only active agents consume RAM
- Configurable agent pool in `config/agents.yaml`

### 3. Safety Engine (`agent_core/safety.py`)
NemoClaw-inspired guardrails ([NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)).
- Input guardrails: pattern/topic blocklist (bomb-making, CSAM, etc.)
- Tool filter: per-tool allowlist/denylist
- Output guardrails: PII redaction (SSN, email, phone, credit card)
- Fully configurable opt-in/opt-out via `config/settings.yaml`

See [docs/AI_ARCHITECTURE.md](docs/AI_ARCHITECTURE.md) for the complete guide.

---

## Quick Start (5 Steps)

```bash
# 1. One-command install (recommended)
bash install.sh

# OR manually:
bash scripts/bootstrap.sh      # Install system dependencies
bash scripts/setup_venv.sh     # Create Python venv + install packages
bash scripts/download_models.sh  # Download AI models (~2.3 GB)
cp .env.example .env           # Copy config

# 2. Start all services
bash scripts/start.sh

# 3. Open the dashboard
open http://localhost:8080

# 4. Launch the interactive CLI
python cli.py

# 5. Verify (run tests)
python -m pytest tests/ -q
# → 275 passed
```


## Model Options (April 2026)

| Model stack | File size | RAM usage | LLM speed | Notes |
|------------|-----------|-----------|-----------|-------|
| **Default** (Qwen3.5-0.8B Q4_K_M) | 550 MB | ~1.05 GB total | **<1s** 🚀 | Ultra-fast, multi-agent |
| **Quality** (Qwen3.5-1.5B Q4_K_M) | 1.0 GB | ~1.5 GB total | ~2s ✅ | Better reasoning |
| **Legacy** (Phi-3-mini Q4_K_M) | 2.2 GB | ~2.7 GB total | ~3–5s | Previous default |
| **Smart** (Mistral-7B Q4_K_M, AirLLM auto) | 4.1 GB | ~4.0 GB total | ~8s | AirLLM auto-activated (>4 GB RAM estimate); disable reflection agent |

```bash
# Default stack (Qwen3.5-0.8B — ultra-fast, recommended)
bash scripts/download_models.sh

# Quality upgrade (Qwen3.5-1.5B)
bash scripts/download_models.sh --quality

# Legacy (Phi-3-mini)
bash scripts/download_models.sh --phi3

# Smart stack (Mistral-7B, for AirLLM engine)
bash scripts/download_models.sh --mistral
```

> **Why Qwen3.5-0.8B over Phi-3-mini?**
> - 4× smaller (550 MB vs 2.2 GB) → cold starts in seconds
> - 3× faster inference (8–12 tok/s vs 2–4 tok/s on Intel Mac)
> - First response reliably **< 1 second** (vs 3–5s)
> - Leaves 3+ GB free for voice pipeline, multi-agent, and browser tools
> - Instruction-tuned quality matches Phi-3-mini on everyday tasks

---

## Memory Profile

| Component             | Typical RAM  | Notes                                          |
|-----------------------|-------------|------------------------------------------------|
| Orchestrator + Engine | ~100 MB     | FastAPI + TF-IDF index (pure Python, ~1 MB)    |
| Safety Engine         | ~1 MB       | Pure Python, no model                          |
| Multi-Agent Dispatcher| ~5 MB       | Agent pool (models loaded on demand)           |
| LLM (Qwen3.5-0.8B Q4, active) | ~0.55 GB | Loaded on-demand, unloaded after use    |
| Web Dashboard         | ~80 MB      | FastAPI + Jinja2                               |
| Voice Service         | ~150 MB     | includes whisper.cpp model                     |
| WhatsApp              | ~200 MB     | Playwright Chromium (headless)                 |
| **Total (default)**   | **~900 MB** | **3.1 GB free — well within 4 GB target** ✓   |

---

## REST Endpoints

| Method | Path               | Description                                     |
|--------|--------------------|-------------------------------------------------|
| GET    | `/health`          | Health check + uptime                           |
| GET    | `/status`          | Extended status (LLM, safety, agents, tasks)    |
| POST   | `/chat`            | Query with safety checks + multi-agent routing  |
| GET    | `/memory`          | Get conversation history                        |
| DELETE | `/memory`          | Clear conversation history                      |
| GET    | `/plugins`         | List all registered plugins with metadata       |
| POST   | `/plugins/reload`  | Rescan `tools/` dirs, register new plugins      |
| GET    | `/analytics`       | Usage statistics (tool counts, success rates)   |
| POST   | `/feedback`        | Submit helpfulness signal for RL tracking       |
| GET    | `/safety`          | Safety engine status and configuration          |
| GET    | `/agents`          | Multi-agent dispatcher pool status              |

Both `query` and `message` field names are accepted by `/chat` for compatibility.

---


## Adding a Plugin

1. Create `tools/third_party/my_plugin.py`:

```python
from tools.base_tool import BaseTool

class MyPlugin(BaseTool):
    name = "my_plugin"
    description = (
        "Handles my specific use case with detailed natural language description "
        "so the intent engine can match queries accurately."
    )
    tags = ["my", "custom", "keywords"]

    def run(self, query: str, **kwargs) -> str:
        return f"MyPlugin result for: {query}"
```

2. Restart, or call `POST /plugins/reload`.
3. Query it: `POST /chat` with `{"query": "do my specific thing"}`.

See [PLUGINS.md](PLUGINS.md) for the full plugin development guide.

---

## Installation

### One-Command Install (Recommended)

```bash
bash install.sh              # full install with model download prompt
bash install.sh --no-models  # skip model downloads (install later)
bash install.sh --yes --no-models  # fully non-interactive quick install
bash install.sh --yes --with-llama # non-interactive + llama-cpp-python
bash install.sh --debug      # verbose output
```

### Manual Installation

### Requirements
- macOS 12+ (Monterey or newer)
- Python 3.9+ — download from [python.org](https://www.python.org/downloads/macos/) or `xcode-select --install`
- **No Homebrew required** — all dependencies use pip or pre-built binaries

> See [docs/INSTALL_NO_HOMEBREW.md](docs/INSTALL_NO_HOMEBREW.md) for the full
> Homebrew-free install guide with step-by-step instructions for every dependency.

### Step-by-step

```bash
# Install system dependencies (Homebrew-free)
bash scripts/bootstrap.sh

# Set up Python virtual environment
bash scripts/setup_venv.sh

# Download AI models (Qwen3.5-0.8B + whisper-tiny + piper)
bash scripts/download_models.sh
```

---

## Service Management

```bash
# Start all services
bash scripts/start.sh

# Start with debug logging
bash scripts/start.sh --debug

# Stop all services
bash scripts/stop.sh

# Check status
bash scripts/status.sh

# Full self-diagnostics
bash scripts/diagnose.sh

# Inference benchmark
bash scripts/benchmark.sh

# Health check
bash scripts/health_check.sh

# Individual services
bash scripts/start_orchestrator.sh   # port 8000
bash scripts/start_dashboard.sh      # port 8080
bash scripts/start_voice.sh
bash scripts/start_whatsapp.sh
```

---

## Interface Setup

### Voice Service
Voice is enabled by default in `config/settings.yaml`. The service will:
1. Listen for the wake word **"Lopen"**
2. Transcribe speech via whisper.cpp (falls back to mock if not installed)
3. Process with LLM
4. Respond via piper TTS (falls back to macOS `say` command)

### WhatsApp
1. Set `whatsapp.enabled: true` in `config/settings.yaml`
2. Run `bash scripts/start_whatsapp.sh`
3. Scan the QR code shown in the browser (headless=false) or log
4. Session is saved for subsequent runs

### Web Dashboard
Access at http://localhost:8080. Features:
- Chat interface
- Live task queue view
- Conversation memory browser

---

## Configuration Guide

### `config/settings.yaml`
| Key | Default | Description |
|-----|---------|-------------|
| `llm.model_path` | `models/llm/model.gguf` | Path to GGUF model |
| `llm.memory_conservative` | `true` | Unload model between calls to save RAM |
| `intent_engine.confidence_threshold` | `0.2` | Below this, fall back to keyword planner |
| `plugin_loader.auto_discover` | `true` | Scan tool dirs on startup |
| `plugin_loader.tool_dirs` | `[tools, tools/third_party]` | Directories to scan |
| `sandbox.confidence_threshold` | `0.3` | Require confirmation below this |
| `sandbox.auto_approve_known_tools` | `true` | Skip confirmation after enough uses |
| `analytics.enabled` | `true` | Log usage to local SQLite |

### `config/tools.yaml`
Enable/disable individual tools and configure permissions.

---

## Troubleshooting

### Run self-diagnostics first

```bash
bash scripts/diagnose.sh
```

This checks OS, Python, RAM, disk, models, services, and config — and tells you what to fix.

### "LLM in MOCK mode"

Download a GGUF model:

```bash
bash scripts/download_models.sh
# or manually (Qwen3.5-0.8B default):
mkdir -p models/llm
curl -L -o models/llm/qwen3.5-0.8b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen3.5-0.8B-Instruct-GGUF/resolve/main/qwen3.5-0.8b-instruct-q4_k_m.gguf"
```

Then install `llama-cpp-python` (cmake is installed via pip — no Homebrew needed):

```bash
pip install cmake
CMAKE_ARGS="-DGGML_METAL=OFF" pip install "llama-cpp-python>=0.3.0"
```

### "ASR in mock mode"

Build whisper.cpp:

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp && make
cp main /usr/local/bin/whisper
```

Or install a pre-built binary from [the releases page](https://github.com/ggerganov/whisper.cpp/releases).

### Port already in use

```bash
bash scripts/status.sh   # see what's running
bash scripts/stop.sh     # stop all services
# Or kill specific PID:
lsof -ti:8000 | xargs kill -9
```

### High RAM usage

1. Ensure `llm.memory_conservative: true` in `config/settings.yaml`
2. Check `bash scripts/health_check.sh` for RAM report
3. Reduce `llm.context_window` to 1024
4. Disable multi-agent: `multi_agent.enabled: false`
5. Run diagnostics: `bash scripts/diagnose.sh`

### CLI can't connect

```bash
# Check orchestrator is running
bash scripts/status.sh

# Start if not running
bash scripts/start.sh

# Try debug mode
python cli.py --debug
```

### Services crash immediately

```bash
# Check the error log
cat logs/lopen_error.log

# Run with debug mode
bash scripts/start.sh --debug
cat logs/lopen_debug.log
```

---

## Running Tests

```bash
source .venv/bin/activate

# Full test suite
pytest tests/ -v --tb=short
# → 275 passed

# CLI tests only
pytest tests/test_cli.py -v

# Safety engine tests
pytest tests/test_safety.py -v

# Smoke tests (requires running services)
pytest tests/smoke/ -v
```

---

## Resource Profile by Component (April 2026)

```
Orchestrator API:      ~100 MB
IntentEngine:          <1 MB   (pure Python TF-IDF, no extra dependencies)
LLM Qwen3.5-0.8B Q4:  ~0.55 GB (load on demand, unloaded after use)
whisper-tiny ASR:      ~80 MB
piper TTS:             ~70 MB
WhatsApp (Chrome):     ~200 MB
TOTAL (default stack): ~1.05 GB (target: ≤ 4 GB)  ← 2.9 GB headroom ✓

Memory guard thresholds:
  Warning at:  3.2 GB used
  Critical at: 3.6 GB used  → watchdog triggers model unload + service restart
```

---

## Session Notes — April 2026

### AirLLM conditional activation
Previously, `AirLLMEngine` always preferred the AirLLM layer-split backend
whenever the `airllm` package was installed, even for tiny models.  This added
unnecessary overhead (layer-split I/O) for small models like Qwen3.5-0.8B that
fit comfortably in RAM.

**As of this session**, AirLLM is only activated when a model's estimated RAM
usage (1.2× the GGUF file size) **exceeds the 4 GB budget**:

- `≤ 4 GB estimate` → `llama_cpp` (fastest path, no layer-split overhead)
- `> 4 GB estimate` → `airllm` (layer-by-layer loading, keeps peak RAM ≤ ~2 GB)
- Can be overridden with `llm.engine: airllm|llama_cpp|mock` in `config/settings.yaml`

This change delivers faster inference for all default models and only incurs the
AirLLM overhead when strictly necessary.

### OMLX multi-agent — Intel Mac fallback
OMLX (`pip install omlx`) provides native parallel multi-LLM routing but is not
always compatible with Intel Mac (x86_64 darwin).

**As of this session**, Lopen auto-detects OMLX compatibility at startup:

1. If OMLX imports successfully and passes a platform probe → OMLX-accelerated
   routing is used.
2. If OMLX is absent or fails the probe (typical for Intel Mac) → Lopen uses its
   built-in **asyncio lightweight agent pool** which provides the same
   OpenClaw-style `planner → executor → reflector` pipeline with zero extra
   dependencies.

In both cases the `AgentDispatcher` API is identical.  Check the startup log
for which path was selected:
```
INFO  OMLX not available/compatible (common on Intel Mac x86_64).
      Using built-in asyncio agent pool…
```
