# Lopen — Local-First General-Purpose Autonomous Assistant

**Lopen** is a production-ready, local-first autonomous assistant framework that runs entirely on your Mac. It uses an **intent-driven, plugin-extensible architecture** to handle any open-ended user request — routing it semantically to the best available tool, without hardcoded mappings or pre-defined task lists.

**New in this release:** AirLLM-backed large-model inference, OMLX-inspired multi-agent orchestration, NemoClaw-inspired safety guardrails, and curated best-fit offline models.

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
- Layer-by-layer CPU scheduling — loads only 1–2 transformer layers at a time
- Supports 7B Q4_K_M models within 4 GB RAM (with AirLLM backend)
- Auto-selects best backend: `airllm` → `llama_cpp` → `mock`
- Configurable via `config/settings.yaml` (`llm.engine`)

### 2. Multi-Agent Dispatcher (`agent_core/multi_agent.py`)
OMLX-inspired parallel agent orchestration ([jundot/omlx](https://github.com/jundot/omlx)).
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
# 1. Install system dependencies (macOS + Homebrew)
bash scripts/bootstrap.sh

# 2. Create Python venv and install packages
bash scripts/setup_venv.sh

# 3. Download AI models (~2.3 GB total)
bash scripts/download_models.sh

# 4. Copy and configure environment
cp .env.example .env
# Edit .env if needed

# 5. Start all services
bash scripts/start.sh
```

Then open http://localhost:8080 in your browser.

**One-command verification:**
```bash
python -m pytest tests/ -q
# → 247 passed (includes safety, multi-agent, and airllm engine tests)
```

---

## Model Options

| Model stack | RAM usage | LLM quality | Notes |
|------------|-----------|-------------|-------|
| **Default** (Phi-3-mini Q4_K_M) | ~2.7 GB | ★★★★☆ | Multi-agent capable |
| **Smart** (Mistral-7B Q4_K_M + AirLLM) | ~4.0 GB | ★★★★★ | Disable reflection agent |

```bash
# Default stack
bash scripts/download_models.sh

# Smart stack (Mistral-7B, for AirLLM engine)
bash scripts/download_models.sh --mistral
```

---

## Memory Profile

| Component             | Typical RAM  | Notes                                          |
|-----------------------|-------------|------------------------------------------------|
| Orchestrator + Engine | ~100 MB     | FastAPI + TF-IDF index (pure Python, ~1 MB)    |
| Safety Engine         | ~1 MB       | Pure Python, no model                          |
| Multi-Agent Dispatcher| ~5 MB       | Agent pool (models loaded on demand)           |
| LLM (Phi-3 Q4, active)| ~2.2 GB     | Loaded on-demand, unloaded after use           |
| Web Dashboard         | ~80 MB      | FastAPI + Jinja2                               |
| Voice Service         | ~150 MB     | includes whisper.cpp model                     |
| WhatsApp              | ~200 MB     | Playwright Chromium (headless)                 |
| **Total**             | **~2.7 GB** | Well within 4 GB target                        |

---

## REST Endpoints

| Method | Path               | Description                                     |
|--------|--------------------|-------------------------------------------------|
| POST   | `/chat`            | Query with safety checks + multi-agent routing  |
| GET    | `/safety`          | Safety engine status and configuration          |
| GET    | `/agents`          | Multi-agent dispatcher pool status              |
| GET    | `/status`          | Extended status (LLM, safety, agents, tasks)    |
| GET    | `/plugins`         | List all registered plugins with metadata       |
| POST   | `/plugins/reload`  | Rescan `tools/` dirs, register new plugins      |
| GET    | `/analytics`       | Usage statistics (tool counts, success rates)   |
| POST   | `/feedback`        | Submit helpfulness signal for RL tracking       |

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

### Requirements
- macOS 12+ (Monterey or newer)
- Python 3.11+
- Homebrew

### Step-by-step

```bash
# Install Homebrew dependencies
bash scripts/bootstrap.sh

# Set up Python virtual environment
bash scripts/setup_venv.sh

# Download AI models
bash scripts/download_models.sh
```

---

## Service Management

```bash
# Start all services
bash scripts/start.sh

# Stop all services
bash scripts/stop.sh

# Check status
bash scripts/status.sh

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

### "LLM in MOCK mode"
Download a GGUF model: `bash scripts/download_models.sh`

### "ASR in mock mode"
Build whisper.cpp from https://github.com/ggerganov/whisper.cpp and place the binary in PATH.

### Port already in use
Check for existing processes: `bash scripts/status.sh` then `bash scripts/stop.sh`

### High RAM usage
1. Ensure `llm.memory_conservative: true` in `config/settings.yaml`
2. Check `bash scripts/health_check.sh` for RAM report
3. Reduce `llm.context_window` to 1024

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v --tb=short
```

---

## Resource Profile by Component

```
Orchestrator API:   ~100 MB
IntentEngine:       <1 MB  (pure Python TF-IDF, no extra dependencies)
LLM (Q4_K_M):      ~2.2 GB (load on demand)
whisper-tiny ASR:  ~150 MB
piper TTS:         ~50 MB
WhatsApp (Chrome): ~200 MB
TOTAL (all):       ~2.7 GB  (target: ≤ 4 GB)
```

