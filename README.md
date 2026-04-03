# Lopen — Local-First Autonomous Assistant

**Lopen** is a production-ready, local-first autonomous assistant framework that runs entirely on your Mac. It orchestrates voice commands, WhatsApp messaging, web research, homework tutoring, coding assistance, and desktop organisation — all without sending your data to the cloud.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Lopen Orchestrator (port 8000)              │
│  ┌─────────────┐  ┌───────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Planner   │→│   Router  │→│Tool Registry│  │ Task Queue  │ │
│  └─────────────┘  └───────────┘  └─────────────┘  └─────────────┘ │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      LLM Adapter (llama.cpp / mock)          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │             Conversation Memory  ←→  SQLite Storage          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
       │               │               │               │
┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼───────────┐
│ Voice Loop  │ │  WhatsApp   │ │    Web     │ │ System Health  │
│ Mic→Wake→  │ │   Bridge    │ │ Dashboard  │ │  RAM/Disk/Hb/  │
│ ASR→LLM→TTS│ │ (Playwright)│ │ (port 8080)│ │  Log Rotation  │
└─────────────┘ └─────────────┘ └────────────┘ └────────────────┘
       │               │               │
┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
│ whisper.cpp │ │ WhatsApp    │ │  FastAPI   │
│ piper TTS   │ │    Web      │ │  Jinja2    │
└─────────────┘ └─────────────┘ └────────────┘

Tools: HomeworkTutor │ Researcher │ CoderAssist │ DesktopOrganizer │ FileOps │ BrowserAutomation
```

---

## Quick Start (5 Steps)

```bash
# 1. Install system dependencies (macOS + Homebrew)
bash scripts/bootstrap.sh

# 2. Create Python venv and install packages
bash scripts/setup_venv.sh

# 3. Download AI models (~2.5 GB total)
bash scripts/download_models.sh

# 4. Copy and configure environment
cp .env.example .env
# Edit .env if needed

# 5. Start all services
bash scripts/start.sh
```

Then open http://localhost:8080 in your browser.

---

## Memory Profile

| Component       | Typical RAM  | Notes                                    |
|-----------------|-------------|------------------------------------------|
| Orchestrator    | ~100 MB     | FastAPI + APScheduler                    |
| LLM (Phi-3 Q4) | ~2.2 GB     | Loaded on-demand, unloaded after use     |
| Web Dashboard   | ~80 MB      | FastAPI + Jinja2                         |
| Voice Service   | ~150 MB     | includes whisper.cpp model               |
| WhatsApp        | ~200 MB     | Playwright Chromium (headless)           |
| **Total**       | **~2.7 GB** | Well within 4 GB target                  |

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
| `llm.context_window` | `2048` | Tokens in context (increase for better reasoning) |
| `llm.memory_conservative` | `true` | Unload model between calls to save RAM |
| `voice.enabled` | `true` | Enable voice service |
| `voice.wake_word` | `Lopen` | Wake word (case-insensitive) |
| `health.ram_threshold_gb` | `4.0` | Critical RAM threshold |

### `config/models.yaml`
Defines download URLs and file locations for all AI models.

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
LLM (Q4_K_M):      ~2.2 GB (load on demand)
whisper-tiny ASR:  ~150 MB
piper TTS:         ~50 MB
WhatsApp (Chrome): ~200 MB
TOTAL (all):       ~2.7 GB  (target: ≤ 4 GB)
```
