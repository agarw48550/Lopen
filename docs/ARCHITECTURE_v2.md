# Lopen Architecture v2 — Complete Student Agent System

> **Target hardware**: 2017 Intel MacBook Pro, 8 GB RAM, ≤4 GB runtime budget  
> **Audience**: Developers, contributors, and curious users

---

## Overview

Lopen is a **local-first autonomous assistant** designed to run 24/7 on a 2017
Intel MacBook Pro with an 8 GB RAM budget of which no more than **4 GB** is
consumed at any one time.  It provides homework tutoring, research assistance,
desktop organisation, coding help, and student-workflow management through three
communication interfaces: WhatsApp, SSH/CLI (MacBook Air), and wake-word voice.

---

## Component Memory Budget

| Component | Strategy | Max RAM |
|-----------|----------|---------|
| **Orchestrator** | FastAPI + TF-IDF, always-on | ~100 MB |
| **LLM** (Qwen3.5-0.8B Q4_K_M) | Load on demand, unload after inference | ~650 MB peak |
| **Safety Engine** | Pure Python, no models | < 1 MB |
| **Notion Sync** | Local SQLite cache, read-only API | ~50 MB |
| **ASR** (whisper-tiny) | Lazy-load, stream-only, unload when idle | ~80 MB |
| **TTS** (piper ryan-high) | Loaded once, stays in memory | ~70 MB |
| **Voice/Audio Model** (LFM2.5-Audio-1.5B) | Optional primary path | ~300 MB |
| **WhatsApp** (Playwright) | Session pickle + message queue | ~30 MB |
| **SSH API** | Stateless stdlib HTTP | ~5 MB |
| **SQLite queue/cache** | mmap-friendly, no heap pressure | ~10 MB |
| **OS Reserve** | Kernel + system daemons | ~1 GB |
| **Total Target** | | **≤ 4 GB** |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT INTERFACES                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │  WhatsApp    │  │  SSH API     │  │  Voice (wake-word) │   │
│  │  (Playwright)│  │  port 8001   │  │  LFM2.5/Whisper    │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘   │
│         │                 │                    │                │
│         └─────────────────┴────────────────────┘                │
│                           │                                     │
│                           ▼                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  ORCHESTRATOR  (FastAPI, port 8000)                    │    │
│  │                                                        │    │
│  │  IntentEngine → ToolSelector → ArgumentComposer        │    │
│  │  SafetyEngine (NemoClaw guardrails)                    │    │
│  │  ConfirmationGate → MultiAgentDispatcher (OMLX)        │    │
│  │  Analytics (SQLite) + Memory (TF-IDF + embeddings)     │    │
│  │                                                        │    │
│  │  GET  /health  /status  /metrics                       │    │
│  │  POST /chat    /feedback                               │    │
│  │  GET  /plugins /analytics /tasks /agents               │    │
│  └──────────────────────────┬─────────────────────────────┘    │
│                              │                                  │
│              ┌───────────────┴───────────────┐                 │
│              ▼                               ▼                 │
│  ┌───────────────────┐        ┌──────────────────────────┐    │
│  │  LLM Backend      │        │  Tool Plugins            │    │
│  │  (LLMAdapter)     │        │  ┌────────────────────┐  │    │
│  │  ┌─────────────┐  │        │  │ HomeworkTutor       │  │    │
│  │  │ llama-cpp   │  │        │  │ NotionIntegration   │  │    │
│  │  │ AirLLM      │  │        │  │ ProjectPulse        │  │    │
│  │  │ Mock        │  │        │  │ Researcher          │  │    │
│  │  └─────────────┘  │        │  │ CoderAssist         │  │    │
│  │  Lazy-load +       │        │  │ DesktopOrganizer    │  │    │
│  │  unload after use  │        │  │ BrowserAutomation   │  │    │
│  └───────────────────┘        │  └────────────────────┘  │    │
│                               └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                           │
             ┌─────────────┴────────────┐
             ▼                          ▼
  ┌─────────────────────┐   ┌──────────────────────────┐
  │  System Health      │   │  Storage                 │
  │  RamWatchdog (3-lvl)│   │  SQLite: memory, analytics│
  │  DiskCheck          │   │  Notion cache            │
  │  LogRotation        │   │  ProjectPulse tasks      │
  │  CacheCleanup       │   │  WhatsApp session        │
  │  Heartbeat          │   └──────────────────────────┘
  │  maintenance.sh     │
  └─────────────────────┘
```

---

## Communication Interfaces

### 1. WhatsApp (Mobile)

- **Transport**: Playwright headless Chromium session to WhatsApp Web
- **Session**: Persisted in `storage/whatsapp_session/` (no re-scan needed)
- **Flow**: Poll for new messages → enqueue to `agent_core.task_queue` → 
  orchestrator processes → sends reply
- **Start**: `bash scripts/start_whatsapp.sh`

### 2. SSH / MacBook Air CLI

New in v2.  A lightweight HTTP API on port **8001** (127.0.0.1 only) that can
be reached from a remote machine via SSH port-forwarding:

```bash
# On MacBook Air:
ssh -L 8001:localhost:8001 your-user@macbook-pro.local

# Then query Lopen from MacBook Air:
curl -H "Authorization: Bearer $LOPEN_SSH_API_KEY" \
     http://localhost:8001/query \
     -d '{"query": "What is the derivative of x^2?"}'
```

- No Paramiko, no SSH daemon — just a Python `http.server` wrapped in FastAPI
  when available (falls back to stdlib)
- Set `LOPEN_SSH_API_KEY` in `.env` to require authentication
- See `interfaces/ssh_service/__init__.py` and `config/settings.yaml`
  (`ssh_api.enabled: true`)

### 3. Voice (Desk / Wake-Word)

- **Primary path**: Raw audio → LiquidAI/LFM2.5-Audio-1.5B → audio reply  
  (no TTS/ASR round-trip — latency ~200–400 ms)
- **Fallback path**: Whisper-tiny ASR → LLM → Piper TTS  
  (latency ~2–4 s on 2017 MacBook)
- **Wake word**: `openwakeword` (falls back to keyword scan in transcript)
- **Start**: `bash scripts/start_voice.sh`

---

## Student Features (New in v2)

### Notion Integration (`tools/notion_integration.py`)

Read-only bridge to your Notion workspace:

| Feature | Details |
|---------|---------|
| **Assignments DB** | Reads tasks with due dates, status, subject |
| **Notes DB** | Full-text search on block content |
| **Deadline Alerts** | Overdue, today, tomorrow, ≤3 days |
| **Local Cache** | SQLite at `storage/notion_cache.db` — works offline |
| **Sync interval** | 1 hour (configurable, non-blocking) |
| **Read-only** | Never writes to Notion — prevents accidental data loss |

Setup:
```bash
# 1. Create a Notion integration at https://www.notion.so/my-integrations
# 2. Add to .env:
NOTION_TOKEN=secret_xxx
NOTION_ASSIGNMENTS_DB=your-database-id
NOTION_NOTES_DB=your-notes-database-id
# 3. Enable in config/settings.yaml:
#    notion.enabled: true
# 4. Install: pip install notion-client
```

### Project Pulse (`tools/project_pulse.py`)

Student task tracking with visual feedback:

```
╔══════════════════ TASK BOARD ══════════════════╗
  📋 Backlog
    #001 Write history essay            [🟠 3d]
    #002 Physics problem set            [🔴 TODAY]
  🔄 In Progress
    #003 Math assignment                [🟡 TOMORROW]
  ✅ Done
    #004 English reading

📊 BURNDOWN CHART (last 4 weeks)
   5│  ██  ██  ██  ██
   4│  ██  ██  ██  ██
   3│      ██  ██  ██
   2│          ██  ██
     ──────────────────
      03/15 03/22 03/29 04/05
```

- **Socratic prompts**: 3 guided questions per task (subject-aware)
- **Notion sync**: Import assignments from Notion with `pulse.sync_from_notion()`
- **Persistent**: SQLite at `storage/project_pulse.db`

---

## System Health (24/7 Operation)

### RAM Watchdog (`system_health/ram_watchdog.py`)

Three-level escalating response:

| Level | Threshold | Action |
|-------|-----------|--------|
| warning | 3.2 GB | `on_warning()` — unload LLM model |
| critical | 3.6 GB | `on_critical()` — restart service (backoff: 1 s → 5 s → 30 s) |
| halt | 3.8 GB | `on_halt()` — emergency stop |

Hysteresis: must drop 10% below threshold before re-triggering.  
Backoff: `_RESTART_BACKOFF = (1, 5, 30)` — prevents tight restart loops.

### Daily Maintenance (`system_health/maintenance.sh`)

Runs automatically at 03:00 AM via launchd:

1. Rotate log files older than 7 days
2. Vacuum SQLite databases
3. Clear Python `__pycache__` directories
4. Clear `~/Library/Caches/Lopen/` stale entries
5. Clean stale Playwright browser logs
6. Disk and RAM usage report

### Metrics Endpoint (`GET /metrics`)

```json
{
  "service": "orchestrator",
  "uptime_seconds": 86400,
  "ram": {"used_gb": 1.8, "total_gb": 8.0, "percent": 22.5},
  "cpu": {"percent": 12.0, "count": 4},
  "llm": {"backend": "llama_cpp", "loaded": false},
  "queue_depth": 0,
  "latency": {"p50_ms": 850, "total_queries": 142}
}
```

---

## Auto-Start on macOS (launchd)

```bash
# One-command setup (from Lopen repo root):
bash install/setup_launchd.sh
```

This installs two LaunchAgents:
- `com.lopen.agent` — orchestrator, restarts on crash (10 s throttle)
- `com.lopen.maintenance` — daily cleanup at 03:00 AM

Logs: `~/Library/Logs/Lopen/`

---

## Install Script Compatibility

`install.sh` is fully compatible with **Bash 3.2** (the default on macOS since
2007 due to GPL licensing).  The `${REPLY,,}` lowercase expansion that requires
Bash 4+ has been replaced throughout with `tr '[:upper:]' '[:lower:]'`.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/agarw48550/Lopen && cd Lopen

# 2. Install (no Homebrew required)
bash install.sh

# 3. Configure
cp .env.example .env && $EDITOR .env   # add NOTION_TOKEN etc.

# 4. Start
bash scripts/start.sh

# 5. Chat
python cli.py

# 6. Enable 24/7 auto-start (macOS only)
bash install/setup_launchd.sh
```

---

## File Structure

```
Lopen/
├── install.sh                 # One-command installer (Bash 3.2 compatible)
├── install/
│   ├── lopen.plist            # macOS LaunchAgent — orchestrator
│   ├── lopen-maintenance.plist # macOS LaunchAgent — daily maintenance
│   └── setup_launchd.sh       # Automated launchd setup
├── orchestrator.py            # FastAPI app (port 8000)
├── cli.py                     # Interactive CLI
├── agent_core/                # Intent, routing, safety, tools, analytics
├── llm/                       # LLM backends (llama-cpp, AirLLM, mock)
├── interfaces/
│   ├── voice_service/         # Wake-word, ASR, TTS, LFM2.5 audio model
│   ├── whatsapp_service/      # Playwright WhatsApp Web bridge
│   ├── ssh_service/           # SSH API (port 8001, stdlib + FastAPI)
│   └── web_dashboard/         # Web UI (port 8080)
├── tools/
│   ├── notion_integration.py  # ← NEW: Notion API bridge (read-only)
│   ├── project_pulse.py       # ← NEW: Task tracking + burndown
│   ├── homework_tutor.py      # Socratic Q&A, subject detection
│   ├── researcher.py          # Web research + caching
│   ├── coder_assist.py        # Code generation + review
│   ├── desktop_organizer.py   # File organisation
│   └── browser_automation.py  # Web automation (Playwright)
├── system_health/
│   ├── ram_watchdog.py        # ← ENHANCED: 3-level + backoff restart
│   ├── maintenance.sh         # ← NEW: Daily cleanup script
│   ├── heartbeat.py           # Service health polling
│   ├── disk_check.py          # Disk space alerts
│   ├── log_rotation.py        # Log file rotation
│   └── cache_cleanup.py       # __pycache__ + temp file cleanup
├── config/
│   ├── settings.yaml          # All tunable parameters
│   ├── agents.yaml            # Multi-agent pool configuration
│   ├── models.yaml            # Model paths and parameters
│   └── logging.yaml           # Structured JSON logging config
├── storage/                   # SQLite databases (local, never uploaded)
├── models/                    # Downloaded AI model files
├── scripts/                   # start/stop/status/diagnose/benchmark
└── docs/
    ├── ARCHITECTURE_v2.md     # This file
    ├── AI_ARCHITECTURE.md     # LLM backend selection guide
    └── INSTALL_NO_HOMEBREW.md # Homebrew-free install guide
```
