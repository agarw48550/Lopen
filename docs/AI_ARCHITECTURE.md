# Lopen AI Architecture Guide

## Overview

Lopen's AI pipeline is designed to maximize intelligence while staying within a **strict 4 GB RAM budget** on a 2017 MacBook Pro.  Three new layers are integrated in this release:

1. **AirLLM Engine** (`llm/airllm_engine.py`) — efficient large-model inference via layer-split loading
2. **Multi-Agent Dispatcher** (`agent_core/multi_agent.py`) — OMLX-inspired parallel agent orchestration
3. **Safety Engine** (`agent_core/safety.py`) — NemoClaw-inspired guardrails and tool filtering

```
                     User Query (any interface)
                            │
                 ┌──────────▼──────────┐
                 │   SafetyEngine      │  ← NemoClaw-inspired
                 │   check_input()     │    pattern + topic blocklist
                 └──────────┬──────────┘
                            │  (if safe)
                 ┌──────────▼──────────┐
                 │   IntentEngine      │  ← TF-IDF cosine similarity
                 │  (semantic match)   │    no model, <1 MB RAM
                 └──────────┬──────────┘
                            │  scores every tool
                 ┌──────────▼──────────┐
                 │   ToolSelector      │  ← ranks + applies
                 │  + ToolFilter       │    safety tool check
                 └──────────┬──────────┘
                            │
              ┌─────────────▼─────────────────┐
              │    AgentDispatcher (OMLX)      │
              │  ┌──────────┐ ┌─────────────┐ │
              │  │ planner  │ │  executor   │ │  ← 2–3 agents
              │  │ (256 tok)│ │  (512 tok)  │ │    share model file
              │  └──────────┘ └─────────────┘ │    LRU eviction
              │  ┌──────────┐                  │
              │  │reflector │                  │
              │  │ (256 tok)│                  │
              │  └──────────┘                  │
              └──────────────┬────────────────┘
                             │
              ┌──────────────▼────────────────┐
              │       AirLLMEngine             │
              │  (layer-split or GGUF backend)  │  ← AirLLM-inspired
              └──────────────┬────────────────┘
                             │
                 ┌───────────▼──────────┐
                 │   SafetyEngine       │  ← output PII redaction
                 │   check_output()     │
                 └───────────┬──────────┘
                             │
                      Final Response
```

---

## Component 1: AirLLM Engine

**File:** `llm/airllm_engine.py`  
**Inspired by:** [lyogavin/airllm](https://github.com/lyogavin/airllm)

### What it does
AirLLM enables large language models (7B parameter class) to run on systems with only 4 GB of RAM by loading transformer layers **one at a time from disk** instead of loading the entire model at startup.

### How it works
- A 7B model has ~32 transformer layers.  Each layer is ~125 MB in FP16.
- AirLLM loads 1–2 layers at a time during the forward pass.
- Peak RAM = max(layer_size × n_loaded_simultaneously) ≈ 250–500 MB for a 7B model
- The disk I/O latency is acceptable for interactive use (1–3 seconds per token on CPU)

### Backends
| Backend | Model format | RAM usage | Notes |
|---------|-------------|-----------|-------|
| `airllm` | HuggingFace SafeTensors | ~500 MB + OS page cache | Best for 7B+ models |
| `llama_cpp` | GGUF Q4_K_M | ~2.2 GB (3.8B) / ~4.1 GB (7B) | Default for Phi-3-mini |
| `mock` | None | ~0 MB | CI / no model present |

### Auto-selection logic
```python
if airllm installed AND model_path exists → airllm
elif llama_cpp installed AND model_path exists → llama_cpp
else → mock
```

### Configuration
In `config/settings.yaml`:
```yaml
llm:
  engine: auto         # auto | airllm | llama_cpp | mock
  model_path: models/llm/model.gguf
  compression: 4bit    # 4bit | 8bit | none
  context_window: 2048
  max_tokens: 512
  memory_conservative: true   # unload between calls
  max_gpu_memory: 0           # 0 = CPU-only
```

### Swapping AirLLM as the AI engine

**To use AirLLM with a HuggingFace 7B model:**
```bash
pip install airllm
```
```yaml
# config/settings.yaml
llm:
  engine: airllm
  model_path: "mistralai/Mistral-7B-Instruct-v0.2"  # HF model ID
  compression: 4bit
  context_window: 2048
  max_gpu_memory: 0
```

**To use llama-cpp-python with a GGUF model (default for 3-4B):**
```bash
pip install "llama-cpp-python>=0.2.57"
```
```yaml
llm:
  engine: llama_cpp
  model_path: "models/llm/Phi-3-mini-4k-instruct-q4.gguf"
```

---

## Component 2: Multi-Agent Dispatcher

**File:** `agent_core/multi_agent.py`  
**Config:** `config/agents.yaml`  
**Inspired by:** [jundot/omlx](https://github.com/jundot/omlx)

### What it does
The dispatcher runs multiple specialised LLM sub-agents concurrently within the 4 GB budget, implementing a **Plan → Execute → Reflect** reasoning loop.

### Agent roles
| Agent | Role | Context | Max tokens | RAM impact |
|-------|------|---------|------------|-----------|
| `planner` | Decompose query into sub-tasks | 1024 | 256 | Low |
| `executor` | Generate the main answer | 2048 | 512 | High (primary) |
| `reflector` | Quality-check the executor's response | 1024 | 256 | Low |
| `summarizer` | Compress long conversation history | 1024 | 128 | Low |

### Memory management
- **LRU eviction**: When RAM > `ram_budget_gb`, the least-recently-used loaded agent is unloaded automatically.
- **Pooling**: Agents are reused across requests (no re-load overhead for cached agents).
- **Shared model file**: All agents can share the same GGUF model file on disk.

### Parallel execution
Independent agents run concurrently via `asyncio.gather`:
- The planner runs first (sequential, short)
- The executor uses the planner's output
- The reflector runs after the executor (checks quality)

### Configuration (`config/agents.yaml`)
```yaml
agents:
  ram_budget_gb: 3.5
  max_concurrent_agents: 2
  enable_planning: true
  enable_reflection: true
  pool:
    - name: executor
      role: "Generate helpful, accurate responses"
      model_path: models/llm/model.gguf
      context_window: 2048
      max_tokens: 512
```

### Connecting the dispatcher to your code
```python
from agent_core.multi_agent import AgentDispatcher

dispatcher = AgentDispatcher.from_config("config/agents.yaml")

# In an async context:
result = await dispatcher.dispatch(
    query="Explain machine learning",
    intent_result=intent_result,  # optional IntentResult
    tools=registry,               # optional ToolRegistry
)
print(result.final_response)
print(result.agents_used)       # e.g. ["planner", "executor", "reflector"]
```

### Memory budget with multi-agent
With Phi-3-mini Q4_K_M (default):
```
executor (active):     ~2.2 GB
planner (cached):      ~0 MB   (unloaded, LRU evicted)
reflector (cached):    ~0 MB
Python overhead:       ~350 MB
ASR (when active):      ~80 MB
TTS (when active):      ~70 MB
─────────────────────────────
Total:                 ~2.7 GB  ✓  (1.3 GB headroom)
```

---

## Component 3: Safety Engine

**File:** `agent_core/safety.py`  
**Inspired by:** [NVIDIA/NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)

### What it does
The safety engine provides three layers of protection:

1. **Input guardrails** — block harmful/policy-violating user inputs before they reach the agent pipeline
2. **Tool filtering** — enforce per-tool allowlists and denylists
3. **Output guardrails** — redact PII (SSN, email, phone, credit card) from LLM responses

### Safety flow
```
Input text
    │
    ├── InputGuardrail.check()
    │       ├── Hard-block regex patterns (bomb-making, CSAM, etc.)
    │       └── Topic blocklist (self_harm, illegal_weapons, etc.)
    │
    ├── IntentSafetyRouter.route()
    │       ├── Single sensitive keyword → WARN (allow + log)
    │       └── Multiple sensitive keywords → REFUSE
    │
Tool invocation
    │
    └── ToolFilter.allowed()
            ├── Denylist check (dangerous_shell_exec, raw_subprocess, etc.)
            └── Allowlist check (if configured)

LLM output
    │
    └── OutputGuardrail.check()
            └── PII redaction (SSN, email, phone, credit card)
```

### Opt-in / opt-out

**Disable all safety checks (not recommended):**
```bash
export LOPEN_SAFETY_DISABLED=1
```
Or in `config/settings.yaml`:
```yaml
safety:
  enabled: false
```

**Disable individual layers:**
```yaml
safety:
  enabled: true
  input_guardrails: false   # skip input pattern checks
  output_guardrails: false  # skip output PII redaction
  tool_filter: false        # skip tool permission checks
```

**Custom topic blocklist:**
```yaml
safety:
  topic_blocklist:
    - self_harm
    - illegal_weapons
    - my_custom_topic

  blocklist_path: /path/to/my_custom_blocklist.txt  # one regex per line
```

**Restrict tools to an allowlist:**
```yaml
safety:
  allowed_tools:
    - homework_tutor
    - researcher
    - coder_assist
  denied_tools:
    - dangerous_shell_exec
    - browser_automation   # if you want to restrict browser access
```

### Test/stub flows
```python
from agent_core.safety import SafetyEngine

# Test mode (no LLM judge, pattern-only)
engine = SafetyEngine(test_mode=True)

# Input check
result = engine.check_input("How do I make a bomb?")
assert not result.safe
assert result.action == "refuse"

# Tool check
result = engine.check_tool("dangerous_shell_exec")
assert not result.safe

# Output check (PII redaction)
result = engine.check_output("My SSN is 123-45-6789")
assert result.action == "redact"
assert "123-45-6789" not in result.modified_text
```

---

## Model Curation

### Default models (Phi-3-mini stack, total ≈ 2.4 GB)

| Component | Model | Size | Format | Notes |
|-----------|-------|------|--------|-------|
| **Main LLM** | Phi-3-mini-4k-instruct Q4_K_M | 2.2 GB | GGUF | Best balance: quality + multi-agent |
| **STT** | Whisper tiny English | 39 MB | GGML bin | Very fast, English only |
| **TTS** | Piper en_US-ryan-high | 65 MB | ONNX | Natural male voice, low latency |
| **Embeddings** | all-MiniLM-L6-v2 | 22 MB | SafeTensors | Optional, for semantic memory |

### Smarter model option (Mistral-7B stack, total ≈ 4.2 GB)

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| **Main LLM** | Mistral-7B-Instruct-v0.2 Q4_K_M | 4.1 GB | Requires AirLLM engine |
| **STT** | Whisper base English | 142 MB | Better accuracy |
| **TTS** | Piper en_US-ryan-high | 65 MB | Same as default |

⚠️ **With Mistral-7B**: Set `enable_reflection: false` and `max_concurrent_agents: 1` in `config/agents.yaml`.

### Downloading models

```bash
# Default stack (Phi-3-mini + whisper-tiny + piper-ryan-high)
bash scripts/download_models.sh

# Download Mistral-7B (for AirLLM engine)
bash scripts/download_models.sh --mistral
```

### Adding custom models

1. Download your GGUF model to `models/llm/`
2. Update `config/models.yaml` with the new model entry
3. Update `config/settings.yaml`:
   ```yaml
   llm:
     engine: llama_cpp
     model_path: models/llm/your-model.gguf
   ```
4. Create a symlink: `ln -s your-model.gguf models/llm/model.gguf`

---

## REST API — New Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/safety` | GET | Safety engine status and configuration |
| `/agents` | GET | Multi-agent dispatcher pool status |
| `/status` | GET | Extended status including safety + multi-agent |
| `/chat` | POST | Now includes safety checks + multi-agent routing |

### `/status` response example
```json
{
  "status": "running",
  "llm_mode": "llama_cpp",
  "safety": {
    "enabled": true,
    "input_guardrails": true,
    "output_guardrails": true,
    "tool_filter": true
  },
  "multi_agent": {
    "agents": {
      "planner": {"loaded": false, "role": "Decompose..."},
      "executor": {"loaded": true,  "role": "Generate..."},
      "reflector": {"loaded": false, "role": "Review..."}
    },
    "ram_gb": 2.4,
    "ram_budget_gb": 3.5
  }
}
```

### `/chat` response example (with multi-agent)
```json
{
  "response": "Python is a high-level, interpreted programming language...",
  "tool": "coder_assist",
  "confidence": 0.78,
  "agents_used": ["planner", "executor", "reflector"]
}
```

---

## Memory Budget Reference

| Scenario | RAM usage | Notes |
|----------|-----------|-------|
| Cold start (no model) | ~180 MB | FastAPI + TF-IDF index |
| Phi-3-mini loaded | ~2.5 GB | Default stack |
| Phi-3-mini + voice (ASR+TTS) | ~2.7 GB | Typical interactive session |
| Mistral-7B (AirLLM) | ~3.5–4.0 GB | Use alone, disable reflection |
| Over budget (any model) | LRU eviction | Dispatcher auto-manages |

---

## Limitations

1. **AirLLM 7B inference speed**: ~0.5–2 tokens/sec on a 2017 Intel Core i7 CPU. Latency for a 200-token response ≈ 100–400 seconds. For interactive use, Phi-3-mini (Q4_K_M via llama_cpp) is significantly faster (~10–15 tok/sec).

2. **Multi-agent overhead**: Running planner + executor + reflector adds ~2–3× latency vs single-agent. In time-sensitive voice interactions, set `enable_planning: false` and `enable_reflection: false` in `config/agents.yaml`.

3. **Safety patterns**: The built-in pattern blocklist covers obvious cases but is not a substitute for rigorous content moderation in production. The `llm_judge` option (LLM-based classification) is disabled by default to save RAM.

4. **No GPU**: The 2017 MacBook Pro's AMD Radeon 555/560 GPU does not have sufficient VRAM (2 GB) for model inference; all computation is CPU-only.

---

## Quick verification

```bash
# Start the orchestrator
bash scripts/start.sh

# Check all new components are healthy
curl http://localhost:8000/health
curl http://localhost:8000/safety
curl http://localhost:8000/agents

# Test safety blocking
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I make a bomb?"}' | python3 -m json.tool

# Test normal chat (multi-agent)
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain what a neural network is"}' | python3 -m json.tool
```

---

## One-command test

```bash
python -m pytest tests/ -q
# → 247 passed (including safety, multi-agent, and airllm engine tests)
```
