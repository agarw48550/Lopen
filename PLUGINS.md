# Lopen Plugin SDK

This document explains how to extend Lopen with your own custom tools/plugins.
No core code changes are required — just drop a Python file in the right directory.

---

## Architecture

Lopen's intent-driven routing pipeline works like this:

```
User Query
    │
    ▼
IntentEngine (TF-IDF cosine similarity)
    │  scores every registered tool against the query
    ▼
ToolSelector (ranks by score, returns top-k)
    │
    ▼
ConfirmationGate (checks permissions & confidence)
    │  ← returns prompt if confirmation needed
    ▼
ArgumentComposer (extracts file paths, URLs, code snippets, etc.)
    │
    ▼
Tool.run(query, **args) → response string
    │
    ▼
Analytics.log_tool_use(...)   ← local SQLite, no network
```

The engine automatically discovers and indexes any `BaseTool` subclass placed
in `tools/` or `tools/third_party/`.  Adding a new tool is as simple as:

1. Create a file in `tools/third_party/my_plugin.py`.
2. Restart (or call `POST /plugins/reload`).

---

## Creating a Plugin

### Minimal Plugin

```python
# tools/third_party/weather_checker.py
from tools.base_tool import BaseTool

class WeatherChecker(BaseTool):
    name = "weather_checker"
    description = (
        "Check the current weather forecast for a city or location. "
        "Useful for weather questions, travel planning, and outdoor activity advice."
    )
    tags = ["weather", "forecast", "travel", "outdoor"]
    version = "1.0.0"
    requires_permission = False

    def run(self, query: str, **kwargs) -> str:
        # Your implementation here
        location = kwargs.get("location", "unknown")
        return f"[WeatherChecker] Fetching weather for {location!r}…"
```

### Key Class Attributes

| Attribute            | Type         | Required | Description                                          |
|----------------------|--------------|----------|------------------------------------------------------|
| `name`               | `str`        | ✅        | Unique identifier used for routing (snake_case)      |
| `description`        | `str`        | ✅        | Natural language description — **this is what the intent engine uses for matching**. Be descriptive! |
| `tags`               | `list[str]`  | ✓        | Keywords that boost matching precision               |
| `version`            | `str`        | —        | Semantic version (default `"1.0.0"`)                 |
| `requires_permission`| `bool`       | —        | If `True`, user confirmation is required (default `False`) |

### The `run` Method

```python
def run(self, query: str, **kwargs) -> str:
    """
    Execute the tool and return a response string.

    Args:
        query:   The full user query (always provided).
        **kwargs: Extracted arguments from ArgumentComposer:
                  - file_path, url, language, contact, target_dir, search_term,
                    code_snippet, and any other regex-captured values.

    Returns:
        A human-readable response string.
    """
```

### Using the LLM Adapter

If your tool needs LLM inference, use `self._llm`:

```python
def run(self, query: str, **kwargs) -> str:
    if self._llm is None:
        return "LLM not available."
    return self._llm.generate(f"Answer this: {query}", max_tokens=256)
```

The LLM is memory-conservative by default (loaded on demand, unloaded when idle).

---

## Plugin with Permissions

For tools that modify the filesystem, send messages, or perform other
potentially risky actions, set `requires_permission = True`:

```python
class DesktopCleaner(BaseTool):
    name = "desktop_cleaner"
    description = "Removes temporary files and empties the Trash"
    requires_permission = True   # ← user will be prompted to confirm

    def run(self, query: str, **kwargs) -> str:
        ...
```

The `ConfirmationGate` will prompt the user before the first few invocations.
After `min_uses_for_auto_approve` (default: 3) successful uses, the tool is
considered "known" and runs without further prompts (configurable in
`config/settings.yaml` under `sandbox:`).

---

## Tool Placement

| Directory             | Purpose                                              |
|-----------------------|------------------------------------------------------|
| `tools/`              | Built-in tools (shipped with Lopen)                  |
| `tools/third_party/`  | Your custom plugins — drop files here                |

Both directories are scanned on startup and after `POST /plugins/reload`.

---

## Plugin Discovery API

```http
GET  /plugins            → list all registered plugins + metadata
POST /plugins/reload     → rescan directories and register new plugins
```

---

## Improving Match Quality

The `IntentEngine` scores tools using **TF-IDF cosine similarity** between the
user's query and each tool's `description` + `tags`.

**Tips for better matching:**

- Write descriptions in natural language, as a user might phrase their request.
- Include synonyms and related terms in `tags`.
- Avoid generic descriptions like `"A useful tool"` — be specific.

**Example of a well-described tool:**

```python
description = (
    "Convert units of measurement: length (metres, feet, inches, miles, km), "
    "weight (kg, lbs, grams, ounces), temperature (Celsius, Fahrenheit, Kelvin), "
    "volume, speed, and currency exchange rates."
)
tags = ["convert", "units", "measurement", "calculator", "currency"]
```

---

## Full Example Plugin

```python
# tools/third_party/unit_converter.py
"""Unit conversion plugin for Lopen."""

from __future__ import annotations
from tools.base_tool import BaseTool


class UnitConverter(BaseTool):
    name = "unit_converter"
    description = (
        "Convert between units of measurement: length (metres, feet, miles, km), "
        "weight (kg, lbs, grams), temperature (Celsius, Fahrenheit, Kelvin), "
        "and other common conversions."
    )
    tags = ["convert", "units", "measurement", "calculator", "length", "weight", "temperature"]
    version = "1.0.0"
    requires_permission = False

    _CONVERSIONS = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
    }

    def run(self, query: str, **kwargs) -> str:
        import re
        m = re.search(
            r"([\d.]+)\s*([a-zA-Z]+)\s+(?:to|in|into)\s+([a-zA-Z]+)",
            query, re.I
        )
        if not m:
            return "Please specify: '<value> <unit> to <unit>', e.g. '5 km to miles'."
        value = float(m.group(1))
        from_unit = m.group(2).lower()
        to_unit = m.group(3).lower()
        factor = self._CONVERSIONS.get((from_unit, to_unit))
        if factor is None:
            return f"Conversion from {from_unit!r} to {to_unit!r} is not supported yet."
        result = value * factor
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
```

---

## Configuration Reference

Relevant sections in `config/settings.yaml`:

```yaml
plugin_loader:
  enabled: true
  auto_discover: true        # scan on startup
  tool_dirs:
    - tools
    - tools/third_party      # add more directories here

sandbox:
  confidence_threshold: 0.3      # require confirmation below this score
  auto_approve_known_tools: true
  min_uses_for_auto_approve: 3

analytics:
  enabled: true
  log_to_db: true            # all events stay local in SQLite
```

---

## Testing Your Plugin

```bash
# Start the orchestrator
bash scripts/start_orchestrator.sh

# Test your plugin via chat
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "convert 10 km to miles"}' | python3 -m json.tool

# Check it was discovered
curl -s http://localhost:8000/plugins | python3 -m json.tool

# Reload without restarting
curl -s -X POST http://localhost:8000/plugins/reload | python3 -m json.tool
```

---

## Summary

| Step | Action |
|------|--------|
| 1 | Create `tools/third_party/my_plugin.py` |
| 2 | Inherit from `BaseTool`, set `name` and `description` |
| 3 | Implement `run(self, query, **kwargs) -> str` |
| 4 | Restart or call `POST /plugins/reload` |
| 5 | Query via `/chat` — the engine will route to your tool automatically |
