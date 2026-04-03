#!/usr/bin/env python3
"""Lopen Interactive CLI — OpenClaw-inspired terminal interface.

Usage
-----
    python cli.py [--debug] [--host HOST] [--port PORT]

Commands (inside the REPL)
--------------------------
    chat <message>       — Send a message to the agent and get a response
    status               — Show service health, RAM, and running services
    system               — Detailed system resource report (RAM/CPU/disk)
    plugins              — List loaded plugins and tools
    tools                — List all available tools with descriptions
    history              — Show recent conversation turns
    summary              — Summarise the current conversation
    clear                — Clear conversation history
    config               — Print active configuration summary
    model [name]         — Show active LLM or switch to a named model
    memory               — Show RAM usage and memory guard status
    fetch <url>          — Fetch a URL and summarise its content
    ingest <file>        — Ingest a local file into the agent's memory
    logs [N]             — Tail the last N lines from the agent log (default 20)
    restart              — Restart the orchestrator service
    debug on|off         — Toggle verbose debug output
    benchmark            — Run a quick inference speed test
    help                 — Show this help
    quit / exit / q      — Exit the CLI

Easter eggs: try  `lopen sing`, `lopen joke`, `lopen haiku`, `lopen about`
"""

from __future__ import annotations

import argparse
import json
import os
import random
import readline  # noqa: F401 – enables arrow-key history in input()
import sys
import time
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    """Wrap *text* in an ANSI colour/style code (auto-stripped when no TTY)."""
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def cyan(t: str) -> str:   return _c("96", t)
def green(t: str) -> str:  return _c("92", t)
def yellow(t: str) -> str: return _c("93", t)
def red(t: str) -> str:    return _c("91", t)
def blue(t: str) -> str:   return _c("94", t)
def bold(t: str) -> str:   return _c("1",  t)
def dim(t: str) -> str:    return _c("2",  t)
def magenta(t: str) -> str: return _c("95", t)


# ---------------------------------------------------------------------------
# ASCII banner
# ---------------------------------------------------------------------------

_BANNER = r"""
  _
 | |    ___  _ __   ___ _ __
 | |   / _ \| '_ \ / _ \ '_ \
 | |__| (_) | |_) |  __/ | | |
 |_____\___/| .__/ \___|_| |_|
            |_|
"""

_TAGLINES = [
    "Your local-first autonomous assistant. No cloud required. 🖥️",
    "Running locally on your Mac, 24/7. Built for intelligence, not bloat.",
    "Intent-driven · Plugin-extensible · Memory-safe · <4 GB RAM",
    "Think big. Run lean. Lopen is on it.",
    "From homework to hacking — your Mac's new best friend.",
    "All local. All private. All yours.",
    "Powered by Qwen3.5 · <1s responses · Sub-600MB model footprint",
]

_GREETINGS = [
    "Ready when you are. Type {help} to get started.",
    "Lopen is online and eager to assist. Try {chat hello} to say hi!",
    "Everything's local — your data stays with you. Let's get to work.",
    "Systems nominal. Type {help} for commands.",
    "Good to see you! Type {chat <message>} to talk to the agent.",
]

# ---------------------------------------------------------------------------
# Easter eggs / fun tidbits
# ---------------------------------------------------------------------------

_JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs. 🐛",
    "A SQL query walks into a bar, walks up to two tables and asks… 'Can I join you?'",
    "There are only 10 types of people in the world: those who understand binary and those who don't.",
    "I asked the AI to tell me a joke. It returned None. That's the joke.",
    "How many developers does it take to change a light bulb? None — that's a hardware problem.",
    "Why did the developer go broke? Because they used up all their cache.",
]

_HAIKU = [
    ("Silent inference,", "4 gigabytes, no more needed,", "Phi-3 whispers back."),
    ("You type a request,", "Intent engine finds the path,", "Tool answers swiftly."),
    ("Old MacBook Pro hums,", "Lopen listens through the mic,", "Wake word: your command."),
    ("Logs rotate at night,", "RAM watchdog guards the memory,", "System stays alive."),
    ("No cloud, no API,", "Everything runs in your house,", "Privacy restored."),
]

_QUOTES = [
    "\"The best interface is one that gets out of the way.\" — Anonymous",
    "\"Make it work, make it right, make it fast.\" — Kent Beck",
    "\"Simplicity is the ultimate sophistication.\" — Leonardo da Vinci",
    "\"Any sufficiently advanced technology is indistinguishable from magic.\" — Arthur C. Clarke",
    "\"The best way to predict the future is to invent it.\" — Alan Kay",
]

_SONGS = [
    "🎵 Running on your Mac, running strong and free,\n"
    "   Lopen's here to help you, no subscription fee!\n"
    "   Intent engine humming, plugins on the shelf,\n"
    "   Local-first forever — Lopen runs itself! 🎵",

    "🎶 Wake word to answer, whisper tiny hears,\n"
    "   Piper speaks in voices soothing tired ears,\n"
    "   Four gigs is enough if you know what to do,\n"
    "   Lopen, Lopen, always here for you! 🎶",
]


# ---------------------------------------------------------------------------
# CLI state
# ---------------------------------------------------------------------------

class CLIState:
    def __init__(self, host: str, port: int, debug: bool) -> None:
        self.host = host
        self.port = port
        self.debug = debug
        self.base_url = f"http://{host}:{port}"
        self.session_id = f"cli-{int(time.time())}"
        self.turn_count = 0

    @property
    def api(self) -> str:
        return self.base_url


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(state: CLIState, path: str, timeout: float = 5.0) -> Optional[dict]:
    if not _HAS_HTTPX:
        print(red("  httpx not installed — run: pip install httpx"))
        return None
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(f"{state.api}{path}")
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        print(red(f"  Cannot reach Lopen at {state.api} — is the orchestrator running?"))
        print(dim("  Start it with:  bash scripts/start.sh"))
        return None
    except httpx.HTTPStatusError as exc:
        print(red(f"  HTTP {exc.response.status_code} from {path}"))
        if state.debug:
            print(dim(f"  Response body: {exc.response.text[:200]}"))
        return None
    except Exception as exc:
        if state.debug:
            print(red(f"  HTTP error: {exc}"))
        return None


def _post(state: CLIState, path: str, payload: dict, timeout: float = 30.0) -> Optional[dict]:
    if not _HAS_HTTPX:
        print(red("  httpx not installed — run: pip install httpx"))
        return None
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{state.api}{path}", json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        print(red(f"  Cannot reach Lopen at {state.api} — is the orchestrator running?"))
        print(dim("  Start it with:  bash scripts/start.sh"))
        return None
    except httpx.HTTPStatusError as exc:
        print(red(f"  HTTP {exc.response.status_code} from {path}"))
        if state.debug:
            print(dim(f"  Response body: {exc.response.text[:200]}"))
        return None
    except Exception as exc:
        if state.debug:
            print(red(f"  HTTP error: {exc}"))
        return None


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _cmd_help(_state: CLIState, _args: str) -> None:
    print()
    print(bold(cyan("  Lopen CLI — Available Commands")))
    print(dim("  ─" * 38))
    cmds = [
        ("chat <message>",       "Send a message to the agent and get a response"),
        ("status",               "Show service health, RAM usage, and running services"),
        ("system",               "Detailed system report (RAM/CPU/disk/processes)"),
        ("plugins",              "List loaded plugins"),
        ("tools",                "List all available tools with descriptions"),
        ("history",              "Show the last 10 conversation turns"),
        ("summary",              "Summarise the current conversation"),
        ("clear",                "Clear conversation history"),
        ("config",               "Print active configuration summary"),
        ("model [name]",         "Show active LLM or switch to a named model"),
        ("memory",               "Show RAM usage and memory guard status"),
        ("fetch <url>",          "Fetch a URL and summarise its content"),
        ("ingest <file>",        "Ingest a local file into the agent's memory"),
        ("logs [N]",             "Tail the last N lines from the agent log (default 20)"),
        ("restart",              "Restart the orchestrator service"),
        ("debug on|off",         "Toggle verbose debug output"),
        ("benchmark",            "Run a quick inference speed test"),
        ("help",                 "Show this help message"),
        ("quit / exit / q",      "Exit the Lopen CLI"),
    ]
    for cmd, desc in cmds:
        print(f"  {cyan(cmd):<38} {dim(desc)}")
    print()
    print(dim("  Fun: lopen joke · lopen haiku · lopen sing · lopen about · lopen quote"))
    print(dim("       lopen fortune · lopen coffee · lopen matrix"))
    print()


def _cmd_status(state: CLIState, _args: str) -> None:
    print()
    print(bold("  Service Status"))
    print(dim("  ─" * 38))

    # Orchestrator health
    data = _get(state, "/health")
    if data:
        st = data.get("status", "?")
        color_fn = green if st in ("healthy", "ok", "running") else yellow
        print(f"  Orchestrator  {color_fn('●')} {color_fn(st.upper())}")
        if "uptime_seconds" in data:
            up = int(data["uptime_seconds"])
            h, m, s = up // 3600, (up % 3600) // 60, up % 60
            print(f"  Uptime        {dim(f'{h:02d}:{m:02d}:{s:02d}')}")
    else:
        print(f"  Orchestrator  {red('●')} {red('OFFLINE')}")

    # System health
    try:
        import psutil  # type: ignore
        mem = psutil.virtual_memory()
        ram_used = mem.used / 1024 ** 3
        ram_total = mem.total / 1024 ** 3
        pct = mem.percent
        bar_filled = int(pct / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        color_fn = green if pct < 75 else (yellow if pct < 90 else red)
        print(f"  RAM           {color_fn(bar)}  {ram_used:.1f}/{ram_total:.1f} GB ({pct:.0f}%)")
        cpu = psutil.cpu_percent(interval=0.5)
        cpu_bar = "█" * int(cpu / 5) + "░" * (20 - int(cpu / 5))
        cpu_fn = green if cpu < 60 else (yellow if cpu < 85 else red)
        print(f"  CPU           {cpu_fn(cpu_bar)}  {cpu:.0f}%")
    except ImportError:
        print(f"  RAM/CPU       {dim('(install psutil for live stats)')}")

    # Dashboard
    try:
        if _HAS_HTTPX:
            with httpx.Client(timeout=2.0) as c:
                r = c.get(f"http://{state.host}:8080/health")
            color_fn = green if r.status_code == 200 else red
            print(f"  Dashboard     {color_fn('●')} {color_fn('ONLINE' if r.status_code == 200 else 'OFFLINE')}  → http://{state.host}:8080")
        else:
            print(f"  Dashboard     {dim('(install httpx to check)')}")
    except Exception:
        print(f"  Dashboard     {red('●')} {red('OFFLINE')}")

    print()


def _cmd_plugins(state: CLIState, _args: str) -> None:
    data = _get(state, "/plugins")
    if not data:
        return
    plugins = data.get("plugins", [])
    print()
    print(bold(f"  Loaded plugins: {len(plugins)}"))
    print(dim("  ─" * 38))
    for p in plugins:
        name = p if isinstance(p, str) else p.get("name", str(p))
        print(f"  {cyan('▸')} {name}")
    print()


def _cmd_history(state: CLIState, _args: str) -> None:
    data = _get(state, "/memory")
    if not data:
        return
    turns = data.get("turns", [])
    print()
    print(bold(f"  Conversation History ({len(turns)} turns)"))
    print(dim("  ─" * 38))
    for t in turns[-10:]:
        role = t.get("role", "?")
        content = t.get("content", "")
        prefix = cyan("You:  ") if role == "user" else green("Lopen:")
        wrapped = textwrap.fill(content, width=70, subsequent_indent="        ")
        print(f"  {prefix} {wrapped}")
    if not turns:
        print(dim("  No conversation history yet. Start with: chat hello"))
    print()


def _cmd_config(state: CLIState, _args: str) -> None:
    data = _get(state, "/status")
    if not data:
        return
    print()
    print(bold("  Active Configuration"))
    print(dim("  ─" * 38))
    for key, val in data.items():
        print(f"  {cyan(key):<28} {dim(str(val))}")
    print()


def _cmd_debug(state: CLIState, args: str) -> None:
    arg = args.strip().lower()
    if arg == "on":
        state.debug = True
        print(green("  Debug mode ON — verbose HTTP and routing details will be shown."))
    elif arg == "off":
        state.debug = False
        print(yellow("  Debug mode OFF."))
    else:
        status = green("ON") if state.debug else yellow("OFF")
        print(f"  Debug mode is currently {status}. Use: debug on  or  debug off")


def _cmd_benchmark(state: CLIState, _args: str) -> None:
    prompts = [
        "What is 2 + 2?",
        "Summarise the water cycle in one sentence.",
        "Write a Python hello-world function.",
    ]
    print()
    print(bold("  Inference Benchmark"))
    print(dim("  ─" * 38))
    total = 0.0
    for i, prompt in enumerate(prompts, 1):
        start = time.perf_counter()
        result = _post(state, "/chat", {"message": prompt, "session_id": state.session_id})
        elapsed = time.perf_counter() - start
        total += elapsed
        status = green(f"{elapsed:.2f}s") if elapsed < 3 else (yellow(f"{elapsed:.2f}s") if elapsed < 8 else red(f"{elapsed:.2f}s"))
        resp = (result or {}).get("response", "(no response)")[:50] if result else "(offline)"
        print(f"  [{i}] {dim(prompt[:40]):<42} {status}")
        if state.debug:
            print(f"      → {dim(resp)}")
    avg = total / len(prompts)
    color_fn = green if avg < 3 else (yellow if avg < 8 else red)
    print(dim("  ─" * 38))
    print(f"  Average:  {color_fn(f'{avg:.2f}s per response')}")
    rating = "🚀 Fast" if avg < 2 else ("✅ Good" if avg < 5 else ("⚠️  Slow" if avg < 10 else "🐢 Very slow"))
    print(f"  Rating:   {rating}")
    print()


def _cmd_chat(state: CLIState, args: str) -> None:
    message = args.strip()
    if not message:
        print(yellow("  Usage: chat <your message>"))
        return

    state.turn_count += 1
    print()
    start = time.perf_counter()
    print(dim(f"  [{datetime.now().strftime('%H:%M:%S')}] Thinking…"), end="", flush=True)
    result = _post(state, "/chat", {"message": message, "session_id": state.session_id})
    elapsed = time.perf_counter() - start
    print(f"\r", end="")  # clear the "Thinking…" line

    if result:
        response = result.get("response", "(no response)")
        tool = result.get("tool_used", "")
        conf = result.get("confidence", None)

        # Pretty-print the response
        wrapped = textwrap.fill(response, width=72, subsequent_indent="       ")
        print(f"  {cyan('You:')}   {message}")
        print(f"  {green('Lopen:')} {wrapped}")
        if state.debug:
            meta = []
            if tool:
                meta.append(f"tool={tool}")
            if conf is not None:
                meta.append(f"conf={conf:.2f}")
            meta.append(f"time={elapsed:.2f}s")
            print(dim(f"         [{' | '.join(meta)}]"))
        print()
    else:
        print()


# ---------------------------------------------------------------------------
# New CLI commands (April 2026 expansion)
# ---------------------------------------------------------------------------

def _cmd_system(state: CLIState, _args: str) -> None:
    """Detailed system resource report."""
    print()
    print(bold("  System Resource Report"))
    print(dim("  ─" * 38))
    try:
        import psutil  # type: ignore
        # RAM
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        ram_used = mem.used / 1024 ** 3
        ram_total = mem.total / 1024 ** 3
        pct = mem.percent
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        color_fn = green if pct < 70 else (yellow if pct < 88 else red)
        print(f"  RAM           {color_fn(bar)}  {ram_used:.2f}/{ram_total:.1f} GB ({pct:.0f}%)")
        guard_status = green("✓ SAFE") if ram_used < 3.5 else red("⚠ NEAR LIMIT")
        print(f"  RAM guard     {guard_status}  {dim('(limit: 3.5 GB active usage)')}")
        print(f"  Swap used     {swap.used / 1024 ** 3:.2f} GB / {swap.total / 1024 ** 3:.1f} GB")
        # CPU
        cpu = psutil.cpu_percent(interval=1.0)
        cpu_bar = "█" * int(cpu / 5) + "░" * (20 - int(cpu / 5))
        cpu_fn = green if cpu < 60 else (yellow if cpu < 85 else red)
        print(f"  CPU           {cpu_fn(cpu_bar)}  {cpu:.0f}%")
        # Disk
        disk = psutil.disk_usage("/")
        disk_pct = disk.percent
        disk_bar = "█" * int(disk_pct / 5) + "░" * (20 - int(disk_pct / 5))
        disk_fn = green if disk_pct < 80 else (yellow if disk_pct < 92 else red)
        print(f"  Disk (/)      {disk_fn(disk_bar)}  {disk.used / 1024**3:.1f}/{disk.total / 1024**3:.1f} GB ({disk_pct:.0f}%)")
        # Top processes by memory
        try:
            procs = [(p.info["name"], p.info["memory_info"].rss / 1024 ** 2)
                     for p in psutil.process_iter(["name", "memory_info"])
                     if p.info.get("memory_info")]
            procs.sort(key=lambda x: x[1], reverse=True)
            print()
            print(f"  {bold('Top memory consumers:')}")
            for name, mb in procs[:5]:
                bar_w = int(min(mb / 500 * 15, 15))
                print(f"    {name[:22]:<22}  {dim('█' * bar_w + '░' * (15 - bar_w))}  {mb:.0f} MB")
        except Exception:
            pass
    except ImportError:
        print(red("  psutil not installed — run: pip install psutil"))
    print()


def _cmd_tools(state: CLIState, _args: str) -> None:
    """List all available tools with descriptions."""
    data = _get(state, "/tools")
    print()
    print(bold("  Available Tools"))
    print(dim("  ─" * 38))
    if data:
        tools = data.get("tools", [])
        for t in tools:
            if isinstance(t, dict):
                name = t.get("name", "?")
                desc = t.get("description", "")
                enabled = t.get("enabled", True)
                icon = green("●") if enabled else dim("○")
                print(f"  {icon} {cyan(name):<28} {dim(desc)}")
            else:
                print(f"  {cyan('▸')} {t}")
    else:
        # Fallback: show known tools from tools/ directory
        known = [
            ("researcher",        "Web search and research tasks"),
            ("coder_assist",      "Code generation, review, and debugging"),
            ("homework_tutor",    "Explain concepts, solve problems, teach"),
            ("desktop_organizer", "Organise files and manage the desktop"),
            ("browser_automation","Control the browser for web tasks"),
            ("file_ops",          "Read, write, move, and manage files"),
        ]
        for name, desc in known:
            print(f"  {green('●')} {cyan(name):<28} {dim(desc)}")
    print()


def _cmd_model(state: CLIState, args: str) -> None:
    """Show or switch the active LLM."""
    target = args.strip()
    print()
    if target:
        # Request model switch
        result = _post(state, "/model/switch", {"model": target})
        if result:
            new_model = result.get("active_model", target)
            print(green(f"  ✔  Switched to model: {bold(new_model)}"))
            if "note" in result:
                print(dim(f"     {result['note']}"))
        else:
            print(yellow(f"  Could not switch model — orchestrator may not be running."))
            print(dim("  Edit config/models.yaml → llm.active to change the default."))
    else:
        # Show current model info
        data = _get(state, "/status")
        if data:
            model_name = data.get("llm_model", data.get("model", "unknown"))
            engine = data.get("llm_engine", "unknown")
            print(f"  Active model  {cyan(model_name)}")
            print(f"  Engine        {dim(engine)}")
        else:
            print(dim("  Orchestrator offline — reading config/models.yaml directly..."))
            _show_model_config()
    print()


def _show_model_config() -> None:
    """Fallback: parse models.yaml directly when orchestrator is offline."""
    try:
        import yaml  # type: ignore
        cfg_path = Path(__file__).parent / "config" / "models.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text())
            active = cfg.get("models", {}).get("llm", {}).get("active", "?")
            model = cfg.get("models", {}).get("llm", {}).get(active, {})
            print(f"  Active model  {cyan(active)}")
            print(f"  Name          {dim(model.get('name', '?'))}")
            print(f"  Size          {dim(str(model.get('size_gb', '?')) + ' GB')}")
            print(f"  Params        {dim(str(model.get('params_b', '?')) + 'B')}")
            print(f"  Format        {dim(model.get('chat_format', 'chatml'))}")
    except Exception:
        pass


def _cmd_memory(state: CLIState, _args: str) -> None:
    """Show RAM usage and memory guard status."""
    print()
    print(bold("  Memory Status"))
    print(dim("  ─" * 38))
    try:
        import psutil  # type: ignore
        mem = psutil.virtual_memory()
        used_gb = mem.used / 1024 ** 3
        total_gb = mem.total / 1024 ** 3
        avail_gb = mem.available / 1024 ** 3
        pct = mem.percent
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        color_fn = green if pct < 70 else (yellow if pct < 88 else red)
        print(f"  Total RAM     {total_gb:.1f} GB")
        print(f"  Used          {color_fn(bar)}  {used_gb:.2f} GB ({pct:.0f}%)")
        print(f"  Available     {avail_gb:.2f} GB")
        print()
        # Memory guard thresholds
        WARN_GB = 3.2
        CRIT_GB = 3.6
        if used_gb < WARN_GB:
            print(f"  Guard status  {green('✓ SAFE')}  {dim(f'({used_gb:.2f} GB / {WARN_GB} GB warn threshold)')}")
        elif used_gb < CRIT_GB:
            print(f"  Guard status  {yellow('⚠ WARNING')}  {dim(f'({used_gb:.2f} GB — approaching {CRIT_GB} GB limit)')}")
            print(yellow("  Recommendation: close browser tabs or restart a service"))
        else:
            print(f"  Guard status  {red('✖ CRITICAL')}  {dim(f'({used_gb:.2f} GB ≥ {CRIT_GB} GB limit)')}")
            print(red("  Action needed: RAM over limit — run: bash scripts/health_check.sh"))
        # Model sizes
        print()
        print(f"  {bold('Model RAM estimates:')}")
        models_info = [
            ("Qwen3.5-0.8B Q4_K_M  (default)", "~0.55 GB"),
            ("Qwen3.5-1.5B Q4_K_M  (quality)",  "~1.0 GB"),
            ("Phi-3-mini Q4_K_M    (legacy)",   "~2.2 GB"),
            ("whisper-tiny (ASR)",               "~0.08 GB"),
            ("Piper TTS",                        "~0.07 GB"),
            ("Python / FastAPI overhead",        "~0.35 GB"),
        ]
        for name, size in models_info:
            print(f"    {dim(name):<38} {cyan(size)}")
    except ImportError:
        print(red("  psutil not installed — run: pip install psutil"))
    print()


def _cmd_fetch(state: CLIState, args: str) -> None:
    """Fetch a URL and summarise its content via the agent."""
    url = args.strip()
    if not url:
        print(yellow("  Usage: fetch <url>"))
        print(dim("  Example: fetch https://en.wikipedia.org/wiki/Quantum_computing"))
        return
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    print()
    print(dim(f"  Fetching {url} …"))
    start = time.perf_counter()
    result = _post(
        state,
        "/chat",
        {
            "message": f"[TOOL:researcher] Fetch and summarise this URL: {url}",
            "session_id": state.session_id,
        },
        timeout=60.0,
    )
    elapsed = time.perf_counter() - start
    if result:
        response = result.get("response", "(no response)")
        wrapped = textwrap.fill(response, width=72, subsequent_indent="       ")
        print(f"  {green('Lopen:')} {wrapped}")
        if state.debug:
            print(dim(f"         [time={elapsed:.2f}s]"))
    print()


def _cmd_ingest(state: CLIState, args: str) -> None:
    """Ingest a local file into the agent's memory."""
    filepath = args.strip()
    if not filepath:
        print(yellow("  Usage: ingest <file_path>"))
        print(dim("  Example: ingest ~/Documents/notes.txt"))
        return
    expanded = os.path.expanduser(filepath)
    if not os.path.isfile(expanded):
        print(red(f"  File not found: {expanded}"))
        return
    size = os.path.getsize(expanded)
    if size > 1024 * 1024:  # 1 MB guard
        print(yellow(f"  File is large ({size // 1024} KB) — truncating to first 8000 chars"))
    try:
        with open(expanded, encoding="utf-8", errors="replace") as fh:
            content = fh.read(8000)
    except Exception as exc:
        print(red(f"  Could not read file: {exc}"))
        return
    print()
    print(dim(f"  Ingesting {os.path.basename(expanded)} ({size} bytes)…"))
    result = _post(
        state,
        "/chat",
        {
            "message": (
                f"[INGEST] I am sending you a file to remember.\n"
                f"Filename: {os.path.basename(expanded)}\n"
                f"Content:\n{content}\n\n"
                "Please acknowledge you have ingested this and summarise it in 2–3 sentences."
            ),
            "session_id": state.session_id,
        },
        timeout=60.0,
    )
    if result:
        response = result.get("response", "(no response)")
        wrapped = textwrap.fill(response, width=72, subsequent_indent="       ")
        print(f"  {green('Lopen:')} {wrapped}")
    print()


def _cmd_summary(state: CLIState, _args: str) -> None:
    """Summarise the current conversation."""
    print()
    print(dim("  Generating conversation summary…"))
    result = _post(
        state,
        "/chat",
        {
            "message": "Please summarise our conversation so far in 3–5 bullet points.",
            "session_id": state.session_id,
        },
        timeout=30.0,
    )
    if result:
        response = result.get("response", "(no response)")
        wrapped = textwrap.fill(response, width=72, subsequent_indent="     ")
        print(f"  {green('Summary:')} {wrapped}")
    else:
        # Fallback: show history from /memory endpoint
        data = _get(state, "/memory")
        if data:
            turns = data.get("turns", [])
            print(bold(f"  Conversation: {len(turns)} turns"))
            if turns:
                print(dim("  (last 5 messages:)"))
                for t in turns[-5:]:
                    role = t.get("role", "?")
                    content = t.get("content", "")[:80]
                    prefix = cyan("You:  ") if role == "user" else green("Lopen:")
                    print(f"  {prefix} {dim(content)}…")
    print()


def _cmd_clear(state: CLIState, _args: str) -> None:
    """Clear conversation history."""
    result = _post(state, "/memory/clear", {"session_id": state.session_id})
    if result:
        print(green("  ✔  Conversation history cleared."))
    else:
        print(yellow("  Could not clear history via API."))
        print(dim("  (Orchestrator may be offline — history will reset when restarted)"))


def _cmd_logs(state: CLIState, args: str) -> None:  # noqa: ARG001
    """Tail the last N lines from the agent log."""
    try:
        n = int(args.strip()) if args.strip().isdigit() else 20
    except (ValueError, AttributeError):
        n = 20
    log_candidates = [
        Path(__file__).parent / "logs" / "lopen.log",
        Path(__file__).parent / "logs" / "orchestrator.log",
        Path("/tmp") / "lopen.log",
    ]
    log_path = None
    for p in log_candidates:
        if p.exists():
            log_path = p
            break
    print()
    if log_path is None:
        print(yellow(f"  No log file found. Checked: {[str(p) for p in log_candidates]}"))
        print(dim("  Start the orchestrator first: bash scripts/start.sh"))
    else:
        print(bold(f"  Last {n} lines of {log_path.name}"))
        print(dim("  ─" * 38))
        try:
            lines = log_path.read_text(errors="replace").splitlines()
            for line in lines[-n:]:
                # Colour-code log levels
                if "ERROR" in line or "CRITICAL" in line:
                    print(f"  {red(line)}")
                elif "WARNING" in line or "WARN" in line:
                    print(f"  {yellow(line)}")
                elif "DEBUG" in line:
                    print(f"  {dim(line)}")
                else:
                    print(f"  {line}")
        except Exception as exc:
            print(red(f"  Could not read log: {exc}"))
    print()


def _cmd_restart(state: CLIState, _args: str) -> None:
    """Restart the orchestrator service."""
    print()
    print(yellow("  Restarting orchestrator…"))
    result = _post(state, "/admin/restart", {})
    if result:
        status = result.get("status", "?")
        print(green(f"  ✔  Restart acknowledged: {status}"))
        print(dim("  Waiting for services to come back online…"))
        time.sleep(2)
        data = _get(state, "/health")
        if data:
            print(green("  ✔  Orchestrator is back online."))
        else:
            print(yellow("  ⚠  Orchestrator not yet reachable — try: bash scripts/start.sh"))
    else:
        print(yellow("  Could not reach orchestrator to request restart."))
        print(dim("  Run manually: bash scripts/stop.sh && bash scripts/start.sh"))
    print()


# ---------------------------------------------------------------------------
# Easter egg commands
# ---------------------------------------------------------------------------

def _easter_egg(state: CLIState, args: str, raw: str) -> bool:  # noqa: ARG001
    """Return True if the input matched an easter egg."""
    lower = raw.lower().strip().lstrip("lopen").strip()

    if lower in ("joke", "tell me a joke", "joke please"):
        print()
        print(f"  {cyan('🎭')} {random.choice(_JOKES)}")
        print()
        return True

    if lower in ("haiku", "write a haiku", "poem"):
        lines = random.choice(_HAIKU)
        print()
        print(f"  {magenta('🌸 Haiku:')}")
        for line in lines:
            print(f"     {italic_dim(line)}")
        print()
        return True

    if lower in ("sing", "song", "sing me a song"):
        print()
        print(random.choice(_SONGS))
        print()
        return True

    if lower in ("quote", "inspire me", "motivation"):
        print()
        print(f"  {yellow('💡')} {random.choice(_QUOTES)}")
        print()
        return True

    if lower in ("about", "what are you", "who are you", "version", "info"):
        _print_about()
        return True

    if lower in ("matrix", "neo", "red pill"):
        _matrix_moment()
        return True

    if lower in ("coffee", "☕", "need coffee"):
        print()
        print(f"  {yellow('☕')} {cyan('Brewing a virtual espresso...')}")
        time.sleep(0.5)
        print(f"  {dim('☕☕☕  Done! Energised and ready to compute.')}")
        print()
        return True

    if lower in ("fortune", "wisdom"):
        fortunes = [
            "The best optimisation is the one you don't have to make.",
            "A model in RAM is worth two on disk.",
            "Trust the intent engine, but verify the tool.",
            "Wake words are the handshake of the future.",
            "4 GB is not a limitation — it is a challenge.",
            "Local is the new cloud.",
        ]
        print()
        print(f"  {cyan('🔮')} {random.choice(fortunes)}")
        print()
        return True

    return False


def italic_dim(t: str) -> str:
    """Return italic + dim text."""
    if not _USE_COLOR:
        return t
    return f"\033[2;3m{t}\033[0m"


def _print_about() -> None:
    print()
    print(bold(cyan("  About Lopen")))
    print(dim("  ─" * 38))
    lines = [
        ("Name",         "Lopen — Local Open Intelligence"),
        ("Version",      "April 2026"),
        ("Wake word",    "\"Lopen\""),
        ("Architecture", "Intent-driven, plugin-extensible, multi-agent"),
        ("LLM",          "Qwen3.5-0.8B-Instruct Q4_K_M (default, <1s responses)"),
        ("LLM alt",      "Qwen3.5-1.5B Q4_K_M (quality) · Phi-3-mini (legacy)"),
        ("ASR",          "whisper.cpp tiny (local, offline)"),
        ("TTS",          "Piper TTS — en_US-ryan-high (natural male voice)"),
        ("Safety",       "NemoClaw-inspired guardrails + PII redaction"),
        ("Memory",       "Target ≤4 GB RAM — designed for 2017 Intel MacBook Pro"),
        ("Install",      "No Homebrew required — pip + direct binaries only"),
        ("Source",       "github.com/agarw48550/Lopen"),
    ]
    for k, v in lines:
        print(f"  {cyan(k):<16} {v}")
    print()


def _matrix_moment() -> None:
    print()
    print(green("  Wake up, Neo..."))
    time.sleep(0.6)
    print(green("  The Matrix has you..."))
    time.sleep(0.6)
    print(green("  Follow the white rabbit."))
    time.sleep(0.6)
    print(dim("  (Lopen is all local — no red pills needed here.)"))
    print()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, object] = {
    "help":      _cmd_help,
    "status":    _cmd_status,
    "system":    _cmd_system,
    "plugins":   _cmd_plugins,
    "tools":     _cmd_tools,
    "history":   _cmd_history,
    "summary":   _cmd_summary,
    "clear":     _cmd_clear,
    "config":    _cmd_config,
    "model":     _cmd_model,
    "memory":    _cmd_memory,
    "fetch":     _cmd_fetch,
    "ingest":    _cmd_ingest,
    "logs":      _cmd_logs,
    "restart":   _cmd_restart,
    "debug":     _cmd_debug,
    "benchmark": _cmd_benchmark,
    "chat":      _cmd_chat,
}

_QUIT_CMDS = {"quit", "exit", "q", "bye", ":q", "exit()", "quit()"}


def _print_banner(state: CLIState) -> None:
    print(cyan(_BANNER))
    print(bold(cyan(f"  {random.choice(_TAGLINES)}")))
    print()
    greeting = random.choice(_GREETINGS)
    greeting = greeting.replace("{help}", bold("help")).replace(
        "{chat hello}", bold("chat hello")
    ).replace("{chat <message>}", bold("chat <message>"))
    print(f"  {greeting}")
    print()
    print(dim(f"  Host: {state.base_url}  |  Session: {state.session_id}"))
    print(dim("  Debug: " + ("ON" if state.debug else "OFF") + "   |   Type 'help' for commands"))
    print()


def repl(state: CLIState) -> None:
    _print_banner(state)

    while True:
        try:
            try:
                raw = input(f"{cyan('lopen')} {dim('›')} ").strip()
            except EOFError:
                break

            if not raw:
                continue

            if raw.lower() in _QUIT_CMDS:
                print()
                print(cyan("  Goodbye! Stay curious. 👋"))
                print()
                break

            # Easter eggs (check before normal dispatch)
            if _easter_egg(state, "", raw):
                continue

            # Split into command + args
            parts = raw.split(None, 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if cmd in _DISPATCH:
                _DISPATCH[cmd](state, args)  # type: ignore[operator]
            elif cmd == "lopen":
                # Handle 'lopen <subcommand>' as shorthand
                sub_parts = args.split(None, 1)
                sub_cmd = sub_parts[0].lower() if sub_parts else ""
                sub_args = sub_parts[1] if len(sub_parts) > 1 else ""
                if sub_cmd in _DISPATCH:
                    _DISPATCH[sub_cmd](state, sub_args)  # type: ignore[operator]
                else:
                    print(yellow(f"  Unknown sub-command: lopen {sub_cmd}. Type 'help' for options."))
            else:
                # Treat unrecognised input as a chat message (convenience)
                _cmd_chat(state, raw)

        except KeyboardInterrupt:
            print()
            print(dim("  (Ctrl-C received — type 'quit' to exit, or press Ctrl-C again)"))
            try:
                input()
            except (KeyboardInterrupt, EOFError):
                print()
                print(cyan("  Bye! 👋"))
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lopen interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python cli.py
              python cli.py --debug
              python cli.py --host localhost --port 8000
        """),
    )
    parser.add_argument("--host", default="localhost", help="Orchestrator host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Orchestrator port (default: 8000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    state = CLIState(host=args.host, port=args.port, debug=args.debug)
    repl(state)


if __name__ == "__main__":
    main()
