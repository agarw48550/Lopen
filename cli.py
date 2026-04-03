#!/usr/bin/env python3
"""Lopen Interactive CLI — OpenClaw-inspired terminal interface.

Usage
-----
    python cli.py [--debug] [--host HOST] [--port PORT]

Commands (inside the REPL)
--------------------------
    chat <message>   — Send a message to the Lopen agent
    status           — Show service health and resource usage
    plugins          — List loaded plugins / tools
    history          — Show recent conversation turns
    config           — Print active configuration summary
    debug on|off     — Toggle verbose debug output
    benchmark        — Run a quick inference speed test
    help             — Show this help
    quit / exit / q  — Exit the CLI

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
        ("chat <message>",   "Send a message to the agent and get a response"),
        ("status",           "Show service health, RAM usage, and running services"),
        ("plugins",          "List loaded plugins and tools"),
        ("history",          "Show the last 10 conversation turns"),
        ("config",           "Print active configuration summary"),
        ("debug on|off",     "Toggle verbose debug output (currently shows HTTP details)"),
        ("benchmark",        "Run a quick inference speed test"),
        ("help",             "Show this help message"),
        ("quit / exit / q",  "Exit the Lopen CLI"),
    ]
    for cmd, desc in cmds:
        print(f"  {cyan(cmd):<35} {dim(desc)}")
    print()
    print(dim("  Fun commands: lopen joke, lopen haiku, lopen sing, lopen about, lopen quote"))
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
        ("Name",        "Lopen — Local Open Intelligence"),
        ("Wake word",   "\"Lopen\""),
        ("Architecture","Intent-driven, plugin-extensible, multi-agent"),
        ("LLM",         "Phi-3-mini Q4_K_M (default) / Mistral-7B Q4 (AirLLM)"),
        ("ASR",         "whisper.cpp tiny (local, offline)"),
        ("TTS",         "Piper TTS — en_US-ryan-high (natural male voice)"),
        ("Safety",      "NemoClaw-inspired guardrails + PII redaction"),
        ("Memory",      "Target ≤4 GB RAM — designed for 2017 Intel MacBook Pro"),
        ("Source",      "github.com/agarw48550/Lopen"),
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
    "plugins":   _cmd_plugins,
    "history":   _cmd_history,
    "config":    _cmd_config,
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
