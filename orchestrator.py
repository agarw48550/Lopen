"""Lopen Orchestrator — main entry point wiring all components together.

New in this version:
  - Dynamic intent recognition via :class:`~agent_core.intent_engine.IntentEngine`
    (TF-IDF cosine similarity, no extra RAM or model downloads).
  - Automatic plugin discovery via :class:`~agent_core.plugin_loader.PluginLoader`
    (drop a file in ``tools/`` or ``tools/third_party/`` and it is picked up).
  - Semantic tool selection via :class:`~agent_core.tool_selector.ToolSelector`
    routing open-ended queries to the best available tool.
  - Argument extraction via :class:`~agent_core.argument_composer.ArgumentComposer`.
  - Confirmation gate via :class:`~agent_core.sandbox.ConfirmationGate` for
    risky / low-confidence tool invocations.
  - Usage analytics via :class:`~agent_core.analytics.Analytics` (local SQLite only).
  - New REST endpoints: ``GET /plugins``, ``POST /plugins/reload``,
    ``GET /analytics``, ``POST /feedback``.
  - AirLLM-backed LLM engine for efficient large-model inference.
  - OMLX-inspired multi-agent dispatcher (planner/executor/reflector).
  - NemoClaw-inspired safety engine (guardrails, tool filter, PII redaction).
"""

from __future__ import annotations

import logging
import logging.config
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter

# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str, default: dict | None = None) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return default or {}


def _configure_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_config_path = "config/logging.yaml"
    if Path(log_config_path).is_file():
        with open(log_config_path) as f:
            cfg = yaml.safe_load(f)
        # Ensure log dir exists before configuring handlers
        for handler in cfg.get("handlers", {}).values():
            fname = handler.get("filename", "")
            if fname:
                Path(fname).parent.mkdir(parents=True, exist_ok=True)
        try:
            logging.config.dictConfig(cfg)
            return
        except Exception:
            pass
    # Fallback: basic config
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}
_scheduler = None  # APScheduler instance
_start_time: float = 0.0

logger = logging.getLogger("lopen.orchestrator")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _startup()
    yield
    _shutdown()


def _startup() -> None:
    global _start_time
    _start_time = time.time()

    settings = _load_yaml("config/settings.yaml")
    lopen_cfg = settings.get("lopen", {})
    log_level = os.environ.get("LOPEN_LOG_LEVEL", lopen_cfg.get("log_level", "INFO"))
    # Debug mode: LOPEN_DEBUG=1 (or "true"/"yes") overrides to DEBUG level
    _debug_env = os.environ.get("LOPEN_DEBUG", "").strip().lower()
    if _debug_env in ("1", "true", "yes"):
        log_level = "DEBUG"
    log_dir = lopen_cfg.get("log_dir", "logs")
    _configure_logging(log_dir=log_dir, level=log_level)

    logger.info("Lopen orchestrator starting up…")

    # Storage
    from storage.database import SQLiteDB
    db = SQLiteDB()
    _state["db"] = db

    # LLM — prefer AirLLMEngine, fall back to LLMAdapter
    llm_cfg = settings.get("llm", {})
    engine_name = os.environ.get("LOPEN_LLM_ENGINE", llm_cfg.get("engine", "auto"))
    model_path = os.environ.get("LOPEN_LLM_MODEL_PATH", llm_cfg.get("model_path"))
    llm = _build_llm(engine_name, model_path, llm_cfg)
    _state["llm"] = llm

    # Safety Engine (NemoClaw-inspired)
    from agent_core.safety import SafetyEngine
    safety_engine = SafetyEngine.from_config("config/settings.yaml")
    _state["safety_engine"] = safety_engine
    logger.info("Safety engine ready (enabled=%s)", safety_engine.enabled)

    # Memory
    mem_cfg = settings.get("memory", {})
    from agent_core.memory import ConversationMemory
    memory = ConversationMemory(
        max_turns=mem_cfg.get("max_turns", 20),
        summary_threshold=mem_cfg.get("summary_threshold", 15),
        llm_adapter=llm,
        db=db,
    )
    memory.load_from_db()
    _state["memory"] = memory

    # Planner + Router
    from agent_core.planner import Planner
    from agent_core.router import Router
    planner = Planner(llm_adapter=llm)
    router_obj = Router()
    _state["planner"] = planner
    _state["router"] = router_obj

    # Tool Registry + register tools
    from agent_core.tool_registry import ToolRegistry, ToolMeta
    registry = ToolRegistry()
    _state["registry"] = registry
    _register_tools(registry, router_obj, llm)

    # Intent Engine — semantic TF-IDF matching (near-zero RAM)
    from agent_core.intent_engine import IntentEngine
    intent_engine = IntentEngine(llm_adapter=llm)
    _state["intent_engine"] = intent_engine

    # Populate intent engine index from already-registered tools
    for tool_meta in registry.list_tools():
        intent_engine.index_tool(tool_meta.name, tool_meta.description, tool_meta.tags)

    # Plugin Loader — dynamic discovery from tools/ and tools/third_party/
    plugin_cfg = settings.get("plugin_loader", {})
    tool_dirs = plugin_cfg.get("tool_dirs", ["tools", "tools/third_party"])
    from agent_core.plugin_loader import PluginLoader
    plugin_loader = PluginLoader(tool_dirs=tool_dirs, llm_adapter=llm)
    _state["plugin_loader"] = plugin_loader

    if plugin_cfg.get("auto_discover", True):
        _discover_plugins(plugin_loader, registry, router_obj, intent_engine)

    # Tool Selector — ranks tools by semantic relevance
    from agent_core.tool_selector import ToolSelector
    tool_selector = ToolSelector(intent_engine)
    _state["tool_selector"] = tool_selector

    # Argument Composer
    from agent_core.argument_composer import ArgumentComposer
    arg_composer = ArgumentComposer(llm_adapter=llm)
    _state["arg_composer"] = arg_composer

    # Analytics (local SQLite only)
    analytics_cfg = settings.get("analytics", {})
    from agent_core.analytics import Analytics
    analytics = Analytics(
        db=db if analytics_cfg.get("log_to_db", True) else None,
        enabled=analytics_cfg.get("enabled", True),
    )
    _state["analytics"] = analytics

    # Confirmation Gate / Sandbox
    sandbox_cfg = settings.get("sandbox", {})
    from agent_core.sandbox import ConfirmationGate
    confirmation_gate = ConfirmationGate(
        confidence_threshold=sandbox_cfg.get("confidence_threshold", 0.3),
        auto_approve_known_tools=sandbox_cfg.get("auto_approve_known_tools", True),
        min_uses_for_auto_approve=sandbox_cfg.get("min_uses_for_auto_approve", 3),
    )
    _state["confirmation_gate"] = confirmation_gate

    # Task Queue
    from agent_core.task_queue import TaskQueue
    task_queue = TaskQueue(max_size=100)
    _state["task_queue"] = task_queue

    # Multi-agent dispatcher (OMLX-inspired)
    ma_cfg = settings.get("multi_agent", {})
    if ma_cfg.get("enabled", True):
        try:
            from agent_core.multi_agent import AgentDispatcher
            dispatcher = AgentDispatcher.from_config(
                ma_cfg.get("config_path", "config/agents.yaml")
            )
            _state["dispatcher"] = dispatcher
            logger.info("Multi-agent dispatcher ready")
        except Exception as exc:
            logger.warning("Multi-agent dispatcher failed to initialise: %s — single-agent mode active", exc)
            _state["dispatcher"] = None
    else:
        _state["dispatcher"] = None

    # Health monitors
    from system_health.ram_watchdog import RamWatchdog
    from system_health.disk_check import DiskCheck
    from system_health.heartbeat import Heartbeat
    from system_health.log_rotation import LogRotation
    from system_health.cache_cleanup import CacheCleanup

    health_cfg = settings.get("health", {})
    _state["ram_watchdog"] = RamWatchdog(
        warning_gb=health_cfg.get("ram_warning_gb", 3.0),
        critical_gb=health_cfg.get("ram_threshold_gb", 4.0),
    )
    _state["disk_check"] = DiskCheck(
        threshold_gb=health_cfg.get("disk_free_threshold_gb", 5.0)
    )
    _state["heartbeat"] = Heartbeat(db=db)
    _state["log_rotation"] = LogRotation(
        threshold_mb=health_cfg.get("log_rotation_threshold_mb", 50)
    )
    _state["cache_cleanup"] = CacheCleanup()

    # Dashboard shared state
    from interfaces.web_dashboard.api import set_shared_state
    set_shared_state({
        "planner": planner,
        "router": router_obj,
        "task_queue": task_queue,
        "memory": memory,
    })

    # APScheduler
    _start_scheduler(settings)
    logger.info("Lopen orchestrator startup complete")


def _build_llm(engine: str, model_path: str | None, llm_cfg: dict) -> Any:
    """Construct the best available LLM backend.

    Priority:
      1. AirLLMEngine (if engine='airllm' or engine='auto' and airllm installed)
      2. LLMAdapter (llama-cpp-python or subprocess llama.cpp)
      3. Mock (logged prominently)
    """
    kwargs = {
        "model_path": model_path,
        "context_window": llm_cfg.get("context_window", 2048),
        "temperature": llm_cfg.get("temperature", 0.7),
        "max_tokens": llm_cfg.get("max_tokens", 512),
        "memory_conservative": llm_cfg.get("memory_conservative", True),
    }
    if engine in ("airllm", "auto"):
        try:
            from llm.airllm_engine import AirLLMEngine
            instance = AirLLMEngine(
                **kwargs,
                engine=None if engine == "auto" else engine,
                compression=llm_cfg.get("compression", "4bit"),
                max_gpu_memory=llm_cfg.get("max_gpu_memory", 0),
            )
            logger.info("LLM backend: AirLLMEngine (engine=%s)", instance.backend)
            return instance
        except Exception as exc:
            logger.warning("AirLLMEngine unavailable (%s), falling back to LLMAdapter", exc)

    from llm.llm_adapter import LLMAdapter
    instance = LLMAdapter(**kwargs)
    logger.info("LLM backend: LLMAdapter (mode=%s)", instance.mode)
    return instance


def _register_tools(registry: Any, router_obj: Any, llm: Any) -> None:
    from agent_core.tool_registry import ToolMeta
    from tools.homework_tutor import HomeworkTutor
    from tools.researcher import Researcher
    from tools.coder_assist import CoderAssist
    from tools.desktop_organizer import DesktopOrganizer
    from tools.file_ops import FileOps
    from tools.browser_automation import BrowserAutomation

    tool_classes = [
        (HomeworkTutor, "homework_tutor", "Educational tutor", False),
        (Researcher, "researcher", "Web research", False),
        (CoderAssist, "coder_assist", "Code assistance", False),
        (DesktopOrganizer, "desktop_organizer", "Desktop file organiser", True),
        (FileOps, "file_ops", "Safe file operations", True),
        (BrowserAutomation, "browser_automation", "Browser automation", True),
    ]
    for ToolClass, name, desc, perm in tool_classes:
        instance = ToolClass(llm_adapter=llm)
        meta = ToolMeta(name=name, description=desc, requires_permission=perm, instance=instance)
        registry.register(meta)
        router_obj.register_handler(name, instance)

    # General LLM handler
    from agent_core.planner import Intent
    def llm_general(query: str, **_: Any) -> str:
        return llm.generate(query)

    router_obj.register_handler("llm_general", llm_general)
    logger.info("Registered %d tools", len(registry))


def _discover_plugins(
    plugin_loader: Any,
    registry: Any,
    router_obj: Any,
    intent_engine: Any,
) -> None:
    """Scan plugin directories, register any new tools found, and index them."""
    try:
        new_metas = plugin_loader.scan(skip_existing=True)
        for meta in new_metas:
            try:
                registry.re_register(meta)
                if meta.instance is not None:
                    router_obj.register_handler(meta.name, meta.instance)
                intent_engine.index_tool(meta.name, meta.description, meta.tags)
                logger.info("Dynamic plugin registered: %s", meta.name)
            except Exception as exc:
                logger.warning("Could not register plugin %s: %s", meta.name, exc)
    except Exception as exc:
        logger.error("Plugin discovery failed: %s", exc)


def _start_scheduler(settings: dict) -> None:
    global _scheduler
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        health_cfg = settings.get("health", {})
        heartbeat_interval = health_cfg.get("heartbeat_interval", 60)
        cache_cleanup_interval = health_cfg.get("cache_cleanup_interval", 86400)

        _scheduler = AsyncIOScheduler()

        def run_heartbeat() -> None:
            hb = _state.get("heartbeat")
            if hb:
                hb.check_all()

        def run_ram_check() -> None:
            wdog = _state.get("ram_watchdog")
            if wdog:
                wdog.check()

        def run_cache_cleanup() -> None:
            cc = _state.get("cache_cleanup")
            if cc:
                cc.run()

        def run_log_rotation() -> None:
            lr = _state.get("log_rotation")
            if lr:
                lr.run()

        _scheduler.add_job(run_heartbeat, "interval", seconds=heartbeat_interval, id="heartbeat")
        _scheduler.add_job(run_ram_check, "interval", seconds=60, id="ram_watchdog")
        _scheduler.add_job(run_cache_cleanup, "interval", seconds=cache_cleanup_interval, id="cache_cleanup")
        _scheduler.add_job(run_log_rotation, "interval", seconds=3600, id="log_rotation")
        _scheduler.start()
        logger.info("APScheduler started with %d jobs", len(_scheduler.get_jobs()))
    except ImportError:
        logger.warning("APScheduler not installed — health monitoring jobs disabled")
    except Exception as exc:
        logger.error("Scheduler start failed: %s", exc)


def _shutdown() -> None:
    global _scheduler
    memory = _state.get("memory")
    if memory:
        memory.save_to_db()
    # Unload multi-agent pool
    dispatcher = _state.get("dispatcher")
    if dispatcher:
        dispatcher.unload_all()
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    logger.info("Lopen orchestrator shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Lopen",
    description="Local-first autonomous assistant orchestrator",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    uptime = time.time() - _start_time if _start_time else 0.0
    return {
        "status": "healthy",
        "service": "orchestrator",
        "uptime_seconds": round(uptime, 1),
    }


@router.get("/status", tags=["Status"])
async def status() -> dict[str, Any]:
    from agent_core.task_queue import TaskStatus
    task_queue = _state.get("task_queue")
    memory = _state.get("memory")
    registry = _state.get("registry")
    safety_engine = _state.get("safety_engine")
    dispatcher = _state.get("dispatcher")
    llm = _state.get("llm")
    # Determine llm_mode/backend from either AirLLMEngine or LLMAdapter
    llm_mode = (
        getattr(llm, "backend", None) or getattr(llm, "mode", "unknown")
        if llm else "unknown"
    )
    result: dict[str, Any] = {
        "status": "running",
        "tasks_pending": sum(
            1 for t in (task_queue.list_tasks() if task_queue else [])
            if t.status == TaskStatus.PENDING
        ),
        "memory_turns": memory.turn_count if memory else 0,
        "tools_registered": len(registry) if registry else 0,
        "llm_mode": llm_mode,
        "safety": safety_engine.status() if safety_engine else {"enabled": False},
        "multi_agent": dispatcher.pool_status() if dispatcher else {"enabled": False},
    }
    return result


@router.post("/chat", tags=["Chat"])
async def chat(request: "Request") -> dict[str, Any]:
    body = await request.json() if hasattr(request, "json") else {}
    # Accept both 'query' and 'message' field names (CLI sends 'message')
    query: str = (body.get("query") or body.get("message") or "").strip()
    if not query:
        return {"response": "Please provide a query."}

    planner = _state.get("planner")
    router_obj = _state.get("router")
    memory = _state.get("memory")
    intent_engine = _state.get("intent_engine")
    tool_selector = _state.get("tool_selector")
    arg_composer = _state.get("arg_composer")
    registry = _state.get("registry")
    analytics = _state.get("analytics")
    confirmation_gate = _state.get("confirmation_gate")
    safety_engine = _state.get("safety_engine")
    dispatcher = _state.get("dispatcher")

    if not planner or not router_obj:
        return {"response": "[Orchestrator] Agent not initialised yet."}

    # --- Safety: check input before any processing ---
    if safety_engine:
        safety_result = safety_engine.check_input(query)
        if not safety_result.safe:
            logger.warning("Safety block on input: %s", safety_result.reason)
            return {
                "response": safety_engine.refusal_message(),
                "safety_blocked": True,
                "reason": safety_result.reason,
            }

    if memory:
        memory.add_turn("user", query)

    start_ms = time.time() * 1000
    selected_tool_name = "llm_general"
    confidence = 0.0

    # --- Dynamic routing via semantic tool selection ---
    if intent_engine and tool_selector and registry:
        available_tools = registry.list_tools(enabled_only=True)
        best_tool, confidence = tool_selector.select_best(query, available_tools)

        if best_tool is not None and confidence > 0.0:
            selected_tool_name = best_tool.name

            # Safety: check tool is permitted
            if safety_engine:
                tool_check = safety_engine.check_tool(selected_tool_name)
                if not tool_check.safe:
                    logger.warning("Safety block on tool '%s': %s", selected_tool_name, tool_check.reason)
                    selected_tool_name = "llm_general"
                    confidence = 0.0

            # Log intent analytics
            if analytics:
                intent_result = intent_engine.analyze(query)
                analytics.log_intent(
                    query=query,
                    structured_intent=intent_result.structured_intent,
                    confidence=confidence,
                    selected_tool=selected_tool_name,
                )

            # Check confirmation gate
            if confirmation_gate:
                confirmation_req = confirmation_gate.check(best_tool, query, confidence)
                if confirmation_req is not None:
                    return {
                        "response": confirmation_req.prompt,
                        "needs_confirmation": True,
                        "tool": selected_tool_name,
                        "confidence": round(confidence, 3),
                    }

    # --- Multi-agent dispatcher (if available and enabled) ---
    if dispatcher is not None:
        try:
            intent_result_obj = intent_engine.analyze(query) if intent_engine else None
            dispatch_result = await dispatcher.dispatch(
                query=query,
                intent_result=intent_result_obj,
                tools=registry,
            )
            response_str = dispatch_result.final_response
            agents_used = dispatch_result.agents_used
            success = True
        except Exception as exc:
            logger.error("Multi-agent dispatch error: %s — falling back to single agent", exc)
            dispatch_result = None
            agents_used = []
            # Fall through to single-agent path below
            response_str, success = await _single_agent_response(
                query, selected_tool_name, planner, router_obj, arg_composer, kwargs={}
            )
    else:
        agents_used = []
        # --- Single-agent path ---
        kwargs: dict[str, Any] = {}
        if arg_composer and selected_tool_name != "llm_general":
            kwargs = arg_composer.compose(query, tool_name=selected_tool_name)
            kwargs.pop("query", None)
        response_str, success = await _single_agent_response(
            query, selected_tool_name, planner, router_obj, arg_composer=None, kwargs=kwargs
        )

    latency_ms = time.time() * 1000 - start_ms

    # --- Safety: check output before returning ---
    if safety_engine:
        output_result = safety_engine.check_output(response_str)
        if output_result.modified_text is not None:
            response_str = output_result.modified_text

    # Record analytics
    if analytics:
        analytics.log_tool_use(selected_tool_name, query, success, latency_ms)

    # Update confirmation gate counter
    if confirmation_gate:
        confirmation_gate.record_use(selected_tool_name, success=success)

    if memory:
        memory.add_turn("assistant", response_str)
        memory.save_to_db()

    result: dict[str, Any] = {
        "response": response_str,
        "tool": selected_tool_name,
        "tool_used": selected_tool_name,
        "confidence": round(confidence, 3),
    }
    if agents_used:
        result["agents_used"] = agents_used
    return result


async def _single_agent_response(
    query: str,
    selected_tool_name: str,
    planner: Any,
    router_obj: Any,
    arg_composer: Any,
    kwargs: dict,
) -> tuple[str, bool]:
    """Execute a single-agent tool invocation, returning (response_str, success)."""
    try:
        handler = router_obj._handlers.get(selected_tool_name) or router_obj._handlers.get("llm_general")
        if handler is None:
            intent = planner.classify_intent(query)
            response = await router_obj.route_async(intent, query)
        else:
            import asyncio
            import inspect
            if inspect.iscoroutinefunction(handler):
                response = await handler(query, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: handler(query, **kwargs))
        return str(response), True
    except Exception as exc:
        logger.error("Single-agent chat error: %s", exc)
        return f"Error processing request: {exc}", False


@router.get("/tasks", tags=["Tasks"])
async def get_tasks() -> dict[str, Any]:
    task_queue = _state.get("task_queue")
    if not task_queue:
        return {"tasks": []}
    return {
        "tasks": [
            {
                "id": t.id,
                "intent": t.intent,
                "status": t.status,
                "payload": t.payload[:80],
                "created_at": t.created_at.isoformat(),
            }
            for t in task_queue.list_tasks()
        ]
    }


@router.get("/plugins", tags=["Plugins"])
async def list_plugins() -> dict[str, Any]:
    """List all registered tools/plugins and their metadata."""
    registry = _state.get("registry")
    if not registry:
        return {"plugins": []}
    return {
        "plugins": [
            {
                "name": t.name,
                "description": t.description,
                "version": t.version,
                "enabled": t.enabled,
                "requires_permission": t.requires_permission,
                "tags": t.tags,
            }
            for t in registry.list_tools()
        ]
    }


@router.post("/plugins/reload", tags=["Plugins"])
async def reload_plugins() -> dict[str, Any]:
    """Re-scan plugin directories and register any newly discovered tools."""
    plugin_loader = _state.get("plugin_loader")
    registry = _state.get("registry")
    router_obj = _state.get("router")
    intent_engine = _state.get("intent_engine")

    if not plugin_loader or not registry or not router_obj:
        return {"status": "error", "message": "Plugin loader not initialised"}

    try:
        new_metas = plugin_loader.scan(skip_existing=False)
        registered: list[str] = []
        for meta in new_metas:
            try:
                registry.re_register(meta)
                if meta.instance is not None:
                    router_obj.register_handler(meta.name, meta.instance)
                if intent_engine:
                    intent_engine.index_tool(meta.name, meta.description, meta.tags)
                registered.append(meta.name)
            except Exception as exc:
                logger.warning("Reload: could not register %s: %s", meta.name, exc)
        return {"status": "ok", "registered": registered, "count": len(registered)}
    except Exception as exc:
        logger.error("Plugin reload failed: %s", exc)
        return {"status": "error", "message": str(exc)}


@router.get("/analytics", tags=["Analytics"])
async def get_analytics() -> dict[str, Any]:
    """Return usage analytics summary."""
    analytics = _state.get("analytics")
    if not analytics:
        return {"error": "Analytics not initialised"}
    return analytics.get_stats()


@router.post("/feedback", tags=["Analytics"])
async def post_feedback(request: "Request") -> dict[str, str]:
    """Record user feedback for a tool response (RL signal)."""
    body = await request.json() if hasattr(request, "json") else {}
    tool_name: str = body.get("tool", "unknown")
    query: str = body.get("query", "")
    was_helpful: bool = bool(body.get("helpful", True))

    analytics = _state.get("analytics")
    if analytics:
        analytics.log_feedback(tool_name, query, was_helpful)

    return {"status": "recorded"}


@router.get("/memory", tags=["Memory"])
async def get_memory() -> dict[str, Any]:
    """Return current conversation history (turns)."""
    memory = _state.get("memory")
    if not memory:
        return {"turns": [], "turn_count": 0}
    turns = [
        {"role": t.role, "content": t.content}
        for t in memory.get_recent(n=50)
    ]
    return {"turns": turns, "turn_count": len(turns)}


@router.delete("/memory", tags=["Memory"])
async def clear_memory() -> dict[str, str]:
    """Clear the conversation history."""
    memory = _state.get("memory")
    if memory and hasattr(memory, "clear"):
        memory.clear()
    return {"status": "cleared"}


@router.get("/safety", tags=["Safety"])
async def get_safety_status() -> dict[str, Any]:
    """Return safety engine status and configuration."""
    safety_engine = _state.get("safety_engine")
    if not safety_engine:
        return {"enabled": False, "message": "Safety engine not initialised"}
    return safety_engine.status()


@router.get("/agents", tags=["Agents"])
async def get_agents_status() -> dict[str, Any]:
    """Return multi-agent dispatcher pool status."""
    dispatcher = _state.get("dispatcher")
    if not dispatcher:
        return {"enabled": False, "message": "Multi-agent dispatcher not enabled"}
    return {"enabled": True, **dispatcher.pool_status()}


app.include_router(router)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    settings = _load_yaml("config/settings.yaml")
    orch_cfg = settings.get("orchestrator", {})
    uvicorn.run(
        "orchestrator:app",
        host=orch_cfg.get("host", "0.0.0.0"),
        port=orch_cfg.get("port", 8000),
        workers=1,
        reload=False,
    )
