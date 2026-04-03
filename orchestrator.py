"""Lopen Orchestrator — main entry point wiring all components together."""

from __future__ import annotations

import logging
import logging.config
import os
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
    settings = _load_yaml("config/settings.yaml")
    lopen_cfg = settings.get("lopen", {})
    log_level = os.environ.get("LOPEN_LOG_LEVEL", lopen_cfg.get("log_level", "INFO"))
    log_dir = lopen_cfg.get("log_dir", "logs")
    _configure_logging(log_dir=log_dir, level=log_level)

    logger.info("Lopen orchestrator starting up…")

    # Storage
    from storage.database import SQLiteDB
    db = SQLiteDB()
    _state["db"] = db

    # LLM
    llm_cfg = settings.get("llm", {})
    from llm.llm_adapter import LLMAdapter
    llm = LLMAdapter(
        model_path=os.environ.get("LOPEN_LLM_MODEL_PATH", llm_cfg.get("model_path")),
        context_window=llm_cfg.get("context_window", 2048),
        temperature=llm_cfg.get("temperature", 0.7),
        max_tokens=llm_cfg.get("max_tokens", 512),
        memory_conservative=llm_cfg.get("memory_conservative", True),
    )
    _state["llm"] = llm

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

    # Task Queue
    from agent_core.task_queue import TaskQueue
    task_queue = TaskQueue(max_size=100)
    _state["task_queue"] = task_queue

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
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "orchestrator"}


@router.get("/status", tags=["Status"])
async def status() -> dict[str, Any]:
    from agent_core.task_queue import TaskStatus
    task_queue = _state.get("task_queue")
    memory = _state.get("memory")
    registry = _state.get("registry")
    return {
        "status": "running",
        "tasks_pending": sum(
            1 for t in (task_queue.list_tasks() if task_queue else [])
            if t.status == TaskStatus.PENDING
        ),
        "memory_turns": memory.turn_count if memory else 0,
        "tools_registered": len(registry) if registry else 0,
        "llm_mode": _state["llm"].mode if "llm" in _state else "unknown",
    }


@router.post("/chat", tags=["Chat"])
async def chat(request: "Request") -> dict[str, str]:
    from fastapi.responses import JSONResponse  # noqa: F401  kept for future use

    body = await request.json() if hasattr(request, "json") else {}
    query: str = body.get("query", "").strip()
    if not query:
        return {"response": "Please provide a query."}

    planner = _state.get("planner")
    router_obj = _state.get("router")
    memory = _state.get("memory")

    if not planner or not router_obj:
        return {"response": "[Orchestrator] Agent not initialised yet."}

    if memory:
        memory.add_turn("user", query)

    intent = planner.classify_intent(query)
    try:
        response = await router_obj.route_async(intent, query)
        response_str = str(response)
    except Exception as exc:
        logger.error("Chat error: %s", exc)
        response_str = f"Error processing request: {exc}"

    if memory:
        memory.add_turn("assistant", response_str)
        memory.save_to_db()

    return {"response": response_str, "intent": intent.value}


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