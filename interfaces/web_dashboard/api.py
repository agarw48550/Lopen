"""REST API routes for the Lopen web dashboard."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# Shared state injected at startup (set from orchestrator)
_state: dict[str, Any] = {}


def set_shared_state(state: dict[str, Any]) -> None:
    """Inject shared runtime objects (task_queue, memory, etc.) into the API."""
    _state.update(state)


def build_router(templates: Jinja2Templates) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse, tags=["UI"])
    async def index(request: Request) -> HTMLResponse:
        tasks = _get_tasks()
        memory_turns = _get_memory_turns()
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "tasks": tasks, "memory_turns": memory_turns},
        )

    @router.get("/health", tags=["Health"])
    async def health() -> dict[str, str]:
        return {"status": "healthy", "service": "dashboard"}

    @router.get("/status", tags=["Status"])
    async def status() -> dict[str, Any]:
        return {
            "status": "running",
            "tasks_pending": len([t for t in _get_tasks() if t.get("status") == "pending"]),
            "memory_turns": len(_get_memory_turns()),
        }

    @router.post("/chat", tags=["Chat"])
    async def chat(request: Request) -> dict[str, str]:
        body = await request.json()
        query: str = body.get("query", "").strip()
        if not query:
            return {"response": "Please provide a query."}
        planner = _state.get("planner")
        router_obj = _state.get("router")
        if planner and router_obj:
            intent = planner.classify_intent(query)
            try:
                response = await router_obj.route_async(intent, query)
                return {"response": str(response)}
            except Exception as exc:
                logger.error("Chat route error: %s", exc)
                return {"response": f"Error: {exc}"}
        return {"response": "[No agent backend configured]"}

    @router.get("/tasks", tags=["Tasks"])
    async def get_tasks() -> dict[str, Any]:
        return {"tasks": _get_tasks()}

    @router.get("/memory", tags=["Memory"])
    async def get_memory() -> dict[str, Any]:
        return {"turns": _get_memory_turns()}

    return router


def _get_tasks() -> list[dict[str, Any]]:
    task_queue = _state.get("task_queue")
    if task_queue is None:
        return []
    return [
        {
            "id": t.id,
            "intent": t.intent,
            "status": t.status,
            "payload": t.payload[:80],
            "created_at": t.created_at.isoformat(),
        }
        for t in task_queue.list_tasks()
    ]


def _get_memory_turns() -> list[dict[str, str]]:
    memory = _state.get("memory")
    if memory is None:
        return []
    return [{"role": t.role, "content": t.content[:120]} for t in memory.get_recent()]
