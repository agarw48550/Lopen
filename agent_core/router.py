"""Routes classified intents to registered tools/services."""

from __future__ import annotations

import logging
from typing import Any, Callable

from agent_core.planner import Intent

logger = logging.getLogger(__name__)

# Maps each intent to the canonical tool name
_DEFAULT_INTENT_TOOL_MAP: dict[Intent, str] = {
    Intent.HOMEWORK: "homework_tutor",
    Intent.RESEARCH: "researcher",
    Intent.CODING: "coder_assist",
    Intent.DESKTOP: "desktop_organizer",
    Intent.COMMUNICATION: "whatsapp",
    Intent.VOICE: "voice_loop",
    Intent.FILE_OPS: "file_ops",
    Intent.GENERAL: "llm_general",
}


class Router:
    """Routes task intents to tool handler callables."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._intent_tool_map: dict[Intent, str] = dict(_DEFAULT_INTENT_TOOL_MAP)
        logger.info("Router initialised")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_handler(self, tool_name: str, handler: Callable[..., Any]) -> None:
        """Register a callable handler under a tool name."""
        self._handlers[tool_name] = handler
        logger.debug("Handler registered for tool: %s", tool_name)

    def unregister_handler(self, tool_name: str) -> bool:
        """Remove a handler. Returns True if it existed."""
        if tool_name in self._handlers:
            del self._handlers[tool_name]
            logger.debug("Handler unregistered: %s", tool_name)
            return True
        return False

    def set_intent_mapping(self, intent: Intent, tool_name: str) -> None:
        """Override the default intent -> tool mapping."""
        self._intent_tool_map[intent] = tool_name

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, intent: Intent, query: str, **kwargs: Any) -> Any:
        """Route a query to the appropriate tool handler synchronously."""
        tool_name = self._intent_tool_map.get(intent, "llm_general")
        handler = self._handlers.get(tool_name)

        if handler is None:
            logger.warning("No handler for tool %r (intent=%s); using fallback", tool_name, intent.value)
            handler = self._handlers.get("llm_general")

        if handler is None:
            logger.error("No fallback handler available for intent %s", intent.value)
            return f"[Router] No handler available for intent '{intent.value}'."

        logger.info("Routing intent=%s to tool=%s", intent.value, tool_name)
        return handler(query, **kwargs)

    async def route_async(self, intent: Intent, query: str, **kwargs: Any) -> Any:
        """Route a query asynchronously (calls sync handler in thread if needed)."""
        import asyncio
        import inspect

        tool_name = self._intent_tool_map.get(intent, "llm_general")
        handler = self._handlers.get(tool_name) or self._handlers.get("llm_general")

        if handler is None:
            return f"[Router] No handler available for intent '{intent.value}'."

        logger.info("Async-routing intent=%s to tool=%s", intent.value, tool_name)
        if inspect.iscoroutinefunction(handler):
            return await handler(query, **kwargs)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: handler(query, **kwargs))

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def available_tools(self) -> list[str]:
        return list(self._handlers.keys())

    def is_available(self, intent: Intent) -> bool:
        tool_name = self._intent_tool_map.get(intent)
        return tool_name in self._handlers
