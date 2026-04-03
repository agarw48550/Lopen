"""Tool registry: register, unregister, and query available tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolMeta:
    name: str
    description: str
    version: str = "1.0.0"
    requires_permission: bool = False
    enabled: bool = True
    tags: list[str] = field(default_factory=list)
    instance: Optional[Any] = field(default=None, repr=False)


class ToolRegistry:
    """Central registry for all Lopen tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolMeta] = {}
        logger.info("ToolRegistry initialised")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, meta: ToolMeta) -> None:
        """Register a tool. Raises ValueError if already registered (by same name)."""
        if meta.name in self._tools:
            raise ValueError(f"Tool '{meta.name}' is already registered.")
        self._tools[meta.name] = meta
        logger.info("Tool registered: %s (enabled=%s)", meta.name, meta.enabled)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns True if it existed."""
        if name in self._tools:
            del self._tools[name]
            logger.info("Tool unregistered: %s", name)
            return True
        logger.warning("Tried to unregister unknown tool: %s", name)
        return False

    def get_tool(self, name: str) -> Optional[ToolMeta]:
        """Return a ToolMeta by name, or None if not found."""
        return self._tools.get(name)

    def enable(self, name: str) -> None:
        tool = self._get_or_raise(name)
        tool.enabled = True
        logger.info("Tool enabled: %s", name)

    def disable(self, name: str) -> None:
        tool = self._get_or_raise(name)
        tool.enabled = False
        logger.info("Tool disabled: %s", name)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_tools(self, enabled_only: bool = False) -> list[ToolMeta]:
        tools = list(self._tools.values())
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        return tools

    def names(self, enabled_only: bool = False) -> list[str]:
        return [t.name for t in self.list_tools(enabled_only=enabled_only)]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_or_raise(self, name: str) -> ToolMeta:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return tool
