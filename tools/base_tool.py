"""Base tool abstract class for all Lopen tools."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base class that all Lopen tools must inherit from."""

    name: str = "base_tool"
    description: str = "A Lopen tool"
    requires_permission: bool = False

    def __init__(self, llm_adapter: Any | None = None) -> None:
        self._llm = llm_adapter
        self._logger = logging.getLogger(f"lopen.tool.{self.name}")
        self._logger.info("Tool initialised: %s", self.name)

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> str:
        """Execute the tool with the given query and return a string result."""

    def __call__(self, query: str, **kwargs: Any) -> str:
        self._logger.info("Tool called: %s | query=%r", self.name, query[:80])
        try:
            result = self.run(query, **kwargs)
            self._logger.debug("Tool result: %r", str(result)[:120])
            return result
        except Exception as exc:
            self._logger.error("Tool %s raised: %s", self.name, exc)
            return f"[{self.name}] Error: {exc}"
