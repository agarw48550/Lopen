"""Dynamic tool selector for Lopen.

Given a free-form user query and a list of available tools, ranks them by
semantic relevance using the :class:`~agent_core.intent_engine.IntentEngine`
and returns an ordered candidate list with confidence scores.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Minimum score for a tool to be considered a viable candidate
_MIN_CANDIDATE_SCORE = 0.05


class ToolSelector:
    """Select the best-matching tool(s) for an open-ended query.

    Usage::

        selector = ToolSelector(intent_engine)
        candidates = selector.select(query, registry.list_tools(enabled_only=True))
        best_tool, confidence = candidates[0]
    """

    def __init__(self, intent_engine: Any) -> None:
        self._engine = intent_engine
        logger.info("ToolSelector initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        query: str,
        tools: list[Any],
        top_k: int = 3,
    ) -> list[tuple[Any, float]]:
        """Rank tools by relevance and return the top-k candidates.

        Args:
            query: Free-form user query.
            tools: List of :class:`~agent_core.tool_registry.ToolMeta` objects.
            top_k: Maximum number of candidates to return.

        Returns:
            Ordered list of ``(ToolMeta, confidence)`` tuples, highest first.
            May be shorter than ``top_k`` if few tools pass the minimum score.
        """
        if not tools:
            return []

        # Ensure all tools are indexed
        for tool in tools:
            if tool.name not in self._engine.indexed_tools():
                self._engine.index_tool(tool.name, tool.description, tool.tags)

        intent_result = self._engine.analyze(query)

        candidates: list[tuple[Any, float]] = []
        for tool in tools:
            if not tool.enabled:
                continue
            score = intent_result.tool_scores.get(tool.name, 0.0)
            if score >= _MIN_CANDIDATE_SCORE:
                candidates.append((tool, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def select_best(self, query: str, tools: list[Any]) -> tuple[Any | None, float]:
        """Return the single best tool and its confidence score.

        Returns ``(None, 0.0)`` if no tool passes the minimum threshold.
        """
        candidates = self.select(query, tools, top_k=1)
        if not candidates:
            return None, 0.0
        return candidates[0]
