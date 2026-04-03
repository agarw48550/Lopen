"""Unit tests for ToolSelector dynamic ranking."""

from __future__ import annotations

import pytest
from agent_core.intent_engine import IntentEngine
from agent_core.tool_registry import ToolMeta
from agent_core.tool_selector import ToolSelector


@pytest.fixture
def engine_with_tools() -> IntentEngine:
    eng = IntentEngine()
    eng.index_tool("homework_tutor", "Educational tutor for homework questions and explanations", ["education"])
    eng.index_tool("researcher", "Web research and information search lookup", ["web", "search"])
    eng.index_tool("coder_assist", "Code generation explanation debugging review python javascript", ["coding"])
    eng.index_tool("file_ops", "File read write create search delete operations filesystem", ["filesystem"])
    return eng


@pytest.fixture
def sample_tools() -> list[ToolMeta]:
    return [
        ToolMeta("homework_tutor", "Educational tutor for homework questions", tags=["education"]),
        ToolMeta("researcher", "Web research and information search", tags=["web", "search"]),
        ToolMeta("coder_assist", "Code generation debugging review", tags=["coding"]),
        ToolMeta("file_ops", "File read write create delete", tags=["filesystem"]),
    ]


@pytest.fixture
def selector(engine_with_tools: IntentEngine) -> ToolSelector:
    return ToolSelector(engine_with_tools)


class TestToolSelector:
    def test_select_returns_list(self, selector: ToolSelector, sample_tools: list[ToolMeta]) -> None:
        result = selector.select("write python code", sample_tools)
        assert isinstance(result, list)

    def test_select_returns_tuples(self, selector: ToolSelector, sample_tools: list[ToolMeta]) -> None:
        result = selector.select("write python code", sample_tools)
        for item in result:
            meta, score = item
            assert isinstance(meta, ToolMeta)
            assert isinstance(score, float)

    def test_scores_in_range(self, selector: ToolSelector, sample_tools: list[ToolMeta]) -> None:
        result = selector.select("write python code", sample_tools)
        for _, score in result:
            assert 0.0 <= score <= 1.0

    def test_coding_query_prefers_coder_assist(
        self, selector: ToolSelector, sample_tools: list[ToolMeta]
    ) -> None:
        result = selector.select("write a python function to sort a list", sample_tools)
        assert len(result) > 0
        top_tool, _ = result[0]
        assert top_tool.name == "coder_assist"

    def test_research_query_prefers_researcher(
        self, selector: ToolSelector, sample_tools: list[ToolMeta]
    ) -> None:
        result = selector.select("search for information about machine learning", sample_tools)
        assert len(result) > 0
        top_tool, _ = result[0]
        assert top_tool.name == "researcher"

    def test_top_k_respected(self, selector: ToolSelector, sample_tools: list[ToolMeta]) -> None:
        result = selector.select("do something", sample_tools, top_k=2)
        assert len(result) <= 2

    def test_select_best_returns_tuple(self, selector: ToolSelector, sample_tools: list[ToolMeta]) -> None:
        tool, score = selector.select_best("write python code", sample_tools)
        assert tool is not None or score == 0.0

    def test_select_best_score_matches_select(
        self, selector: ToolSelector, sample_tools: list[ToolMeta]
    ) -> None:
        best_tool, best_score = selector.select_best("write python code", sample_tools)
        ranked = selector.select("write python code", sample_tools)
        if ranked:
            assert ranked[0][1] == pytest.approx(best_score, abs=1e-6)

    def test_empty_tools_returns_empty(self, selector: ToolSelector) -> None:
        result = selector.select("any query", [])
        assert result == []

    def test_disabled_tools_excluded(self, selector: ToolSelector) -> None:
        tools = [
            ToolMeta("coder_assist", "Code generation", enabled=True, tags=["coding"]),
            ToolMeta("researcher", "Web research", enabled=False, tags=["web"]),
        ]
        result = selector.select("search the web", tools)
        names = [m.name for m, _ in result]
        assert "researcher" not in names

    def test_results_ordered_descending(self, selector: ToolSelector, sample_tools: list[ToolMeta]) -> None:
        result = selector.select("write python code", sample_tools)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)
