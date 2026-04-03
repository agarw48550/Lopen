"""Unit tests for IntentEngine semantic matching."""

from __future__ import annotations

import pytest
from agent_core.intent_engine import IntentEngine, IntentResult, _tokenize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> IntentEngine:
    eng = IntentEngine(llm_adapter=None)
    # Index sample tools
    eng.index_tool("homework_tutor", "Educational tutor for homework questions, lessons and explanations", ["education", "llm"])
    eng.index_tool("researcher", "Web research and information lookup using DuckDuckGo search", ["web", "search", "information"])
    eng.index_tool("coder_assist", "Code explanation review generation debugging python javascript", ["coding", "llm", "debug"])
    eng.index_tool("desktop_organizer", "Organise Desktop files into folders by type sort clean", ["filesystem", "macos", "organize"])
    eng.index_tool("file_ops", "Safe file read write create search delete within approved directories", ["filesystem", "files"])
    eng.index_tool("browser_automation", "Playwright based browser automation web scraping forms", ["web", "automation", "browser"])
    return eng


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_tokenization(self) -> None:
        tokens = _tokenize("Write a Python function")
        assert "python" in tokens
        assert "function" in tokens

    def test_stop_words_removed(self) -> None:
        tokens = _tokenize("what is the speed of light")
        assert "the" not in tokens
        assert "of" not in tokens
        assert "is" not in tokens

    def test_short_tokens_removed(self) -> None:
        tokens = _tokenize("a an in on")
        assert tokens == []  # all stop words or too short

    def test_empty_string(self) -> None:
        assert _tokenize("") == []


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

class TestIndexManagement:
    def test_index_tool(self) -> None:
        eng = IntentEngine()
        eng.index_tool("my_tool", "A useful tool for testing")
        assert "my_tool" in eng.indexed_tools()

    def test_remove_tool(self) -> None:
        eng = IntentEngine()
        eng.index_tool("my_tool", "A tool")
        eng.remove_tool("my_tool")
        assert "my_tool" not in eng.indexed_tools()

    def test_update_existing_tool(self) -> None:
        eng = IntentEngine()
        eng.index_tool("my_tool", "First description")
        eng.index_tool("my_tool", "Second description with new keywords")
        # Should not raise; just updates
        assert "my_tool" in eng.indexed_tools()


# ---------------------------------------------------------------------------
# Intent analysis
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_returns_intent_result(self, engine: IntentEngine) -> None:
        result = engine.analyze("explain derivatives")
        assert isinstance(result, IntentResult)

    def test_raw_intent_preserved(self, engine: IntentEngine) -> None:
        query = "write a python function to sort a list"
        result = engine.analyze(query)
        assert result.raw_intent == query

    def test_keywords_extracted(self, engine: IntentEngine) -> None:
        result = engine.analyze("write a python function")
        assert "python" in result.keywords or "function" in result.keywords

    def test_empty_query_returns_zero_confidence(self, engine: IntentEngine) -> None:
        result = engine.analyze("")
        assert result.confidence == 0.0

    def test_coding_query_suggests_coder(self, engine: IntentEngine) -> None:
        result = engine.analyze("write a python function to sort a list")
        assert "coder_assist" in result.suggested_tools[:3]

    def test_research_query_suggests_researcher(self, engine: IntentEngine) -> None:
        result = engine.analyze("search for information about climate change")
        assert "researcher" in result.suggested_tools[:3]

    def test_file_query_suggests_file_ops(self, engine: IntentEngine) -> None:
        result = engine.analyze("read this file and write a new one")
        assert "file_ops" in result.suggested_tools[:3]

    def test_organize_query_suggests_desktop(self, engine: IntentEngine) -> None:
        result = engine.analyze("organize and sort files on my desktop")
        assert "desktop_organizer" in result.suggested_tools[:2]

    def test_tool_scores_present(self, engine: IntentEngine) -> None:
        result = engine.analyze("debug my python code")
        assert len(result.tool_scores) > 0
        assert all(0.0 <= s <= 1.0 for s in result.tool_scores.values())

    def test_score_tool_direct(self, engine: IntentEngine) -> None:
        score = engine.score_tool("write python code", "coder_assist")
        assert score > 0.0

    def test_no_index_returns_safely(self) -> None:
        eng = IntentEngine()
        result = eng.analyze("some query")
        assert result.confidence == 0.0
        assert result.suggested_tools == []
