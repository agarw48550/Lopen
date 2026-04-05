"""Tests for FastLLM singleton and mock fallback."""

from __future__ import annotations

import pytest
from llm.fast_llm import FastLLM, get_fast_llm, _heuristic_complexity


# ---------------------------------------------------------------------------
# Singleton pattern
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_same_instance(self) -> None:
        a = FastLLM()
        b = FastLLM()
        assert a is b

    def test_get_fast_llm_returns_singleton(self) -> None:
        a = get_fast_llm()
        b = get_fast_llm()
        assert a is b

    def test_get_fast_llm_is_fast_llm(self) -> None:
        assert isinstance(get_fast_llm(), FastLLM)


# ---------------------------------------------------------------------------
# Mock mode (model file not present in test env)
# ---------------------------------------------------------------------------

class TestMockMode:
    @pytest.fixture
    def llm(self) -> FastLLM:
        return get_fast_llm()

    def test_is_mock_when_no_model(self, llm: FastLLM) -> None:
        # In CI there is no model file, so mock mode must be active
        assert llm.is_mock is True

    def test_acknowledge_returns_string(self, llm: FastLLM) -> None:
        result = llm.acknowledge("Set a timer for 5 minutes")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_acknowledge_short(self, llm: FastLLM) -> None:
        result = llm.acknowledge("hello")
        # Should be a brief phrase, not a paragraph
        assert len(result) < 50

    def test_route_returns_int(self, llm: FastLLM) -> None:
        score = llm.route("What is 2+2?")
        assert isinstance(score, int)

    def test_route_in_range(self, llm: FastLLM) -> None:
        for query in ["hi", "explain quantum entanglement step by step", "debug my code"]:
            score = llm.route(query)
            assert 0 <= score <= 10, f"score {score} out of range for: {query!r}"

    def test_generate_returns_string(self, llm: FastLLM) -> None:
        result = llm.generate("Hello world")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_returns_shorter_text(self, llm: FastLLM) -> None:
        long_text = " ".join(["word"] * 100)
        summary = llm.summarize(long_text)
        assert isinstance(summary, str)
        assert len(summary) < len(long_text)

    def test_summarize_empty_string(self, llm: FastLLM) -> None:
        result = llm.summarize("")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Heuristic complexity helper
# ---------------------------------------------------------------------------

class TestHeuristicComplexity:
    def test_greeting_low(self) -> None:
        assert _heuristic_complexity("hello") <= 2

    def test_complex_high(self) -> None:
        score = _heuristic_complexity("explain step by step how to debug memory leaks in C++")
        assert score >= 5

    def test_simple_question_moderate(self) -> None:
        score = _heuristic_complexity("what is Python?")
        assert 0 <= score <= 10
