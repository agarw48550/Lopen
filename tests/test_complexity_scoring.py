"""Tests for IntentEngine.complexity_score()."""

from __future__ import annotations

import pytest
from agent_core.intent_engine import IntentEngine


@pytest.fixture
def engine() -> IntentEngine:
    return IntentEngine(llm_adapter=None)


# ---------------------------------------------------------------------------
# Simple queries → low score (0-3)
# ---------------------------------------------------------------------------

class TestSimpleQueries:
    def test_greeting_hello(self, engine: IntentEngine) -> None:
        assert engine.complexity_score("hello") <= 3

    def test_greeting_hi(self, engine: IntentEngine) -> None:
        assert engine.complexity_score("hi") <= 3

    def test_status_check(self, engine: IntentEngine) -> None:
        assert engine.complexity_score("status") <= 3

    def test_ping(self, engine: IntentEngine) -> None:
        assert engine.complexity_score("ping") <= 3

    def test_thanks(self, engine: IntentEngine) -> None:
        assert engine.complexity_score("thanks") <= 3

    def test_short_question(self, engine: IntentEngine) -> None:
        assert engine.complexity_score("what time is it") <= 3


# ---------------------------------------------------------------------------
# Complex queries → high score (7-10)
# ---------------------------------------------------------------------------

class TestComplexQueries:
    def test_step_by_step(self, engine: IntentEngine) -> None:
        score = engine.complexity_score("explain step by step how to debug memory leaks")
        assert score >= 7

    def test_multi_keyword(self, engine: IntentEngine) -> None:
        score = engine.complexity_score(
            "research and analyse the algorithm, then implement and code a solution"
        )
        assert score >= 7

    def test_architecture_design(self, engine: IntentEngine) -> None:
        score = engine.complexity_score("design the architecture for a microservice system")
        assert score >= 5

    def test_complex_research(self, engine: IntentEngine) -> None:
        score = engine.complexity_score("research quantum computing and explain the key concepts")
        assert score >= 5


# ---------------------------------------------------------------------------
# Medium queries → medium score (4-6)
# ---------------------------------------------------------------------------

class TestMediumQueries:
    def test_moderate_length(self, engine: IntentEngine) -> None:
        query = "tell me about Python programming and how to write functions properly"
        score = engine.complexity_score(query)
        assert 0 <= score <= 10  # valid range

    def test_explain_once(self, engine: IntentEngine) -> None:
        score = engine.complexity_score("explain how lists work in Python")
        # Single complexity keyword → 5
        assert score == 5

    def test_single_debug(self, engine: IntentEngine) -> None:
        score = engine.complexity_score("debug this code snippet")
        assert score >= 4


# ---------------------------------------------------------------------------
# Return type and range
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_int(self, engine: IntentEngine) -> None:
        assert isinstance(engine.complexity_score("hello"), int)

    def test_always_in_range(self, engine: IntentEngine) -> None:
        queries = [
            "",
            "hi",
            "write a sorting algorithm",
            "explain step by step how to implement and debug and analyse complex distributed systems",
        ]
        for q in queries:
            score = engine.complexity_score(q)
            assert 0 <= score <= 10, f"score {score} out of range for: {q!r}"
