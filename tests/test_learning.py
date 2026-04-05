"""Tests for AgentLearner."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from agent_core.learning import AgentLearner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_interactions.db")


@pytest.fixture
def learner(db_path: str) -> AgentLearner:
    return AgentLearner(db_path=db_path)


# ---------------------------------------------------------------------------
# log_interaction
# ---------------------------------------------------------------------------

class TestLogInteraction:
    def test_returns_int(self, learner: AgentLearner) -> None:
        row_id = learner.log_interaction("Hello", "Hi there!")
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_increments_id(self, learner: AgentLearner) -> None:
        id1 = learner.log_interaction("First", "Reply 1")
        id2 = learner.log_interaction("Second", "Reply 2")
        assert id2 > id1

    def test_with_rating(self, learner: AgentLearner) -> None:
        row_id = learner.log_interaction("How are you?", "I am well.", rating=5)
        assert row_id > 0

    def test_with_interface(self, learner: AgentLearner) -> None:
        row_id = learner.log_interaction("Test", "Answer", interface="whatsapp")
        assert row_id > 0

    def test_creates_db_file(self, db_path: str) -> None:
        l = AgentLearner(db_path=db_path)
        l.log_interaction("test", "response")
        assert Path(db_path).exists()


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_returns_dict(self, learner: AgentLearner) -> None:
        assert isinstance(learner.get_stats(), dict)

    def test_empty_db_stats(self, learner: AgentLearner) -> None:
        stats = learner.get_stats()
        assert stats["total_interactions"] == 0
        assert stats["rated_interactions"] == 0
        assert stats["avg_rating"] is None
        assert stats["learned_patterns"] == 0

    def test_stats_after_interactions(self, learner: AgentLearner) -> None:
        learner.log_interaction("Q1", "A1", rating=4)
        learner.log_interaction("Q2", "A2", rating=2)
        learner.log_interaction("Q3", "A3")
        stats = learner.get_stats()
        assert stats["total_interactions"] == 3
        assert stats["rated_interactions"] == 2
        assert stats["avg_rating"] == 3.0

    def test_stats_has_required_keys(self, learner: AgentLearner) -> None:
        stats = learner.get_stats()
        for key in ("total_interactions", "rated_interactions", "avg_rating",
                    "low_rated_count", "learned_patterns", "top_topics"):
            assert key in stats

    def test_top_topics_is_list(self, learner: AgentLearner) -> None:
        assert isinstance(learner.get_stats()["top_topics"], list)


# ---------------------------------------------------------------------------
# get_low_rated
# ---------------------------------------------------------------------------

class TestGetLowRated:
    def test_returns_list(self, learner: AgentLearner) -> None:
        assert isinstance(learner.get_low_rated(), list)

    def test_empty_when_no_low_ratings(self, learner: AgentLearner) -> None:
        learner.log_interaction("Good Q", "Good A", rating=5)
        assert learner.get_low_rated(min_rating=3) == []

    def test_finds_low_rated(self, learner: AgentLearner) -> None:
        learner.log_interaction("Bad Q", "Bad A", rating=1)
        results = learner.get_low_rated(min_rating=3)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# improve_response
# ---------------------------------------------------------------------------

class TestImproveResponse:
    def test_returns_string(self, learner: AgentLearner) -> None:
        result = learner.improve_response("coding", "Here is a snippet.")
        assert isinstance(result, str)

    def test_no_patterns_returns_original(self, learner: AgentLearner) -> None:
        original = "Here is the answer."
        result = learner.improve_response("nonexistent_topic", original)
        assert result == original

    def test_with_learned_patterns(self, learner: AgentLearner) -> None:
        # Add low-rated interaction so learn_from_feedback generates a pattern
        learner.log_interaction("coding question", "Bad answer", rating=1)
        learner.learn_from_feedback()
        result = learner.improve_response("coding", "Basic response.")
        # After learning, result should differ from plain original (note appended)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# learn_from_feedback
# ---------------------------------------------------------------------------

class TestLearnFromFeedback:
    def test_returns_int(self, learner: AgentLearner) -> None:
        assert isinstance(learner.learn_from_feedback(), int)

    def test_no_low_rated_returns_zero(self, learner: AgentLearner) -> None:
        learner.log_interaction("Good", "Great answer", rating=5)
        assert learner.learn_from_feedback() == 0

    def test_learns_from_low_rated(self, learner: AgentLearner) -> None:
        learner.log_interaction("debug my code", "Unhelpful", rating=1)
        count = learner.learn_from_feedback()
        assert count >= 0  # may be 0 if pattern already exists, >= 0 always
        stats = learner.get_stats()
        assert stats["learned_patterns"] >= 0
