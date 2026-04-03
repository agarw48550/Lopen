"""Unit tests for Analytics event logging and stats."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from agent_core.analytics import Analytics


# ---------------------------------------------------------------------------
# Minimal in-memory DB stub
# ---------------------------------------------------------------------------

class _MockDB:
    """Minimal SQLite-backed stub that mimics the real SQLiteDB interface."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row

    def execute(self, sql: str, params: tuple = ()) -> list:
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        try:
            return cur.fetchall()
        except Exception:
            return []


@pytest.fixture
def db() -> _MockDB:
    return _MockDB()


@pytest.fixture
def analytics(db: _MockDB) -> Analytics:
    return Analytics(db=db, enabled=True)


class TestAnalyticsLogging:
    def test_log_tool_use_inserts_row(self, analytics: Analytics, db: _MockDB) -> None:
        analytics.log_tool_use("coder_assist", "write python code", True, 50.0)
        rows = db.execute("SELECT * FROM analytics WHERE event_type='tool_use'")
        assert len(rows) == 1

    def test_log_intent_inserts_row(self, analytics: Analytics, db: _MockDB) -> None:
        analytics.log_intent("find info", "web research", 0.8, "researcher")
        rows = db.execute("SELECT * FROM analytics WHERE event_type='intent'")
        assert len(rows) == 1

    def test_log_feedback_inserts_row(self, analytics: Analytics, db: _MockDB) -> None:
        analytics.log_feedback("researcher", "climate news", True)
        rows = db.execute("SELECT * FROM analytics WHERE event_type='feedback'")
        assert len(rows) == 1

    def test_data_serialised_as_json(self, analytics: Analytics, db: _MockDB) -> None:
        analytics.log_tool_use("file_ops", "read my file", False, 10.0)
        rows = db.execute("SELECT data FROM analytics WHERE event_type='tool_use'")
        data = json.loads(rows[0]["data"])
        assert data["tool_name"] == "file_ops"
        assert data["success"] is False

    def test_disabled_analytics_does_not_write(self, db: _MockDB) -> None:
        an = Analytics(db=db, enabled=False)
        an.log_tool_use("coder_assist", "some query", True, 20.0)
        rows = db.execute("SELECT COUNT(*) as cnt FROM analytics")
        assert rows[0]["cnt"] == 0

    def test_no_db_does_not_raise(self) -> None:
        an = Analytics(db=None, enabled=True)
        an.log_tool_use("coder_assist", "some query", True, 20.0)
        an.log_intent("q", "intent", 0.5, "tool")

    def test_query_truncated_at_200(self, analytics: Analytics, db: _MockDB) -> None:
        long_query = "x" * 500
        analytics.log_tool_use("tool", long_query, True, 0.0)
        rows = db.execute("SELECT data FROM analytics WHERE event_type='tool_use'")
        data = json.loads(rows[0]["data"])
        assert len(data["query"]) <= 200


class TestAnalyticsStats:
    def test_get_stats_returns_dict(self, analytics: Analytics) -> None:
        stats = analytics.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_event_counts(self, analytics: Analytics, db: _MockDB) -> None:
        analytics.log_tool_use("coder_assist", "q1", True, 10.0)
        analytics.log_tool_use("researcher", "q2", False, 20.0)
        analytics.log_intent("q1", "coding", 0.9, "coder_assist")
        stats = analytics.get_stats()
        assert stats["event_counts"].get("tool_use", 0) == 2
        assert stats["event_counts"].get("intent", 0) == 1

    def test_get_stats_tool_usage(self, analytics: Analytics, db: _MockDB) -> None:
        analytics.log_tool_use("coder_assist", "q1", True, 10.0)
        analytics.log_tool_use("coder_assist", "q2", True, 15.0)
        analytics.log_tool_use("researcher", "q3", True, 5.0)
        stats = analytics.get_stats()
        assert stats["tool_usage"]["coder_assist"] == 2
        assert stats["tool_usage"]["researcher"] == 1

    def test_get_stats_no_db(self) -> None:
        an = Analytics(db=None, enabled=True)
        stats = an.get_stats()
        assert "error" in stats
