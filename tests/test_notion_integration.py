"""Tests for NotionIntegration (mock mode — no API calls)."""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from tools.notion_integration import (
    NotionIntegration,
    Assignment,
    NotePage,
    _NotionCache,
)


def _temp_db() -> str:
    """Return a path for a temporary SQLite database (deleted when process exits)."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# _NotionCache tests
# ---------------------------------------------------------------------------

class TestNotionCache:
    def _make_cache(self) -> _NotionCache:
        return _NotionCache(_temp_db())

    def test_upsert_and_get_assignments(self) -> None:
        cache = self._make_cache()
        a = Assignment(
            id="page-1",
            title="Write essay",
            due_date=date.today() + timedelta(days=3),
            status="in_progress",
            subject="English",
        )
        cache.upsert_assignment(a)
        results = cache.get_assignments()
        assert len(results) == 1
        assert results[0].title == "Write essay"
        assert results[0].subject == "English"

    def test_upsert_replaces_existing(self) -> None:
        cache = self._make_cache()
        a = Assignment(id="p1", title="Original")
        cache.upsert_assignment(a)
        a2 = Assignment(id="p1", title="Updated")
        cache.upsert_assignment(a2)
        results = cache.get_assignments()
        assert len(results) == 1
        assert results[0].title == "Updated"

    def test_search_notes_by_title(self) -> None:
        cache = self._make_cache()
        note = NotePage(id="n1", title="Calculus notes", content="derivatives and integrals")
        cache.upsert_note(note)
        results = cache.search_notes("calculus")
        assert len(results) == 1
        assert results[0].title == "Calculus notes"

    def test_search_notes_by_content(self) -> None:
        cache = self._make_cache()
        note = NotePage(id="n2", title="Math", content="The derivative of x^2 is 2x")
        cache.upsert_note(note)
        results = cache.search_notes("derivative")
        assert len(results) == 1

    def test_search_notes_no_match(self) -> None:
        cache = self._make_cache()
        note = NotePage(id="n3", title="History", content="World War 2")
        cache.upsert_note(note)
        results = cache.search_notes("calculus")
        assert len(results) == 0

    def test_cache_age_on_empty_table(self) -> None:
        cache = self._make_cache()
        age = cache.cache_age_seconds("assignments")
        assert age == float("inf")

    def test_set_and_get_meta(self) -> None:
        cache = self._make_cache()
        cache.set_meta("last_sync", "2026-01-01T00:00:00")
        assert cache.get_meta("last_sync") == "2026-01-01T00:00:00"
        assert cache.get_meta("missing", "default") == "default"


# ---------------------------------------------------------------------------
# Assignment model tests
# ---------------------------------------------------------------------------

class TestAssignment:
    def test_days_until_due_future(self) -> None:
        a = Assignment("id", "task", due_date=date.today() + timedelta(days=5))
        assert a.days_until_due() == 5

    def test_days_until_due_today(self) -> None:
        a = Assignment("id", "task", due_date=date.today())
        assert a.days_until_due() == 0
        assert a.is_due_today()

    def test_days_until_due_past(self) -> None:
        a = Assignment("id", "task", due_date=date.today() - timedelta(days=2))
        assert a.days_until_due() == -2

    def test_days_until_due_no_date(self) -> None:
        a = Assignment("id", "task", due_date=None)
        assert a.days_until_due() is None
        assert not a.is_due_today()


# ---------------------------------------------------------------------------
# NotionIntegration (mock mode — no network) tests
# ---------------------------------------------------------------------------

class TestNotionIntegrationMock:
    def _make_integration(self) -> NotionIntegration:
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        f.close()
        # No token → mock mode
        return NotionIntegration(token="", cache_db_path=f.name)

    @pytest.mark.asyncio
    async def test_get_assignments_returns_empty_in_mock(self) -> None:
        ni = self._make_integration()
        assignments = await ni.get_assignments()
        assert isinstance(assignments, list)

    @pytest.mark.asyncio
    async def test_get_due_today_empty(self) -> None:
        ni = self._make_integration()
        result = await ni.get_due_today()
        assert result == []

    @pytest.mark.asyncio
    async def test_deadline_alerts_empty(self) -> None:
        ni = self._make_integration()
        alerts = await ni.deadline_alerts()
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_search_notes_empty_cache(self) -> None:
        ni = self._make_integration()
        results = await ni.search_notes("calculus")
        assert results == []

    @pytest.mark.asyncio
    async def test_upcoming_with_cached_data(self) -> None:
        ni = self._make_integration()
        # Seed the cache directly
        a = Assignment("p1", "Essay", due_date=date.today() + timedelta(days=2))
        ni._cache.upsert_assignment(a)
        upcoming = await ni.get_upcoming(days=7)
        assert len(upcoming) == 1

    @pytest.mark.asyncio
    async def test_deadline_alerts_include_overdue(self) -> None:
        ni = self._make_integration()
        a = Assignment("p2", "Late homework", due_date=date.today() - timedelta(days=3))
        ni._cache.upsert_assignment(a)
        alerts = await ni.deadline_alerts()
        assert any("OVERDUE" in alert for alert in alerts)

    @pytest.mark.asyncio
    async def test_deadline_alerts_include_today(self) -> None:
        ni = self._make_integration()
        a = Assignment("p3", "Due now", due_date=date.today())
        ni._cache.upsert_assignment(a)
        alerts = await ni.deadline_alerts()
        assert any("TODAY" in alert for alert in alerts)

    def test_extract_title_helper(self) -> None:
        ni = self._make_integration()
        props = {
            "Name": {"title": [{"plain_text": "My Task"}]}
        }
        assert ni._extract_title(props) == "My Task"

    def test_extract_title_empty(self) -> None:
        ni = self._make_integration()
        assert ni._extract_title({}) == ""
