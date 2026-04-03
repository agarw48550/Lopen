"""Unit tests for SQLite storage layer."""

import pytest
import os
from storage.database import SQLiteDB


@pytest.fixture
def db(tmp_path) -> SQLiteDB:
    """Create a temporary SQLiteDB for each test."""
    return SQLiteDB(db_path=str(tmp_path / "test.db"))


class TestTaskCRUD:
    def test_insert_and_retrieve_task(self, db: SQLiteDB) -> None:
        db.insert_task("task-1", "coding", "write a function")
        tasks = db.get_tasks()
        assert len(tasks) == 1
        assert tasks[0]["id"] == "task-1"
        assert tasks[0]["intent"] == "coding"
        assert tasks[0]["status"] == "pending"

    def test_update_task_status_done(self, db: SQLiteDB) -> None:
        db.insert_task("task-2", "research", "find AI news")
        db.update_task_status("task-2", "done", result="Some results")
        tasks = db.get_tasks()
        assert tasks[0]["status"] == "done"
        assert tasks[0]["result"] == "Some results"

    def test_update_task_status_failed(self, db: SQLiteDB) -> None:
        db.insert_task("task-3", "homework", "math problem")
        db.update_task_status("task-3", "failed", error="LLM error")
        tasks = db.get_tasks()
        assert tasks[0]["status"] == "failed"
        assert tasks[0]["error"] == "LLM error"

    def test_insert_duplicate_ignored(self, db: SQLiteDB) -> None:
        db.insert_task("task-x", "general", "hello")
        db.insert_task("task-x", "general", "hello again")  # OR IGNORE
        assert len(db.get_tasks()) == 1

    def test_get_tasks_limit(self, db: SQLiteDB) -> None:
        for i in range(5):
            db.insert_task(f"t{i}", "general", f"payload {i}")
        tasks = db.get_tasks(limit=3)
        assert len(tasks) == 3


class TestMemoryCRUD:
    def test_upsert_and_get_memory(self, db: SQLiteDB) -> None:
        db.upsert_memory("session-1", '{"turns": [], "summary": ""}')
        raw = db.get_memory("session-1")
        assert raw is not None
        assert "turns" in raw

    def test_get_nonexistent_memory(self, db: SQLiteDB) -> None:
        assert db.get_memory("no-session") is None

    def test_upsert_updates_existing(self, db: SQLiteDB) -> None:
        db.upsert_memory("s1", '{"turns": [], "summary": "old"}')
        db.upsert_memory("s1", '{"turns": [], "summary": "new"}')
        raw = db.get_memory("s1")
        assert "new" in raw


class TestSettingsCRUD:
    def test_set_and_get_setting(self, db: SQLiteDB) -> None:
        db.set_setting("theme", "dark")
        val = db.get_setting("theme")
        assert val == "dark"

    def test_get_nonexistent_returns_default(self, db: SQLiteDB) -> None:
        assert db.get_setting("missing", default="fallback") == "fallback"

    def test_set_complex_value(self, db: SQLiteDB) -> None:
        db.set_setting("config", {"a": 1, "b": [1, 2, 3]})
        val = db.get_setting("config")
        assert val == {"a": 1, "b": [1, 2, 3]}

    def test_upsert_setting(self, db: SQLiteDB) -> None:
        db.set_setting("counter", 1)
        db.set_setting("counter", 2)
        assert db.get_setting("counter") == 2


class TestHeartbeatCRUD:
    def test_record_and_retrieve_heartbeat(self, db: SQLiteDB) -> None:
        db.record_heartbeat("orchestrator", True)
        records = db.get_recent_heartbeats("orchestrator")
        assert len(records) == 1
        assert records[0]["service"] == "orchestrator"
        assert records[0]["healthy"] == 1

    def test_multiple_heartbeats_limited(self, db: SQLiteDB) -> None:
        for _ in range(15):
            db.record_heartbeat("svc", True)
        records = db.get_recent_heartbeats("svc", limit=5)
        assert len(records) == 5
