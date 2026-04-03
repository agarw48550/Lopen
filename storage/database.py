"""SQLite storage layer for Lopen: tasks, memory, settings, heartbeats, logs."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "storage/lopen.db"


class SQLiteDB:
    """Thread-safe SQLite wrapper with tables for tasks, memory, settings, heartbeats, logs."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("SQLiteDB initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    intent TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    result TEXT,
                    error TEXT,
                    priority INTEGER DEFAULT 5,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );

                CREATE TABLE IF NOT EXISTS memory (
                    session_id TEXT PRIMARY KEY,
                    data JSON NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS heartbeats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service TEXT NOT NULL,
                    healthy INTEGER NOT NULL,
                    error TEXT,
                    recorded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
            """)

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def insert_task(self, task_id: str, intent: str, payload: str, priority: int = 5) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO tasks (id, intent, payload, status, priority, created_at) VALUES (?,?,?,?,?,?)",
                (task_id, intent, payload, "pending", priority, datetime.now(timezone.utc).isoformat()),
            )

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """UPDATE tasks SET status=?, result=?, error=?,
                   started_at=COALESCE(started_at, ?),
                   finished_at=CASE WHEN ? IN ('done','failed') THEN ? ELSE finished_at END
                   WHERE id=?""",
                (status, result, error, now, status, now, task_id),
            )

    def get_tasks(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def upsert_memory(self, session_id: str, data: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory (session_id, data, updated_at) VALUES (?,?,?)",
                (session_id, data, datetime.now(timezone.utc).isoformat()),
            )

    def get_memory(self, session_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM memory WHERE session_id=?", (session_id,)
            ).fetchone()
        return row["data"] if row else None

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def set_setting(self, key: str, value: Any) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?,?,?)",
                (key, json.dumps(value), datetime.now(timezone.utc).isoformat()),
            )

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    # ------------------------------------------------------------------
    # Heartbeats
    # ------------------------------------------------------------------

    def record_heartbeat(self, service: str, healthy: bool, error: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO heartbeats (service, healthy, error, recorded_at) VALUES (?,?,?,?)",
                (service, int(healthy), error, datetime.now(timezone.utc).isoformat()),
            )

    def get_recent_heartbeats(self, service: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM heartbeats WHERE service=? ORDER BY recorded_at DESC LIMIT ?",
                (service, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def insert_log(self, level: str, message: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO logs (level, message, recorded_at) VALUES (?,?,?)",
                (level, message, datetime.now(timezone.utc).isoformat()),
            )
