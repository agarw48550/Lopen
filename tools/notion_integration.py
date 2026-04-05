"""Notion integration: read-only bridge to Notion databases for student workflows.

Connects to the Notion API to read:
  - Assignments database (tasks with deadlines)
  - Notes database (full-text content for homework help)
  - Calendar property on tasks (today's schedule / deadline alerts)

Design decisions:
  - Read-only: no writes to prevent accidental data loss
  - Local cache (SQLite via storage layer) updated every hour
  - Async-friendly: all network calls wrapped in asyncio.to_thread
  - Graceful fallback: returns cached data if API is unreachable
  - Memory-conservative: processes pages in small batches (never loads
    the entire database into RAM at once)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

_NOTION_AVAILABLE = False
_NotionClient: Any = None

try:
    from notion_client import Client as _NotionClient  # type: ignore
    _NOTION_AVAILABLE = True
except ImportError:
    pass

_CACHE_TTL_SECONDS = 3600  # 1 hour
_BATCH_SIZE = 50            # pages per Notion API request


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Assignment:
    id: str
    title: str
    due_date: Optional[date] = None
    status: str = "not_started"
    subject: str = ""
    url: str = ""

    # ------------------------------------------------------------------
    def is_due_today(self) -> bool:
        return self.due_date == date.today()

    def days_until_due(self) -> Optional[int]:
        if self.due_date is None:
            return None
        return (self.due_date - date.today()).days


@dataclass
class NotePage:
    id: str
    title: str
    content: str = ""
    last_edited: Optional[datetime] = None
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Local cache (SQLite)
# ---------------------------------------------------------------------------

class _NotionCache:
    """Lightweight SQLite cache so Notion data is available offline."""

    def __init__(self, db_path: str = "storage/notion_cache.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS assignments (
                id TEXT PRIMARY KEY,
                title TEXT,
                due_date TEXT,
                status TEXT,
                subject TEXT,
                url TEXT,
                cached_at REAL
            );
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                last_edited TEXT,
                tags TEXT,
                cached_at REAL
            );
            CREATE TABLE IF NOT EXISTS sync_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Assignments
    # ------------------------------------------------------------------

    def upsert_assignment(self, a: Assignment) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO assignments
              (id, title, due_date, status, subject, url, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                a.id,
                a.title,
                a.due_date.isoformat() if a.due_date else None,
                a.status,
                a.subject,
                a.url,
                time.time(),
            ),
        )
        self._conn.commit()

    def get_assignments(self) -> list[Assignment]:
        rows = self._conn.execute(
            "SELECT id, title, due_date, status, subject, url FROM assignments"
        ).fetchall()
        result = []
        for r in rows:
            due = date.fromisoformat(r[2]) if r[2] else None
            result.append(Assignment(r[0], r[1], due, r[3], r[4], r[5]))
        return result

    def cache_age_seconds(self, table: str) -> float:
        row = self._conn.execute(
            "SELECT MAX(cached_at) FROM " + table  # noqa: S608 — table is internal constant
        ).fetchone()
        if row and row[0]:
            return time.time() - row[0]
        return float("inf")

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    def upsert_note(self, note: NotePage) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO notes
              (id, title, content, last_edited, tags, cached_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                note.id,
                note.title,
                note.content,
                note.last_edited.isoformat() if note.last_edited else None,
                json.dumps(note.tags),
                time.time(),
            ),
        )
        self._conn.commit()

    def search_notes(self, query: str) -> list[NotePage]:
        query_lower = f"%{query.lower()}%"
        rows = self._conn.execute(
            """
            SELECT id, title, content, last_edited, tags
            FROM notes
            WHERE LOWER(title) LIKE ? OR LOWER(content) LIKE ?
            """,
            (query_lower, query_lower),
        ).fetchall()
        return [
            NotePage(
                r[0],
                r[1],
                r[2],
                datetime.fromisoformat(r[3]) if r[3] else None,
                json.loads(r[4]) if r[4] else [],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Sync metadata
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM sync_meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default


# ---------------------------------------------------------------------------
# Main integration class
# ---------------------------------------------------------------------------

class NotionIntegration:
    """Read-only Notion API bridge with local caching.

    Configuration (via environment variables or explicit kwargs):
      NOTION_TOKEN        — Notion integration token (required)
      NOTION_ASSIGNMENTS_DB — Database ID for assignments/homework
      NOTION_NOTES_DB     — Database ID for notes/study materials
    """

    def __init__(
        self,
        token: Optional[str] = None,
        assignments_db_id: Optional[str] = None,
        notes_db_id: Optional[str] = None,
        cache_db_path: str = "storage/notion_cache.db",
        cache_ttl: int = _CACHE_TTL_SECONDS,
    ) -> None:
        self._token = token or os.environ.get("NOTION_TOKEN", "")
        self._assignments_db = assignments_db_id or os.environ.get("NOTION_ASSIGNMENTS_DB", "")
        self._notes_db = notes_db_id or os.environ.get("NOTION_NOTES_DB", "")
        self._cache = _NotionCache(cache_db_path)
        self._cache_ttl = cache_ttl
        self._client: Any = None
        self._mock = not _NOTION_AVAILABLE or not self._token
        if self._mock:
            logger.warning(
                "NotionIntegration in MOCK mode — "
                "install notion-client and set NOTION_TOKEN to enable live sync"
            )
        else:
            self._client = _NotionClient(auth=self._token)
            logger.info("NotionIntegration ready (live mode)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_assignments(self, force_refresh: bool = False) -> list[Assignment]:
        """Return all assignments, refreshing from Notion if cache is stale."""
        if not force_refresh and self._cache.cache_age_seconds("assignments") < self._cache_ttl:
            return self._cache.get_assignments()
        if not self._mock and self._assignments_db:
            await asyncio.to_thread(self._sync_assignments)
        return self._cache.get_assignments()

    async def get_due_today(self) -> list[Assignment]:
        assignments = await self.get_assignments()
        return [a for a in assignments if a.is_due_today()]

    async def get_upcoming(self, days: int = 7) -> list[Assignment]:
        assignments = await self.get_assignments()
        cutoff = date.today() + timedelta(days=days)
        return [
            a
            for a in assignments
            if a.due_date is not None and date.today() <= a.due_date <= cutoff
        ]

    async def search_notes(self, query: str, force_refresh: bool = False) -> list[NotePage]:
        if not force_refresh and self._cache.cache_age_seconds("notes") < self._cache_ttl:
            return self._cache.search_notes(query)
        if not self._mock and self._notes_db:
            await asyncio.to_thread(self._sync_notes)
        return self._cache.search_notes(query)

    async def deadline_alerts(self) -> list[str]:
        """Return human-readable alert strings for imminent deadlines."""
        assignments = await self.get_assignments()
        alerts: list[str] = []
        for a in sorted(assignments, key=lambda x: x.due_date or date.max):
            days = a.days_until_due()
            if days is None:
                continue
            if days < 0:
                alerts.append(f"⚠️  OVERDUE ({-days}d): {a.title}")
            elif days == 0:
                alerts.append(f"🔴 DUE TODAY: {a.title}")
            elif days == 1:
                alerts.append(f"🟡 Due tomorrow: {a.title}")
            elif days <= 3:
                alerts.append(f"🟠 Due in {days}d: {a.title}")
        return alerts

    # ------------------------------------------------------------------
    # Sync helpers (run in thread pool to stay async-friendly)
    # ------------------------------------------------------------------

    def _sync_assignments(self) -> None:
        """Fetch all pages from the assignments database and cache them."""
        if not self._client or not self._assignments_db:
            return
        try:
            cursor: Optional[str] = None
            while True:
                kwargs: dict[str, Any] = {
                    "database_id": self._assignments_db,
                    "page_size": _BATCH_SIZE,
                }
                if cursor:
                    kwargs["start_cursor"] = cursor
                resp = self._client.databases.query(**kwargs)
                for page in resp.get("results", []):
                    assignment = self._parse_assignment(page)
                    if assignment:
                        self._cache.upsert_assignment(assignment)
                if not resp.get("has_more"):
                    break
                cursor = resp.get("next_cursor")
            self._cache.set_meta("assignments_last_sync", datetime.now(timezone.utc).isoformat())
            logger.info("Notion assignments synced")
        except Exception as exc:
            logger.error("Notion assignments sync failed: %s", exc)

    def _sync_notes(self) -> None:
        """Fetch all pages from the notes database and cache them."""
        if not self._client or not self._notes_db:
            return
        try:
            cursor: Optional[str] = None
            while True:
                kwargs: dict[str, Any] = {
                    "database_id": self._notes_db,
                    "page_size": _BATCH_SIZE,
                }
                if cursor:
                    kwargs["start_cursor"] = cursor
                resp = self._client.databases.query(**kwargs)
                for page in resp.get("results", []):
                    note = self._parse_note(page)
                    if note:
                        self._cache.upsert_note(note)
                if not resp.get("has_more"):
                    break
                cursor = resp.get("next_cursor")
            self._cache.set_meta("notes_last_sync", datetime.now(timezone.utc).isoformat())
            logger.info("Notion notes synced")
        except Exception as exc:
            logger.error("Notion notes sync failed: %s", exc)

    # ------------------------------------------------------------------
    # Page parsers
    # ------------------------------------------------------------------

    def _parse_assignment(self, page: dict[str, Any]) -> Optional[Assignment]:
        try:
            props = page.get("properties", {})
            title = self._extract_title(props)
            if not title:
                return None

            due_date: Optional[date] = None
            for key in ("Due Date", "Due", "Deadline", "Date"):
                if key in props and props[key].get("date"):
                    raw = props[key]["date"].get("start", "")
                    if raw:
                        due_date = date.fromisoformat(raw[:10])
                    break

            status = "not_started"
            for key in ("Status", "State", "Progress"):
                if key in props:
                    sel = props[key].get("select") or props[key].get("status", {})
                    if sel:
                        status = (sel.get("name") or "not_started").lower().replace(" ", "_")
                    break

            subject = ""
            for key in ("Subject", "Course", "Class", "Category"):
                if key in props:
                    sel = props[key].get("select") or {}
                    subject = sel.get("name", "")
                    break

            return Assignment(
                id=page["id"],
                title=title,
                due_date=due_date,
                status=status,
                subject=subject,
                url=page.get("url", ""),
            )
        except Exception as exc:
            logger.debug("Failed to parse assignment page: %s", exc)
            return None

    def _parse_note(self, page: dict[str, Any]) -> Optional[NotePage]:
        try:
            props = page.get("properties", {})
            title = self._extract_title(props)
            if not title:
                return None

            last_edited_raw = page.get("last_edited_time", "")
            last_edited = (
                datetime.fromisoformat(last_edited_raw.replace("Z", "+00:00"))
                if last_edited_raw
                else None
            )

            tags: list[str] = []
            for key in ("Tags", "Labels", "Topics"):
                if key in props and props[key].get("multi_select"):
                    tags = [t["name"] for t in props[key]["multi_select"]]
                    break

            # Fetch rich text content from page body
            content = self._fetch_page_content(page["id"])

            return NotePage(
                id=page["id"],
                title=title,
                content=content,
                last_edited=last_edited,
                tags=tags,
            )
        except Exception as exc:
            logger.debug("Failed to parse note page: %s", exc)
            return None

    def _fetch_page_content(self, page_id: str) -> str:
        """Fetch plain text from the page's block children (first 20 blocks)."""
        if not self._client:
            return ""
        try:
            blocks = self._client.blocks.children.list(block_id=page_id, page_size=20)
            texts: list[str] = []
            for block in blocks.get("results", []):
                block_type = block.get("type", "")
                block_data = block.get(block_type, {})
                for rt in block_data.get("rich_text", []):
                    texts.append(rt.get("plain_text", ""))
            return " ".join(texts)[:2000]  # cap at 2000 chars to limit RAM
        except Exception:
            return ""

    @staticmethod
    def _extract_title(props: dict[str, Any]) -> str:
        for key in ("Name", "Title", "title"):
            if key in props:
                rich = props[key].get("title") or props[key].get("rich_text", [])
                if rich:
                    return "".join(r.get("plain_text", "") for r in rich)
        return ""
