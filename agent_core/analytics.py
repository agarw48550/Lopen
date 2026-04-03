"""Usage analytics and RL evaluation hooks for Lopen.

All events are written to the local SQLite database — no network calls.
The RL reward signal is a simple boolean (was_helpful) that can be used
to improve intent classification or tool selection over time.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event data classes
# ---------------------------------------------------------------------------

@dataclass
class ToolUseEvent:
    tool_name: str
    query: str
    success: bool
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentEvent:
    query: str
    structured_intent: str
    confidence: float
    selected_tool: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class FeedbackEvent:
    tool_name: str
    query: str
    was_helpful: bool
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class Analytics:
    """Lightweight usage logger and RL hook store.

    Events are appended to a dedicated ``analytics`` table in the SQLite DB.
    If the DB is unavailable, events are written to the Python logger only
    (graceful degradation).

    Usage::

        analytics = Analytics(db=sqlite_db)
        analytics.log_tool_use("researcher", "find info on AI", True, 120.0)
        analytics.log_intent("find info on AI", "web research", 0.87, "researcher")
        stats = analytics.get_stats()
    """

    def __init__(self, db: Any | None = None, enabled: bool = True) -> None:
        self._db = db
        self._enabled = enabled
        self._ensure_table()
        logger.info("Analytics initialised (enabled=%s, db=%s)", enabled, db is not None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_tool_use(
        self,
        tool_name: str,
        query: str,
        success: bool,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return
        event = ToolUseEvent(
            tool_name=tool_name,
            query=query[:200],
            success=success,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self._write("tool_use", asdict(event))

    def log_intent(
        self,
        query: str,
        structured_intent: str,
        confidence: float,
        selected_tool: str,
    ) -> None:
        if not self._enabled:
            return
        event = IntentEvent(
            query=query[:200],
            structured_intent=structured_intent[:200],
            confidence=confidence,
            selected_tool=selected_tool,
        )
        self._write("intent", asdict(event))

    def log_feedback(self, tool_name: str, query: str, was_helpful: bool) -> None:
        """Record user feedback for RL signal.

        This is the hook for future reinforcement learning / preference tuning.
        For now it simply persists the signal to the DB.
        """
        if not self._enabled:
            return
        event = FeedbackEvent(
            tool_name=tool_name,
            query=query[:200],
            was_helpful=was_helpful,
        )
        self._write("feedback", asdict(event))

    def get_stats(self) -> dict[str, Any]:
        """Return a summary of usage statistics."""
        if self._db is None:
            return {"error": "no database configured"}
        try:
            rows = self._db.execute(
                "SELECT event_type, COUNT(*) as cnt FROM analytics GROUP BY event_type"
            )
            counts = {row["event_type"]: row["cnt"] for row in (rows or [])}

            # Tool usage breakdown
            tool_rows = self._db.execute(
                "SELECT data FROM analytics WHERE event_type='tool_use' ORDER BY ts DESC LIMIT 200"
            )
            tool_counts: dict[str, int] = {}
            success_counts: dict[str, int] = {}
            for row in (tool_rows or []):
                try:
                    data = json.loads(row["data"])
                    t = data.get("tool_name", "unknown")
                    tool_counts[t] = tool_counts.get(t, 0) + 1
                    if data.get("success"):
                        success_counts[t] = success_counts.get(t, 0) + 1
                except Exception:
                    pass

            return {
                "event_counts": counts,
                "tool_usage": tool_counts,
                "tool_success": success_counts,
            }
        except Exception as exc:
            logger.warning("Analytics get_stats failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write(self, event_type: str, data: dict[str, Any]) -> None:
        """Persist an event to the DB and also log it."""
        logger.debug("Analytics event [%s]: %s", event_type, data)
        if self._db is None:
            return
        try:
            self._db.execute(
                "INSERT INTO analytics (event_type, data, ts) VALUES (?, ?, ?)",
                (event_type, json.dumps(data), time.time()),
            )
        except Exception as exc:
            logger.warning("Analytics write failed: %s", exc)

    def _ensure_table(self) -> None:
        """Create the analytics table if it does not exist."""
        if self._db is None:
            return
        try:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS analytics "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                " event_type TEXT NOT NULL, "
                " data TEXT NOT NULL, "
                " ts REAL NOT NULL)"
            )
        except Exception as exc:
            logger.warning("Could not create analytics table: %s", exc)
