"""Self-improvement learning module for Lopen.

Logs interactions, analyzes patterns, improves responses over time.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "storage/interactions.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    user_input  TEXT    NOT NULL,
    response    TEXT    NOT NULL,
    rating      INTEGER,
    interface   TEXT    NOT NULL DEFAULT 'cli',
    topic       TEXT
);

CREATE TABLE IF NOT EXISTS learned_patterns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT    NOT NULL,
    pattern     TEXT    NOT NULL,
    improvement TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    UNIQUE(topic, pattern)
);
"""


# ---------------------------------------------------------------------------
# AgentLearner
# ---------------------------------------------------------------------------

class AgentLearner:
    """Records interactions and applies learned improvements to responses.

    Uses a local SQLite database so no external services are required.
    All heavy analysis runs in-process with stdlib only.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        """Initialise the learner and create the database if necessary.

        Args:
            db_path: Path to the SQLite database file.  Parent directories
                     are created automatically.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("AgentLearner initialised (db=%s)", self._db_path)

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not yet exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Interaction logging
    # ------------------------------------------------------------------

    def log_interaction(
        self,
        user_input: str,
        response: str,
        rating: int | None = None,
        interface: str = "cli",
    ) -> int:
        """Store an interaction in the database.

        Args:
            user_input: The user's original message.
            response: The agent's reply.
            rating: Optional quality rating (e.g. 1-5).
            interface: Source interface (``"cli"``, ``"whatsapp"``, etc.).

        Returns:
            The rowid of the inserted record.
        """
        topic = _extract_topic(user_input)
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO interactions (timestamp, user_input, response, rating, interface, topic) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (now, user_input, response, rating, interface, topic),
            )
            row_id: int = cursor.lastrowid  # type: ignore[assignment]
        logger.debug("AgentLearner: logged interaction id=%d rating=%s", row_id, rating)
        return row_id

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_low_rated(self, days: int = 7, min_rating: int = 3) -> list[dict[str, Any]]:
        """Return poorly-rated interactions from the last *days* days.

        Args:
            days: Look-back window in days.
            min_rating: Interactions with a rating strictly less than this
                        value are considered low-rated.

        Returns:
            List of dicts with keys ``id``, ``timestamp``, ``user_input``,
            ``response``, ``rating``, ``interface``, ``topic``.
        """
        since = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM interactions "
                "WHERE rating IS NOT NULL AND rating < ? AND timestamp >= ? "
                "ORDER BY timestamp DESC",
                (min_rating, since),
            ).fetchall()
        return [dict(r) for r in rows]

    def learn_from_feedback(self) -> int:
        """Analyse low-rated interactions and store improvement patterns.

        Returns:
            Number of new patterns stored.
        """
        low_rated = self.get_low_rated()
        if not low_rated:
            logger.info("AgentLearner: no low-rated interactions to learn from")
            return 0

        # Group by topic
        topic_inputs: dict[str, list[str]] = {}
        for row in low_rated:
            topic = row.get("topic") or "general"
            topic_inputs.setdefault(topic, []).append(row["user_input"])

        new_count = 0
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        with self._connect() as conn:
            for topic, inputs in topic_inputs.items():
                # Derive a simple pattern from common words
                pattern = _common_words_pattern(inputs)
                improvement = (
                    f"For '{topic}' queries matching '{pattern}': "
                    "provide more detail, include examples, and verify accuracy."
                )
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO learned_patterns "
                        "(topic, pattern, improvement, created_at) VALUES (?, ?, ?, ?)",
                        (topic, pattern, improvement, now),
                    )
                    new_count += conn.execute(
                        "SELECT changes()"
                    ).fetchone()[0]
                except sqlite3.Error as exc:
                    logger.warning("AgentLearner: could not store pattern: %s", exc)

        logger.info("AgentLearner: stored %d new improvement patterns", new_count)
        return new_count

    def improve_response(self, topic: str, response: str) -> str:
        """Apply any learned improvements for *topic* to *response*.

        Args:
            topic: The topic/domain of the response.
            response: The original response text.

        Returns:
            Potentially improved response string.  If no patterns are
            found the original response is returned unchanged.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT improvement FROM learned_patterns WHERE topic = ? LIMIT 5",
                (topic,),
            ).fetchall()

        if not rows:
            return response

        notes = "; ".join(r["improvement"] for r in rows)
        improved = f"{response}\n\n[Improvement note: {notes}]"
        logger.debug("AgentLearner: applied improvements for topic '%s'", topic)
        return improved

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about the learning database.

        Returns:
            Dict with keys: ``total_interactions``, ``rated_interactions``,
            ``avg_rating``, ``low_rated_count``, ``learned_patterns``,
            ``top_topics``.
        """
        with self._connect() as conn:
            total: int = conn.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()[0]
            rated: int = conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE rating IS NOT NULL"
            ).fetchone()[0]
            avg_row = conn.execute(
                "SELECT AVG(rating) FROM interactions WHERE rating IS NOT NULL"
            ).fetchone()
            avg_rating: float | None = round(avg_row[0], 2) if avg_row[0] is not None else None
            low_rated: int = conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE rating IS NOT NULL AND rating < 3"
            ).fetchone()[0]
            patterns: int = conn.execute(
                "SELECT COUNT(*) FROM learned_patterns"
            ).fetchone()[0]
            topic_rows = conn.execute(
                "SELECT topic, COUNT(*) as cnt FROM interactions "
                "WHERE topic IS NOT NULL "
                "GROUP BY topic ORDER BY cnt DESC LIMIT 5"
            ).fetchall()
            top_topics = [{"topic": r["topic"], "count": r["cnt"]} for r in topic_rows]

        return {
            "total_interactions": total,
            "rated_interactions": rated,
            "avg_rating": avg_rating,
            "low_rated_count": low_rated,
            "learned_patterns": patterns,
            "top_topics": top_topics,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "it", "to", "of", "and", "or", "for",
    "in", "on", "at", "my", "me", "i", "you", "can", "will", "do",
    "please", "help", "want", "need", "get", "make", "show",
})


def _extract_topic(text: str) -> str:
    """Extract a simple topic label from the query text."""
    tokens = [
        t for t in re.findall(r"\b[a-z][a-z0-9]*\b", text.lower())
        if t not in _STOP_WORDS and len(t) > 2
    ]
    if not tokens:
        return "general"
    return tokens[0]


def _common_words_pattern(texts: list[str]) -> str:
    """Find the most common word across a list of texts."""
    counts: dict[str, int] = {}
    for text in texts:
        for word in re.findall(r"\b[a-z][a-z0-9]*\b", text.lower()):
            if word not in _STOP_WORDS and len(word) > 2:
                counts[word] = counts.get(word, 0) + 1
    if not counts:
        return "general"
    return max(counts, key=lambda w: counts[w])
