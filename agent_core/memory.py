"""Conversation memory with summary compression and SQLite persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    role: str          # "user" or "assistant"
    content: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class ConversationMemory:
    """Stores recent conversation turns and compresses old ones into a summary."""

    def __init__(
        self,
        max_turns: int = 20,
        summary_threshold: int = 15,
        llm_adapter: Any | None = None,
        db: Any | None = None,
        session_id: str = "default",
    ) -> None:
        self.max_turns = max_turns
        self.summary_threshold = summary_threshold
        self._llm = llm_adapter
        self._db = db
        self.session_id = session_id

        self._turns: list[Turn] = []
        self._summary: str = ""
        logger.info(
            "ConversationMemory initialised (session=%s, max_turns=%d, summary_threshold=%d)",
            session_id,
            max_turns,
            summary_threshold,
        )

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))
        if len(self._turns) >= self.summary_threshold:
            self._maybe_summarise()

    def get_recent(self, n: int | None = None) -> list[Turn]:
        turns = self._turns[-(n or self.max_turns):]
        return turns

    def get_context_string(self) -> str:
        """Return a formatted context block for LLM prompting."""
        parts: list[str] = []
        if self._summary:
            parts.append(f"[Summary of earlier conversation]\n{self._summary}\n")
        for turn in self.get_recent():
            parts.append(f"{turn.role.capitalize()}: {turn.content}")
        return "\n".join(parts)

    def clear(self) -> None:
        self._turns.clear()
        self._summary = ""

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def summary(self) -> str:
        return self._summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_db(self) -> None:
        if self._db is None:
            logger.debug("No DB attached — skipping memory persistence")
            return
        data = {
            "turns": [asdict(t) for t in self._turns],
            "summary": self._summary,
        }
        self._db.upsert_memory(self.session_id, json.dumps(data))
        logger.debug("Memory saved to DB for session=%s", self.session_id)

    def load_from_db(self) -> None:
        if self._db is None:
            return
        raw = self._db.get_memory(self.session_id)
        if raw is None:
            return
        try:
            data: dict[str, Any] = json.loads(raw)
            self._turns = [Turn(**t) for t in data.get("turns", [])]
            self._summary = data.get("summary", "")
            logger.info("Memory loaded from DB for session=%s (%d turns)", self.session_id, len(self._turns))
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load memory from DB: %s", exc)

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def _maybe_summarise(self) -> None:
        """Compress oldest turns into a summary string."""
        to_summarise = self._turns[: self.summary_threshold // 2]
        self._turns = self._turns[self.summary_threshold // 2:]

        if self._llm is not None:
            prompt = (
                "Summarise the following conversation turns concisely (2-3 sentences):\n"
                + "\n".join(f"{t.role}: {t.content}" for t in to_summarise)
            )
            try:
                new_summary = self._llm.generate(prompt, max_tokens=150).strip()
            except Exception as exc:
                logger.warning("LLM summarisation failed: %s", exc)
                new_summary = self._simple_summary(to_summarise)
        else:
            new_summary = self._simple_summary(to_summarise)

        if self._summary:
            self._summary = f"{self._summary} | {new_summary}"
        else:
            self._summary = new_summary

        logger.info(
            "Summarised %d old turns for session=%s", len(to_summarise), self.session_id
        )

    @staticmethod
    def _simple_summary(turns: list[Turn]) -> str:
        topics = [t.content[:60] for t in turns if t.role == "user"]
        return "Earlier topics: " + "; ".join(topics[:5])
