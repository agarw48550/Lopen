"""Intent classification and task decomposition for the Lopen agent."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    HOMEWORK = "homework"
    RESEARCH = "research"
    CODING = "coding"
    DESKTOP = "desktop"
    COMMUNICATION = "communication"
    VOICE = "voice"
    FILE_OPS = "file_ops"
    GENERAL = "general"


@dataclass
class TaskPlan:
    intent: Intent
    original_query: str
    steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


_KEYWORD_MAP: dict[Intent, list[str]] = {
    Intent.HOMEWORK: [
        "homework", "explain", "teach", "tutor", "lesson", "derivative", "integral",
        "algebra", "geometry", "biology", "chemistry", "history", "essay",
        "equation", "formula", "definition", "what is", "how does", "why does",
        "solve", "calculate", "math", "science", "physics",
    ],
    Intent.RESEARCH: [
        "research", "find info", "look up", "search for", "investigate",
        "climate change", "tell me about", "what happened", "news about",
        "latest on", "article", "information about", "who is", "facts about",
    ],
    Intent.CODING: [
        "code", "program", "script", "function", "class", "debug",
        "python", "javascript", "typescript", "rust", "golang", "java",
        "write a", "implement", "refactor", "review my code", "fix this",
        "explain this code", "what does this code", "error in code",
    ],
    Intent.DESKTOP: [
        "organize", "desktop", "clean up", "sort files", "move files",
        "finder", "folder", "rename files", "delete old", "tidy",
        "my desktop is", "organize my",
    ],
    Intent.COMMUNICATION: [
        "send message", "whatsapp", "text", "contact", "reply to",
        "message", "notify", "alert", "communicate",
    ],
    Intent.VOICE: [
        "speak", "say", "voice", "listen", "wake word", "microphone",
        "audio", "tts", "asr",
    ],
    Intent.FILE_OPS: [
        "read file", "write file", "create file", "open file", "save",
        "list files", "find file", "search file", "delete file",
        "documents", "downloads",
    ],
}


class Planner:
    """Classifies user intents and decomposes tasks into executable steps."""

    def __init__(self, llm_adapter: Any | None = None) -> None:
        self._llm = llm_adapter
        logger.info("Planner initialised (llm_adapter=%s)", type(llm_adapter).__name__ if llm_adapter else "None")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_intent(self, query: str) -> Intent:
        """Classify a user query into an Intent using keyword matching with LLM fallback."""
        normalised = query.lower()
        scores: dict[Intent, int] = {intent: 0 for intent in Intent}

        for intent, keywords in _KEYWORD_MAP.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", normalised):
                    scores[intent] += 1

        best_intent = max(scores, key=lambda i: scores[i])
        best_score = scores[best_intent]

        if best_score == 0 and self._llm is not None:
            logger.debug("No keyword match for %r — falling back to LLM classification", query)
            best_intent = self._llm_classify(query)
        elif best_score == 0:
            best_intent = Intent.GENERAL

        logger.info("Classified %r as %s (score=%d)", query, best_intent.value, best_score)
        return best_intent

    def decompose(self, query: str, intent: Intent | None = None) -> TaskPlan:
        """Break a query into ordered execution steps."""
        if intent is None:
            intent = self.classify_intent(query)

        steps = self._default_steps(intent, query)
        plan = TaskPlan(intent=intent, original_query=query, steps=steps)
        logger.debug("Task plan: %s", plan)
        return plan

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _llm_classify(self, query: str) -> Intent:
        """Use the LLM adapter to classify intent when keyword matching fails."""
        prompt = (
            "Classify the following user query into exactly one of these intents: "
            "homework, research, coding, desktop, communication, voice, file_ops, general.\n"
            f"Query: {query}\n"
            "Reply with a single word."
        )
        try:
            response = self._llm.generate(prompt, max_tokens=10).strip().lower()
            return Intent(response)
        except (ValueError, Exception) as exc:
            logger.warning("LLM intent classification failed: %s", exc)
            return Intent.GENERAL

    @staticmethod
    def _default_steps(intent: Intent, query: str) -> list[str]:
        templates: dict[Intent, list[str]] = {
            Intent.HOMEWORK: [
                "Understand the subject matter from the query",
                "Retrieve relevant educational context",
                "Generate step-by-step explanation",
                "Verify the answer for accuracy",
            ],
            Intent.RESEARCH: [
                "Extract key search terms from the query",
                "Perform web search",
                "Aggregate and summarise results",
                "Return findings to user",
            ],
            Intent.CODING: [
                "Parse the programming task or question",
                "Generate or explain code",
                "Review for correctness",
                "Return code with explanation",
            ],
            Intent.DESKTOP: [
                "Identify target directory / desktop",
                "Analyse file types present",
                "Group files by category",
                "Move files to organised folders",
            ],
            Intent.COMMUNICATION: [
                "Identify target contact and message body",
                "Check WhatsApp service availability",
                "Send message via bridge",
                "Confirm delivery",
            ],
            Intent.VOICE: [
                "Check voice service state",
                "Process voice command or query",
                "Route to appropriate handler",
            ],
            Intent.FILE_OPS: [
                "Validate file path against allowed directories",
                "Perform requested file operation",
                "Return result to user",
            ],
            Intent.GENERAL: [
                "Process query with LLM",
                "Return response to user",
            ],
        }
        return templates.get(intent, ["Process query", "Return response"])
