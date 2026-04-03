"""Homework tutor tool: educational explanations via LLM."""

from __future__ import annotations

import logging
from typing import Any

from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

_SUBJECT_PROMPTS: dict[str, str] = {
    "math": "You are a patient, expert math tutor. Explain step-by-step.",
    "science": "You are a clear, engaging science tutor. Use examples.",
    "history": "You are an insightful history tutor. Provide context and significance.",
    "english": "You are an encouraging English/literature tutor.",
    "coding": "You are a friendly programming mentor who teaches with simple analogies.",
    "default": "You are a knowledgeable, patient tutor. Explain clearly and step-by-step.",
}

_SUBJECT_KEYWORDS: dict[str, list[str]] = {
    "math": ["math", "derivative", "integral", "algebra", "geometry", "calculus", "equation", "formula", "trigonometry"],
    "science": ["science", "biology", "chemistry", "physics", "atom", "molecule", "force", "energy", "evolution"],
    "history": ["history", "war", "civilization", "emperor", "revolution", "century", "ancient", "historical"],
    "english": ["essay", "poem", "grammar", "sentence", "verb", "noun", "literature", "shakespeare"],
    "coding": ["code", "program", "algorithm", "function", "variable", "loop", "class"],
}


class HomeworkTutor(BaseTool):
    """Answers homework and learning questions with subject-aware LLM prompting."""

    name = "homework_tutor"
    description = "Educational tutor for homework help in math, science, history, coding and more."

    def run(self, query: str, **kwargs: Any) -> str:
        subject = self._detect_subject(query)
        system_prompt = _SUBJECT_PROMPTS.get(subject, _SUBJECT_PROMPTS["default"])
        full_prompt = f"{system_prompt}\n\nStudent question: {query}\n\nTutor answer:"

        if self._llm is not None:
            try:
                return self._llm.generate(full_prompt, max_tokens=400)
            except Exception as exc:
                logger.error("LLM tutor failed: %s", exc)

        # Mock fallback
        return (
            f"[HomeworkTutor MOCK - {subject}] Great question! "
            f"Here is a structured answer for: '{query}'. "
            "Step 1: Understand the concept. Step 2: Apply the method. Step 3: Verify your answer. "
            "(Connect a real LLM model for full explanations.)"
        )

    def _detect_subject(self, query: str) -> str:
        lower = query.lower()
        for subject, keywords in _SUBJECT_KEYWORDS.items():
            for kw in keywords:
                if kw in lower:
                    return subject
        return "default"
