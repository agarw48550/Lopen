"""Coder assist tool: code explanation, review, and generation."""

from __future__ import annotations

import logging
import re
from typing import Any

from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

_LANG_KEYWORDS: dict[str, list[str]] = {
    "python": ["python", "def ", "import ", "class ", "pip ", ".py"],
    "javascript": ["javascript", "js", "const ", "let ", "function ", "=>", "node"],
    "typescript": ["typescript", "ts", "interface ", "type ", ": string", ": number"],
    "rust": ["rust", "fn ", "let mut", "cargo ", ".rs"],
    "go": ["golang", " go ", "func ", "package ", ":=", ".go"],
    "java": ["java", "public class", "System.out", "void main"],
    "bash": ["bash", "shell", "#!/bin/", "chmod", "export "],
}

_TASK_PATTERNS: dict[str, list[str]] = {
    "explain": ["explain", "what does", "what is", "how does", "describe"],
    "review": ["review", "check", "improve", "refactor", "optimise", "optimize", "fix"],
    "generate": ["write", "create", "generate", "implement", "make a", "code a"],
    "debug": ["debug", "error", "bug", "exception", "traceback", "not working"],
}


class CoderAssist(BaseTool):
    """Provides code explanation, review, and generation using LLM."""

    name = "coder_assist"
    description = "Code assistant: explain, review, generate, and debug code."

    def run(self, query: str, **kwargs: Any) -> str:
        task = self._detect_task(query)
        lang = self._detect_language(query)
        system = self._build_system_prompt(task, lang)
        full_prompt = f"{system}\n\n{query}\n\n"

        if self._llm is not None:
            try:
                return self._llm.generate(full_prompt, max_tokens=512)
            except Exception as exc:
                logger.error("LLM coder assist failed: %s", exc)

        # Mock fallback
        return (
            f"[CoderAssist MOCK - {task}/{lang}] "
            f"For your {task} request: '{query[:80]}'\n"
            "Connect a real LLM model (see config/models.yaml) for full code assistance.\n"
            + self._mock_example(task, lang)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_task(self, query: str) -> str:
        lower = query.lower()
        for task, patterns in _TASK_PATTERNS.items():
            if any(p in lower for p in patterns):
                return task
        return "general"

    def _detect_language(self, query: str) -> str:
        lower = query.lower()
        for lang, keywords in _LANG_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return lang
        return "python"  # default

    def _build_system_prompt(self, task: str, lang: str) -> str:
        prompts = {
            "explain": f"You are an expert {lang} developer. Explain the following code clearly.",
            "review": f"You are a senior {lang} engineer. Review the code and suggest improvements.",
            "generate": f"You are an expert {lang} developer. Write clean, well-commented {lang} code.",
            "debug": f"You are an expert {lang} debugger. Identify and fix the issue.",
            "general": f"You are an expert {lang} developer. Help with the following.",
        }
        return prompts.get(task, prompts["general"])

    @staticmethod
    def _mock_example(task: str, lang: str) -> str:
        if task == "generate" and lang == "python":
            return "\n```python\ndef example_function(x: int) -> int:\n    \"\"\"Example stub.\"\"\"\n    return x * 2\n```"
        return ""
