"""FastLLM: Always-on singleton wrapping TinyLlama 1.1B Q4.

RAM: ~350 MB  |  Speed: 20-30 tok/s on Intel 2017 Mac
Purpose: Immediate routing, brief responses, wake acknowledgments.

Model file: models/llm/tinyllama-1.1b-chat-q4_k_m.gguf
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("models/llm/tinyllama-1.1b-chat-q4_k_m.gguf")

_LLAMA_CPP_AVAILABLE = False
try:
    from llama_cpp import Llama  # type: ignore
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Mock responses
# ---------------------------------------------------------------------------

_ACKNOWLEDGE_RESPONSES = [
    "Got it!",
    "On it!",
    "Sure thing!",
    "Looking into that...",
    "Understood!",
    "Roger that!",
]

_MOCK_GENERATE_PREFIX = "[mock-fast] "


# ---------------------------------------------------------------------------
# FastLLM singleton
# ---------------------------------------------------------------------------

class FastLLM:
    """Always-on singleton LLM for fast routing and brief responses.

    Wraps TinyLlama 1.1B Q4 when available; falls back to deterministic
    mock responses when the model file is absent or llama-cpp-python is
    not installed.
    """

    _instance: Optional["FastLLM"] = None

    def __new__(cls) -> "FastLLM":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        self._initialised: bool = True
        self._model: Any = None
        self._mock_mode: bool = True
        self._load()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Attempt to load the TinyLlama model; fall back to mock on failure."""
        if not _LLAMA_CPP_AVAILABLE:
            logger.info("FastLLM: llama-cpp-python not available, using mock mode")
            return
        model_path = _MODEL_PATH
        if not model_path.exists():
            logger.info("FastLLM: model not found at %s, using mock mode", model_path)
            return
        try:
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=512,
                n_threads=4,
                verbose=False,
            )
            self._mock_mode = False
            logger.info("FastLLM: loaded TinyLlama from %s", model_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("FastLLM: failed to load model (%s), using mock mode", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acknowledge(self, query: str) -> str:  # noqa: ARG002
        """Return an instant 1-2 word acknowledgment.

        Args:
            query: The user query (used only in non-mock mode for context).

        Returns:
            Short acknowledgment string, e.g. "Got it!" or "On it!".
        """
        if self._mock_mode:
            return random.choice(_ACKNOWLEDGE_RESPONSES)
        prompt = (
            "<|system|>You are a brief assistant. Reply with 1-3 words only.</s>"
            f"<|user|>Acknowledge this request concisely: {query[:80]}</s>"
            "<|assistant|>"
        )
        return self._infer(prompt, max_tokens=8).strip() or random.choice(_ACKNOWLEDGE_RESPONSES)

    def route(self, query: str) -> int:
        """Return a complexity score 0-10 for routing purposes.

        0-3  → FastLLM handles alone
        4-6  → moderate; FastLLM draft + HeavyLLM refine
        7-10 → complex; HeavyLLM full answer

        Args:
            query: The user query.

        Returns:
            Integer in the range [0, 10].
        """
        if self._mock_mode:
            return _heuristic_complexity(query)
        prompt = (
            "<|system|>Rate query complexity 0-10. Reply with a single integer only.</s>"
            f"<|user|>{query[:120]}</s>"
            "<|assistant|>"
        )
        raw = self._infer(prompt, max_tokens=4).strip()
        try:
            score = int("".join(c for c in raw if c.isdigit())[:2] or "0")
            return max(0, min(10, score))
        except ValueError:
            return _heuristic_complexity(query)

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        """Generate a fast short response.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """
        if self._mock_mode:
            return f"{_MOCK_GENERATE_PREFIX}{prompt[:60]}..."
        return self._infer(prompt, max_tokens=max_tokens)

    def summarize(self, long_text: str) -> str:
        """Return a concise summary of *long_text*.

        Args:
            long_text: Text to summarize.

        Returns:
            A shorter summary string.
        """
        if self._mock_mode:
            words = long_text.split()
            # Return roughly first 20 words as a mock summary
            return " ".join(words[:20]) + ("..." if len(words) > 20 else "")
        prompt = (
            "<|system|>Summarise the following text in 1-2 sentences.</s>"
            f"<|user|>{long_text[:800]}</s>"
            "<|assistant|>"
        )
        return self._infer(prompt, max_tokens=80)

    @property
    def is_mock(self) -> bool:
        """True when the model is not loaded and mock responses are used."""
        return self._mock_mode

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer(self, prompt: str, max_tokens: int) -> str:
        """Run inference and return the generated text."""
        try:
            result = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.1,
                stop=["</s>", "<|user|>", "<|system|>"],
                echo=False,
            )
            return result["choices"][0]["text"]
        except Exception as exc:  # pragma: no cover
            logger.warning("FastLLM inference error: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Complexity heuristic (used in mock mode and by IntentEngine)
# ---------------------------------------------------------------------------

# Import shared keyword sets from intent_engine to avoid duplication.
# Use a try/except to handle potential circular-import edge cases at startup.
try:
    from agent_core.intent_engine import (  # type: ignore
        _COMPLEX_QUERY_KEYWORDS as _HIGH_COMPLEXITY_KEYWORDS,
        _SIMPLE_QUERY_KEYWORDS as _LOW_COMPLEXITY_KEYWORDS,
    )
except ImportError:
    _HIGH_COMPLEXITY_KEYWORDS: frozenset[str] = frozenset({
        "multi-step", "step by step", "step-by-step", "explain", "research",
        "implement", "design", "architecture", "analyse", "analyze", "debug",
        "code", "script", "algorithm", "compare", "evaluate", "review",
        "translate", "prove", "theorem", "calculate", "derive",
        "write a", "build a", "create a", "generate a", "help me understand",
        "why does", "how does",
    })
    _LOW_COMPLEXITY_KEYWORDS: frozenset[str] = frozenset({
        "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
        "status", "ping", "ok", "yes", "no", "sure",
    })


def _heuristic_complexity(query: str) -> int:
    """Estimate query complexity via keyword heuristics (0-10)."""
    lower = query.lower()
    for kw in _LOW_COMPLEXITY_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", lower):
            return 1
    high_hits = 0
    for kw in _HIGH_COMPLEXITY_KEYWORDS:
        if " " in kw:
            if kw in lower:
                high_hits += 1
        else:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower):
                high_hits += 1
    if high_hits >= 4:
        return 9
    if high_hits == 3:
        return 8
    if high_hits == 2:
        return 7
    if high_hits == 1:
        return 5
    # Length-based fallback
    word_count = len(query.split())
    if word_count > 30:
        return 5
    if word_count > 10:
        return 3
    return 2


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def get_fast_llm() -> FastLLM:
    """Return the process-wide FastLLM singleton."""
    return FastLLM()
