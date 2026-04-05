"""Semantic intent recognition engine for Lopen.

Uses lightweight TF-IDF cosine similarity to match open-ended user queries
against available tool descriptions — no model downloads, no GPU, near-zero RAM.
Falls back to an LLM for structured intent summarisation when available.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop words (common English words that carry little signal)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "it", "to", "of", "and", "or", "for", "in",
    "on", "at", "by", "with", "from", "as", "be", "this", "that", "my",
    "me", "i", "you", "we", "they", "can", "will", "do", "does", "did",
    "have", "has", "had", "are", "was", "were", "so", "if", "but", "not",
    "no", "up", "out", "about", "into", "than", "then", "there", "what",
    "when", "where", "how", "who", "which", "its", "also", "just", "some",
    "would", "could", "should", "please", "help", "want", "need", "like",
    "get", "make", "use", "let", "give", "show", "tell", "find", "know",
    "go", "come", "see", "take", "put", "try", "say", "ask",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    """Result of intent analysis for a user query."""

    raw_intent: str
    """The original user query (unmodified)."""

    structured_intent: str
    """A summarised/structured version of the intent."""

    confidence: float
    """Overall confidence score in the range [0.0, 1.0]."""

    suggested_tools: list[str] = field(default_factory=list)
    """Tool names ranked by relevance (most relevant first)."""

    keywords: list[str] = field(default_factory=list)
    """Key tokens extracted from the query."""

    tool_scores: dict[str, float] = field(default_factory=dict)
    """Raw similarity score per tool name."""


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lower-case tokeniser that strips stop words and short tokens."""
    tokens = re.findall(r"\b[a-z][a-z0-9]*\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]


def _build_document(name: str, description: str, tags: list[str]) -> str:
    """Concatenate tool metadata into a searchable document string."""
    tag_str = " ".join(tags)
    # Repeat name and tags a few times to boost their weight
    return f"{name} {name} {description} {tag_str} {tag_str}"


def _tf(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency (normalised by document length)."""
    if not tokens:
        return {}
    total = len(tokens)
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return {w: c / total for w, c in counts.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a * mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Complexity scoring keyword sets (used by IntentEngine.complexity_score)
# ---------------------------------------------------------------------------

_SIMPLE_QUERY_KEYWORDS: frozenset[str] = frozenset({
    "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
    "status", "ping", "ok", "yes", "no", "sure", "what time",
})

_COMPLEX_QUERY_KEYWORDS: frozenset[str] = frozenset({
    "multi-step", "step by step", "step-by-step", "explain", "research",
    "implement", "design", "architecture", "analyse", "analyze",
    "debug", "code", "script", "algorithm", "compare", "evaluate",
    "review", "translate", "prove", "calculate", "derive",
    "write a", "build a", "create a", "generate a",
    "help me understand", "why does", "how does",
})


# ---------------------------------------------------------------------------
# IntentEngine
# ---------------------------------------------------------------------------

class IntentEngine:
    """Match open-ended user queries to available tools using TF-IDF similarity.

    Architecture:
        1. Index all registered tool documents into TF-IDF vectors.
        2. On query: tokenise → TF-IDF → cosine similarity per tool.
        3. Boost scores using query–keyword heuristics per tool.
        4. Optionally ask the LLM to summarise the structured intent.

    Memory footprint: < 1 MB (pure Python dicts, no model loading).
    """

    def __init__(self, llm_adapter: Any | None = None) -> None:
        self._llm = llm_adapter
        # tool_name → (document_tokens, tf_vector)
        self._index: dict[str, tuple[list[str], dict[str, float]]] = {}
        # IDF values computed over the whole corpus
        self._idf: dict[str, float] = {}
        logger.info("IntentEngine initialised (llm_adapter=%s)", type(llm_adapter).__name__ if llm_adapter else "None")

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def index_tool(self, name: str, description: str, tags: list[str] | None = None) -> None:
        """Add (or update) a tool in the semantic index."""
        doc = _build_document(name, description, tags or [])
        tokens = _tokenize(doc)
        tf = _tf(tokens)
        self._index[name] = (tokens, tf)
        self._recompute_idf()
        logger.debug("Indexed tool: %s (%d tokens)", name, len(tokens))

    def remove_tool(self, name: str) -> None:
        """Remove a tool from the index."""
        if name in self._index:
            del self._index[name]
            self._recompute_idf()

    def indexed_tools(self) -> list[str]:
        """Return names of all indexed tools."""
        return list(self._index.keys())

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, query: str) -> IntentResult:
        """Analyse a query and return intent + ranked tool suggestions.

        Args:
            query: Free-form user query.

        Returns:
            IntentResult with suggested_tools ranked by relevance.
        """
        keywords = _tokenize(query)
        if not keywords:
            return IntentResult(
                raw_intent=query,
                structured_intent=query,
                confidence=0.0,
                suggested_tools=list(self._index.keys()),
                keywords=[],
                tool_scores={},
            )

        query_tf = _tf(keywords)
        query_tfidf = self._apply_idf(query_tf)

        scores: dict[str, float] = {}
        for tool_name, (_, tool_tf) in self._index.items():
            tool_tfidf = self._apply_idf(tool_tf)
            score = _cosine_similarity(query_tfidf, tool_tfidf)
            # Keyword boost: if a keyword directly appears in the tool name/doc
            for kw in keywords:
                if kw in tool_name:
                    score += 0.15
            scores[tool_name] = min(score, 1.0)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_confidence = ranked[0][1] if ranked else 0.0
        suggested = [name for name, _ in ranked if scores[name] > 0.0]

        structured = self._structure_intent(query) if self._llm is not None else query

        result = IntentResult(
            raw_intent=query,
            structured_intent=structured,
            confidence=top_confidence,
            suggested_tools=suggested,
            keywords=keywords,
            tool_scores=scores,
        )
        logger.info(
            "Intent analysis: confidence=%.3f top_tool=%s query=%r",
            top_confidence,
            suggested[0] if suggested else "none",
            query[:60],
        )
        return result

    def score_tool(self, query: str, tool_name: str) -> float:
        """Return the relevance score for a single tool given a query."""
        result = self.analyze(query)
        return result.tool_scores.get(tool_name, 0.0)

    def complexity_score(self, query: str) -> int:
        """Estimate the complexity of *query* as an integer in [0, 10].

        Scoring bands:
            0-3  → Simple Q&A (FastLLM handles alone)
            4-6  → Moderate (FastLLM draft + HeavyLLM refine)
            7-10 → Complex reasoning (HeavyLLM full answer)

        The score is based on keyword heuristics; no model inference is
        performed, so it is always fast and works without any LLM.

        Args:
            query: The user query string.

        Returns:
            Integer in the range [0, 10].
        """
        import re as _re
        lower = query.lower()

        # Greetings and status checks → very low complexity
        # Use word-boundary matching to avoid false positives on substrings
        for kw in _SIMPLE_QUERY_KEYWORDS:
            pattern = r"\b" + _re.escape(kw) + r"\b"
            if _re.search(pattern, lower):
                return 1

        # Count high-complexity indicator keywords (phrase or word-boundary)
        high_hits = 0
        for kw in _COMPLEX_QUERY_KEYWORDS:
            if " " in kw:
                if kw in lower:
                    high_hits += 1
            else:
                if _re.search(r"\b" + _re.escape(kw) + r"\b", lower):
                    high_hits += 1

        if high_hits >= 4:
            return 9
        if high_hits == 3:
            return 8
        if high_hits == 2:
            return 7
        if high_hits == 1:
            return 5

        # Medium heuristics: question words + moderate length
        word_count = len(query.split())
        if word_count > 30:
            return 4
        if word_count > 15:
            return 3
        if word_count > 5:
            return 2
        return 1

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recompute_idf(self) -> None:
        """Recompute IDF values over the current corpus."""
        n = len(self._index)
        if n == 0:
            self._idf = {}
            return
        # Count document frequency for each term
        df: dict[str, int] = {}
        for tokens, _ in self._index.values():
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        # Smooth IDF: log((N+1)/(df+1)) + 1
        self._idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}

    def _apply_idf(self, tf_vec: dict[str, float]) -> dict[str, float]:
        """Multiply TF values by their IDF weights."""
        return {term: tf * self._idf.get(term, 1.0) for term, tf in tf_vec.items()}

    def _structure_intent(self, query: str) -> str:
        """Use the LLM to produce a concise action description (1 sentence)."""
        prompt = (
            "In one sentence, describe what the user wants to accomplish:\n"
            f'"{query}"\n'
            "Action:"
        )
        try:
            return self._llm.generate(prompt, max_tokens=40).strip()
        except Exception as exc:
            logger.debug("LLM intent structuring failed: %s", exc)
            return query
