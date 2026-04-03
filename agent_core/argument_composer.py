"""Argument composer for Lopen.

Extracts structured arguments from a free-form user query so that a tool
can be invoked with the right parameters.  Uses lightweight regex patterns
for common argument types (file paths, URLs, names, code snippets) and
falls back to returning the full query as a ``query`` argument.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Filesystem paths
    ("file_path", re.compile(r'(?:^|[\s"\'])(/[^\s"\']+\.[a-zA-Z]{2,6})', re.M)),
    # Relative paths like docs/file.txt or ./folder/file.py
    ("file_path", re.compile(r'(?:^|[\s"\'])(\.?\.?/[^\s"\']+\.[a-zA-Z]{2,6})', re.M)),
    # URLs
    ("url", re.compile(r"https?://[^\s]+")),
    # Programming language
    ("language", re.compile(
        r"\b(python|javascript|typescript|rust|golang|go|java|c\+\+|cpp|bash|shell|"
        r"ruby|php|swift|kotlin|scala|r|sql|html|css)\b",
        re.I,
    )),
    # Contact name: "message/text/send [to] <Name>"
    ("contact", re.compile(r"(?:message|text|send|whatsapp)\s+(?:to\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")),
    # Desktop/folder path keywords
    ("target_dir", re.compile(r"(?:in|from|inside|within|folder|directory|dir)\s+[\"']?([^\s\"']+)[\"']?", re.I)),
]


# ---------------------------------------------------------------------------
# ArgumentComposer
# ---------------------------------------------------------------------------

class ArgumentComposer:
    """Extract structured arguments from a free-form query.

    The extracted dict always contains at least ``{"query": original_query}``
    so every tool can handle the call even if specific arguments aren't found.

    Usage::

        composer = ArgumentComposer(llm_adapter=llm)
        args = composer.compose("debug my python function in /tmp/foo.py")
        # → {"query": "debug ...", "file_path": "/tmp/foo.py", "language": "python"}
    """

    def __init__(self, llm_adapter: Any | None = None) -> None:
        self._llm = llm_adapter
        logger.info("ArgumentComposer initialised (llm_adapter=%s)", type(llm_adapter).__name__ if llm_adapter else "None")

    def compose(self, query: str, tool_name: str = "") -> dict[str, Any]:
        """Extract arguments relevant to *tool_name* from *query*.

        Args:
            query: The original user query.
            tool_name: Name of the target tool (used to tailor extraction).

        Returns:
            Dict of argument name → extracted value.  Always includes
            ``"query"`` with the full original query.
        """
        args: dict[str, Any] = {"query": query}

        # Apply all regex patterns
        for arg_name, pattern in _PATTERNS:
            match = pattern.search(query)
            if match:
                value = match.group(1) if pattern.groups else match.group(0)
                # Avoid overwriting a more specific match
                if arg_name not in args:
                    args[arg_name] = value.strip()

        # Tool-specific enrichment
        enriched = self._tool_specific(query, tool_name, args)
        args.update(enriched)

        logger.debug("Composed args for tool=%r: %s", tool_name, list(args.keys()))
        return args

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tool_specific(
        self, query: str, tool_name: str, current: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply tool-specific extraction heuristics."""
        extra: dict[str, Any] = {}

        if tool_name in ("file_ops", "desktop_organizer"):
            # Extract explicit directory names after common prepositions
            m = re.search(r"(?:organize|sort|clean|scan)\s+([~/.\w/]+)", query, re.I)
            if m and "target_dir" not in current:
                extra["target_dir"] = m.group(1)

        if tool_name == "researcher":
            # Capture quoted search phrases
            m = re.search(r'["\']([^"\']{5,})["\']', query)
            if m:
                extra["search_term"] = m.group(1)

        if tool_name == "coder_assist":
            # Extract code block (triple backtick)
            m = re.search(r"```(?:[a-z]*\n)?(.*?)```", query, re.S)
            if m:
                extra["code_snippet"] = m.group(1).strip()

        return extra
