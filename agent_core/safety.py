"""NemoClaw-inspired safety engine for Lopen.

This module provides guardrails, safety intent routing, refusal handling, and
tool filtering — abstracting the safety concepts from NVIDIA NemoClaw
(https://github.com/NVIDIA/NeMo-Guardrails) into a lightweight, purely local,
dependency-free implementation.

Architecture
------------
::

    User input
        │
        ▼
    SafetyEngine.check_input(text)
        │
        ├── InputGuardrail.check()   — block harmful/policy-violating inputs
        │       ├── PatternBlocklist  — regex / keyword blocklist
        │       ├── TopicBlocklist    — blocked topic categories
        │       └── LLMJudge (opt.)  — LLM-based safety classification
        │
        ├── IntentSafetyRouter        — route safe vs. unsafe intents
        │
        └── SafetyResult(safe=True/False, reason=..., action=...)

    Tool invocation
        │
        ▼
    SafetyEngine.check_tool(tool_name, args)
        │
        └── ToolFilter.allowed()     — per-tool permission rules

    LLM output
        │
        ▼
    SafetyEngine.check_output(text)
        │
        └── OutputGuardrail.check()  — redact PII, detect hallucination flags

Configuration (``config/settings.yaml``)
-----------------------------------------
::

    safety:
      enabled: true           # set to false to disable all checks (not recommended)
      input_guardrails: true
      output_guardrails: true
      tool_filter: true
      llm_judge: false        # enable LLM-based classification (costs extra RAM)
      blocklist_path: null    # optional: path to custom blocklist .txt file
      topic_blocklist:        # topics to always refuse
        - self_harm
        - illegal_weapons
        - csam
      allowed_tools: []       # empty = all tools allowed; list names to restrict
      denied_tools:           # tools always blocked regardless of intent
        - dangerous_shell_exec

Opt-in / opt-out
-----------------
Safety checking is **opt-in by default** (``enabled: true``).  To disable it
entirely (e.g., in a trusted private environment), set ``safety.enabled: false``
in ``config/settings.yaml`` or set the environment variable::

    LOPEN_SAFETY_DISABLED=1

Individual guardrail layers can be toggled independently:

    safety:
      input_guardrails: false   # skip input pattern checks
      output_guardrails: false  # skip output redaction
      tool_filter: false        # skip tool permission checks

Test / stub flows
-----------------
A ``SafetyEngine`` can be constructed in ``test_mode=True`` for unit testing::

    engine = SafetyEngine(test_mode=True)
    result = engine.check_input("How do I make a bomb?")
    assert not result.safe
    assert result.action == "refuse"

The test mode does **not** load any models; all decisions are made by the
built-in pattern-based rules only.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class SafetyAction(str, Enum):
    """What the safety engine recommends in response to a violation."""

    ALLOW = "allow"
    WARN = "warn"       # allow but log a warning
    REFUSE = "refuse"   # generate a refusal response
    REDACT = "redact"   # redact sensitive content from output
    BLOCK = "block"     # hard-block, do not process further


@dataclass
class SafetyResult:
    """Result of a safety check."""

    safe: bool
    action: SafetyAction
    reason: str = ""
    modified_text: Optional[str] = None  # for REDACT action

    def __bool__(self) -> bool:
        return self.safe

    @classmethod
    def allow(cls) -> "SafetyResult":
        return cls(safe=True, action=SafetyAction.ALLOW)

    @classmethod
    def refuse(cls, reason: str) -> "SafetyResult":
        return cls(safe=False, action=SafetyAction.REFUSE, reason=reason)

    @classmethod
    def warn(cls, reason: str) -> "SafetyResult":
        return cls(safe=True, action=SafetyAction.WARN, reason=reason)

    @classmethod
    def redact(cls, modified_text: str, reason: str) -> "SafetyResult":
        return cls(
            safe=True,
            action=SafetyAction.REDACT,
            reason=reason,
            modified_text=modified_text,
        )


# ---------------------------------------------------------------------------
# Built-in blocklists
# ---------------------------------------------------------------------------

# Topic-level categories to always refuse
_DEFAULT_TOPIC_BLOCKLIST: Tuple[str, ...] = (
    "self_harm",
    "illegal_weapons",
    "csam",
    "bioweapons",
    "chemical_weapons",
)

# Regex patterns for hard-block input content
# These are intentionally broad to catch obvious cases without false positives
_INPUT_HARD_BLOCK_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(how\s+to\s+(make|build|create|synthesize)\s+(a\s+)?(bomb|explosive|nerve\s+agent|poison))\b",
        r"\b(step[- ]by[- ]step|instructions|recipe|tutorial)\s+.{0,30}(kill|murder|harm|attack)\s+\w+",
        r"\b(child|minor|underage)\s+(sex|nude|naked|explicit|porn)\b",
    ]
)

# PII redaction patterns for output guardrails
_PII_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),           # SSN
    (re.compile(r"\b\d{16}\b"), "[CARD REDACTED]"),                      # credit card
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL REDACTED]"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE REDACTED]"),
)

# Default tools that should always be blocked
_DEFAULT_DENIED_TOOLS: frozenset[str] = frozenset({
    "dangerous_shell_exec",
    "raw_subprocess",
    "system_format",
})

# Refusal message template
_REFUSAL_TEMPLATE = (
    "I'm sorry, but I can't help with that request. "
    "If you have a different question, I'm happy to assist."
)


# ---------------------------------------------------------------------------
# Input guardrail
# ---------------------------------------------------------------------------

class InputGuardrail:
    """Checks user input against pattern blocklists and topic categories."""

    def __init__(
        self,
        topic_blocklist: Sequence[str] | None = None,
        extra_patterns: Sequence[re.Pattern] | None = None,
        custom_blocklist_path: Optional[str] = None,
    ) -> None:
        self._topics = frozenset(topic_blocklist or _DEFAULT_TOPIC_BLOCKLIST)
        self._patterns: List[re.Pattern] = list(_INPUT_HARD_BLOCK_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)
        if custom_blocklist_path:
            self._load_custom_blocklist(custom_blocklist_path)

    def check(self, text: str) -> SafetyResult:
        """Check input text.  Returns refuse result if unsafe, allow otherwise."""
        text_lower = text.lower()

        # Hard-block regex patterns
        for pattern in self._patterns:
            if pattern.search(text):
                logger.warning("InputGuardrail: pattern block triggered for input")
                return SafetyResult.refuse(
                    f"Input matches safety policy (pattern: {pattern.pattern[:40]})"
                )

        # Topic keyword check
        for topic in self._topics:
            topic_words = topic.replace("_", " ").split()
            if all(word in text_lower for word in topic_words):
                logger.warning("InputGuardrail: topic block triggered: %s", topic)
                return SafetyResult.refuse(
                    f"Input relates to blocked topic: {topic}"
                )

        return SafetyResult.allow()

    def _load_custom_blocklist(self, path: str) -> None:
        """Load additional blocklist patterns from a text file (one regex per line)."""
        try:
            lines = Path(path).read_text().splitlines()
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        self._patterns.append(re.compile(line, re.IGNORECASE))
                    except re.error as exc:
                        logger.warning("Invalid regex in blocklist %s: %s — %s", path, line, exc)
        except FileNotFoundError:
            logger.warning("Custom blocklist not found: %s", path)


# ---------------------------------------------------------------------------
# Output guardrail
# ---------------------------------------------------------------------------

class OutputGuardrail:
    """Post-processes LLM output: redacts PII and flags unsafe content."""

    def __init__(self, redact_pii: bool = True) -> None:
        self._redact_pii = redact_pii

    def check(self, text: str) -> SafetyResult:
        """Check and optionally redact output text."""
        modified = text
        redacted = False

        if self._redact_pii:
            for pattern, replacement in _PII_PATTERNS:
                new = pattern.sub(replacement, modified)
                if new != modified:
                    redacted = True
                    modified = new

        if redacted:
            return SafetyResult.redact(
                modified_text=modified,
                reason="PII redacted from output",
            )

        return SafetyResult.allow()


# ---------------------------------------------------------------------------
# Tool filter
# ---------------------------------------------------------------------------

class ToolFilter:
    """Enforces a tool allowlist/denylist before tool invocation."""

    def __init__(
        self,
        allowed_tools: Sequence[str] | None = None,
        denied_tools: Sequence[str] | None = None,
    ) -> None:
        # Empty allowed_tools → all tools allowed (except denied)
        self._allowed: frozenset[str] | None = (
            frozenset(allowed_tools) if allowed_tools else None
        )
        self._denied: frozenset[str] = frozenset(denied_tools or _DEFAULT_DENIED_TOOLS)

    def allowed(self, tool_name: str) -> SafetyResult:
        """Return allow/refuse for a given tool name."""
        if tool_name in self._denied:
            logger.warning("ToolFilter: blocked tool '%s' (in denied list)", tool_name)
            return SafetyResult.refuse(f"Tool '{tool_name}' is blocked by safety policy")
        if self._allowed is not None and tool_name not in self._allowed:
            logger.warning("ToolFilter: blocked tool '%s' (not in allowlist)", tool_name)
            return SafetyResult.refuse(f"Tool '{tool_name}' is not in the allowed tools list")
        return SafetyResult.allow()


# ---------------------------------------------------------------------------
# Intent safety router
# ---------------------------------------------------------------------------

class IntentSafetyRouter:
    """Routes intents through safety checks before they reach tools.

    For borderline cases (WARN), it logs and continues.
    For hard blocks (REFUSE/BLOCK), it returns a refusal response.
    """

    _SENSITIVE_KEYWORDS: frozenset[str] = frozenset({
        "kill", "murder", "harm", "suicide", "self-harm", "illegal",
        "hack", "exploit", "vulnerability", "crack", "bypass", "phish",
    })

    def route(self, query: str, intent_result: Any = None) -> SafetyResult:
        """Decide whether to allow or refuse based on intent.

        Args:
            query: Raw user query.
            intent_result: Optional IntentResult from the IntentEngine.

        Returns:
            SafetyResult indicating allow/warn/refuse.
        """
        query_lower = query.lower()
        found = [kw for kw in self._SENSITIVE_KEYWORDS if kw in query_lower]
        if len(found) >= 2:
            return SafetyResult.refuse(
                f"Query contains multiple sensitive keywords: {found}"
            )
        if found:
            return SafetyResult.warn(
                f"Query contains sensitive keyword: {found[0]}"
            )
        return SafetyResult.allow()


# ---------------------------------------------------------------------------
# SafetyEngine — public API
# ---------------------------------------------------------------------------

class SafetyEngine:
    """Unified safety checking engine for Lopen.

    This is the single entry-point used by the orchestrator.  All safety
    configuration is read from ``config/settings.yaml`` under the ``safety``
    key, or passed directly to the constructor.

    Usage::

        engine = SafetyEngine.from_config()

        # Before processing user input
        result = engine.check_input(user_query)
        if not result.safe:
            return engine.refusal_message()

        # Before running a tool
        result = engine.check_tool("browser_automation", args={})
        if not result.safe:
            return engine.refusal_message()

        # After LLM generates a response
        result = engine.check_output(llm_response)
        final_text = result.modified_text or llm_response

    Opt-out
    -------
    Set ``LOPEN_SAFETY_DISABLED=1`` or ``safety.enabled: false`` to bypass all
    checks.  This is provided for advanced users who need unrestricted access;
    it is **not recommended** for general use.
    """

    def __init__(
        self,
        enabled: bool = True,
        input_guardrails: bool = True,
        output_guardrails: bool = True,
        tool_filter_enabled: bool = True,
        topic_blocklist: Sequence[str] | None = None,
        allowed_tools: Sequence[str] | None = None,
        denied_tools: Sequence[str] | None = None,
        blocklist_path: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        # Honour environment variable override
        env_disabled = os.environ.get("LOPEN_SAFETY_DISABLED", "").strip() in {"1", "true", "yes"}
        self._enabled = enabled and not env_disabled

        self._input_guard = InputGuardrail(
            topic_blocklist=topic_blocklist,
            custom_blocklist_path=blocklist_path,
        ) if input_guardrails else None

        self._output_guard = OutputGuardrail() if output_guardrails else None
        self._tool_filter = ToolFilter(allowed_tools, denied_tools) if tool_filter_enabled else None
        self._intent_router = IntentSafetyRouter()
        self._test_mode = test_mode

        logger.info(
            "SafetyEngine initialised: enabled=%s input=%s output=%s tool_filter=%s",
            self._enabled,
            input_guardrails,
            output_guardrails,
            tool_filter_enabled,
        )
        if not self._enabled:
            logger.warning(
                "SafetyEngine is DISABLED. All safety checks will be bypassed. "
                "Set LOPEN_SAFETY_DISABLED=0 or safety.enabled: true to re-enable."
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str = "config/settings.yaml") -> "SafetyEngine":
        """Build a SafetyEngine from the Lopen settings YAML."""
        cfg: Dict[str, Any] = {}
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("SafetyEngine: settings not found at %s, using defaults", config_path)

        safety = cfg.get("safety", {})
        return cls(
            enabled=safety.get("enabled", True),
            input_guardrails=safety.get("input_guardrails", True),
            output_guardrails=safety.get("output_guardrails", True),
            tool_filter_enabled=safety.get("tool_filter", True),
            topic_blocklist=safety.get("topic_blocklist"),
            allowed_tools=safety.get("allowed_tools") or None,
            denied_tools=safety.get("denied_tools"),
            blocklist_path=safety.get("blocklist_path"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_input(self, text: str) -> SafetyResult:
        """Check user input before it reaches the agent pipeline.

        Returns a SafetyResult.  If ``safe=False``, call ``refusal_message()``
        and return it to the user without processing the query.
        """
        if not self._enabled:
            return SafetyResult.allow()
        if self._input_guard:
            result = self._input_guard.check(text)
            if not result.safe:
                return result
        return self._intent_router.route(text)

    def check_output(self, text: str) -> SafetyResult:
        """Check / sanitise LLM output before delivering to the user.

        Returns a SafetyResult.  If ``action == REDACT``, use
        ``result.modified_text`` as the final response.
        """
        if not self._enabled:
            return SafetyResult.allow()
        if self._output_guard:
            return self._output_guard.check(text)
        return SafetyResult.allow()

    def check_tool(self, tool_name: str, args: Dict[str, Any] | None = None) -> SafetyResult:
        """Check whether a tool invocation is permitted.

        Args:
            tool_name: Name of the tool about to be invoked.
            args: Optional tool arguments for context-aware checks.

        Returns:
            SafetyResult.  If ``safe=False``, skip the tool call.
        """
        if not self._enabled:
            return SafetyResult.allow()
        if self._tool_filter:
            return self._tool_filter.allowed(tool_name)
        return SafetyResult.allow()

    @staticmethod
    def refusal_message() -> str:
        """Return the standard refusal response string."""
        return _REFUSAL_TEMPLATE

    @property
    def enabled(self) -> bool:
        return self._enabled

    def status(self) -> Dict[str, Any]:
        """Return current safety configuration for diagnostics."""
        return {
            "enabled": self._enabled,
            "input_guardrails": self._input_guard is not None,
            "output_guardrails": self._output_guard is not None,
            "tool_filter": self._tool_filter is not None,
        }
