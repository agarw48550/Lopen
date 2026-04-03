"""Confirmation gate / sandbox for Lopen.

Before executing a tool that requires permission, is unfamiliar, or has a
low confidence score, the ConfirmationGate can intercept the request and
return a human-readable confirmation prompt.

In 24/7 autonomous mode, the gate can be configured to auto-approve known
safe tools so the assistant is never blocked on routine tasks.

All confirmation decisions are logged for audit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConfirmationRequest:
    """A pending confirmation request returned to the caller."""

    tool_name: str
    query: str
    reason: str
    prompt: str
    """Human-readable question to display to the user."""

    requires_explicit_ok: bool = True
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# ConfirmationGate
# ---------------------------------------------------------------------------

class ConfirmationGate:
    """Intercept high-risk or low-confidence tool invocations for user approval.

    Configuration:
        confidence_threshold (float):
            Tools selected with a confidence below this value require
            confirmation even if they do not declare ``requires_permission``.
            Default: 0.3.

        auto_approve_known_tools (bool):
            Skip confirmation for tools that are already well-known (i.e.,
            have been used successfully at least ``min_uses_for_auto_approve``
            times).  Default: True.

        min_uses_for_auto_approve (int):
            Minimum successful uses before a tool is considered "known".
            Default: 3.

    Usage::

        gate = ConfirmationGate(confidence_threshold=0.3)
        req = gate.check(tool_meta, query, confidence=0.15)
        if req is not None:
            return {"response": req.prompt, "needs_confirmation": True}
        # else: proceed with execution
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        auto_approve_known_tools: bool = True,
        min_uses_for_auto_approve: int = 3,
    ) -> None:
        self._threshold = confidence_threshold
        self._auto_approve = auto_approve_known_tools
        self._min_uses = min_uses_for_auto_approve
        # tool_name → successful use count (in-memory; can be backed by DB)
        self._use_counts: dict[str, int] = {}
        logger.info(
            "ConfirmationGate initialised (threshold=%.2f, auto_approve=%s)",
            self._threshold,
            self._auto_approve,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        tool_meta: Any,
        query: str,
        confidence: float = 1.0,
    ) -> ConfirmationRequest | None:
        """Decide whether a confirmation is needed before executing a tool.

        Returns:
            A :class:`ConfirmationRequest` if the call needs approval,
            or ``None`` if execution may proceed immediately.
        """
        tool_name: str = getattr(tool_meta, "name", str(tool_meta))
        requires_perm: bool = getattr(tool_meta, "requires_permission", False)

        # Auto-approve well-known tools to avoid blocking 24/7 operations
        if self._auto_approve and self._is_known(tool_name):
            logger.debug("Auto-approved known tool: %s", tool_name)
            return None

        # Check explicit permission requirement
        if requires_perm:
            return self._build_request(
                tool_name, query, reason="requires_permission",
                prompt=(
                    f"The tool '{tool_name}' requires your permission to run.\n"
                    f"Query: {query}\n"
                    "Reply 'yes' to proceed or 'no' to cancel."
                ),
            )

        # Check confidence
        if confidence < self._threshold:
            return self._build_request(
                tool_name, query, reason="low_confidence",
                prompt=(
                    f"I'm not very confident (score={confidence:.0%}) that "
                    f"'{tool_name}' is the right tool for:\n  \"{query}\"\n"
                    "Reply 'yes' to proceed anyway or 'no' to try a different approach."
                ),
            )

        return None

    def record_use(self, tool_name: str, success: bool = True) -> None:
        """Update the known-tool counter after a tool execution."""
        if success:
            self._use_counts[tool_name] = self._use_counts.get(tool_name, 0) + 1

    def reset_known(self, tool_name: str) -> None:
        """Reset the auto-approve counter for a tool (e.g., after update)."""
        self._use_counts.pop(tool_name, None)

    def is_auto_approved(self, tool_name: str) -> bool:
        """Return True if the tool would be auto-approved right now."""
        return self._auto_approve and self._is_known(tool_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_known(self, tool_name: str) -> bool:
        return self._use_counts.get(tool_name, 0) >= self._min_uses

    @staticmethod
    def _build_request(
        tool_name: str,
        query: str,
        reason: str,
        prompt: str,
    ) -> ConfirmationRequest:
        logger.info(
            "Confirmation required: tool=%s reason=%s query=%r",
            tool_name,
            reason,
            query[:60],
        )
        return ConfirmationRequest(
            tool_name=tool_name,
            query=query,
            reason=reason,
            prompt=prompt,
        )
