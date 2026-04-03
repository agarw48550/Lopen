"""Unit tests for ConfirmationGate sandbox logic."""

from __future__ import annotations

import pytest
from agent_core.sandbox import ConfirmationGate, ConfirmationRequest
from agent_core.tool_registry import ToolMeta


@pytest.fixture
def gate() -> ConfirmationGate:
    return ConfirmationGate(
        confidence_threshold=0.3,
        auto_approve_known_tools=True,
        min_uses_for_auto_approve=3,
    )


@pytest.fixture
def safe_tool() -> ToolMeta:
    return ToolMeta("safe_tool", "A safe tool", requires_permission=False)


@pytest.fixture
def risky_tool() -> ToolMeta:
    return ToolMeta("risky_tool", "A risky tool", requires_permission=True)


class TestConfirmationGate:
    def test_no_confirmation_for_high_confidence_safe_tool(
        self, gate: ConfirmationGate, safe_tool: ToolMeta
    ) -> None:
        req = gate.check(safe_tool, "some query", confidence=0.9)
        assert req is None

    def test_confirmation_required_for_permission_tool(
        self, gate: ConfirmationGate, risky_tool: ToolMeta
    ) -> None:
        req = gate.check(risky_tool, "delete my files", confidence=0.9)
        assert req is not None
        assert isinstance(req, ConfirmationRequest)
        assert req.tool_name == "risky_tool"

    def test_confirmation_required_for_low_confidence(
        self, gate: ConfirmationGate, safe_tool: ToolMeta
    ) -> None:
        req = gate.check(safe_tool, "some query", confidence=0.1)
        assert req is not None
        assert "confidence" in req.reason or "low" in req.reason

    def test_reason_is_low_confidence(
        self, gate: ConfirmationGate, safe_tool: ToolMeta
    ) -> None:
        req = gate.check(safe_tool, "some query", confidence=0.15)
        assert req is not None
        assert req.reason == "low_confidence"

    def test_reason_is_requires_permission(
        self, gate: ConfirmationGate, risky_tool: ToolMeta
    ) -> None:
        req = gate.check(risky_tool, "run something", confidence=0.99)
        assert req is not None
        assert req.reason == "requires_permission"

    def test_auto_approve_after_enough_uses(
        self, gate: ConfirmationGate, risky_tool: ToolMeta
    ) -> None:
        # Record 3 successful uses → auto-approve kicks in
        for _ in range(3):
            gate.record_use(risky_tool.name, success=True)
        req = gate.check(risky_tool, "do something", confidence=0.9)
        assert req is None

    def test_auto_approve_not_triggered_too_few_uses(
        self, gate: ConfirmationGate, risky_tool: ToolMeta
    ) -> None:
        gate.record_use(risky_tool.name, success=True)
        gate.record_use(risky_tool.name, success=True)
        # Only 2 uses — still below threshold
        req = gate.check(risky_tool, "do something", confidence=0.9)
        assert req is not None

    def test_reset_known_clears_counter(
        self, gate: ConfirmationGate, risky_tool: ToolMeta
    ) -> None:
        for _ in range(3):
            gate.record_use(risky_tool.name, success=True)
        gate.reset_known(risky_tool.name)
        # Now should require confirmation again
        req = gate.check(risky_tool, "do something", confidence=0.9)
        assert req is not None

    def test_is_auto_approved(
        self, gate: ConfirmationGate, safe_tool: ToolMeta
    ) -> None:
        for _ in range(3):
            gate.record_use(safe_tool.name, success=True)
        assert gate.is_auto_approved(safe_tool.name) is True

    def test_not_auto_approved_when_disabled(
        self, risky_tool: ToolMeta
    ) -> None:
        g = ConfirmationGate(auto_approve_known_tools=False)
        for _ in range(10):
            g.record_use(risky_tool.name, success=True)
        assert g.is_auto_approved(risky_tool.name) is False

    def test_failed_uses_do_not_count(
        self, gate: ConfirmationGate, safe_tool: ToolMeta
    ) -> None:
        for _ in range(5):
            gate.record_use(safe_tool.name, success=False)
        # Failed uses should not count toward auto-approve
        assert gate.is_auto_approved(safe_tool.name) is False

    def test_confirmation_request_has_prompt(
        self, gate: ConfirmationGate, risky_tool: ToolMeta
    ) -> None:
        req = gate.check(risky_tool, "delete files", confidence=0.9)
        assert req is not None
        assert len(req.prompt) > 10

    def test_confidence_at_exact_threshold_no_confirmation(
        self, gate: ConfirmationGate, safe_tool: ToolMeta
    ) -> None:
        # Exactly at threshold (0.3) — should NOT require confirmation
        req = gate.check(safe_tool, "some query", confidence=0.3)
        assert req is None

    def test_confidence_below_threshold_needs_confirmation(
        self, gate: ConfirmationGate, safe_tool: ToolMeta
    ) -> None:
        req = gate.check(safe_tool, "some query", confidence=0.29)
        assert req is not None
