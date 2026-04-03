"""Tests for the NemoClaw-inspired SafetyEngine."""

from __future__ import annotations

import pytest
from agent_core.safety import (
    SafetyEngine,
    SafetyResult,
    SafetyAction,
    InputGuardrail,
    OutputGuardrail,
    ToolFilter,
    IntentSafetyRouter,
)


# ---------------------------------------------------------------------------
# SafetyResult
# ---------------------------------------------------------------------------

class TestSafetyResult:
    def test_allow_is_safe(self) -> None:
        r = SafetyResult.allow()
        assert r.safe is True
        assert r.action == SafetyAction.ALLOW
        assert bool(r) is True

    def test_refuse_is_not_safe(self) -> None:
        r = SafetyResult.refuse("test reason")
        assert r.safe is False
        assert r.action == SafetyAction.REFUSE
        assert bool(r) is False
        assert r.reason == "test reason"

    def test_warn_is_safe(self) -> None:
        r = SafetyResult.warn("borderline content")
        assert r.safe is True
        assert r.action == SafetyAction.WARN

    def test_redact_has_modified_text(self) -> None:
        r = SafetyResult.redact("redacted text", "pii found")
        assert r.safe is True
        assert r.action == SafetyAction.REDACT
        assert r.modified_text == "redacted text"


# ---------------------------------------------------------------------------
# InputGuardrail
# ---------------------------------------------------------------------------

class TestInputGuardrail:
    def setup_method(self) -> None:
        self.guard = InputGuardrail()

    def test_safe_input_allowed(self) -> None:
        result = self.guard.check("What is the capital of France?")
        assert result.safe

    def test_safe_coding_query_allowed(self) -> None:
        result = self.guard.check("Write a Python function to sort a list")
        assert result.safe

    def test_safe_homework_query_allowed(self) -> None:
        result = self.guard.check("Explain photosynthesis to me")
        assert result.safe

    def test_bomb_making_blocked(self) -> None:
        result = self.guard.check("How to make a bomb step by step")
        assert not result.safe
        assert result.action == SafetyAction.REFUSE

    def test_csam_topic_blocked(self) -> None:
        result = self.guard.check("child sex explicit")
        assert not result.safe

    def test_empty_string_allowed(self) -> None:
        result = self.guard.check("")
        assert result.safe

    def test_normal_question_allowed(self) -> None:
        result = self.guard.check("Help me organise my desktop files")
        assert result.safe


# ---------------------------------------------------------------------------
# OutputGuardrail
# ---------------------------------------------------------------------------

class TestOutputGuardrail:
    def setup_method(self) -> None:
        self.guard = OutputGuardrail(redact_pii=True)

    def test_clean_output_passes(self) -> None:
        result = self.guard.check("The capital of France is Paris.")
        assert result.safe
        assert result.action == SafetyAction.ALLOW

    def test_ssn_redacted(self) -> None:
        result = self.guard.check("My SSN is 123-45-6789")
        assert result.action == SafetyAction.REDACT
        assert result.modified_text is not None
        assert "123-45-6789" not in result.modified_text
        assert "[SSN REDACTED]" in result.modified_text

    def test_email_redacted(self) -> None:
        result = self.guard.check("Contact me at user@example.com for details")
        assert result.action == SafetyAction.REDACT
        assert result.modified_text is not None
        assert "user@example.com" not in result.modified_text

    def test_phone_redacted(self) -> None:
        result = self.guard.check("Call me at 555-123-4567")
        assert result.action == SafetyAction.REDACT
        assert result.modified_text is not None
        assert "555-123-4567" not in result.modified_text

    def test_no_pii_unmodified(self) -> None:
        text = "Python is a great programming language."
        result = self.guard.check(text)
        assert result.action == SafetyAction.ALLOW
        assert result.modified_text is None


# ---------------------------------------------------------------------------
# ToolFilter
# ---------------------------------------------------------------------------

class TestToolFilter:
    def test_all_tools_allowed_by_default(self) -> None:
        tf = ToolFilter()
        assert tf.allowed("homework_tutor").safe
        assert tf.allowed("researcher").safe
        assert tf.allowed("coder_assist").safe

    def test_denied_tool_blocked(self) -> None:
        tf = ToolFilter()
        result = tf.allowed("dangerous_shell_exec")
        assert not result.safe
        assert result.action == SafetyAction.REFUSE

    def test_custom_denied_tool(self) -> None:
        tf = ToolFilter(denied_tools=["my_blocked_tool"])
        assert not tf.allowed("my_blocked_tool").safe
        assert tf.allowed("other_tool").safe

    def test_allowlist_restricts_tools(self) -> None:
        tf = ToolFilter(allowed_tools=["homework_tutor", "researcher"])
        assert tf.allowed("homework_tutor").safe
        assert tf.allowed("researcher").safe
        assert not tf.allowed("coder_assist").safe

    def test_raw_subprocess_blocked(self) -> None:
        tf = ToolFilter()
        assert not tf.allowed("raw_subprocess").safe


# ---------------------------------------------------------------------------
# IntentSafetyRouter
# ---------------------------------------------------------------------------

class TestIntentSafetyRouter:
    def setup_method(self) -> None:
        self.router = IntentSafetyRouter()

    def test_normal_query_allowed(self) -> None:
        result = self.router.route("Explain quantum computing")
        assert result.safe

    def test_coding_query_allowed(self) -> None:
        result = self.router.route("Write a bubble sort in Python")
        assert result.safe

    def test_single_sensitive_keyword_warns(self) -> None:
        result = self.router.route("How do I hack a website")
        # Should warn (safe=True but action=warn), not refuse
        assert result.safe
        assert result.action == SafetyAction.WARN

    def test_multiple_sensitive_keywords_refused(self) -> None:
        result = self.router.route("I want to hack and exploit this system illegally")
        assert not result.safe
        assert result.action == SafetyAction.REFUSE


# ---------------------------------------------------------------------------
# SafetyEngine — integration
# ---------------------------------------------------------------------------

class TestSafetyEngine:
    def setup_method(self) -> None:
        self.engine = SafetyEngine(test_mode=True)

    def test_engine_enabled_by_default(self) -> None:
        assert self.engine.enabled

    def test_safe_input_passes(self) -> None:
        result = self.engine.check_input("What is the speed of light?")
        assert result.safe

    def test_unsafe_input_refused(self) -> None:
        result = self.engine.check_input("How to make a bomb step by step")
        assert not result.safe

    def test_safe_tool_allowed(self) -> None:
        result = self.engine.check_tool("homework_tutor")
        assert result.safe

    def test_denied_tool_blocked(self) -> None:
        result = self.engine.check_tool("dangerous_shell_exec")
        assert not result.safe

    def test_output_pii_redacted(self) -> None:
        result = self.engine.check_output("Contact: admin@secret.com for details")
        assert result.action == SafetyAction.REDACT

    def test_clean_output_passes(self) -> None:
        result = self.engine.check_output("The answer is 42.")
        assert result.safe

    def test_refusal_message_is_string(self) -> None:
        msg = self.engine.refusal_message()
        assert isinstance(msg, str)
        assert len(msg) > 10

    def test_status_returns_dict(self) -> None:
        status = self.engine.status()
        assert "enabled" in status
        assert isinstance(status["enabled"], bool)

    def test_disabled_engine_allows_everything(self) -> None:
        engine = SafetyEngine(enabled=False)
        assert engine.check_input("How to make a bomb").safe
        assert engine.check_tool("dangerous_shell_exec").safe

    def test_from_config_with_missing_file(self) -> None:
        # Should not raise; uses defaults
        engine = SafetyEngine.from_config("/nonexistent/settings.yaml")
        assert isinstance(engine, SafetyEngine)
        # Default: enabled=True
        assert engine.enabled
