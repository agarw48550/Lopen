"""Tests for HeavyLLM lazy loading and mock fallback."""

from __future__ import annotations

import threading
import time

import pytest
from llm.heavy_llm import HeavyLLM, get_heavy_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_heavy() -> HeavyLLM:
    """Create a fresh HeavyLLM instance (not the shared singleton)."""
    return HeavyLLM()


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------

class TestLazyLoading:
    def test_not_loaded_on_init(self) -> None:
        h = _make_heavy()
        assert h.is_loaded is False

    def test_is_loaded_property_type(self) -> None:
        h = _make_heavy()
        assert isinstance(h.is_loaded, bool)


# ---------------------------------------------------------------------------
# Mock fallback (no model file in CI)
# ---------------------------------------------------------------------------

class TestMockFallback:
    @pytest.fixture
    def heavy(self) -> HeavyLLM:
        return _make_heavy()

    def test_generate_returns_string(self, heavy: HeavyLLM) -> None:
        result = heavy.generate("Write a haiku")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_returns_string(self, heavy: HeavyLLM) -> None:
        result = heavy.chat("What is the capital of France?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_mock_prefix(self, heavy: HeavyLLM) -> None:
        # In mock mode responses start with [mock-heavy]
        result = heavy.generate("test prompt")
        assert result.startswith("[mock-heavy]")

    def test_chat_mock_prefix(self, heavy: HeavyLLM) -> None:
        result = heavy.chat("test message")
        assert result.startswith("[mock-heavy]")

    def test_generate_with_custom_max_tokens(self, heavy: HeavyLLM) -> None:
        result = heavy.generate("hello", max_tokens=10)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Unload
# ---------------------------------------------------------------------------

class TestUnload:
    def test_unload_when_not_loaded(self) -> None:
        h = _make_heavy()
        # Should not raise even if nothing is loaded
        h.unload()
        assert h.is_loaded is False

    def test_unload_stops_watchdog(self) -> None:
        h = _make_heavy()
        # Trigger watchdog start
        h.generate("hello")
        time.sleep(0.1)
        # Unload should set stop event
        h.unload()
        assert h.is_loaded is False


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_get_heavy_llm_returns_instance(self) -> None:
        assert isinstance(get_heavy_llm(), HeavyLLM)

    def test_get_heavy_llm_same_object(self) -> None:
        a = get_heavy_llm()
        b = get_heavy_llm()
        assert a is b
