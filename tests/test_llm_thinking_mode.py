"""Tests for Qwen3.5 thinking mode switching in LLMAdapter."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from llm.llm_adapter import (
    ThinkingMode,
    LLMAdapter,
    _build_prompt,
    _infer_thinking_mode,
    _strip_think_tags,
    _THINKING_ENABLE_SUFFIX,
    _THINKING_DISABLE_SUFFIX,
)


class TestThinkingModeEnum:
    def test_values(self) -> None:
        assert ThinkingMode.AUTO.value == "auto"
        assert ThinkingMode.THINKING.value == "thinking"
        assert ThinkingMode.NON_THINKING.value == "non_thinking"

    def test_string_coercible(self) -> None:
        assert ThinkingMode("thinking") is ThinkingMode.THINKING
        assert ThinkingMode("non_thinking") is ThinkingMode.NON_THINKING


class TestInferThinkingMode:
    def test_simple_question_non_thinking(self) -> None:
        mode = _infer_thinking_mode("What time is it?")
        assert mode == ThinkingMode.NON_THINKING

    def test_reasoning_keyword_thinking(self) -> None:
        mode = _infer_thinking_mode("Can you explain how black holes form?")
        assert mode == ThinkingMode.THINKING

    def test_plan_keyword_thinking(self) -> None:
        mode = _infer_thinking_mode("Help me plan my week.")
        assert mode == ThinkingMode.THINKING

    def test_code_keyword_thinking(self) -> None:
        mode = _infer_thinking_mode("Write a Python script to parse JSON.")
        assert mode == ThinkingMode.THINKING

    def test_debug_keyword_thinking(self) -> None:
        mode = _infer_thinking_mode("Debug this function for me.")
        assert mode == ThinkingMode.THINKING

    def test_translate_keyword_thinking(self) -> None:
        mode = _infer_thinking_mode("Translate this text to French.")
        assert mode == ThinkingMode.THINKING

    def test_case_insensitive(self) -> None:
        mode = _infer_thinking_mode("EXPLAIN quantum entanglement")
        assert mode == ThinkingMode.THINKING

    def test_empty_prompt_non_thinking(self) -> None:
        mode = _infer_thinking_mode("")
        assert mode == ThinkingMode.NON_THINKING


class TestBuildPromptThinkingMode:
    def test_non_thinking_appends_no_think(self) -> None:
        prompt = _build_prompt("Hello", "chatml", "", ThinkingMode.NON_THINKING)
        assert _THINKING_DISABLE_SUFFIX in prompt

    def test_thinking_appends_think(self) -> None:
        prompt = _build_prompt("Explain recursion", "chatml", "", ThinkingMode.THINKING)
        assert _THINKING_ENABLE_SUFFIX in prompt

    def test_auto_not_appended_to_prompt(self) -> None:
        # AUTO should not add any suffix — resolved before _build_prompt is called
        prompt = _build_prompt("Hello", "chatml", "", ThinkingMode.AUTO)
        assert _THINKING_ENABLE_SUFFIX not in prompt
        assert _THINKING_DISABLE_SUFFIX not in prompt

    def test_phi3_format_unaffected(self) -> None:
        prompt = _build_prompt("Hello", "phi3", "", ThinkingMode.THINKING)
        # phi3 format does not support thinking tokens
        assert "<|user|>" in prompt
        assert _THINKING_ENABLE_SUFFIX not in prompt

    def test_chatml_structure(self) -> None:
        prompt = _build_prompt("Hi there", "chatml", "You are helpful.", ThinkingMode.NON_THINKING)
        assert "<|im_start|>system" in prompt
        assert "You are helpful." in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt

    def test_system_prompt_included(self) -> None:
        prompt = _build_prompt("Q", "chatml", "Custom system.", ThinkingMode.NON_THINKING)
        assert "Custom system." in prompt


class TestStripThinkTags:
    def test_strips_think_block(self) -> None:
        raw = "<think>This is internal reasoning.</think>Final answer."
        assert _strip_think_tags(raw) == "Final answer."

    def test_strips_multiline_think_block(self) -> None:
        raw = "<think>\nLine 1\nLine 2\n</think>Answer here."
        assert _strip_think_tags(raw) == "Answer here."

    def test_no_think_block_unchanged(self) -> None:
        raw = "Just a plain response."
        assert _strip_think_tags(raw) == "Just a plain response."

    def test_empty_string(self) -> None:
        assert _strip_think_tags("") == ""

    def test_multiple_think_blocks(self) -> None:
        raw = "<think>A</think>Middle<think>B</think>End"
        result = _strip_think_tags(raw)
        assert "A" not in result
        assert "B" not in result
        assert "Middle" in result
        assert "End" in result


class TestLLMAdapterThinkingMode:
    def _make_mock(self, thinking_mode: ThinkingMode = ThinkingMode.AUTO) -> LLMAdapter:
        return LLMAdapter(
            model_path="/nonexistent/model.gguf",
            thinking_mode=thinking_mode,
        )

    def test_default_thinking_mode_is_auto(self) -> None:
        adapter = self._make_mock()
        assert adapter.thinking_mode == ThinkingMode.AUTO

    def test_custom_thinking_mode_stored(self) -> None:
        adapter = self._make_mock(ThinkingMode.THINKING)
        assert adapter.thinking_mode == ThinkingMode.THINKING

    def test_chat_returns_string_non_thinking(self) -> None:
        adapter = self._make_mock(ThinkingMode.NON_THINKING)
        result = adapter.chat("Hello")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_returns_string_thinking(self) -> None:
        adapter = self._make_mock(ThinkingMode.THINKING)
        result = adapter.chat("Explain how sorting works.")
        assert isinstance(result, str)

    def test_chat_per_call_override(self) -> None:
        adapter = self._make_mock(ThinkingMode.NON_THINKING)
        result = adapter.chat("Explain recursion.", thinking_mode=ThinkingMode.THINKING)
        assert isinstance(result, str)

    def test_chat_auto_infers_mode(self) -> None:
        adapter = self._make_mock(ThinkingMode.AUTO)
        # Should not raise; mode is inferred internally
        result = adapter.chat("What is 2+2?")
        assert isinstance(result, str)

    def test_mode_is_mock(self) -> None:
        adapter = self._make_mock()
        assert adapter.mode == "mock"
