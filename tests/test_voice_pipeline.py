"""Tests for the updated VoiceLoop with AudioModel integration."""

from __future__ import annotations

import sys
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from interfaces.voice_service.voice_loop import VoiceLoop
from interfaces.voice_service.audio_model import (
    AudioModel,
    AudioResponse,
    EmotionLabel,
    Language,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wake_detector(wake_word: str = "lopen") -> MagicMock:
    wd = MagicMock()
    wd.wake_word = wake_word
    wd.check_transcript = MagicMock(return_value=False)
    wd.start = MagicMock()
    wd.stop = MagicMock()
    return wd


def _make_mock_audio_response(**kwargs) -> AudioResponse:
    defaults = dict(
        text="Hello from audio model",
        audio_bytes=None,
        emotion=EmotionLabel.NEUTRAL,
        reply_emotion=EmotionLabel.NEUTRAL,
        tool_call=None,
        used_llm_fallback=False,
    )
    defaults.update(kwargs)
    return AudioResponse(**defaults)


def _make_voice_loop(
    audio_model=None,
    on_query=None,
    language: str = "en",
) -> VoiceLoop:
    wake = _make_wake_detector()
    asr = MagicMock()
    asr.transcribe = MagicMock(return_value="hello")
    tts = MagicMock()
    tts.speak = MagicMock(return_value=True)
    llm = MagicMock()
    llm.generate = MagicMock(return_value="LLM reply")
    return VoiceLoop(
        wake_word_detector=wake,
        asr_adapter=asr,
        tts_adapter=tts,
        llm_adapter=llm,
        on_query=on_query,
        audio_model=audio_model,
        language=language,
    )


# ---------------------------------------------------------------------------
# VoiceLoop constructor
# ---------------------------------------------------------------------------

class TestVoiceLoopInit:
    def test_basic_construction(self) -> None:
        loop = _make_voice_loop()
        assert loop is not None

    def test_audio_model_stored(self) -> None:
        am = AudioModel(mock_mode=True)
        loop = _make_voice_loop(audio_model=am)
        assert loop.audio_model is am

    def test_language_stored(self) -> None:
        loop = _make_voice_loop(language="hi")
        assert loop.language == "hi"

    def test_default_language_en(self) -> None:
        loop = _make_voice_loop()
        assert loop.language == "en"

    def test_no_audio_model_by_default(self) -> None:
        loop = _make_voice_loop()
        assert loop.audio_model is None

    def test_stop_sets_running_false(self) -> None:
        loop = _make_voice_loop()
        loop._running = True
        loop.stop()
        assert loop._running is False


# ---------------------------------------------------------------------------
# VoiceLoop._handle_audio_chunk
# ---------------------------------------------------------------------------

class TestHandleAudioChunk:
    def test_calls_audio_model_process_audio(self) -> None:
        am = MagicMock(spec=AudioModel)
        am.process_audio = MagicMock(return_value=_make_mock_audio_response())
        loop = _make_voice_loop(audio_model=am)

        asyncio.run(loop._handle_audio_chunk(b"\x00" * 32))
        am.process_audio.assert_called_once()

    def test_calls_tts_when_no_audio_bytes(self) -> None:
        am = MagicMock(spec=AudioModel)
        am.process_audio = MagicMock(
            return_value=_make_mock_audio_response(text="Hello!", audio_bytes=None)
        )
        loop = _make_voice_loop(audio_model=am)

        asyncio.run(loop._handle_audio_chunk(b"\x00" * 32))
        loop.tts.speak.assert_called_once_with("Hello!")

    def test_dispatches_tool_call_to_on_query(self) -> None:
        tool = {"name": "search", "args": {"q": "test"}}
        am = MagicMock(spec=AudioModel)
        am.process_audio = MagicMock(
            return_value=_make_mock_audio_response(tool_call=tool)
        )
        query_calls = []
        on_query = MagicMock(side_effect=lambda x: query_calls.append(x) or "ok")
        loop = _make_voice_loop(audio_model=am, on_query=on_query)

        asyncio.run(loop._handle_audio_chunk(b"\x00" * 32))
        assert any("[TOOL_CALL]" in call for call in query_calls)

    def test_language_en_resolved(self) -> None:
        am = MagicMock(spec=AudioModel)
        am.process_audio = MagicMock(return_value=_make_mock_audio_response())
        loop = _make_voice_loop(audio_model=am, language="en")

        asyncio.run(loop._handle_audio_chunk(b"\x00" * 32))
        _, kwargs = am.process_audio.call_args
        assert kwargs.get("language") == Language.ENGLISH

    def test_language_hi_resolved(self) -> None:
        am = MagicMock(spec=AudioModel)
        am.process_audio = MagicMock(return_value=_make_mock_audio_response())
        loop = _make_voice_loop(audio_model=am, language="hi")

        asyncio.run(loop._handle_audio_chunk(b"\x00" * 32))
        _, kwargs = am.process_audio.call_args
        assert kwargs.get("language") == Language.HINDI

    def test_language_zh_resolved(self) -> None:
        am = MagicMock(spec=AudioModel)
        am.process_audio = MagicMock(return_value=_make_mock_audio_response())
        loop = _make_voice_loop(audio_model=am, language="zh")

        asyncio.run(loop._handle_audio_chunk(b"\x00" * 32))
        _, kwargs = am.process_audio.call_args
        assert kwargs.get("language") == Language.CHINESE

    def test_unknown_language_falls_back_to_english(self) -> None:
        am = MagicMock(spec=AudioModel)
        am.process_audio = MagicMock(return_value=_make_mock_audio_response())
        loop = _make_voice_loop(audio_model=am, language="xx")

        asyncio.run(loop._handle_audio_chunk(b"\x00" * 32))
        _, kwargs = am.process_audio.call_args
        assert kwargs.get("language") == Language.ENGLISH

    def test_no_crash_on_model_error(self) -> None:
        am = MagicMock(spec=AudioModel)
        am.process_audio = MagicMock(side_effect=RuntimeError("boom"))
        loop = _make_voice_loop(audio_model=am)

        # Should not raise
        asyncio.run(loop._handle_audio_chunk(b"\x00" * 32))


# ---------------------------------------------------------------------------
# VoiceLoop._handle_query (legacy text path)
# ---------------------------------------------------------------------------

class TestHandleQuery:
    def test_on_query_called(self) -> None:
        on_query = MagicMock(return_value="Response text")
        loop = _make_voice_loop(on_query=on_query)
        asyncio.run(loop._handle_query("hello world"))
        on_query.assert_called_once_with("hello world")

    def test_tts_speak_called_with_response(self) -> None:
        loop = _make_voice_loop(on_query=lambda q: "Hi there!")
        asyncio.run(loop._handle_query("test"))
        loop.tts.speak.assert_called_once_with("Hi there!")

    def test_llm_called_when_no_on_query(self) -> None:
        loop = _make_voice_loop(on_query=None)
        asyncio.run(loop._handle_query("test"))
        loop.llm.generate.assert_called_once()

    def test_error_in_on_query_handled_gracefully(self) -> None:
        def bad_handler(q):
            raise ValueError("oops")
        loop = _make_voice_loop(on_query=bad_handler)
        asyncio.run(loop._handle_query("hi"))
        # tts.speak should be called with an error message
        loop.tts.speak.assert_called_once()


# ---------------------------------------------------------------------------
# VoiceLoop._strip_wake_word
# ---------------------------------------------------------------------------

class TestStripWakeWord:
    def test_strips_prefix(self) -> None:
        loop = _make_voice_loop()
        loop.wake_detector.wake_word = "lopen"
        result = loop._strip_wake_word("lopen what is the weather")
        assert result == "what is the weather"

    def test_no_wake_word_returns_stripped(self) -> None:
        loop = _make_voice_loop()
        loop.wake_detector.wake_word = "lopen"
        result = loop._strip_wake_word("what is the weather")
        assert result == "what is the weather"

    def test_case_insensitive_strip(self) -> None:
        loop = _make_voice_loop()
        loop.wake_detector.wake_word = "Lopen"
        result = loop._strip_wake_word("LOPEN hello there")
        assert result == "hello there"


# ---------------------------------------------------------------------------
# VoiceLoop wake word callback
# ---------------------------------------------------------------------------

class TestOnWakeWord:
    def test_sets_wake_detected(self) -> None:
        loop = _make_voice_loop()
        assert loop._wake_detected is False
        loop._on_wake_word()
        assert loop._wake_detected is True
