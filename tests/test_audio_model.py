"""Tests for the AudioModel (LFM2.5-Audio wrapper) in mock mode."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from interfaces.voice_service.audio_model import (
    AudioModel,
    AudioResponse,
    EmotionLabel,
    Language,
    _EMOTION_PROSODY_HINTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_model(**kwargs) -> AudioModel:
    """Return an AudioModel forced into mock mode."""
    return AudioModel(mock_mode=True, **kwargs)


# ---------------------------------------------------------------------------
# AudioModel: constructor and properties
# ---------------------------------------------------------------------------

class TestAudioModelInit:
    def test_default_model_id(self) -> None:
        m = _mock_model()
        assert m.model_id == "LiquidAI/LFM2.5-Audio-1.5B"

    def test_custom_model_id(self) -> None:
        m = _mock_model(model_id="liquid-ai/lfm2.5-audio-1.5b")
        assert m.model_id == "liquid-ai/lfm2.5-audio-1.5b"

    def test_is_mock_true(self) -> None:
        m = _mock_model()
        assert m.is_mock is True

    def test_default_language(self) -> None:
        m = _mock_model()
        assert m.language == Language.ENGLISH

    def test_custom_language_hindi(self) -> None:
        m = _mock_model(language=Language.HINDI)
        assert m.language == Language.HINDI

    def test_custom_language_chinese(self) -> None:
        m = _mock_model(language=Language.CHINESE)
        assert m.language == Language.CHINESE

    def test_emotion_mode_default_true(self) -> None:
        m = _mock_model()
        assert m.emotion_mode is True

    def test_emotion_mode_can_be_disabled(self) -> None:
        m = _mock_model(emotion_mode=False)
        assert m.emotion_mode is False

    def test_llm_fallback_stored(self) -> None:
        cb = lambda text: "answer"
        m = _mock_model(llm_fallback=cb)
        assert m.llm_fallback is cb


# ---------------------------------------------------------------------------
# AudioModel: process_audio (mock path)
# ---------------------------------------------------------------------------

class TestAudioModelProcessAudio:
    def test_returns_audio_response(self) -> None:
        m = _mock_model()
        resp = m.process_audio(b"\x00" * 100)
        assert isinstance(resp, AudioResponse)

    def test_mock_text_contains_byte_count(self) -> None:
        m = _mock_model()
        audio = b"\x01" * 512
        resp = m.process_audio(audio)
        assert "512" in resp.text

    def test_mock_audio_bytes_is_none(self) -> None:
        m = _mock_model()
        resp = m.process_audio(b"\x00" * 64)
        assert resp.audio_bytes is None

    def test_emotion_default_neutral(self) -> None:
        m = _mock_model()
        resp = m.process_audio(b"\x00" * 64)
        assert resp.emotion == EmotionLabel.NEUTRAL

    def test_reply_emotion_default_neutral(self) -> None:
        m = _mock_model()
        resp = m.process_audio(b"\x00" * 64)
        assert resp.reply_emotion == EmotionLabel.NEUTRAL

    def test_tool_call_none_in_mock(self) -> None:
        m = _mock_model()
        resp = m.process_audio(b"\x00" * 64)
        assert resp.tool_call is None

    def test_used_llm_fallback_false_in_mock(self) -> None:
        m = _mock_model()
        resp = m.process_audio(b"\x00" * 64)
        assert resp.used_llm_fallback is False

    def test_language_override_per_call(self) -> None:
        m = _mock_model(language=Language.ENGLISH)
        resp = m.process_audio(b"\x00" * 64, language=Language.HINDI)
        assert "hi" in resp.text

    def test_empty_audio(self) -> None:
        m = _mock_model()
        resp = m.process_audio(b"")
        assert isinstance(resp, AudioResponse)


# ---------------------------------------------------------------------------
# AudioModel: synthesise (mock path)
# ---------------------------------------------------------------------------

class TestAudioModelSynthesise:
    def test_returns_none_in_mock(self) -> None:
        m = _mock_model()
        result = m.synthesise("Hello world")
        assert result is None

    def test_does_not_raise_in_mock(self) -> None:
        m = _mock_model()
        m.synthesise("Test", emotion=EmotionLabel.HAPPY)  # should not raise


# ---------------------------------------------------------------------------
# AudioModel: recognise_emotion (mock/no-emotion-mode path)
# ---------------------------------------------------------------------------

class TestAudioModelRecogniseEmotion:
    def test_mock_returns_neutral(self) -> None:
        m = _mock_model()
        label = m.recognise_emotion(b"\x00" * 100)
        assert label == EmotionLabel.NEUTRAL

    def test_emotion_mode_disabled_returns_neutral(self) -> None:
        m = _mock_model(emotion_mode=False)
        label = m.recognise_emotion(b"\x00" * 100)
        assert label == EmotionLabel.NEUTRAL


# ---------------------------------------------------------------------------
# AudioModel: unload
# ---------------------------------------------------------------------------

class TestAudioModelUnload:
    def test_unload_noop_when_not_loaded(self) -> None:
        m = _mock_model()
        m.unload()  # should not raise
        assert m._model is None
        assert m._processor is None


# ---------------------------------------------------------------------------
# AudioResponse
# ---------------------------------------------------------------------------

class TestAudioResponse:
    def _make(self, **kwargs) -> AudioResponse:
        defaults = dict(
            text="Hello",
            audio_bytes=None,
            emotion=EmotionLabel.NEUTRAL,
            reply_emotion=EmotionLabel.NEUTRAL,
            tool_call=None,
            used_llm_fallback=False,
        )
        defaults.update(kwargs)
        return AudioResponse(**defaults)

    def test_repr_contains_text(self) -> None:
        resp = self._make(text="Test reply")
        assert "Test reply" in repr(resp)

    def test_has_audio_true(self) -> None:
        resp = self._make(audio_bytes=b"\x00" * 10)
        assert "has_audio=True" in repr(resp)

    def test_has_audio_false(self) -> None:
        resp = self._make(audio_bytes=None)
        assert "has_audio=False" in repr(resp)

    def test_tool_call_present_in_repr(self) -> None:
        resp = self._make(tool_call={"name": "search", "args": {"q": "test"}})
        assert "tool_call=True" in repr(resp)

    def test_llm_fallback_in_repr(self) -> None:
        resp = self._make(used_llm_fallback=True)
        assert "llm_fallback=True" in repr(resp)


# ---------------------------------------------------------------------------
# EmotionLabel enum
# ---------------------------------------------------------------------------

class TestEmotionLabel:
    def test_all_labels_defined(self) -> None:
        labels = {e.value for e in EmotionLabel}
        expected = {"neutral", "happy", "sad", "excited", "frustrated", "calm", "surprised", "angry"}
        assert expected == labels

    def test_prosody_hints_for_all_emotions(self) -> None:
        # Every EmotionLabel must have a prosody hint entry
        for emotion in EmotionLabel:
            assert emotion in _EMOTION_PROSODY_HINTS


# ---------------------------------------------------------------------------
# Language enum
# ---------------------------------------------------------------------------

class TestLanguage:
    def test_values(self) -> None:
        assert Language.ENGLISH.value == "en"
        assert Language.HINDI.value == "hi"
        assert Language.CHINESE.value == "zh"

    def test_from_string(self) -> None:
        assert Language("en") == Language.ENGLISH
        assert Language("hi") == Language.HINDI
        assert Language("zh") == Language.CHINESE


# ---------------------------------------------------------------------------
# AudioModel._map_input_to_reply_emotion
# ---------------------------------------------------------------------------

class TestEmotionMapping:
    def test_happy_maps_to_happy(self) -> None:
        result = AudioModel._map_input_to_reply_emotion(EmotionLabel.HAPPY)
        assert result == EmotionLabel.HAPPY

    def test_sad_maps_to_calm(self) -> None:
        result = AudioModel._map_input_to_reply_emotion(EmotionLabel.SAD)
        assert result == EmotionLabel.CALM

    def test_frustrated_maps_to_calm(self) -> None:
        result = AudioModel._map_input_to_reply_emotion(EmotionLabel.FRUSTRATED)
        assert result == EmotionLabel.CALM

    def test_angry_maps_to_calm(self) -> None:
        result = AudioModel._map_input_to_reply_emotion(EmotionLabel.ANGRY)
        assert result == EmotionLabel.CALM

    def test_excited_maps_to_excited(self) -> None:
        result = AudioModel._map_input_to_reply_emotion(EmotionLabel.EXCITED)
        assert result == EmotionLabel.EXCITED

    def test_neutral_maps_to_neutral(self) -> None:
        result = AudioModel._map_input_to_reply_emotion(EmotionLabel.NEUTRAL)
        assert result == EmotionLabel.NEUTRAL


# ---------------------------------------------------------------------------
# AudioModel._extract_tool_call
# ---------------------------------------------------------------------------

class TestExtractToolCall:
    def test_valid_tool_call(self) -> None:
        text = 'Sure! [TOOL_CALL] {"name": "search", "args": {"q": "AI"}}'
        result = AudioModel._extract_tool_call(text)
        assert result == {"name": "search", "args": {"q": "AI"}}

    def test_no_tool_call(self) -> None:
        text = "Just a normal response."
        assert AudioModel._extract_tool_call(text) is None

    def test_malformed_json_returns_none(self) -> None:
        text = "[TOOL_CALL] {broken json}"
        assert AudioModel._extract_tool_call(text) is None


# ---------------------------------------------------------------------------
# AudioModel._needs_llm_fallback
# ---------------------------------------------------------------------------

class TestNeedsLLMFallback:
    def test_deep_think_trigger(self) -> None:
        assert AudioModel._needs_llm_fallback("Let me think... [DEEP_THINK]") is True

    def test_reasoning_needed_trigger(self) -> None:
        assert AudioModel._needs_llm_fallback("[REASONING_NEEDED] complex task") is True

    def test_llm_fallback_trigger(self) -> None:
        assert AudioModel._needs_llm_fallback("[LLM_FALLBACK] please help") is True

    def test_plain_text_no_fallback(self) -> None:
        assert AudioModel._needs_llm_fallback("Hello, how are you?") is False

    def test_empty_string_no_fallback(self) -> None:
        assert AudioModel._needs_llm_fallback("") is False
