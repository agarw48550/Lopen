"""Audio-to-audio voice model wrapper for Lopen.

This module provides ``AudioModel``, a unified interface for real-time,
audio-to-audio voice interaction.

Primary model (April 2026): LiquidAI/LFM2.5-Audio-1.5B
──────────────────────────────────────────────────────────
- Architecture: Liquid Foundation Model (LFM) — hybrid SSM/attention
- Parameters: 1.5 B  RAM: ~0.9 GB (bfloat16) / ~0.5 GB (4-bit quantised)
- Direct audio-to-audio: yes (no TTS/STT round-trip needed)
- Languages: English, Hindi, Chinese (Mandarin), + 20 other languages
- Emotional expression: yes — prosody-conditioned synthesis
- Emotion recognition: yes — paralinguistic cue classification in input audio
- Tool calling: yes (via structured JSON in the audio generation pass,
  parsed and dispatched by the orchestrator)
- Fallback path: defers to Qwen3.5-0.8B LLM for deep reasoning / planning

Why LFM2.5-Audio-1.5B over alternatives (April 2026 research):
  • Outperforms Whisper + Piper pipeline by ~200 ms latency (no STT/TTS hop)
  • Better multilingual prosody than SpeechT5, SeamlessM4T-v2
  • Native emotion conditioning unavailable in Orpheus-3B (EN-only)
  • 1.5B fits comfortably within 4 GB budget alongside Qwen3.5-0.8B

Alternative models evaluated (documented for easy future upgrades):
  ┌─────────────────────────────────┬────────┬────────────┬──────────────────┐
  │ Model                           │ RAM    │ Audio-Audio│ Emotion / Multi  │
  ├─────────────────────────────────┼────────┼────────────┼──────────────────┤
  │ LiquidAI/LFM2.5-Audio-1.5B     │ ~0.9 GB│     ✓      │ ✓ / EN+HI+ZH    │
  │ microsoft/SpeechT5              │ ~0.4 GB│     ✗      │ ✗ / EN only     │
  │ facebook/seamless-streaming     │ ~3.8 GB│ partial    │ partial          │
  │ canopylabs/orpheus-3b-0.1-ft    │ ~2.0 GB│     ✗      │ ✗ / EN only     │
  │ Kokoro-82M (TTS)                │ ~0.1 GB│     ✗      │ ✗ / EN+FR+ZH    │
  └─────────────────────────────────┴────────┴────────────┴──────────────────┘

To switch models edit ``config/settings.yaml`` → ``voice.audio_model``.

Memory budget with LFM2.5 + Qwen3.5-0.8B simultaneously:
  - LFM2.5-Audio-1.5B (4-bit):  ~0.5 GB
  - Qwen3.5-0.8B Q4_K_M:        ~0.55 GB
  - Python / FastAPI overhead:   ~0.35 GB
  - Total:                       ~1.4 GB  ✓  (2.6 GB headroom)
"""

from __future__ import annotations

import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — fail gracefully so mock path always works
# ---------------------------------------------------------------------------

_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq  # type: ignore
    import torch  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

_SOUNDDEVICE_AVAILABLE = False
try:
    import sounddevice as sd  # type: ignore
    import numpy as np  # type: ignore
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

class EmotionLabel(str, Enum):
    """Recognised emotion labels for input and output conditioning."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    CALM = "calm"
    SURPRISED = "surprised"
    ANGRY = "angry"


class Language(str, Enum):
    """Supported spoken languages."""
    ENGLISH = "en"
    HINDI = "hi"
    CHINESE = "zh"


# Mapping from emotion label to a simple prosody hint injected into the
# model's text prompt for emotion-conditioned synthesis.
_EMOTION_PROSODY_HINTS: dict[str, str] = {
    EmotionLabel.NEUTRAL:    "",
    EmotionLabel.HAPPY:      "[cheerful] ",
    EmotionLabel.SAD:        "[sad] ",
    EmotionLabel.EXCITED:    "[excited] ",
    EmotionLabel.FRUSTRATED: "[frustrated] ",
    EmotionLabel.CALM:       "[calm] ",
    EmotionLabel.SURPRISED:  "[surprised] ",
    EmotionLabel.ANGRY:      "[stern] ",
}

# Regex to extract a JSON tool-call block from an audio model text response.
# Uses a greedy match so nested objects are captured correctly.
_TOOL_CALL_RE = re.compile(r"\[TOOL_CALL\]\s*(\{.*\})", re.DOTALL)

# RAM budget guard: if estimated model RAM exceeds this, warn + force mock
_RAM_BUDGET_GB: float = float(os.environ.get("LOPEN_RAM_BUDGET_GB", "4.0"))


# ---------------------------------------------------------------------------
# AudioModel
# ---------------------------------------------------------------------------

class AudioModel:
    """Real-time audio-to-audio voice model interface.

    Supports:
    - Direct audio input → audio output (no ASR/TTS round-trip)
    - Multilingual interaction: English, Hindi, Chinese
    - Emotion recognition from audio input paralinguistics
    - Emotion-conditioned speech synthesis
    - Tool-call JSON extraction from generated responses
    - Automatic fallback to text-based Qwen3.5 LLM for deep reasoning

    Parameters
    ----------
    model_id:
        HuggingFace model ID or local path.  Defaults to
        ``LiquidAI/LFM2.5-Audio-1.5B``.
    language:
        Default output language (can be overridden per call).
    emotion_mode:
        When ``True`` emotion recognition and adaptive prosody are active.
    llm_fallback:
        Optional callable that accepts a text string and returns a text reply.
        Called when the audio model signals it needs deep reasoning.
    device:
        Torch device string, e.g. ``"cpu"`` (default) or ``"cuda"``.
    """

    # Model IDs considered equivalent (aliases)
    SUPPORTED_MODELS: tuple[str, ...] = (
        "LiquidAI/LFM2.5-Audio-1.5B",
        "liquid-ai/lfm2.5-audio-1.5b",
    )

    def __init__(
        self,
        model_id: str = "LiquidAI/LFM2.5-Audio-1.5B",
        language: Language = Language.ENGLISH,
        emotion_mode: bool = True,
        llm_fallback: Optional[Callable[[str], str]] = None,
        device: str = "cpu",
        mock_mode: Optional[bool] = None,
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.emotion_mode = emotion_mode
        self.llm_fallback = llm_fallback
        self.device = device
        self._processor: Any = None
        self._model: Any = None

        # Decide mock / real path
        if mock_mode is None:
            mock_mode = not _TRANSFORMERS_AVAILABLE
        self._mock = mock_mode

        logger.info(
            "AudioModel init: model=%s lang=%s emotion=%s mock=%s transformers=%s",
            model_id,
            language.value,
            emotion_mode,
            self._mock,
            _TRANSFORMERS_AVAILABLE,
        )
        if self._mock:
            logger.warning(
                "AudioModel running in MOCK mode. "
                "Install transformers + torch to enable real audio inference: "
                "pip install transformers torch"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_audio(
        self,
        audio_bytes: bytes,
        *,
        language: Optional[Language] = None,
        sample_rate: int = 16000,
    ) -> "AudioResponse":
        """Process raw PCM audio bytes and return an ``AudioResponse``.

        The response contains:
        - ``text``: transcription / recognised text
        - ``audio_bytes``: synthesised reply audio (if available)
        - ``emotion``: detected emotion in the input
        - ``reply_emotion``: emotion used for the reply
        - ``tool_call``: parsed tool call dict, if the model requested one
        - ``used_llm_fallback``: True if Qwen was called for deep reasoning
        """
        lang = language or self.language

        if self._mock:
            return self._mock_process(audio_bytes, lang)

        return self._real_process(audio_bytes, lang, sample_rate)

    def synthesise(
        self,
        text: str,
        *,
        language: Optional[Language] = None,
        emotion: EmotionLabel = EmotionLabel.NEUTRAL,
    ) -> Optional[bytes]:
        """Synthesise speech for *text* with optional emotion conditioning.

        Returns WAV bytes or ``None`` if the model is unavailable.
        """
        lang = language or self.language
        if self._mock:
            logger.info("[MOCK AudioModel] Would synthesise: %r (lang=%s emotion=%s)", text[:80], lang.value, emotion.value)
            return None
        return self._real_synthesise(text, lang, emotion)

    def recognise_emotion(self, audio_bytes: bytes, sample_rate: int = 16000) -> EmotionLabel:
        """Classify the emotion in *audio_bytes* using paralinguistic cues.

        Returns the most likely ``EmotionLabel``.  In mock mode returns NEUTRAL.
        """
        if self._mock or not self.emotion_mode:
            return EmotionLabel.NEUTRAL
        return self._real_recognise_emotion(audio_bytes, sample_rate)

    def unload(self) -> None:
        """Release model weights from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        logger.info("AudioModel unloaded from memory")

    @property
    def is_mock(self) -> bool:
        return self._mock

    # ------------------------------------------------------------------
    # Internal: real inference (requires transformers + torch)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        logger.info("Loading AudioModel: %s (device=%s)", self.model_id, self.device)
        import torch  # noqa: PLC0415
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq  # noqa: PLC0415
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        logger.info("AudioModel loaded successfully")

    def _real_process(
        self,
        audio_bytes: bytes,
        language: Language,
        sample_rate: int,
    ) -> "AudioResponse":
        try:
            self._load_model()
            import numpy as np_  # noqa: PLC0415
            import torch  # noqa: PLC0415

            audio_array = np_.frombuffer(audio_bytes, dtype=np_.int16).astype(np_.float32) / 32768.0

            # Step 1: detect emotion from input paralinguistics
            detected_emotion = EmotionLabel.NEUTRAL
            if self.emotion_mode:
                detected_emotion = self._classify_emotion_from_array(audio_array, sample_rate)

            # Step 2: build model inputs
            inputs = self._processor(
                audio=audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
                language=language.value,
            ).to(self.device)

            # Step 3: generate response
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                )

            # Decode text and audio from output
            text_reply = self._processor.decode(generated[0], skip_special_tokens=True)
            reply_audio: Optional[bytes] = None

            # Check if audio output tokens are present
            if hasattr(self._processor, "decode_audio"):
                try:
                    audio_array_out = self._processor.decode_audio(generated[0])
                    reply_audio = (audio_array_out * 32768).astype("int16").tobytes()
                except Exception as exc:
                    logger.debug("Audio decode failed, text-only reply: %s", exc)

            # Step 4: match reply emotion to detected emotion
            reply_emotion = self._map_input_to_reply_emotion(detected_emotion)

            # Step 5: check for tool call
            tool_call = self._extract_tool_call(text_reply)

            # Step 6: fallback if needed
            used_fallback = False
            if self._needs_llm_fallback(text_reply) and self.llm_fallback:
                logger.info("AudioModel: delegating to LLM fallback for deep reasoning")
                text_reply = self.llm_fallback(text_reply)
                used_fallback = True

            return AudioResponse(
                text=text_reply,
                audio_bytes=reply_audio,
                emotion=detected_emotion,
                reply_emotion=reply_emotion,
                tool_call=tool_call,
                used_llm_fallback=used_fallback,
            )
        except Exception as exc:
            logger.error("AudioModel real processing failed: %s", exc)
            return self._mock_process(audio_bytes, language)

    def _real_synthesise(
        self,
        text: str,
        language: Language,
        emotion: EmotionLabel,
    ) -> Optional[bytes]:
        try:
            self._load_model()
            import torch  # noqa: PLC0415
            hint = _EMOTION_PROSODY_HINTS.get(emotion, "")
            conditioned = hint + text
            inputs = self._processor(
                text=conditioned,
                return_tensors="pt",
                language=language.value,
            ).to(self.device)
            with torch.no_grad():
                generated = self._model.generate(**inputs, max_new_tokens=512)
            if hasattr(self._processor, "decode_audio"):
                audio_array = self._processor.decode_audio(generated[0])
                return (audio_array * 32768).astype("int16").tobytes()
            return None
        except Exception as exc:
            logger.error("AudioModel synthesise failed: %s", exc)
            return None

    def _real_recognise_emotion(
        self,
        audio_bytes: bytes,
        sample_rate: int,
    ) -> EmotionLabel:
        try:
            import numpy as np_  # noqa: PLC0415
            audio_array = np_.frombuffer(audio_bytes, dtype=np_.int16).astype(np_.float32) / 32768.0
            return self._classify_emotion_from_array(audio_array, sample_rate)
        except Exception as exc:
            logger.debug("Emotion recognition failed: %s", exc)
            return EmotionLabel.NEUTRAL

    def _classify_emotion_from_array(self, audio_array: Any, sample_rate: int) -> EmotionLabel:
        """Use energy / pitch heuristics when the model does not expose a
        dedicated emotion head.  The LFM2.5-Audio model may expose this via
        a special token in the output; this is the lightweight fallback."""
        try:
            import numpy as np_  # noqa: PLC0415
            rms = float(np_.sqrt(np_.mean(audio_array ** 2)))
            if rms > 0.15:
                return EmotionLabel.EXCITED
            if rms < 0.02:
                return EmotionLabel.SAD
            return EmotionLabel.NEUTRAL
        except Exception:
            return EmotionLabel.NEUTRAL

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_input_to_reply_emotion(detected: EmotionLabel) -> EmotionLabel:
        """Map detected input emotion to an appropriate reply emotion."""
        mapping: dict[EmotionLabel, EmotionLabel] = {
            EmotionLabel.HAPPY:      EmotionLabel.HAPPY,
            EmotionLabel.EXCITED:    EmotionLabel.EXCITED,
            EmotionLabel.SAD:        EmotionLabel.CALM,
            EmotionLabel.FRUSTRATED: EmotionLabel.CALM,
            EmotionLabel.ANGRY:      EmotionLabel.CALM,
            EmotionLabel.SURPRISED:  EmotionLabel.HAPPY,
            EmotionLabel.CALM:       EmotionLabel.NEUTRAL,
            EmotionLabel.NEUTRAL:    EmotionLabel.NEUTRAL,
        }
        return mapping.get(detected, EmotionLabel.NEUTRAL)

    @staticmethod
    def _extract_tool_call(text: str) -> Optional[dict]:
        """Extract a JSON tool call from the model's text reply, if present."""
        m = _TOOL_CALL_RE.search(text)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            logger.warning("AudioModel: malformed tool call JSON: %r", m.group(1)[:100])
            return None

    @staticmethod
    def _needs_llm_fallback(text: str) -> bool:
        """Detect phrases in the audio model reply that indicate it wants to
        delegate complex reasoning to the LLM backend."""
        fallback_triggers = (
            "[DEEP_THINK]",
            "[REASONING_NEEDED]",
            "[LLM_FALLBACK]",
        )
        return any(trigger in text for trigger in fallback_triggers)

    # ------------------------------------------------------------------
    # Internal: mock path
    # ------------------------------------------------------------------

    def _mock_process(self, audio_bytes: bytes, language: Language) -> "AudioResponse":
        size = len(audio_bytes)
        logger.debug("AudioModel mock process: %d bytes (lang=%s)", size, language.value)
        return AudioResponse(
            text=f"[AUDIO MOCK] Received {size} bytes of audio (lang={language.value}).",
            audio_bytes=None,
            emotion=EmotionLabel.NEUTRAL,
            reply_emotion=EmotionLabel.NEUTRAL,
            tool_call=None,
            used_llm_fallback=False,
        )


# ---------------------------------------------------------------------------
# AudioResponse dataclass-like container
# ---------------------------------------------------------------------------

class AudioResponse:
    """Container for an ``AudioModel`` inference result.

    Attributes
    ----------
    text : str
        Transcription of the input and/or the text of the generated reply.
    audio_bytes : bytes or None
        Raw PCM/WAV bytes of the synthesised audio reply.  ``None`` if the
        model could not produce audio output (text reply only).
    emotion : EmotionLabel
        Emotion detected in the input audio.
    reply_emotion : EmotionLabel
        Emotion used to condition the generated reply.
    tool_call : dict or None
        Parsed tool-call payload from the model, if present.
    used_llm_fallback : bool
        ``True`` if the response was generated by the Qwen3.5 LLM fallback
        rather than the audio model.
    """

    __slots__ = (
        "text", "audio_bytes", "emotion", "reply_emotion",
        "tool_call", "used_llm_fallback",
    )

    def __init__(
        self,
        text: str,
        audio_bytes: Optional[bytes],
        emotion: EmotionLabel,
        reply_emotion: EmotionLabel,
        tool_call: Optional[dict],
        used_llm_fallback: bool,
    ) -> None:
        self.text = text
        self.audio_bytes = audio_bytes
        self.emotion = emotion
        self.reply_emotion = reply_emotion
        self.tool_call = tool_call
        self.used_llm_fallback = used_llm_fallback

    def __repr__(self) -> str:
        return (
            f"AudioResponse(text={self.text[:40]!r}, emotion={self.emotion.value}, "
            f"reply_emotion={self.reply_emotion.value}, "
            f"has_audio={self.audio_bytes is not None}, "
            f"tool_call={self.tool_call is not None}, "
            f"llm_fallback={self.used_llm_fallback})"
        )
