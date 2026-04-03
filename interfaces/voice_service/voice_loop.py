"""Main voice service loop: microphone → wake word → AudioModel (audio-to-audio) → speaker.

Architecture (April 2026):
  - PRIMARY path: raw audio → LiquidAI/LFM2.5-Audio-1.5B → audio reply
    No TTS/STT round-trip: latency ~200-400 ms end-to-end.
  - FALLBACK path (text): when AudioModel is unavailable, falls back to
    the classic ASR → LLM → TTS pipeline.
  - LLM OFFLOAD: AudioModel automatically delegates deep reasoning to the
    Qwen3.5-0.8B LLM when it detects the query needs it.
  - EMOTION ADAPTATION: input emotion is recognised and the reply prosody
    is adapted to match the user's energy.
  - TOOL CALLS: tool-call JSON emitted by the audio model is extracted and
    dispatched through the on_query handler.
  - LANGUAGE: English, Hindi, Chinese supported end-to-end.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_SOUNDDEVICE_AVAILABLE = False
try:
    import sounddevice as sd  # type: ignore
    import numpy as np
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    pass


class VoiceLoop:
    """Orchestrates microphone input, wake word, audio model inference, and speaker output.

    Primary (audio-to-audio) pipeline::

        mic → AudioModel.process_audio() → play audio_bytes  (fastest path)
                    ↓ (if deep reasoning needed)
               LLM.chat() → AudioModel.synthesise() → play audio_bytes

    Legacy fallback pipeline (when AudioModel unavailable)::

        mic → ASRAdapter.transcribe() → LLM → TTSAdapter.speak()
    """

    def __init__(
        self,
        wake_word_detector: Any,
        asr_adapter: Any,
        tts_adapter: Any,
        llm_adapter: Any | None = None,
        on_query: Optional[Callable[[str], str]] = None,
        sample_rate: int = 16000,
        microphone_index: Optional[int] = None,
        chunk_seconds: float = 2.0,
        audio_model: Any | None = None,
        language: str = "en",
    ) -> None:
        self.wake_detector = wake_word_detector
        self.asr = asr_adapter
        self.tts = tts_adapter
        self.llm = llm_adapter
        self.on_query = on_query  # external handler for the transcribed text
        self.sample_rate = sample_rate
        self.microphone_index = microphone_index
        self.chunk_seconds = chunk_seconds
        self._running = False
        self._wake_detected = False
        self.audio_model = audio_model  # LFM2.5-Audio (audio-to-audio model)
        self.language = language  # "en" | "hi" | "zh"

        logger.info(
            "VoiceLoop initialised (sounddevice=%s, sample_rate=%d, "
            "audio_model=%s, lang=%s)",
            _SOUNDDEVICE_AVAILABLE,
            sample_rate,
            type(audio_model).__name__ if audio_model else "None",
            language,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self.wake_detector.callback = self._on_wake_word
        self.wake_detector.start()
        logger.info("VoiceLoop started")
        await self._main_loop()

    def stop(self) -> None:
        self._running = False
        self.wake_detector.stop()
        logger.info("VoiceLoop stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        while self._running:
            if not _SOUNDDEVICE_AVAILABLE:
                logger.warning("sounddevice not installed — voice loop in idle mode (polling every 5 s)")
                await asyncio.sleep(5)
                continue

            audio_chunk = await self._record_chunk()

            # ---------- Audio-to-audio path (LFM2.5-Audio) ----------
            if self.audio_model is not None:
                await self._handle_audio_chunk(audio_chunk)
                continue

            # ---------- Legacy ASR + LLM + TTS path ----------
            transcript = self.asr.transcribe(audio_chunk)
            if not transcript:
                await asyncio.sleep(0.1)
                continue

            if self.wake_detector.check_transcript(transcript) or self._wake_detected:
                self._wake_detected = False
                query = self._strip_wake_word(transcript)
                if not query:
                    audio_chunk = await self._record_chunk()
                    query = self.asr.transcribe(audio_chunk)

                if query:
                    await self._handle_query(query)

    # ------------------------------------------------------------------
    # Audio-to-audio handler (primary path)
    # ------------------------------------------------------------------

    async def _handle_audio_chunk(self, audio_bytes: bytes) -> None:
        """Process audio through the LFM2.5-Audio model and play the reply."""
        from interfaces.voice_service.audio_model import Language  # lazy import

        # Map language string to enum
        try:
            lang = Language(self.language)
        except ValueError:
            lang = Language.ENGLISH

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.audio_model.process_audio(
                    audio_bytes,
                    language=lang,
                    sample_rate=self.sample_rate,
                ),
            )
        except Exception as exc:
            logger.error("AudioModel.process_audio failed: %s", exc)
            return

        logger.info(
            "AudioModel response: emotion=%s reply_emotion=%s llm_fallback=%s tool=%s text=%r",
            response.emotion.value,
            response.reply_emotion.value,
            response.used_llm_fallback,
            response.tool_call,
            response.text[:80],
        )

        # Dispatch tool call if present
        if response.tool_call and self.on_query:
            tool_json = str(response.tool_call)
            try:
                result = self.on_query(f"[TOOL_CALL] {tool_json}")
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("Tool call dispatch failed: %s", exc)

        # Play audio reply if available, otherwise fall back to TTS
        if response.audio_bytes:
            await self._play_audio_bytes(response.audio_bytes)
        elif response.text:
            self.tts.speak(response.text)

    # ------------------------------------------------------------------
    # Legacy text query handler (fallback path)
    # ------------------------------------------------------------------

    async def _handle_query(self, query: str) -> None:
        logger.info("Voice query: %r", query)
        response = "[No response]"
        if self.on_query:
            try:
                result = self.on_query(query)
                if asyncio.iscoroutine(result):
                    response = await result
                else:
                    response = result
            except Exception as exc:
                logger.error("Query handler failed: %s", exc)
                response = "Sorry, I encountered an error."
        elif self.llm:
            try:
                response = self.llm.generate(query, max_tokens=256)
            except Exception as exc:
                logger.error("LLM failed: %s", exc)
                response = "Sorry, I couldn't process that."

        self.tts.speak(response)

    # ------------------------------------------------------------------
    # Audio I/O helpers
    # ------------------------------------------------------------------

    async def _record_chunk(self) -> bytes:
        """Record a chunk of audio and return raw PCM bytes."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._blocking_record)

    def _blocking_record(self) -> bytes:
        frames = int(self.sample_rate * self.chunk_seconds)
        try:
            recording = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=self.microphone_index,
            )
            sd.wait()
            return recording.tobytes()
        except Exception as exc:
            logger.error("Audio recording failed: %s", exc)
            time.sleep(self.chunk_seconds)
            return b""

    async def _play_audio_bytes(self, audio_bytes: bytes) -> None:
        """Play raw PCM int16 audio bytes through the default output device."""
        if not _SOUNDDEVICE_AVAILABLE or not audio_bytes:
            logger.debug("VoiceLoop: cannot play audio (sounddevice=%s)", _SOUNDDEVICE_AVAILABLE)
            return
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._blocking_play, audio_bytes)
        except Exception as exc:
            logger.error("Audio playback failed: %s", exc)

    def _blocking_play(self, audio_bytes: bytes) -> None:
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio_array, samplerate=self.sample_rate)
            sd.wait()
        except Exception as exc:
            logger.error("_blocking_play failed: %s", exc)

    # ------------------------------------------------------------------
    # Wake word helpers
    # ------------------------------------------------------------------

    def _strip_wake_word(self, transcript: str) -> str:
        lower = transcript.lower()
        ww = self.wake_detector.wake_word.lower()
        if lower.startswith(ww):
            return transcript[len(ww):].strip()
        return transcript.strip()

    def _on_wake_word(self) -> None:
        self._wake_detected = True
        logger.info("Wake word callback fired")
