"""Main voice service loop: microphone → wake word → ASR → LLM → TTS."""

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
    """Orchestrates microphone input, wake word detection, ASR, LLM inference, and TTS output."""

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

        logger.info(
            "VoiceLoop initialised (sounddevice=%s, sample_rate=%d)",
            _SOUNDDEVICE_AVAILABLE,
            sample_rate,
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

            # First pass: check for wake word via transcript
            transcript = self.asr.transcribe(audio_chunk)
            if not transcript:
                await asyncio.sleep(0.1)
                continue

            if self.wake_detector.check_transcript(transcript) or self._wake_detected:
                self._wake_detected = False
                # Strip wake word from transcript and process
                query = self._strip_wake_word(transcript)
                if not query:
                    # Record another chunk for the actual command
                    audio_chunk = await self._record_chunk()
                    query = self.asr.transcribe(audio_chunk)

                if query:
                    await self._handle_query(query)

    # ------------------------------------------------------------------
    # Helpers
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

    def _strip_wake_word(self, transcript: str) -> str:
        lower = transcript.lower()
        ww = self.wake_detector.wake_word.lower()
        if lower.startswith(ww):
            return transcript[len(ww):].strip()
        return transcript.strip()

    def _on_wake_word(self) -> None:
        self._wake_detected = True
        logger.info("Wake word callback fired")
