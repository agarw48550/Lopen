"""Wake word detection adapter for Lopen.

Uses openwakeword if available; falls back to simple keyword search in transcript.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_OWW_AVAILABLE = False
try:
    import openwakeword  # type: ignore
    _OWW_AVAILABLE = True
except ImportError:
    pass


class WakeWordDetector:
    """Detect the configured wake word and call a callback when heard."""

    def __init__(
        self,
        wake_word: str = "lopen",
        callback: Optional[Callable[[], None]] = None,
        sensitivity: float = 0.5,
    ) -> None:
        self.wake_word = wake_word.lower()
        self.callback = callback
        self.sensitivity = sensitivity
        self._running = False
        self._thread: Optional[threading.Thread] = None
        logger.info(
            "WakeWordDetector initialised (wake_word=%r, oww_available=%s)",
            wake_word,
            _OWW_AVAILABLE,
        )

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="wake-word-detector")
        self._thread.start()
        logger.info("WakeWordDetector started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("WakeWordDetector stopped")

    # ------------------------------------------------------------------
    # Detection helpers (public for testing)
    # ------------------------------------------------------------------

    def check_transcript(self, transcript: str) -> bool:
        """Return True if the wake word appears in the transcript."""
        detected = self.wake_word in transcript.lower()
        if detected:
            logger.info("Wake word %r detected in transcript", self.wake_word)
            self._fire_callback()
        return detected

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        if _OWW_AVAILABLE:
            self._run_oww()
        else:
            logger.warning(
                "openwakeword not installed — wake word detection in transcript-only mode"
            )
            # In transcript-only mode we just sleep; check_transcript is called externally
            while self._running:
                time.sleep(1)

    def _run_oww(self) -> None:
        """Run openwakeword in a background loop (stub — full integration requires audio stream)."""
        logger.info("openwakeword available but audio-stream integration not configured; waiting…")
        while self._running:
            time.sleep(1)

    def _fire_callback(self) -> None:
        if self.callback:
            try:
                self.callback()
            except Exception as exc:
                logger.error("Wake word callback raised: %s", exc)
