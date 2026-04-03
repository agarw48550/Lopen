"""TTS adapter: convert text to speech.

Priority:
  1. piper binary (male voice: en_US-ryan-high)
  2. macOS 'say' command
  3. Mock (log only)
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _find_piper_binary(override: Optional[str] = None) -> Optional[str]:
    candidates = [override] if override else []
    candidates += [
        os.environ.get("LOPEN_PIPER_BINARY", ""),
        str(Path.home() / ".local" / "bin" / "piper"),
        "/usr/local/bin/piper",
        "/opt/homebrew/bin/piper",
        "piper",
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    try:
        result = subprocess.run(["which", "piper"], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _say_available() -> bool:
    return platform.system() == "Darwin"


class TTSAdapter:
    """Convert text to audible speech."""

    def __init__(
        self,
        binary_path: Optional[str] = None,
        voice_model: str = "en_US-ryan-high",
        model_dir: str = "models/tts",
    ) -> None:
        self.binary = _find_piper_binary(binary_path)
        self.voice_model = voice_model
        self.model_path = str(Path(model_dir) / f"{voice_model}.onnx")
        self._say_available = _say_available()

        if self.binary and Path(self.model_path).is_file():
            self._mode = "piper"
        elif self._say_available:
            self._mode = "say"
        else:
            self._mode = "mock"

        logger.info("TTSAdapter mode=%s (piper=%s, say=%s)", self._mode, self.binary, self._say_available)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> bool:
        """Speak the given text. Returns True on success."""
        if not text.strip():
            return False
        logger.info("TTS speak: %r (mode=%s)", text[:80], self._mode)
        if self._mode == "piper":
            return self._speak_piper(text)
        elif self._mode == "say":
            return self._speak_say(text)
        else:
            logger.info("[MOCK TTS] Would say: %s", text)
            return True

    def synthesise(self, text: str) -> Optional[bytes]:
        """Return WAV bytes for the given text (piper only; None if unavailable)."""
        if self._mode != "piper":
            return None
        return self._piper_to_bytes(text)

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _speak_piper(self, text: str) -> bool:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [self.binary, "--model", self.model_path, "--output_file", tmp_path],
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error("piper error: %s", result.stderr[:200])
                return False
            self._play_wav(tmp_path)
            return True
        except Exception as exc:
            logger.error("TTS piper failed: %s", exc)
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _piper_to_bytes(self, text: str) -> Optional[bytes]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [self.binary, "--model", self.model_path, "--output_file", tmp_path],
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None
            with open(tmp_path, "rb") as f:
                return f.read()
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _speak_say(self, text: str) -> bool:
        try:
            subprocess.run(["say", text], check=True, timeout=30)
            return True
        except Exception as exc:
            logger.error("macOS say failed: %s", exc)
            return False

    @staticmethod
    def _play_wav(path: str) -> None:
        """Play a WAV file using available player."""
        for cmd in [["afplay", path], ["aplay", path], ["play", path]]:
            try:
                subprocess.run(cmd, check=True, timeout=30, capture_output=True)
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        logger.warning("No audio player found to play %s", path)
