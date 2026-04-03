"""ASR adapter: transcribe audio bytes to text.

Uses whisper.cpp binary if available; falls back to mock transcription.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _find_whisper_binary(override: Optional[str] = None) -> Optional[str]:
    """Locate the whisper.cpp binary."""
    candidates = [override] if override else []
    candidates += [
        os.environ.get("LOPEN_WHISPER_BINARY", ""),
        str(Path.home() / ".local" / "bin" / "whisper"),
        "/usr/local/bin/whisper",
        "/opt/homebrew/bin/whisper",
        "whisper",
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    # Check PATH
    try:
        result = subprocess.run(["which", "whisper"], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


class ASRAdapter:
    """Transcribe raw PCM/WAV audio bytes to text."""

    def __init__(
        self,
        binary_path: Optional[str] = None,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        mock_response: str = "this is a mock transcription",
    ) -> None:
        self.binary = _find_whisper_binary(binary_path)
        self.model_path = model_path or str(Path("models/asr/ggml-tiny.en.bin").resolve())
        self.sample_rate = sample_rate
        self._mock_response = mock_response
        self._mock_mode = self.binary is None or not Path(self.model_path).is_file()

        if self._mock_mode:
            logger.warning(
                "ASRAdapter running in MOCK mode (binary=%s, model_exists=%s)",
                self.binary,
                Path(self.model_path).is_file(),
            )
        else:
            logger.info("ASRAdapter using whisper binary: %s", self.binary)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, *, language: str = "en") -> str:
        """Transcribe audio bytes to text. Returns empty string on error."""
        if self._mock_mode:
            logger.debug("ASR mock transcription returned")
            return self._mock_response

        return self._whisper_transcribe(audio_bytes, language=language)

    @property
    def is_mock(self) -> bool:
        return self._mock_mode

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _whisper_transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        """Write audio to a temp WAV file and call whisper.cpp binary."""
        # Write bytes to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            try:
                with wave.open(tmp_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_bytes)
            except Exception:
                # If audio_bytes is already a valid WAV, just write raw
                with open(tmp_path, "wb") as f:
                    f.write(audio_bytes)

        try:
            result = subprocess.run(
                [
                    self.binary,
                    "-m", self.model_path,
                    "-f", tmp_path,
                    "-l", language,
                    "--no-timestamps",
                    "-otxt",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error("whisper.cpp error: %s", result.stderr[:200])
                return ""
            transcript = result.stdout.strip()
            logger.info("ASR transcribed: %r", transcript[:80])
            return transcript
        except subprocess.TimeoutExpired:
            logger.error("whisper.cpp timed out")
            return ""
        except Exception as exc:
            logger.error("ASR failed: %s", exc)
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
