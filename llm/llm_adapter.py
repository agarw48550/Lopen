"""LLM adapter: abstraction over llama.cpp / llama-cpp-python with mock fallback.

Priority:
  1. llama-cpp-python (if installed and model file exists)
  2. subprocess call to llama.cpp binary
  3. Mock response (clearly logged)
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_LLAMA_CPP_PYTHON_AVAILABLE = False
try:
    from llama_cpp import Llama  # type: ignore
    _LLAMA_CPP_PYTHON_AVAILABLE = True
except ImportError:
    pass


def _find_llama_binary(override: Optional[str] = None) -> Optional[str]:
    candidates = [override] if override else []
    candidates += [
        os.environ.get("LOPEN_LLAMA_BINARY", ""),
        str(Path.home() / ".local" / "bin" / "llama"),
        "/usr/local/bin/llama",
        "/opt/homebrew/bin/llama",
        "llama",
        "llama-cli",
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    try:
        for name in ["llama", "llama-cli"]:
            result = subprocess.run(["which", name], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception:
        pass
    return None


class LLMAdapter:
    """Abstraction over llama.cpp for text generation with automatic mock fallback."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        context_window: int = 2048,
        temperature: float = 0.7,
        max_tokens: int = 512,
        memory_conservative: bool = True,
        llama_binary: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or str(Path("models/llm/model.gguf"))
        self.context_window = context_window
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_conservative = memory_conservative
        self._llama_binary = _find_llama_binary(llama_binary)
        self._llama_instance: Optional[Any] = None

        model_exists = Path(self.model_path).is_file()
        if _LLAMA_CPP_PYTHON_AVAILABLE and model_exists:
            self._mode = "llama_cpp_python"
        elif self._llama_binary and model_exists:
            self._mode = "subprocess"
        else:
            self._mode = "mock"

        logger.info(
            "LLMAdapter mode=%s (model_exists=%s, llama_cpp_python=%s, binary=%s)",
            self._mode,
            model_exists,
            _LLAMA_CPP_PYTHON_AVAILABLE,
            self._llama_binary,
        )
        if self._mode == "mock":
            logger.warning(
                "LLMAdapter in MOCK mode. Download a GGUF model to %s to enable real inference.",
                self.model_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: Optional[int] = None, **kwargs: Any) -> str:
        """Generate text from a prompt."""
        tokens = max_tokens or self.max_tokens
        if self._mode == "llama_cpp_python":
            return self._generate_llama_cpp_python(prompt, tokens)
        elif self._mode == "subprocess":
            return self._generate_subprocess(prompt, tokens)
        else:
            return self._mock_response(prompt, tokens)

    def unload(self) -> None:
        """Free the loaded model from memory (used when memory_conservative=True)."""
        if self._llama_instance is not None:
            del self._llama_instance
            self._llama_instance = None
            logger.info("LLMAdapter: model unloaded from memory")

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _generate_llama_cpp_python(self, prompt: str, max_tokens: int) -> str:
        try:
            if self._llama_instance is None:
                logger.info("Loading LLM model: %s", self.model_path)
                self._llama_instance = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_window,
                    verbose=False,
                )
            output = self._llama_instance(
                prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                stop=["</s>", "\n\nUser:", "\n\nHuman:"],
            )
            text = output["choices"][0]["text"].strip()
            if self.memory_conservative:
                self.unload()
            return text
        except Exception as exc:
            logger.error("llama-cpp-python generation failed: %s", exc)
            return self._mock_response(prompt, max_tokens)

    def _generate_subprocess(self, prompt: str, max_tokens: int) -> str:
        try:
            result = subprocess.run(
                [
                    self._llama_binary,
                    "-m", self.model_path,
                    "-p", prompt,
                    "-n", str(max_tokens),
                    "--temp", str(self.temperature),
                    "--ctx-size", str(self.context_window),
                    "--log-disable",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.error("llama.cpp subprocess error: %s", result.stderr[:200])
                return self._mock_response(prompt, max_tokens)
            # llama.cpp echoes the prompt; strip it
            output = result.stdout
            if output.startswith(prompt):
                output = output[len(prompt):]
            return output.strip()
        except subprocess.TimeoutExpired:
            logger.error("llama.cpp timed out")
            return "[LLM] Request timed out."
        except Exception as exc:
            logger.error("llama.cpp subprocess failed: %s", exc)
            return self._mock_response(prompt, max_tokens)

    @staticmethod
    def _mock_response(prompt: str, max_tokens: int) -> str:
        short_prompt = prompt[:80].replace("\n", " ")
        return (
            f"[LLM MOCK] I received your prompt: '{short_prompt}…' "
            "To get real AI responses, download a GGUF model and set the model_path in config/settings.yaml. "
            "See scripts/download_models.sh for instructions."
        )


# Allow type hints without circular import
from typing import Any
