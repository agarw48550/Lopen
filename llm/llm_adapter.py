"""LLM adapter: abstraction over llama.cpp / llama-cpp-python with mock fallback.

Priority:
  1. llama-cpp-python (if installed and model file exists)
  2. subprocess call to llama.cpp binary
  3. Mock response (clearly logged)

Default model (April 2026): Qwen2.5-0.5B-Instruct Q4_K_M
  - File:   models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf
  - RAM:    ~360 MB
  - Speed:  8-12 tok/s on Intel Mac → first response <1s
  - Format: ChatML  (<|im_start|>system\n…<|im_end|>)

Supported chat formats:
  - "chatml"  — Qwen2.5, Qwen2, Mistral-Instruct (default)
  - "phi3"    — Phi-3-mini  (<|user|>…<|end|>)
  - "llama2"  — Llama-2/Mistral legacy  ([INST] … [/INST])
  - "raw"     — no special tokens (use for base/non-instruct models)
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LLAMA_CPP_PYTHON_AVAILABLE = False
try:
    from llama_cpp import Llama  # type: ignore
    _LLAMA_CPP_PYTHON_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Chat format helpers
# ---------------------------------------------------------------------------

def _build_prompt(text: str, chat_format: str = "chatml", system: str = "") -> str:
    """Wrap *text* in the correct instruct template for the active model."""
    sys_msg = system or (
        "You are Lopen, a helpful local AI assistant running on macOS. "
        "Be concise and accurate."
    )
    if chat_format == "chatml":
        return (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    if chat_format == "phi3":
        return f"<|system|>\n{sys_msg}<|end|>\n<|user|>\n{text}<|end|>\n<|assistant|>\n"
    if chat_format == "llama2":
        return f"[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{text} [/INST]"
    # raw — no wrapping
    return text


def _default_stop_tokens(chat_format: str) -> list[str]:
    if chat_format == "chatml":
        return ["<|im_end|>", "<|endoftext|>", "\n\nUser:", "\n\nHuman:"]
    if chat_format == "phi3":
        return ["<|end|>", "<|user|>"]
    if chat_format == "llama2":
        return ["[INST]", "</s>"]
    return ["</s>", "\n\nUser:"]


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def _find_llama_binary(override: Optional[str] = None) -> Optional[str]:
    candidates = [override] if override else []
    candidates += [
        os.environ.get("LOPEN_LLAMA_BINARY", ""),
        str(Path.home() / ".local" / "bin" / "llama-cli"),
        str(Path.home() / ".local" / "bin" / "llama"),
        "/usr/local/bin/llama-cli",
        "/usr/local/bin/llama",
        "llama-cli",
        "llama",
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    try:
        for name in ["llama-cli", "llama"]:
            result = subprocess.run(["which", name], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# LLMAdapter
# ---------------------------------------------------------------------------

class LLMAdapter:
    """Abstraction over llama.cpp for text generation with automatic mock fallback.

    Speed tuning for <2s responses on 2017 Intel MacBook Pro:
    - Default context_window=2048 (was 4096) — halves memory & first-token latency
    - Default max_tokens=256 (was 512) — faster replies for most tasks
    - Qwen2.5-0.5B Q4_K_M: ~360 MB, 8-12 tok/s → first response <1s
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        context_window: int = 2048,
        temperature: float = 0.7,
        max_tokens: int = 256,
        memory_conservative: bool = True,
        llama_binary: Optional[str] = None,
        chat_format: str = "chatml",
        system_prompt: str = "",
    ) -> None:
        # Resolve model path: prefer explicit arg, then env, then default location
        self.model_path = (
            model_path
            or os.environ.get("LOPEN_MODEL_PATH", "")
            or str(Path("models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf"))
        )
        self.context_window = context_window
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_conservative = memory_conservative
        self.chat_format = chat_format
        self.system_prompt = system_prompt
        self._stop_tokens = _default_stop_tokens(chat_format)
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
            "LLMAdapter mode=%s model=%s ctx=%d max_tok=%d format=%s",
            self._mode,
            Path(self.model_path).name,
            self.context_window,
            self.max_tokens,
            self.chat_format,
        )
        if self._mode == "mock":
            logger.warning(
                "LLMAdapter in MOCK mode. Download a GGUF model to %s to enable real inference. "
                "Run: bash scripts/download_models.sh",
                self.model_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: Optional[int] = None, **kwargs: Any) -> str:
        """Generate text from a raw prompt string.

        For chat/instruct models, prefer ``chat()`` which applies the correct
        template automatically.
        """
        tokens = max_tokens or self.max_tokens
        if self._mode == "llama_cpp_python":
            return self._generate_llama_cpp_python(prompt, tokens)
        elif self._mode == "subprocess":
            return self._generate_subprocess(prompt, tokens)
        else:
            return self._mock_response(prompt, tokens)

    def chat(self, message: str, max_tokens: Optional[int] = None, **kwargs: Any) -> str:
        """Generate a chat response.

        Wraps *message* in the correct instruct template for the active model
        (ChatML for Qwen2.5, Phi-3 format for Phi-3-mini, etc.) then calls
        ``generate()``.  Use this instead of ``generate()`` for all conversational
        or task-oriented queries.
        """
        prompt = _build_prompt(message, self.chat_format, self.system_prompt)
        return self.generate(prompt, max_tokens=max_tokens, **kwargs)

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
                logger.info("Loading LLM model: %s (ctx=%d)", self.model_path, self.context_window)
                self._llama_instance = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_window,
                    n_threads=max(1, (os.cpu_count() or 1) - 1),  # leave one core free; default 1 if unknown
                    verbose=False,
                )
            output = self._llama_instance(
                prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                stop=self._stop_tokens,
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
                    "--model", self.model_path,
                    "--prompt", prompt,
                    "--n-predict", str(max_tokens),
                    "--temp", str(self.temperature),
                    "--ctx-size", str(self.context_window),
                    "--threads", str(max(1, (os.cpu_count() or 1) - 1)),
                    "--log-disable",
                    "--no-display-prompt",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.error("llama.cpp subprocess error: %s", result.stderr[:200])
                return self._mock_response(prompt, max_tokens)
            output = result.stdout.strip()
            # Some llama.cpp builds echo the prompt; strip it
            if output.startswith(prompt):
                output = output[len(prompt):].strip()
            # Strip any trailing stop tokens
            for token in self._stop_tokens:
                if output.endswith(token):
                    output = output[: -len(token)].strip()
            return output
        except subprocess.TimeoutExpired:
            logger.error("llama.cpp timed out after 60s")
            return "[LLM] Request timed out — try a smaller max_tokens or simpler prompt."
        except Exception as exc:
            logger.error("llama.cpp subprocess failed: %s", exc)
            return self._mock_response(prompt, max_tokens)

    @staticmethod
    def _mock_response(prompt: str, max_tokens: int) -> str:  # noqa: ARG004
        short_prompt = prompt[:80].replace("\n", " ")
        return (
            f"[LLM MOCK] Received: '{short_prompt}…' "
            "To enable real AI responses, download a model and install llama-cpp-python: "
            "bash scripts/download_models.sh && pip install llama-cpp-python"
        )
