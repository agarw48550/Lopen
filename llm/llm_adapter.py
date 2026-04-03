"""LLM adapter: abstraction over llama.cpp / llama-cpp-python with mock fallback.

Priority:
  1. llama-cpp-python (if installed and model file exists)
  2. subprocess call to llama.cpp binary
  3. Mock response (clearly logged)

Default model (April 2026): Qwen3.5-0.8B-Instruct Q4_K_M
  - File:   models/llm/qwen3.5-0.8b-instruct-q4_k_m.gguf
  - RAM:    ~0.55 GB
  - Speed:  8-12 tok/s on Intel Mac → first response <1s
  - Format: ChatML  (<|im_start|>system\n…<|im_end|>)

Supported chat formats:
  - "chatml"  — Qwen3.5, Qwen2, Mistral-Instruct (default)
  - "phi3"    — Phi-3-mini  (<|user|>…<|end|>)
  - "llama2"  — Llama-2/Mistral legacy  ([INST] … [/INST])
  - "raw"     — no special tokens (use for base/non-instruct models)

Thinking modes (Qwen3.5-0.8B):
  - ThinkingMode.AUTO       — automatic selection based on query complexity
  - ThinkingMode.THINKING   — enables <think> chain-of-thought block; best for
                              reasoning, planning, tool use, and multi-step tasks
  - ThinkingMode.NON_THINKING — disables CoT; fastest, lowest latency; best for
                                simple Q&A, completions, and voice replies
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thinking mode
# ---------------------------------------------------------------------------

class ThinkingMode(str, Enum):
    """Qwen3.5 thinking / non-thinking mode selector.

    AUTO selects the mode based on keyword heuristics applied to the user
    prompt at call time.  The other values force a specific mode.
    """
    AUTO = "auto"
    THINKING = "thinking"
    NON_THINKING = "non_thinking"


# Keywords that suggest a task needs deep reasoning (→ THINKING mode)
_THINKING_KEYWORDS: frozenset[str] = frozenset({
    "plan", "reason", "explain", "analyse", "analyze", "compare", "evaluate",
    "debug", "fix", "code", "implement", "design", "architecture", "review",
    "research", "summarise", "summarize", "translate", "calculate", "prove",
    "step by step", "step-by-step", "think", "help me understand", "how does",
    "why does", "what is the difference", "write a", "create a", "build a",
    "tool", "function", "script",
})


def _infer_thinking_mode(prompt: str) -> ThinkingMode:
    """Heuristic: detect whether a prompt needs deep reasoning."""
    lower = prompt.lower()
    # If more than one reasoning keyword appears, enable thinking
    hits = sum(1 for kw in _THINKING_KEYWORDS if kw in lower)
    return ThinkingMode.THINKING if hits >= 1 else ThinkingMode.NON_THINKING

_LLAMA_CPP_PYTHON_AVAILABLE = False
try:
    from llama_cpp import Llama  # type: ignore
    _LLAMA_CPP_PYTHON_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Thinking-mode prompt helpers (Qwen3.5 specific)
# ---------------------------------------------------------------------------

_THINKING_ENABLE_SUFFIX = "\n/think"   # appended to user turn to enable CoT
_THINKING_DISABLE_SUFFIX = "\n/no_think"  # appended to user turn to disable CoT

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove Qwen3.5 <think>…</think> block from a response."""
    return _THINK_TAG_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Chat format helpers
# ---------------------------------------------------------------------------

def _build_prompt(
    text: str,
    chat_format: str = "chatml",
    system: str = "",
    thinking_mode: ThinkingMode = ThinkingMode.NON_THINKING,
) -> str:
    """Wrap *text* in the correct instruct template for the active model.

    For Qwen3.5 (chatml format) the ``thinking_mode`` controls whether a
    chain-of-thought ``<think>`` block is generated:
    - THINKING     → appends ``/think`` to enable deep reasoning
    - NON_THINKING → appends ``/no_think`` to skip CoT (fastest)
    - AUTO         → determined upstream via ``_infer_thinking_mode``
    """
    sys_msg = system or (
        "You are Lopen, a helpful local AI assistant running on macOS. "
        "Be concise and accurate."
    )
    if chat_format == "chatml":
        # Qwen3.5 thinking/non-thinking control
        if thinking_mode == ThinkingMode.THINKING:
            user_turn = text + _THINKING_ENABLE_SUFFIX
        elif thinking_mode == ThinkingMode.NON_THINKING:
            user_turn = text + _THINKING_DISABLE_SUFFIX
        else:
            user_turn = text
        return (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_turn}<|im_end|>\n"
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
    - Qwen3.5-0.8B Q4_K_M: ~0.55 GB, 8-12 tok/s → first response <1s

    Thinking mode (Qwen3.5):
    - ThinkingMode.AUTO         — automatically choose based on query content
    - ThinkingMode.THINKING     — enable <think> CoT block (deep reasoning)
    - ThinkingMode.NON_THINKING — skip CoT (fastest, default for simple tasks)
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
        thinking_mode: ThinkingMode = ThinkingMode.AUTO,
    ) -> None:
        # Resolve model path: prefer explicit arg, then env, then default location
        self.model_path = (
            model_path
            or os.environ.get("LOPEN_MODEL_PATH", "")
            or str(Path("models/llm/qwen3.5-0.8b-instruct-q4_k_m.gguf"))
        )
        self.context_window = context_window
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_conservative = memory_conservative
        self.chat_format = chat_format
        self.system_prompt = system_prompt
        self.thinking_mode = thinking_mode
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
            "LLMAdapter mode=%s model=%s ctx=%d max_tok=%d format=%s thinking=%s",
            self._mode,
            Path(self.model_path).name,
            self.context_window,
            self.max_tokens,
            self.chat_format,
            self.thinking_mode.value,
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

    def chat(
        self,
        message: str,
        max_tokens: Optional[int] = None,
        thinking_mode: Optional[ThinkingMode] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat response.

        Wraps *message* in the correct instruct template for the active model
        (ChatML for Qwen3.5, Phi-3 format for Phi-3-mini, etc.) then calls
        ``generate()``.  Use this instead of ``generate()`` for all conversational
        or task-oriented queries.

        Args:
            message: The user message / query.
            max_tokens: Override token budget for this call.
            thinking_mode: Override the adapter-level thinking mode for this
                call.  ``None`` uses the adapter default (``self.thinking_mode``).
                Pass ``ThinkingMode.AUTO`` to auto-detect from *message* content.
        """
        # Resolve effective thinking mode for this call
        effective_mode = thinking_mode if thinking_mode is not None else self.thinking_mode
        if effective_mode == ThinkingMode.AUTO:
            effective_mode = _infer_thinking_mode(message)
            logger.debug(
                "ThinkingMode.AUTO resolved to %s for prompt: %r",
                effective_mode.value,
                message[:60],
            )

        prompt = _build_prompt(message, self.chat_format, self.system_prompt, effective_mode)
        raw = self.generate(prompt, max_tokens=max_tokens, **kwargs)

        # Strip <think>…</think> blocks from the user-facing reply so that the
        # chain-of-thought is transparent but not shown in the final answer.
        if effective_mode == ThinkingMode.THINKING and self.chat_format == "chatml":
            cleaned = _strip_think_tags(raw)
            if cleaned:
                return cleaned
        return raw

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
