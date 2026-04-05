"""HeavyLLM: Lazy-loaded heavy model for complex reasoning tasks.

Model: models/llm/qwen3.5-0.8b-instruct-q4_k_m.gguf
RAM: ~650 MB  |  Lazy-loaded on demand  |  Auto-unloaded after 30 s idle
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("models/llm/qwen3.5-0.8b-instruct-q4_k_m.gguf")
_IDLE_TIMEOUT: float = 30.0  # seconds before auto-unload

_LLAMA_CPP_AVAILABLE = False
try:
    from llama_cpp import Llama  # type: ignore
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    pass

_MOCK_GENERATE_PREFIX = "[mock-heavy] "


# ---------------------------------------------------------------------------
# HeavyLLM
# ---------------------------------------------------------------------------

class HeavyLLM:
    """Lazy-loading heavy LLM with automatic idle unload.

    The model is only loaded when ``generate()`` or ``chat()`` is first
    called.  A background watchdog thread tracks the last-use timestamp
    and calls ``unload()`` after ``_IDLE_TIMEOUT`` seconds of inactivity.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()
        self._last_used: float = 0.0
        self._watchdog: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._mock_mode: bool = not _LLAMA_CPP_AVAILABLE or not _MODEL_PATH.exists()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True when the model is currently loaded in memory."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a response for *prompt*.

        Loads the model on first call; resets idle timer on every call.

        Args:
            prompt: Raw prompt text.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """
        self._touch()
        if self._mock_mode:
            return f"{_MOCK_GENERATE_PREFIX}{prompt[:80]}..."
        model = self._ensure_loaded()
        return self._infer(model, prompt, max_tokens=max_tokens)

    def chat(self, message: str, max_tokens: int = 256) -> str:
        """Send a chat message and return the assistant reply.

        Wraps *message* in the ChatML instruct template before calling
        the model.

        Args:
            message: User message text.
            max_tokens: Maximum tokens to generate.

        Returns:
            Assistant reply string.
        """
        self._touch()
        if self._mock_mode:
            return f"{_MOCK_GENERATE_PREFIX}Re: {message[:60]}..."
        model = self._ensure_loaded()
        prompt = (
            "<|im_start|>system\n"
            "You are Lopen, a helpful local AI assistant. Be accurate and concise.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{message}\n/no_think\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return self._infer(model, prompt, max_tokens=max_tokens)

    def unload(self) -> None:
        """Release the model from memory and stop the watchdog thread."""
        with self._lock:
            if self._model is not None:
                try:
                    del self._model
                except Exception:  # pragma: no cover
                    pass
                self._model = None
                logger.info("HeavyLLM: model unloaded")
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _touch(self) -> None:
        """Update last-used timestamp and ensure watchdog is running."""
        self._last_used = time.monotonic()
        if self._watchdog is None or not self._watchdog.is_alive():
            self._stop_event.clear()
            self._watchdog = threading.Thread(
                target=self._watchdog_loop,
                daemon=True,
                name="heavy-llm-watchdog",
            )
            self._watchdog.start()

    def _ensure_loaded(self) -> Any:
        """Load the model if not already loaded, then return it."""
        with self._lock:
            if self._model is None:
                self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        """Load and return the Llama model instance."""
        logger.info("HeavyLLM: loading model from %s", _MODEL_PATH)
        try:
            model = Llama(  # type: ignore[name-defined]
                model_path=str(_MODEL_PATH),
                n_ctx=2048,
                n_threads=4,
                verbose=False,
            )
            logger.info("HeavyLLM: model loaded")
            return model
        except Exception as exc:  # pragma: no cover
            logger.error("HeavyLLM: failed to load model: %s", exc)
            raise

    def _infer(self, model: Any, prompt: str, max_tokens: int) -> str:
        """Run inference and return the generated text."""
        try:
            result = model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.2,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )
            return result["choices"][0]["text"].strip()
        except Exception as exc:  # pragma: no cover
            logger.warning("HeavyLLM inference error: %s", exc)
            return ""

    def _watchdog_loop(self) -> None:
        """Background thread: unload model after IDLE_TIMEOUT seconds of inactivity."""
        while not self._stop_event.is_set():
            time.sleep(5)
            if self._model is not None:
                idle = time.monotonic() - self._last_used
                if idle >= _IDLE_TIMEOUT:
                    logger.info("HeavyLLM: idle for %.1f s, unloading", idle)
                    self.unload()
                    return


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

_heavy_llm_instance: Optional[HeavyLLM] = None
_heavy_llm_lock = threading.Lock()


def get_heavy_llm() -> HeavyLLM:
    """Return a shared HeavyLLM instance (one per process, not a strict singleton)."""
    global _heavy_llm_instance
    with _heavy_llm_lock:
        if _heavy_llm_instance is None:
            _heavy_llm_instance = HeavyLLM()
    return _heavy_llm_instance
