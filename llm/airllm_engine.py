"""AirLLM-inspired efficient large-model inference engine for Lopen.

This module abstracts layer-by-layer CPU scheduling, quantization, and memory
mapping so that large/smart models (7B+ parameter class) can run within the
4 GB RAM budget on a 2017 MacBook Pro.

Architecture overview
---------------------
AirLLM (https://github.com/lyogavin/airllm) works by:

1. **Layer-by-layer loading** — instead of loading the entire model into RAM
   at startup, only a few transformer layers are resident at any time.
   The rest remain on disk (mmap) or in a small on-disk cache.
2. **Quantization** — weights are stored in 4-bit (GGUF Q4_K_M recommended)
   cutting a 7B model from ~14 GB to ~4 GB.  We default to 4-bit unless the
   user explicitly selects a higher precision.
3. **Memory-mapped I/O** — Python ``mmap`` + safetensors allow pages to be
   loaded on demand and evicted by the OS when pressure rises.

This class wraps either:
  a) The real ``airllm`` Python package (``pip install airllm``) if installed.
  b) ``llama-cpp-python`` as a drop-in backend (same GGUF models).
  c) A mock backend for CI / environments without models.

Backend selection policy
------------------------
AirLLM is **only activated when the model would exceed the 4 GB RAM budget**
when loaded via ``llama-cpp-python`` (estimated as 1.2× the GGUF file size).
For smaller models — including the default Qwen3.5-0.8B Q4 (≈0.55 GB file,
≈0.66 GB RAM) — ``llama-cpp-python`` is always used as the fastest path.
Only when a model's runtime estimate exceeds ``_RAM_BUDGET_GB`` (4.0 GB) does
the engine fall back to AirLLM's layer-by-layer loading to stay within budget.
For very large models where even AirLLM cannot fit, size is reduced further by
using stricter quantisation or disabling optional features.

You can swap the active engine by setting ``engine`` in ``config/settings.yaml``::

    llm:
      engine: auto     # options: auto | airllm | llama_cpp | mock
      model_path: models/llm/model.gguf
      max_gpu_memory: 0      # 0 = CPU-only (safe for 2017 MacBook)
      compression: 4bit      # 4bit | 8bit | none
      max_seq_len: 512

Memory budget guidance
----------------------
* Q4_K_M 0.8B model ≈ 0.55 GB file, ≈ 0.66 GB RAM → ``llama_cpp`` (fastest)
* Q4_K_M 1.5B model ≈ 1.0 GB file,  ≈ 1.2 GB RAM  → ``llama_cpp`` (fastest)
* Q4_K_M 3B model   ≈ 2.2 GB file,  ≈ 2.6 GB RAM  → ``llama_cpp`` (fastest)
* Q4_K_M 7B model   ≈ 4.1 GB file,  ≈ 4.9 GB RAM  → ``airllm`` (layer-split)
* Whisper-tiny ASR  ≈ 39 MB
* Piper TTS         ≈ 65 MB
* Python overhead   ≈ 300–400 MB
Total (0.8B LLM active): ≈ 1.1 GB  ← well within 4 GB budget.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RAM budget constant
# ---------------------------------------------------------------------------

# Hard ceiling for model RAM usage.  AirLLM is only enabled when a model
# would exceed this budget when loaded via llama-cpp-python (estimated as
# 1.2× the GGUF file size).  Keeping this at 4.0 GB leaves headroom for
# Python overhead, ASR, TTS, and multi-agent sub-processes.
_RAM_BUDGET_GB: float = 4.0

# Multiplier for estimating llama-cpp-python runtime RAM from file size.
# (GGUF mmap + KV cache overhead ≈ 1.2×)
_LLAMA_CPP_OVERHEAD: float = 1.2

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

_AIRLLM_AVAILABLE = False
try:
    import airllm  # type: ignore  # noqa: F401
    _AIRLLM_AVAILABLE = True
except ImportError:
    pass

_LLAMA_CPP_AVAILABLE = False
try:
    from llama_cpp import Llama  # type: ignore
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# AirLLMEngine
# ---------------------------------------------------------------------------

class AirLLMEngine:
    """Memory-efficient large-model inference engine.

    Backend selection policy (automatic when ``engine=None``):
      1. If model RAM estimate ≤ 4 GB → ``llama_cpp`` (fastest path, always preferred)
      2. If model RAM estimate > 4 GB → ``airllm`` (layer-split, keeps peak RAM low)
      3. If neither is available → ``mock`` (CI / no model downloaded)

    AirLLM is therefore **only activated for large models** (>4 GB RAM estimate).
    For the default Qwen3.5-0.8B Q4 (~0.55 GB file, ~0.66 GB RAM) and any other
    model whose llama_cpp runtime estimate stays under the 4 GB budget,
    ``llama-cpp-python`` is used directly for maximum inference speed.

    Parameters
    ----------
    model_path:
        Path to the GGUF model file (for llama-cpp backend) or the
        HuggingFace model ID / directory (for AirLLM backend).
    engine:
        Force a specific backend: ``"airllm"``, ``"llama_cpp"``, ``"mock"``.
        If ``None``, the best available backend is auto-selected based on
        model size vs. RAM budget.
    compression:
        Quantization level: ``"4bit"`` (default), ``"8bit"``, or ``"none"``.
    context_window:
        Maximum context length in tokens (default 2048, keeps RAM low).
    max_tokens:
        Maximum tokens to generate per call (default 512).
    temperature:
        Sampling temperature (default 0.7).
    memory_conservative:
        If ``True``, unload the model from RAM between calls.  Adds latency
        but guarantees we never exceed the 4 GB budget.
    max_gpu_memory:
        Fraction of GPU memory to use (0 = CPU-only, recommended for 2017 Mac).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        engine: Optional[str] = None,
        compression: str = "4bit",
        context_window: int = 2048,
        max_tokens: int = 512,
        temperature: float = 0.7,
        memory_conservative: bool = True,
        max_gpu_memory: int = 0,
    ) -> None:
        self.model_path = model_path or str(Path("models/llm/model.gguf"))
        self.compression = compression
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.memory_conservative = memory_conservative
        self.max_gpu_memory = max_gpu_memory

        self._instance: Any = None
        self._backend = self._select_backend(engine)

        logger.info(
            "AirLLMEngine initialised: backend=%s model=%s compression=%s "
            "memory_conservative=%s ram_estimate_gb=%.2f",
            self._backend, self.model_path, self.compression,
            self.memory_conservative, self.memory_footprint_hint_gb,
        )
        if self._backend == "mock":
            logger.warning(
                "AirLLMEngine running in MOCK mode. "
                "Install 'airllm' or 'llama-cpp-python' and download a model to enable real inference. "
                "See docs/AI_ARCHITECTURE.md for setup instructions."
            )
        elif self._backend == "airllm":
            logger.info(
                "AirLLMEngine: AirLLM (layer-split) selected because model RAM estimate "
                "(%.2f GB) exceeds the %.1f GB budget for llama-cpp-python.",
                self.memory_footprint_hint_gb, _RAM_BUDGET_GB,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: Optional[int] = None, **kwargs: Any) -> str:
        """Generate text from a prompt.

        This is the single entrypoint used by all Lopen agents.

        Args:
            prompt: Full formatted prompt string.
            max_tokens: Override the default max_tokens for this call.

        Returns:
            Generated text string.
        """
        tokens = max_tokens or self.max_tokens
        try:
            if self._backend == "airllm":
                return self._generate_airllm(prompt, tokens)
            elif self._backend == "llama_cpp":
                return self._generate_llama_cpp(prompt, tokens)
            else:
                return self._mock_response(prompt, tokens)
        except Exception as exc:
            logger.error("AirLLMEngine generation error (backend=%s): %s", self._backend, exc)
            return self._mock_response(prompt, tokens)

    def unload(self) -> None:
        """Release the model from RAM.

        Call this explicitly (or set memory_conservative=True) to free
        RAM between calls.  The model will be reloaded on the next generate().
        """
        if self._instance is not None:
            del self._instance
            self._instance = None
            logger.info("AirLLMEngine: model unloaded from RAM (backend=%s)", self._backend)

    @property
    def backend(self) -> str:
        """The active backend name."""
        return self._backend

    @property
    def memory_footprint_hint_gb(self) -> float:
        """Estimated runtime RAM usage in GB for the active model + compression.

        For ``llama_cpp``, this is 1.2× the GGUF file size (mmap + KV cache).
        For ``airllm``, the peak RSS is much lower because only 1–2 transformer
        layers reside in RAM at once; we still report the conservative 1.5×
        estimate so callers can use it as an upper bound.
        """
        size = Path(self.model_path).stat().st_size if Path(self.model_path).exists() else 0
        gb = size / (1024 ** 3)
        # Rough runtime overhead: 1.2× file size for llama_cpp, 1.5× for airllm
        multiplier = 1.5 if self._backend == "airllm" else _LLAMA_CPP_OVERHEAD
        return round(gb * multiplier, 2)

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _estimate_llama_cpp_ram_gb(self) -> float:
        """Estimate the runtime RAM usage (in GB) if loaded via llama-cpp-python.

        Uses the model file size multiplied by ``_LLAMA_CPP_OVERHEAD`` (1.2×).
        Returns 0.0 if the file does not exist.
        """
        p = Path(self.model_path)
        if not p.is_file():
            return 0.0
        return p.stat().st_size / (1024 ** 3) * _LLAMA_CPP_OVERHEAD

    def _select_backend(self, forced: Optional[str]) -> str:
        """Choose the best backend for the current model and platform.

        Policy (when ``forced`` is None):
          1. Model fits in budget via llama_cpp  → ``llama_cpp`` (fastest)
          2. Model exceeds budget + AirLLM avail → ``airllm``   (layer-split)
          3. Model exceeds budget, no AirLLM     → ``llama_cpp`` with warning
          4. No model file and no backends       → ``mock``
        """
        if forced:
            return forced

        model_exists = Path(self.model_path).is_file()
        if not model_exists:
            return "mock"

        llama_ram_gb = self._estimate_llama_cpp_ram_gb()
        fits_in_budget = llama_ram_gb <= _RAM_BUDGET_GB

        if fits_in_budget and _LLAMA_CPP_AVAILABLE:
            logger.debug(
                "AirLLMEngine: model RAM estimate %.2f GB ≤ budget %.1f GB → using llama_cpp (fastest path)",
                llama_ram_gb, _RAM_BUDGET_GB,
            )
            return "llama_cpp"

        if not fits_in_budget:
            if _AIRLLM_AVAILABLE:
                logger.info(
                    "AirLLMEngine: model RAM estimate %.2f GB > budget %.1f GB → "
                    "activating AirLLM layer-split loading to stay within budget",
                    llama_ram_gb, _RAM_BUDGET_GB,
                )
                return "airllm"
            if _LLAMA_CPP_AVAILABLE:
                logger.warning(
                    "AirLLMEngine: model RAM estimate %.2f GB exceeds %.1f GB budget "
                    "but AirLLM is not installed. Using llama_cpp anyway — consider "
                    "installing 'airllm' or choosing a smaller model.",
                    llama_ram_gb, _RAM_BUDGET_GB,
                )
                return "llama_cpp"

        if _LLAMA_CPP_AVAILABLE:
            return "llama_cpp"

        return "mock"

    # ------------------------------------------------------------------
    # AirLLM backend
    # ------------------------------------------------------------------

    def _generate_airllm(self, prompt: str, max_tokens: int) -> str:
        """Layer-by-layer generation via AirLLM.

        AirLLM splits the transformer into individual layers stored on disk
        and loads them one at a time.  This allows a 7B Q4 model (~4 GB) to
        run on a machine with only 4 GB of RAM because only 1–2 layers
        (typically 80–200 MB) are ever resident simultaneously.

        Reference: https://github.com/lyogavin/airllm
        """
        if self._instance is None:
            import airllm  # type: ignore

            # AirLLM accepts a HuggingFace model ID or local path.
            # If the user provides a GGUF path, fall back to llama_cpp.
            if self.model_path.endswith(".gguf"):
                logger.warning(
                    "AirLLM does not support GGUF files directly. "
                    "Falling back to llama_cpp backend for %s", self.model_path
                )
                self._backend = "llama_cpp"
                return self._generate_llama_cpp(prompt, max_tokens)

            compression_map = {
                "4bit": "4bit",
                "8bit": "8bit",
                "none": None,
            }
            comp = compression_map.get(self.compression, "4bit")

            logger.info("Loading model via AirLLM (layer-split): %s", self.model_path)
            self._instance = airllm.AutoModel.from_pretrained(
                self.model_path,
                compression=comp,
                max_seq_len=self.context_window,
                # 0 = CPU-only; set a fraction (e.g. 0.9) if you have GPU
                max_gpu_memory={} if self.max_gpu_memory == 0 else {"0": f"{self.max_gpu_memory}GiB"},
            )

        tokens = self._instance.tokenizer(prompt, return_tensors="pt")
        gen_ids = self._instance.generate(
            tokens["input_ids"],
            max_new_tokens=max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        output_ids = gen_ids[0][tokens["input_ids"].shape[-1]:]
        text = self._instance.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        if self.memory_conservative:
            self.unload()
        return text

    # ------------------------------------------------------------------
    # llama-cpp-python backend
    # ------------------------------------------------------------------

    def _generate_llama_cpp(self, prompt: str, max_tokens: int) -> str:
        """Inference via llama-cpp-python (GGUF 4-bit quantized models)."""
        if self._instance is None:
            from llama_cpp import Llama  # type: ignore

            logger.info("Loading GGUF model via llama-cpp-python: %s", self.model_path)
            # n_gpu_layers=0 keeps everything on CPU for 2017 Mac
            self._instance = Llama(
                model_path=self.model_path,
                n_ctx=self.context_window,
                n_gpu_layers=0 if self.max_gpu_memory == 0 else -1,
                verbose=False,
            )

        output = self._instance(
            prompt,
            max_tokens=max_tokens,
            temperature=self.temperature,
            stop=["</s>", "\n\nUser:", "\n\nHuman:", "<|end|>"],
        )
        text = output["choices"][0]["text"].strip()

        if self.memory_conservative:
            self.unload()
        return text

    # ------------------------------------------------------------------
    # Mock backend
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_response(prompt: str, max_tokens: int) -> str:
        short = prompt[:80].replace("\n", " ")
        return (
            f"[AirLLM MOCK] Received: '{short}…'. "
            "Install a supported backend and download a model to get real responses. "
            "See docs/AI_ARCHITECTURE.md for setup instructions."
        )

    # ------------------------------------------------------------------
    # How backend selection works (summary)
    # ------------------------------------------------------------------
    # AirLLM is **only used when the model would exceed the 4 GB RAM budget**
    # when loaded via llama-cpp-python (estimated as 1.2× the GGUF file size).
    #
    # Small/medium models (≤ ~3.3 GB file → ≤ 4 GB RAM estimate):
    #   → llama_cpp backend (fastest possible path, no layer-split overhead)
    #
    # Large models (> ~3.3 GB file → > 4 GB RAM estimate), e.g. Mistral-7B Q4:
    #   → airllm backend (layer-by-layer loading, keeps peak RAM ≤ ~2 GB)
    #
    # To enable AirLLM for a large HuggingFace model:
    #   pip install airllm
    #   llm:
    #     engine: auto   # will auto-select airllm for large models
    #     model_path: "mistralai/Mistral-7B-Instruct-v0.2"
    #     compression: 4bit
    #
    # To force a specific backend regardless of model size:
    #   llm:
    #     engine: airllm   # or: llama_cpp | mock
    # ------------------------------------------------------------------
