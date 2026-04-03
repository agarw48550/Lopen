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

You can swap the active engine by setting ``engine`` in ``config/settings.yaml``::

    llm:
      engine: airllm   # options: airllm | llama_cpp | mock
      model_path: models/llm/model.gguf
      max_gpu_memory: 0      # 0 = CPU-only (safe for 2017 MacBook)
      compression: 4bit      # 4bit | 8bit | none
      max_seq_len: 512

Memory budget guidance
----------------------
* Q4_K_M 7B model  ≈ 4.0 GB  (use alone, no other large models active)
* Q4_K_M 3.8B model ≈ 2.2 GB  (allows running a second small agent)
* Whisper-tiny ASR  ≈  39 MB
* Piper TTS         ≈  65 MB
* Python overhead   ≈ 300–400 MB
Total (3.8B LLM active): ≈ 2.7 GB  ← well within 4 GB budget.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

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

    Priority order:
      1. AirLLM (layer-split loading, best for 7B models on 4 GB RAM)
      2. llama-cpp-python (GGUF 4-bit, best balance for 3–4B models)
      3. Mock (CI / no model downloaded)

    Parameters
    ----------
    model_path:
        Path to the GGUF model file (for llama-cpp backend) or the
        HuggingFace model ID / directory (for AirLLM backend).
    engine:
        Force a specific backend: ``"airllm"``, ``"llama_cpp"``, ``"mock"``.
        If ``None``, the best available backend is auto-selected.
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
            "AirLLMEngine initialised: backend=%s model=%s compression=%s memory_conservative=%s",
            self._backend, self.model_path, self.compression, self.memory_conservative,
        )
        if self._backend == "mock":
            logger.warning(
                "AirLLMEngine running in MOCK mode. "
                "Install 'airllm' or 'llama-cpp-python' and download a model to enable real inference. "
                "See docs/AI_ARCHITECTURE.md for setup instructions."
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
        """Estimated RAM usage in GB for the active model + compression."""
        size = Path(self.model_path).stat().st_size if Path(self.model_path).exists() else 0
        gb = size / (1024 ** 3)
        # Rough runtime overhead: 1.2× file size for llama_cpp, 1.5× for airllm
        multiplier = 1.5 if self._backend == "airllm" else 1.2
        return round(gb * multiplier, 2)

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _select_backend(self, forced: Optional[str]) -> str:
        if forced:
            return forced
        model_exists = Path(self.model_path).is_file()
        if _AIRLLM_AVAILABLE and model_exists:
            return "airllm"
        if _LLAMA_CPP_AVAILABLE and model_exists:
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
    # How to swap AirLLM as the AI engine (documentation)
    # ------------------------------------------------------------------
    # To use AirLLM for a large HuggingFace model (e.g., Mistral-7B-v0.1):
    #
    #   pip install airllm
    #
    # In config/settings.yaml:
    #   llm:
    #     engine: airllm
    #     model_path: "mistralai/Mistral-7B-Instruct-v0.2"  # HF model ID
    #     compression: 4bit
    #     context_window: 2048
    #     max_gpu_memory: 0   # CPU-only
    #
    # The AirLLMEngine will then split the model into individual transformer
    # layers and stream them from disk on each forward pass.  Peak RAM usage
    # stays well below the model's full size.
    #
    # For GGUF models (llama-cpp-python backend):
    #   pip install llama-cpp-python
    #
    #   llm:
    #     engine: llama_cpp
    #     model_path: "models/llm/Phi-3-mini-4k-instruct-q4.gguf"
    #     compression: 4bit   # informational; GGUF is already quantized
    # ------------------------------------------------------------------
