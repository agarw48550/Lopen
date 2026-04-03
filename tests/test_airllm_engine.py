"""Tests for the AirLLM-inspired inference engine."""

from __future__ import annotations

import os
import tempfile

import pytest
from llm.airllm_engine import (
    AirLLMEngine,
    _RAM_BUDGET_GB,
    _LLAMA_CPP_OVERHEAD,
    _LLAMA_CPP_AVAILABLE,
)


class TestAirLLMEngineBackendSelection:
    def test_mock_backend_when_no_model(self) -> None:
        engine = AirLLMEngine(model_path="/nonexistent/model.gguf", engine=None)
        assert engine.backend == "mock"

    def test_forced_mock_backend(self) -> None:
        engine = AirLLMEngine(model_path="/nonexistent/model.gguf", engine="mock")
        assert engine.backend == "mock"

    def test_forced_llama_cpp_backend(self) -> None:
        engine = AirLLMEngine(model_path="/nonexistent/model.gguf", engine="llama_cpp")
        assert engine.backend == "llama_cpp"

    def test_forced_airllm_backend(self) -> None:
        engine = AirLLMEngine(model_path="/nonexistent/model.gguf", engine="airllm")
        assert engine.backend == "airllm"


class TestAirLLMSmartBackendSelection:
    """Validate that AirLLM is only activated for large (>4 GB RAM) models."""

    def _make_model_file(self, size_bytes: int) -> str:
        """Create a temporary dummy model file of the given byte size.

        Uses ``os.ftruncate`` to create a sparse file so no actual data is
        written to disk — safe for large sizes (e.g. 3.5 GB) in CI.
        """
        fd, path = tempfile.mkstemp(suffix=".gguf")
        try:
            os.ftruncate(fd, size_bytes)  # sparse file — no real disk writes
        finally:
            os.close(fd)
        return path

    def test_small_model_does_not_use_airllm(self) -> None:
        """A tiny model file (< 4 GB RAM estimate) must not select airllm backend."""
        # 100 MB file → ~0.12 GB RAM estimate — well under budget
        path = self._make_model_file(100 * 1024 * 1024)
        try:
            engine = AirLLMEngine(model_path=path, engine=None)
            # Should prefer llama_cpp or mock; never airllm for a small model
            assert engine.backend != "airllm", (
                "AirLLM should not be selected for models that fit within the 4 GB budget"
            )
        finally:
            os.unlink(path)

    def test_ram_budget_constant_is_4gb(self) -> None:
        """The RAM budget ceiling must be 4.0 GB."""
        assert _RAM_BUDGET_GB == 4.0

    def test_llama_cpp_overhead_constant(self) -> None:
        """The llama_cpp RAM overhead multiplier must be 1.2×."""
        assert _LLAMA_CPP_OVERHEAD == 1.2

    def test_estimate_llama_cpp_ram_gb_no_file(self) -> None:
        """Estimate returns 0.0 when model file does not exist."""
        engine = AirLLMEngine(model_path="/nonexistent/model.gguf", engine="mock")
        assert engine._estimate_llama_cpp_ram_gb() == 0.0

    def test_estimate_llama_cpp_ram_gb_with_file(self) -> None:
        """Estimate returns correct value (1.2× file size in GB)."""
        size_bytes = 200 * 1024 * 1024  # 200 MB
        path = self._make_model_file(size_bytes)
        try:
            engine = AirLLMEngine(model_path=path, engine="mock")
            expected = (size_bytes / (1024 ** 3)) * _LLAMA_CPP_OVERHEAD
            assert abs(engine._estimate_llama_cpp_ram_gb() - expected) < 1e-6
        finally:
            os.unlink(path)

    def test_large_model_selects_airllm_when_available(self, monkeypatch) -> None:
        """A file whose RAM estimate exceeds 4 GB should trigger AirLLM (mocked)."""
        # 3.5 GB file → 3.5 * 1.2 = 4.2 GB RAM estimate → exceeds budget
        size_bytes = int(3.5 * 1024 ** 3)
        path = self._make_model_file(size_bytes)
        try:
            # Pretend AirLLM is installed
            import llm.airllm_engine as ae
            monkeypatch.setattr(ae, "_AIRLLM_AVAILABLE", True)
            engine = AirLLMEngine(model_path=path, engine=None)
            assert engine.backend == "airllm", (
                "AirLLM should be selected for models whose RAM estimate exceeds 4 GB"
            )
        finally:
            os.unlink(path)

    def test_large_model_falls_back_to_llama_cpp_when_airllm_unavailable(self, monkeypatch) -> None:
        """When AirLLM is not installed and model is large, warn + use llama_cpp."""
        size_bytes = int(3.5 * 1024 ** 3)  # 4.2 GB RAM estimate
        path = self._make_model_file(size_bytes)
        try:
            import llm.airllm_engine as ae
            monkeypatch.setattr(ae, "_AIRLLM_AVAILABLE", False)
            monkeypatch.setattr(ae, "_LLAMA_CPP_AVAILABLE", True)
            engine = AirLLMEngine(model_path=path, engine=None)
            assert engine.backend == "llama_cpp"
        finally:
            os.unlink(path)


class TestAirLLMEngineMockGeneration:
    def test_mock_generate_returns_string(self) -> None:
        engine = AirLLMEngine(model_path="/nonexistent/model.gguf", engine="mock")
        result = engine.generate("Hello, who are you?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_response_contains_prompt_fragment(self) -> None:
        engine = AirLLMEngine(model_path="/nonexistent/model.gguf", engine="mock")
        result = engine.generate("What is the capital of France?")
        assert "AirLLM MOCK" in result or "MOCK" in result

    def test_mock_generate_with_max_tokens(self) -> None:
        engine = AirLLMEngine(model_path="/nonexistent/model.gguf", engine="mock")
        result = engine.generate("test prompt", max_tokens=10)
        assert isinstance(result, str)


class TestAirLLMEngineProperties:
    def test_backend_property(self) -> None:
        engine = AirLLMEngine(model_path="/nonexistent/model.gguf", engine="mock")
        assert engine.backend in ("airllm", "llama_cpp", "mock")

    def test_memory_footprint_hint(self) -> None:
        engine = AirLLMEngine(model_path="/nonexistent/model.gguf", engine="mock")
        hint = engine.memory_footprint_hint_gb
        assert isinstance(hint, float)
        assert hint >= 0.0

    def test_unload_noop_when_not_loaded(self) -> None:
        engine = AirLLMEngine(model_path="/nonexistent/model.gguf", engine="mock")
        # Should not raise
        engine.unload()

    def test_default_parameters(self) -> None:
        engine = AirLLMEngine(engine="mock")
        assert engine.context_window == 2048
        assert engine.max_tokens == 512
        assert 0 < engine.temperature <= 1.0
        assert engine.memory_conservative is True
        assert engine.max_gpu_memory == 0

    def test_custom_parameters(self) -> None:
        engine = AirLLMEngine(
            engine="mock",
            context_window=1024,
            max_tokens=128,
            temperature=0.3,
            memory_conservative=False,
        )
        assert engine.context_window == 1024
        assert engine.max_tokens == 128
        assert engine.temperature == 0.3
        assert engine.memory_conservative is False
