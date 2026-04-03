"""Tests for the AirLLM-inspired inference engine."""

from __future__ import annotations

import pytest
from llm.airllm_engine import AirLLMEngine


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
