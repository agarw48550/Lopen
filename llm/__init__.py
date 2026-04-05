"""LLM package."""
from llm.llm_adapter import LLMAdapter
from llm.fast_llm import FastLLM, get_fast_llm
from llm.heavy_llm import HeavyLLM, get_heavy_llm

__all__ = ["LLMAdapter", "FastLLM", "get_fast_llm", "HeavyLLM", "get_heavy_llm"]
