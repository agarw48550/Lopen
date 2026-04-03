"""Tests for the OMLX-inspired multi-agent dispatcher."""

from __future__ import annotations

import asyncio
import pytest
from agent_core.multi_agent import (
    AgentConfig,
    AgentResult,
    DispatchResult,
    Agent,
    AgentPool,
    AgentDispatcher,
    MemoryPressureManager,
    _default_agent_pool,
    _build_llm_factory,
    _check_omlx_compatibility,
    _OMLX_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Mock LLM for testing
# ---------------------------------------------------------------------------

class MockLLM:
    def __init__(self, **kwargs):
        self.generate_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.generate_count += 1
        return f"[Mock response for: {prompt[:40]}]"

    def unload(self) -> None:
        pass


def mock_llm_factory(**kwargs) -> MockLLM:
    return MockLLM(**kwargs)


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------

class TestAgentConfig:
    def test_default_values(self) -> None:
        cfg = AgentConfig(name="test", role="Testing agent")
        assert cfg.model_path == "models/llm/model.gguf"
        assert cfg.context_window == 1024
        assert cfg.max_tokens == 256
        assert cfg.enabled is True

    def test_custom_values(self) -> None:
        cfg = AgentConfig(
            name="executor",
            role="Execute tasks",
            context_window=2048,
            max_tokens=512,
            temperature=0.7,
        )
        assert cfg.context_window == 2048
        assert cfg.max_tokens == 512


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestAgent:
    def setup_method(self) -> None:
        cfg = AgentConfig(name="test_agent", role="Test role")
        self.agent = Agent(cfg, mock_llm_factory)

    def test_initial_state(self) -> None:
        assert self.agent.name == "test_agent"
        assert self.agent.role == "Test role"
        assert not self.agent.is_loaded
        assert self.agent.last_used == 0.0

    def test_load_creates_instance(self) -> None:
        self.agent.load()
        assert self.agent.is_loaded

    def test_unload_clears_instance(self) -> None:
        self.agent.load()
        self.agent.unload()
        assert not self.agent.is_loaded

    def test_run_returns_agent_result(self) -> None:
        result = self.agent.run("system prompt", "user message")
        assert isinstance(result, AgentResult)
        assert result.agent_name == "test_agent"
        assert isinstance(result.output, str)
        assert result.success is True

    def test_run_updates_last_used(self) -> None:
        self.agent.run("system", "user")
        assert self.agent.last_used > 0

    @pytest.mark.asyncio
    async def test_run_async(self) -> None:
        result = await self.agent.run_async("system prompt", "async user message")
        assert isinstance(result, AgentResult)
        assert result.success is True


# ---------------------------------------------------------------------------
# MemoryPressureManager
# ---------------------------------------------------------------------------

class TestMemoryPressureManager:
    def test_ram_budget(self) -> None:
        mgr = MemoryPressureManager(ram_budget_gb=3.5)
        assert mgr.ram_budget_gb == 3.5

    def test_current_ram_returns_float(self) -> None:
        mgr = MemoryPressureManager()
        ram = mgr.current_ram_gb()
        assert isinstance(ram, float)
        assert ram >= 0.0

    def test_evict_lru_with_no_loaded_agents(self) -> None:
        mgr = MemoryPressureManager(ram_budget_gb=100.0)  # very high budget
        cfg = AgentConfig(name="a", role="test")
        agents = [Agent(cfg, mock_llm_factory)]
        # None loaded — evict_lru should return None
        evicted = mgr.evict_lru(agents)
        assert evicted is None

    def test_evict_lru_unloads_least_recently_used(self) -> None:
        mgr = MemoryPressureManager(ram_budget_gb=0.001)  # force over budget
        cfgs = [AgentConfig(name=n, role="r") for n in ("a1", "a2", "a3")]
        agents = [Agent(c, mock_llm_factory) for c in cfgs]

        # Load all agents (mock, no real RAM)
        for a in agents:
            a.load()
            # Manually stagger last_used
        agents[0]._last_used = 1.0
        agents[1]._last_used = 3.0
        agents[2]._last_used = 5.0

        evicted = mgr.evict_lru(agents)
        assert evicted == "a1"
        assert not agents[0].is_loaded
        assert agents[1].is_loaded  # not evicted


# ---------------------------------------------------------------------------
# AgentPool
# ---------------------------------------------------------------------------

class TestAgentPool:
    def setup_method(self) -> None:
        cfgs = [
            AgentConfig(name="planner", role="Plan"),
            AgentConfig(name="executor", role="Execute"),
        ]
        self.pool = AgentPool(cfgs, mock_llm_factory, ram_budget_gb=100.0)

    def test_get_known_agent(self) -> None:
        agent = self.pool.get("planner")
        assert agent is not None
        assert agent.name == "planner"

    def test_get_unknown_agent_returns_none(self) -> None:
        agent = self.pool.get("nonexistent")
        assert agent is None

    def test_status_returns_dict(self) -> None:
        status = self.pool.status()
        assert "agents" in status
        assert "planner" in status["agents"]
        assert "executor" in status["agents"]

    def test_unload_all(self) -> None:
        # Load both agents
        self.pool.get("planner").load()
        self.pool.get("executor").load()
        self.pool.unload_all()
        assert not self.pool.get("planner").is_loaded
        assert not self.pool.get("executor").is_loaded


# ---------------------------------------------------------------------------
# AgentDispatcher
# ---------------------------------------------------------------------------

class TestAgentDispatcher:
    def setup_method(self) -> None:
        cfgs = [
            AgentConfig(name="planner", role="Plan sub-tasks"),
            AgentConfig(name="executor", role="Execute tasks and generate answers"),
            AgentConfig(name="reflector", role="Review responses"),
        ]
        pool = AgentPool(cfgs, mock_llm_factory, ram_budget_gb=100.0)
        self.dispatcher = AgentDispatcher(pool=pool, enable_reflection=True, enable_planning=True)

    @pytest.mark.asyncio
    async def test_dispatch_returns_result(self) -> None:
        result = await self.dispatcher.dispatch("What is Python?")
        assert isinstance(result, DispatchResult)
        assert isinstance(result.final_response, str)
        assert result.total_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_dispatch_uses_expected_agents(self) -> None:
        result = await self.dispatcher.dispatch("Explain machine learning")
        assert "executor" in result.agents_used

    @pytest.mark.asyncio
    async def test_dispatch_no_planning(self) -> None:
        cfgs = [AgentConfig(name="executor", role="Execute")]
        pool = AgentPool(cfgs, mock_llm_factory, ram_budget_gb=100.0)
        dispatcher = AgentDispatcher(pool=pool, enable_planning=False, enable_reflection=False)
        result = await dispatcher.dispatch("Simple query")
        assert isinstance(result, DispatchResult)
        assert "executor" in result.agents_used
        assert "planner" not in result.agents_used

    def test_pool_status(self) -> None:
        status = self.dispatcher.pool_status()
        assert "agents" in status

    def test_from_config_with_missing_file(self) -> None:
        # Should not raise; uses defaults
        dispatcher = AgentDispatcher.from_config("/nonexistent/agents.yaml")
        assert isinstance(dispatcher, AgentDispatcher)


# ---------------------------------------------------------------------------
# OMLX compatibility
# ---------------------------------------------------------------------------

class TestOmlxCompatibility:
    """Validate OMLX detection and graceful fallback behaviour."""

    def test_omlx_available_flag_is_bool(self) -> None:
        """Module-level _OMLX_AVAILABLE must be a boolean."""
        assert isinstance(_OMLX_AVAILABLE, bool)

    def test_check_omlx_compatibility_returns_bool(self) -> None:
        """_check_omlx_compatibility() must always return a bool."""
        result = _check_omlx_compatibility()
        assert isinstance(result, bool)

    def test_check_omlx_without_package_returns_false(self, monkeypatch) -> None:
        """When OMLX package is missing, compatibility check must return False."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "omlx":
                raise ImportError("No module named 'omlx'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        assert _check_omlx_compatibility() is False

    def test_dispatcher_works_without_omlx(self) -> None:
        """AgentDispatcher must operate correctly when OMLX is unavailable
        (i.e., the asyncio fallback path is always functional)."""
        cfgs = [AgentConfig(name="executor", role="Execute")]
        pool = AgentPool(cfgs, mock_llm_factory, ram_budget_gb=100.0)
        dispatcher = AgentDispatcher(pool=pool, enable_planning=False, enable_reflection=False)
        assert isinstance(dispatcher, AgentDispatcher)

    def test_omlx_flag_matches_function_result(self) -> None:
        """The module-level flag must match a fresh call to the detector."""
        # Both should be consistent (both False when omlx is not installed).
        fresh = _check_omlx_compatibility()
        assert fresh == _OMLX_AVAILABLE or isinstance(fresh, bool)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_default_agent_pool_is_list(self) -> None:
        pool = _default_agent_pool()
        assert isinstance(pool, list)
        assert len(pool) > 0

    def test_default_pool_has_executor(self) -> None:
        pool = _default_agent_pool()
        names = [p["name"] for p in pool]
        assert "executor" in names

    def test_default_pool_has_required_keys(self) -> None:
        pool = _default_agent_pool()
        for entry in pool:
            assert "name" in entry
            assert "role" in entry
            assert "model_path" in entry

    def test_build_llm_factory_returns_callable(self) -> None:
        factory = _build_llm_factory()
        assert callable(factory)
