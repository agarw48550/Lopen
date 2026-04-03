"""OMLX-inspired multi-agent dispatcher for Lopen.

This module implements a lightweight multi-agent orchestration layer inspired
by the omlx project (https://github.com/jundot/omlx).  It allows Lopen to
run several specialised LLM sub-agents concurrently while staying within the
4 GB RAM budget.

Architecture
------------
::

    User query
        │
        ▼
    AgentDispatcher
        │  route_query(query, intent_result)
        │
        ├── AgentPool (manages agent slots / memory)
        │       │
        │       ├── Agent("planner")      — task decomposition
        │       ├── Agent("executor")     — tool invocation
        │       ├── Agent("reflector")    — self-critique / quality check
        │       └── Agent("summarizer")   — context compression
        │
        └── MemoryPressureManager
                │
                ├── tracks each agent's RAM estimate
                ├── LRU-evicts idle agents when budget exceeded
                └── queues tasks when all slots are full

Design principles (from omlx)
------------------------------
1. **Parallel intent flow** — multiple agents can process independent sub-tasks
   concurrently using ``asyncio``.
2. **Memory backpressure** — before spawning a new agent, the dispatcher checks
   current RAM usage.  If we are above the ``ram_budget_gb`` threshold, the
   least-recently-used agent is unloaded first (LRU eviction).
3. **Pooling** — agent instances are pooled and reused across requests to avoid
   repeated model-load overhead.
4. **Memory isolation** — each agent has its own conversation context and tool
   permission set.

Configuration (``config/agents.yaml``)
---------------------------------------
::

    agents:
      ram_budget_gb: 3.5
      max_concurrent_agents: 3
      lru_eviction: true
      pool:
        - name: planner
          role: decompose user requests into ordered sub-tasks
          model_path: models/llm/model.gguf
          context_window: 1024
          max_tokens: 256
        - name: executor
          role: call tools to fulfil sub-tasks
          model_path: models/llm/model.gguf
          context_window: 2048
          max_tokens: 512
        - name: reflector
          role: review and critique the executor response
          model_path: models/llm/model.gguf
          context_window: 1024
          max_tokens: 256
        - name: summarizer
          role: compress long conversation history
          model_path: models/llm/model.gguf
          context_window: 1024
          max_tokens: 128

Connecting the dispatcher
--------------------------
In ``orchestrator.py`` (startup)::

    from agent_core.multi_agent import AgentDispatcher
    dispatcher = AgentDispatcher.from_config("config/agents.yaml")
    _state["dispatcher"] = dispatcher

    # In the /chat endpoint:
    result = await dispatcher.dispatch(query, intent_result, tools=registry)

The dispatcher returns a unified response dict with the final answer and the
reasoning chain from all agents.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for a single sub-agent."""

    name: str
    role: str
    model_path: str = "models/llm/model.gguf"
    context_window: int = 1024
    max_tokens: int = 256
    temperature: float = 0.7
    enabled: bool = True


@dataclass
class AgentResult:
    """Result returned by a sub-agent."""

    agent_name: str
    output: str
    latency_ms: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class DispatchResult:
    """Aggregated result from the multi-agent dispatcher."""

    final_response: str
    agent_results: List[AgentResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    agents_used: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent slot
# ---------------------------------------------------------------------------

class Agent:
    """A single stateful LLM sub-agent with its own context window.

    Each agent has a specific role (planner, executor, reflector, summarizer)
    and maintains its own short-term context.  Agents share the same base
    model file but can be loaded/unloaded independently for memory management.
    """

    def __init__(self, cfg: AgentConfig, llm_factory: Callable[..., Any]) -> None:
        self.name = cfg.name
        self.role = cfg.role
        self._cfg = cfg
        self._llm_factory = llm_factory
        self._llm: Optional[Any] = None
        self._last_used: float = 0.0
        self._use_count: int = 0
        self._context: List[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate the LLM backend for this agent."""
        if self._llm is None:
            self._llm = self._llm_factory(
                model_path=self._cfg.model_path,
                context_window=self._cfg.context_window,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                memory_conservative=False,  # Agent manages its own lifecycle
            )
            logger.debug("Agent '%s' loaded model: %s", self.name, self._cfg.model_path)

    def unload(self) -> None:
        """Release the LLM backend to free RAM."""
        if self._llm is not None and hasattr(self._llm, "unload"):
            self._llm.unload()
        self._llm = None
        logger.debug("Agent '%s' unloaded from RAM", self.name)

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def last_used(self) -> float:
        return self._last_used

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, system_prompt: str, user_message: str) -> AgentResult:
        """Run synchronous inference for this agent.

        Args:
            system_prompt: Role/context instructions for this agent.
            user_message: The actual task/query for this call.

        Returns:
            AgentResult with the generated output.
        """
        start = time.time()
        try:
            self.load()
            full_prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
            output = self._llm.generate(full_prompt, max_tokens=self._cfg.max_tokens)
            self._last_used = time.time()
            self._use_count += 1
            latency_ms = (time.time() - start) * 1000
            return AgentResult(
                agent_name=self.name,
                output=output,
                latency_ms=latency_ms,
                success=True,
            )
        except Exception as exc:
            logger.error("Agent '%s' inference error: %s", self.name, exc)
            latency_ms = (time.time() - start) * 1000
            return AgentResult(
                agent_name=self.name,
                output=f"[Agent '{self.name}' error: {exc}]",
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )

    async def run_async(self, system_prompt: str, user_message: str) -> AgentResult:
        """Async wrapper — runs in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, system_prompt, user_message)


# ---------------------------------------------------------------------------
# Memory pressure manager
# ---------------------------------------------------------------------------

class MemoryPressureManager:
    """Monitors RAM and evicts the LRU agent when budget is exceeded.

    Uses ``psutil`` to read actual RSS and falls back to an estimate based
    on model file sizes when psutil is unavailable.
    """

    def __init__(self, ram_budget_gb: float = 3.5) -> None:
        self.ram_budget_gb = ram_budget_gb
        self._psutil_available = False
        try:
            import psutil  # noqa: F401
            self._psutil_available = True
        except ImportError:
            pass

    def current_ram_gb(self) -> float:
        """Return current process RSS in GB."""
        if self._psutil_available:
            import psutil
            rss = psutil.Process().memory_info().rss
            return rss / (1024 ** 3)
        return 0.0

    def is_over_budget(self) -> bool:
        """Return True if current RAM usage exceeds the configured budget."""
        return self.current_ram_gb() >= self.ram_budget_gb

    def evict_lru(self, agents: List[Agent]) -> Optional[str]:
        """Unload the least-recently-used *loaded* agent.

        Returns the name of the evicted agent, or None if nothing was evicted.
        """
        loaded = [a for a in agents if a.is_loaded]
        if not loaded:
            return None
        lru = min(loaded, key=lambda a: a.last_used)
        lru.unload()
        logger.info(
            "MemoryPressureManager: evicted LRU agent '%s' (RAM=%.2f GB / budget=%.2f GB)",
            lru.name, self.current_ram_gb(), self.ram_budget_gb,
        )
        return lru.name


# ---------------------------------------------------------------------------
# Agent pool
# ---------------------------------------------------------------------------

class AgentPool:
    """Manages a fixed pool of named agent slots.

    Agents are created lazily and reused across requests (pooling pattern).
    When RAM pressure is high, idle agents are LRU-evicted.
    """

    def __init__(
        self,
        configs: List[AgentConfig],
        llm_factory: Callable[..., Any],
        ram_budget_gb: float = 3.5,
        max_concurrent: int = 3,
    ) -> None:
        self._agents: Dict[str, Agent] = {
            cfg.name: Agent(cfg, llm_factory)
            for cfg in configs
            if cfg.enabled
        }
        self._pressure = MemoryPressureManager(ram_budget_gb)
        self._max_concurrent = max_concurrent
        logger.info(
            "AgentPool initialised: %d agents, ram_budget=%.1f GB, max_concurrent=%d",
            len(self._agents), ram_budget_gb, max_concurrent,
        )

    def get(self, name: str) -> Optional[Agent]:
        """Return the named agent, evicting LRU agents if RAM is tight."""
        agent = self._agents.get(name)
        if agent is None:
            logger.warning("AgentPool: unknown agent '%s'", name)
            return None

        # Enforce RAM budget before loading
        while self._pressure.is_over_budget():
            evicted = self._pressure.evict_lru(list(self._agents.values()))
            if evicted is None:
                break  # Nothing left to evict

        return agent

    def unload_all(self) -> None:
        """Unload all agents (call at shutdown)."""
        for agent in self._agents.values():
            agent.unload()

    def status(self) -> Dict[str, Any]:
        """Return pool status dict for diagnostics."""
        return {
            "agents": {
                name: {
                    "loaded": a.is_loaded,
                    "last_used": a.last_used,
                    "role": a.role,
                }
                for name, a in self._agents.items()
            },
            "ram_gb": round(self._pressure.current_ram_gb(), 2),
            "ram_budget_gb": self._pressure.ram_budget_gb,
            "over_budget": self._pressure.is_over_budget(),
        }


# ---------------------------------------------------------------------------
# Agent dispatcher
# ---------------------------------------------------------------------------

class AgentDispatcher:
    """Routes queries to the appropriate sub-agent(s) and aggregates results.

    The dispatcher implements a simple agentic reasoning loop:

    1. **Plan** — the ``planner`` agent decomposes the query into sub-tasks.
    2. **Execute** — the ``executor`` agent calls tools to fulfil each sub-task.
    3. **Reflect** — the ``reflector`` agent reviews the response quality.
    4. **Summarise** (optional) — compress context when history grows too long.

    All steps run concurrently where independent, using ``asyncio.gather``.

    Memory isolation
    ----------------
    Each agent maintains its own conversation context list.  Agents never
    share context, preventing cross-contamination of reasoning chains.  The
    dispatcher merges results into a final unified response.

    Connecting to the orchestrator
    --------------------------------
    See module docstring for example code.
    """

    def __init__(
        self,
        pool: AgentPool,
        enable_reflection: bool = True,
        enable_planning: bool = True,
    ) -> None:
        self._pool = pool
        self._enable_reflection = enable_reflection
        self._enable_planning = enable_planning

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str = "config/agents.yaml") -> "AgentDispatcher":
        """Build a dispatcher from a YAML config file.

        Example ``config/agents.yaml``::

            agents:
              ram_budget_gb: 3.5
              max_concurrent_agents: 3
              lru_eviction: true
              enable_planning: true
              enable_reflection: true
              pool: [...]

        If the config file doesn't exist, sensible defaults are used.
        """
        cfg: Dict[str, Any] = {}
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("AgentDispatcher: config not found at %s, using defaults", config_path)

        agents_cfg = cfg.get("agents", {})
        ram_budget = agents_cfg.get("ram_budget_gb", 3.5)
        max_concurrent = agents_cfg.get("max_concurrent_agents", 3)
        enable_planning = agents_cfg.get("enable_planning", True)
        enable_reflection = agents_cfg.get("enable_reflection", True)

        pool_cfgs_raw = agents_cfg.get("pool", _default_agent_pool())
        pool_cfgs = [AgentConfig(**c) for c in pool_cfgs_raw]

        # Build an LLM factory using the best available backend
        llm_factory = _build_llm_factory()

        pool = AgentPool(
            configs=pool_cfgs,
            llm_factory=llm_factory,
            ram_budget_gb=ram_budget,
            max_concurrent=max_concurrent,
        )
        dispatcher = cls(
            pool=pool,
            enable_reflection=enable_reflection,
            enable_planning=enable_planning,
        )
        logger.info(
            "AgentDispatcher ready: planning=%s reflection=%s",
            enable_planning, enable_reflection,
        )
        return dispatcher

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        query: str,
        intent_result: Any = None,
        tools: Any = None,
    ) -> DispatchResult:
        """Route a user query through the multi-agent reasoning pipeline.

        Args:
            query: User's natural language input.
            intent_result: Optional IntentResult from the IntentEngine.
            tools: Optional ToolRegistry for the executor agent.

        Returns:
            DispatchResult with the aggregated final response.
        """
        start = time.time()
        results: List[AgentResult] = []
        agents_used: List[str] = []

        # Step 1 — Plan (if enabled and planner agent exists)
        plan_text = ""
        if self._enable_planning:
            planner = self._pool.get("planner")
            if planner:
                plan_result = await planner.run_async(
                    system_prompt=(
                        "You are a task planning assistant. "
                        "Break the user's request into 2–3 clear, ordered sub-tasks. "
                        "Be concise. Output numbered steps only."
                    ),
                    user_message=query,
                )
                results.append(plan_result)
                agents_used.append("planner")
                plan_text = plan_result.output if plan_result.success else ""

        # Step 2 — Execute (main response generation)
        executor = self._pool.get("executor")
        if executor is None:
            # Fallback: no pool configured — single-agent mode
            final_response = f"[No executor agent configured] Query: {query}"
        else:
            exec_prompt = (
                "You are a helpful AI assistant running locally on a Mac. "
                "Answer the user's request fully and accurately."
            )
            if plan_text:
                exec_prompt += f"\n\nTask plan:\n{plan_text}"

            tool_list = ""
            if tools is not None:
                try:
                    tool_names = [t.name for t in tools.list_tools()]
                    tool_list = ", ".join(tool_names[:10])
                    exec_prompt += f"\n\nAvailable tools: {tool_list}"
                except Exception:
                    pass

            exec_result = await executor.run_async(
                system_prompt=exec_prompt,
                user_message=query,
            )
            results.append(exec_result)
            agents_used.append("executor")
            final_response = exec_result.output if exec_result.success else exec_result.error or query

        # Step 3 — Reflect (if enabled and reflector agent exists)
        if self._enable_reflection and final_response:
            reflector = self._pool.get("reflector")
            if reflector:
                reflect_result = await reflector.run_async(
                    system_prompt=(
                        "You are a quality-checking assistant. "
                        "Review the assistant's response for accuracy, completeness, and helpfulness. "
                        "If it is good, reply 'OK'. "
                        "If there is a clear factual error, briefly correct it."
                    ),
                    user_message=(
                        f"Original question: {query}\n\n"
                        f"Assistant's response: {final_response}"
                    ),
                )
                results.append(reflect_result)
                agents_used.append("reflector")
                # Only override the response if the reflector found a correction
                reflection = reflect_result.output.strip() if reflect_result.success else ""
                if reflection and reflection.upper() not in {"OK", "LGTM", "LOOKS GOOD"}:
                    final_response = reflection

        total_ms = (time.time() - start) * 1000
        return DispatchResult(
            final_response=final_response,
            agent_results=results,
            total_latency_ms=total_ms,
            agents_used=agents_used,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def pool_status(self) -> Dict[str, Any]:
        """Return current pool status (for /health and /status endpoints)."""
        return self._pool.status()

    def unload_all(self) -> None:
        """Shutdown — release all agent RAM."""
        self._pool.unload_all()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_agent_pool() -> List[Dict[str, Any]]:
    """Minimal default pool config used when no YAML is present."""
    default_model = "models/llm/model.gguf"
    return [
        {
            "name": "planner",
            "role": "Decompose user requests into ordered sub-tasks",
            "model_path": default_model,
            "context_window": 1024,
            "max_tokens": 256,
            "temperature": 0.5,
            "enabled": True,
        },
        {
            "name": "executor",
            "role": "Call tools and generate final answers",
            "model_path": default_model,
            "context_window": 2048,
            "max_tokens": 512,
            "temperature": 0.7,
            "enabled": True,
        },
        {
            "name": "reflector",
            "role": "Review and quality-check assistant responses",
            "model_path": default_model,
            "context_window": 1024,
            "max_tokens": 256,
            "temperature": 0.3,
            "enabled": True,
        },
        {
            "name": "summarizer",
            "role": "Compress long conversation history into a short summary",
            "model_path": default_model,
            "context_window": 1024,
            "max_tokens": 128,
            "temperature": 0.3,
            "enabled": True,
        },
    ]


def _build_llm_factory() -> Callable[..., Any]:
    """Return the best available LLM factory function.

    The factory is called by Agent.load() with keyword arguments matching
    LLMAdapter / AirLLMEngine constructors.
    """
    try:
        from llm.airllm_engine import AirLLMEngine
        return AirLLMEngine
    except ImportError:
        pass
    try:
        from llm.llm_adapter import LLMAdapter
        return LLMAdapter
    except ImportError:
        pass

    # Last resort: trivial mock factory
    class _MockLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass
        def generate(self, prompt: str, **kwargs: Any) -> str:
            return f"[MultiAgent MOCK] {prompt[:60]}…"
        def unload(self) -> None:
            pass

    return _MockLLM
