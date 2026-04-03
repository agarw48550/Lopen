"""Unit tests for ToolRegistry."""

import pytest
from agent_core.tool_registry import ToolRegistry, ToolMeta


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def sample_meta() -> ToolMeta:
    return ToolMeta(
        name="test_tool",
        description="A test tool",
        requires_permission=False,
        enabled=True,
        tags=["test"],
    )


class TestToolRegistration:
    def test_register_tool(self, registry: ToolRegistry, sample_meta: ToolMeta) -> None:
        registry.register(sample_meta)
        assert "test_tool" in registry

    def test_register_duplicate_raises(self, registry: ToolRegistry, sample_meta: ToolMeta) -> None:
        registry.register(sample_meta)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(sample_meta)

    def test_unregister_existing(self, registry: ToolRegistry, sample_meta: ToolMeta) -> None:
        registry.register(sample_meta)
        result = registry.unregister("test_tool")
        assert result is True
        assert "test_tool" not in registry

    def test_unregister_nonexistent(self, registry: ToolRegistry) -> None:
        result = registry.unregister("nonexistent")
        assert result is False

    def test_get_tool(self, registry: ToolRegistry, sample_meta: ToolMeta) -> None:
        registry.register(sample_meta)
        tool = registry.get_tool("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"

    def test_get_nonexistent_returns_none(self, registry: ToolRegistry) -> None:
        assert registry.get_tool("missing") is None


class TestToolListing:
    def test_list_all_tools(self, registry: ToolRegistry) -> None:
        registry.register(ToolMeta("tool_a", "A", enabled=True))
        registry.register(ToolMeta("tool_b", "B", enabled=False))
        tools = registry.list_tools()
        assert len(tools) == 2

    def test_list_enabled_only(self, registry: ToolRegistry) -> None:
        registry.register(ToolMeta("tool_a", "A", enabled=True))
        registry.register(ToolMeta("tool_b", "B", enabled=False))
        enabled = registry.list_tools(enabled_only=True)
        assert len(enabled) == 1
        assert enabled[0].name == "tool_a"

    def test_names(self, registry: ToolRegistry) -> None:
        registry.register(ToolMeta("alpha", "Alpha"))
        registry.register(ToolMeta("beta", "Beta"))
        names = registry.names()
        assert "alpha" in names
        assert "beta" in names

    def test_len(self, registry: ToolRegistry) -> None:
        assert len(registry) == 0
        registry.register(ToolMeta("x", "X"))
        assert len(registry) == 1


class TestToolEnableDisable:
    def test_disable_tool(self, registry: ToolRegistry, sample_meta: ToolMeta) -> None:
        registry.register(sample_meta)
        registry.disable("test_tool")
        tool = registry.get_tool("test_tool")
        assert tool is not None
        assert tool.enabled is False

    def test_enable_tool(self, registry: ToolRegistry, sample_meta: ToolMeta) -> None:
        sample_meta.enabled = False
        registry.register(sample_meta)
        registry.enable("test_tool")
        tool = registry.get_tool("test_tool")
        assert tool is not None
        assert tool.enabled is True

    def test_enable_unknown_raises(self, registry: ToolRegistry) -> None:
        with pytest.raises(KeyError):
            registry.enable("ghost")
