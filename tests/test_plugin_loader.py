"""Unit tests for PluginLoader dynamic discovery."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent_core.plugin_loader import PluginLoader


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    """Create a temporary plugin directory with sample tool files."""
    plugin_d = tmp_path / "plugins"
    plugin_d.mkdir()

    # A valid BaseTool subclass
    (plugin_d / "sample_plugin.py").write_text(
        textwrap.dedent("""\
            from tools.base_tool import BaseTool

            class SamplePlugin(BaseTool):
                name = "sample_plugin"
                description = "A sample plugin for testing dynamic loading"
                tags = ["test", "sample"]
                version = "2.0.0"
                requires_permission = False

                def run(self, query: str, **kwargs) -> str:
                    return f"SamplePlugin: {query}"
        """)
    )

    # A plugin with requires_permission = True
    (plugin_d / "perm_plugin.py").write_text(
        textwrap.dedent("""\
            from tools.base_tool import BaseTool

            class PermPlugin(BaseTool):
                name = "perm_plugin"
                description = "A plugin that requires user permission"
                requires_permission = True

                def run(self, query: str, **kwargs) -> str:
                    return "PermPlugin ran"
        """)
    )

    # A file that is NOT a valid plugin (no BaseTool subclass)
    (plugin_d / "not_a_plugin.py").write_text("x = 42\n")

    # Private file that should be skipped
    (plugin_d / "_private.py").write_text(
        textwrap.dedent("""\
            from tools.base_tool import BaseTool
            class _PrivateTool(BaseTool):
                name = "_private"
                description = "Should be skipped"
                def run(self, query, **kwargs): return ""
        """)
    )

    return plugin_d


class TestPluginLoader:
    def test_scan_discovers_valid_plugins(self, plugin_dir: Path) -> None:
        loader = PluginLoader(tool_dirs=[str(plugin_dir)])
        metas = loader.scan()
        names = [m.name for m in metas]
        assert "sample_plugin" in names
        assert "perm_plugin" in names

    def test_scan_skips_private_files(self, plugin_dir: Path) -> None:
        loader = PluginLoader(tool_dirs=[str(plugin_dir)])
        metas = loader.scan()
        names = [m.name for m in metas]
        assert "_private" not in names

    def test_scan_ignores_non_plugin_files(self, plugin_dir: Path) -> None:
        loader = PluginLoader(tool_dirs=[str(plugin_dir)])
        metas = loader.scan()
        # not_a_plugin.py has no BaseTool subclass, so no meta for it
        assert len(metas) == 2  # only sample_plugin + perm_plugin

    def test_meta_attributes_populated(self, plugin_dir: Path) -> None:
        loader = PluginLoader(tool_dirs=[str(plugin_dir)])
        metas = loader.scan()
        sample = next(m for m in metas if m.name == "sample_plugin")
        assert sample.description == "A sample plugin for testing dynamic loading"
        assert sample.version == "2.0.0"
        assert "test" in sample.tags
        assert sample.requires_permission is False

    def test_permission_plugin_flagged(self, plugin_dir: Path) -> None:
        loader = PluginLoader(tool_dirs=[str(plugin_dir)])
        metas = loader.scan()
        perm = next(m for m in metas if m.name == "perm_plugin")
        assert perm.requires_permission is True

    def test_instance_created(self, plugin_dir: Path) -> None:
        loader = PluginLoader(tool_dirs=[str(plugin_dir)])
        metas = loader.scan()
        for meta in metas:
            assert meta.instance is not None

    def test_tool_class_stored(self, plugin_dir: Path) -> None:
        loader = PluginLoader(tool_dirs=[str(plugin_dir)])
        metas = loader.scan()
        for meta in metas:
            assert meta.tool_class is not None

    def test_scan_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        loader = PluginLoader(tool_dirs=[str(tmp_path / "does_not_exist")])
        assert loader.scan() == []

    def test_load_file_directly(self, plugin_dir: Path) -> None:
        loader = PluginLoader(tool_dirs=[])
        metas = loader.load_file(plugin_dir / "sample_plugin.py")
        assert len(metas) == 1
        assert metas[0].name == "sample_plugin"

    def test_loaded_files_tracking(self, plugin_dir: Path) -> None:
        loader = PluginLoader(tool_dirs=[str(plugin_dir)])
        loader.scan()
        loaded = loader.loaded_files()
        assert any("sample_plugin" in p for p in loaded)

    def test_plugin_run_callable(self, plugin_dir: Path) -> None:
        loader = PluginLoader(tool_dirs=[str(plugin_dir)])
        metas = loader.scan()
        sample = next(m for m in metas if m.name == "sample_plugin")
        result = sample.instance.run("hello world")
        assert "SamplePlugin" in result
        assert "hello world" in result
