"""Dynamic plugin loader for Lopen.

Scans ``tools/`` and ``tools/third_party/`` (or any configured directories)
for Python files that contain :class:`~tools.base_tool.BaseTool` subclasses,
imports them, and returns ready-to-register :class:`~agent_core.tool_registry.ToolMeta`
instances.

This makes Lopen genuinely open-ended: drop a new ``.py`` file in ``tools/``
and it is automatically discovered, indexed in the :class:`~agent_core.intent_engine.IntentEngine`,
and made available for routing — without any code changes to the core.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency at module load time
_BaseTool: Any = None
_ToolMeta: Any = None
_ToolRegistry: Any = None


def _get_base_tool() -> Any:
    global _BaseTool
    if _BaseTool is None:
        from tools.base_tool import BaseTool  # type: ignore[import]
        _BaseTool = BaseTool
    return _BaseTool


def _get_tool_meta() -> Any:
    global _ToolMeta
    if _ToolMeta is None:
        from agent_core.tool_registry import ToolMeta  # type: ignore[import]
        _ToolMeta = ToolMeta
    return _ToolMeta


# ---------------------------------------------------------------------------
# PluginLoader
# ---------------------------------------------------------------------------

class PluginLoader:
    """Discover and load :class:`~tools.base_tool.BaseTool` subclasses from disk.

    Usage::

        loader = PluginLoader(tool_dirs=["tools", "tools/third_party"])
        metas = loader.scan()          # list[ToolMeta] — ready to register
        for meta in metas:
            registry.register(meta)    # idempotent — skip duplicates

    Adding a plugin is as simple as:

    1. Create ``tools/third_party/my_plugin.py`` with a class inheriting
       :class:`~tools.base_tool.BaseTool`.
    2. Set ``name``, ``description``, and (optionally) ``tags`` / ``version``
       as class attributes.
    3. Restart the orchestrator (or call :meth:`scan` again).
    """

    def __init__(
        self,
        tool_dirs: list[str] | None = None,
        llm_adapter: Any | None = None,
    ) -> None:
        self._tool_dirs: list[str] = tool_dirs or ["tools", "tools/third_party"]
        self._llm = llm_adapter
        # Track which files have already been loaded (path → list[class_name])
        self._loaded: dict[str, list[str]] = {}
        logger.info("PluginLoader initialised with dirs: %s", self._tool_dirs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, skip_existing: bool = True) -> list[Any]:
        """Scan all configured directories and return discovered ToolMeta objects.

        Args:
            skip_existing: If True, skip tool classes already loaded.

        Returns:
            List of :class:`~agent_core.tool_registry.ToolMeta` ready to register.
        """
        discovered: list[Any] = []
        for dir_path in self._tool_dirs:
            p = Path(dir_path)
            if not p.exists() or not p.is_dir():
                logger.debug("Plugin dir not found, skipping: %s", dir_path)
                continue
            for py_file in sorted(p.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue  # skip __init__, __pycache__, etc.
                metas = self._load_from_file(py_file, skip_existing=skip_existing)
                discovered.extend(metas)

        logger.info("Plugin scan complete: %d new tool(s) found", len(discovered))
        return discovered

    def load_file(self, filepath: str | Path) -> list[Any]:
        """Load tool classes from a single file.

        Args:
            filepath: Absolute or relative path to a Python plugin file.

        Returns:
            List of :class:`~agent_core.tool_registry.ToolMeta` objects.
        """
        return self._load_from_file(Path(filepath), skip_existing=False)

    def loaded_files(self) -> list[str]:
        """Return paths of all files that have been loaded."""
        return list(self._loaded.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_file(self, filepath: Path, skip_existing: bool = True) -> list[Any]:
        """Import a Python file and extract all BaseTool subclasses."""
        filepath = filepath.resolve()
        BaseTool = _get_base_tool()
        ToolMeta = _get_tool_meta()

        already_loaded = self._loaded.get(str(filepath), [])

        try:
            spec = importlib.util.spec_from_file_location(
                f"lopen_plugin_{filepath.stem}",
                filepath,
            )
            if spec is None or spec.loader is None:
                logger.warning("Cannot create spec for %s", filepath)
                return []

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Failed to import plugin file %s: %s", filepath, exc)
            return []

        metas: list[Any] = []
        class_names: list[str] = []

        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            # Must subclass BaseTool, not be BaseTool itself, and not be abstract
            if (
                not issubclass(obj, BaseTool)
                or obj is BaseTool
                or inspect.isabstract(obj)
            ):
                continue

            class_name = obj.__name__
            if skip_existing and class_name in already_loaded:
                logger.debug("Skipping already-loaded class: %s", class_name)
                continue

            tool_name = getattr(obj, "name", obj.__name__.lower().replace("tool", ""))
            description = getattr(obj, "description", f"Plugin: {tool_name}")
            tags = getattr(obj, "tags", [])
            version = getattr(obj, "version", "1.0.0")
            requires_perm = getattr(obj, "requires_permission", False)

            # Instantiate the tool with the llm_adapter if available
            try:
                instance = obj(llm_adapter=self._llm)
            except Exception as exc:
                logger.warning("Could not instantiate %s: %s — skipping", class_name, exc)
                continue

            meta = ToolMeta(
                name=tool_name,
                description=description,
                version=version,
                requires_permission=requires_perm,
                tags=list(tags) if tags else [],
                instance=instance,
                tool_class=obj,
            )
            metas.append(meta)
            class_names.append(class_name)
            logger.info(
                "Plugin loaded: %s v%s from %s (permission_required=%s)",
                tool_name,
                version,
                filepath.name,
                requires_perm,
            )

        self._loaded[str(filepath)] = already_loaded + class_names
        return metas
