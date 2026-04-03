"""Safe file operations tool: read, write, search, list, move, delete."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

# Allowed root directories for file operations
_ALLOWED_ROOTS: list[Path] = [
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path.home() / "Downloads",
]


def _is_allowed(path: Path) -> bool:
    """Return True if path is within one of the allowed root directories."""
    resolved = path.resolve()
    return any(
        str(resolved).startswith(str(root.resolve()))
        for root in _ALLOWED_ROOTS
    )


class FileOps(BaseTool):
    """Permission-checked file operations within ~/Documents, ~/Desktop, ~/Downloads."""

    name = "file_ops"
    description = "Safe file read/write/search/list/move/delete within approved directories."
    requires_permission = True

    def run(self, query: str, **kwargs: Any) -> str:
        """Dispatch based on 'action' kwarg or auto-detect from query."""
        action = kwargs.get("action") or self._detect_action(query)
        path_str: str = kwargs.get("path", "")
        content: str = kwargs.get("content", "")
        dest_str: str = kwargs.get("destination", "")

        dispatch = {
            "read": lambda: self.read_file(path_str),
            "write": lambda: self.write_file(path_str, content),
            "list": lambda: self.list_dir(path_str or str(Path.home() / "Desktop")),
            "search": lambda: self.search_files(path_str or str(Path.home()), query),
            "move": lambda: self.move_file(path_str, dest_str),
            "delete": lambda: self.delete_file(path_str),
        }
        fn = dispatch.get(action)
        if fn:
            return fn()
        return f"Unknown file action: {action}"

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def read_file(self, path: str) -> str:
        p = Path(path).expanduser().resolve()
        if not _is_allowed(p):
            return f"Access denied: '{p}' is outside allowed directories."
        if not p.is_file():
            return f"File not found: {p}"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            logger.info("Read file: %s (%d chars)", p, len(text))
            return text[:10000]  # cap at 10KB
        except Exception as exc:
            return f"Error reading file: {exc}"

    def write_file(self, path: str, content: str) -> str:
        p = Path(path).expanduser().resolve()
        if not _is_allowed(p):
            return f"Access denied: '{p}' is outside allowed directories."
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            logger.info("Wrote file: %s (%d chars)", p, len(content))
            return f"File written: {p}"
        except Exception as exc:
            return f"Error writing file: {exc}"

    def list_dir(self, path: str) -> str:
        p = Path(path).expanduser().resolve()
        if not _is_allowed(p) and p != Path.home().resolve():
            return f"Access denied: '{p}'"
        if not p.is_dir():
            return f"Not a directory: {p}"
        items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = [f"{'DIR ' if i.is_dir() else 'FILE'} {i.name}" for i in items[:100]]
        return f"Contents of {p} ({len(items)} items):\n" + "\n".join(lines)

    def search_files(self, base_path: str, pattern: str) -> str:
        base = Path(base_path).expanduser().resolve()
        if not _is_allowed(base):
            return f"Access denied: '{base}'"
        results: list[str] = []
        try:
            for p in base.rglob(f"*{pattern}*"):
                results.append(str(p))
                if len(results) >= 50:
                    break
        except Exception as exc:
            return f"Search error: {exc}"
        if not results:
            return f"No files matching '{pattern}' found in {base}."
        return f"Found {len(results)} files:\n" + "\n".join(results)

    def move_file(self, source: str, destination: str) -> str:
        src = Path(source).expanduser().resolve()
        dst = Path(destination).expanduser().resolve()
        if not _is_allowed(src) or not _is_allowed(dst):
            return "Access denied: path outside allowed directories."
        if not src.exists():
            return f"Source not found: {src}"
        try:
            shutil.move(str(src), str(dst))
            logger.info("Moved %s -> %s", src, dst)
            return f"Moved '{src.name}' to '{dst}'"
        except Exception as exc:
            return f"Move failed: {exc}"

    def delete_file(self, path: str) -> str:
        p = Path(path).expanduser().resolve()
        if not _is_allowed(p):
            return f"Access denied: '{p}'"
        if not p.exists():
            return f"Not found: {p}"
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            logger.info("Deleted: %s", p)
            return f"Deleted: {p}"
        except Exception as exc:
            return f"Delete failed: {exc}"

    # ------------------------------------------------------------------
    # Auto-detect
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_action(query: str) -> str:
        lower = query.lower()
        if any(w in lower for w in ["read", "open", "show", "display"]):
            return "read"
        if any(w in lower for w in ["write", "save", "create"]):
            return "write"
        if any(w in lower for w in ["list", "show files", "what's in"]):
            return "list"
        if any(w in lower for w in ["search", "find", "look for"]):
            return "search"
        if any(w in lower for w in ["move", "move to"]):
            return "move"
        if any(w in lower for w in ["delete", "remove", "trash"]):
            return "delete"
        return "list"
