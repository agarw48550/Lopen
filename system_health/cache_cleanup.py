"""Cache cleanup: purge Lopen temp cache directories."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIRS: list[Path] = [
    Path("storage/__pycache__"),
    Path("agent_core/__pycache__"),
    Path("tools/__pycache__"),
    Path("llm/__pycache__"),
]


class CacheCleanup:
    """Purges Python __pycache__ and other temp files for Lopen."""

    def __init__(
        self,
        extra_dirs: Optional[list[str]] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.extra_dirs: list[Path] = [Path(d) for d in (extra_dirs or [])]

    def run(self) -> dict[str, int]:
        """Perform cleanup. Returns dict with files_removed and dirs_removed counts."""
        targets = list(_DEFAULT_CACHE_DIRS) + self.extra_dirs
        files_removed = 0
        dirs_removed = 0

        # Also collect all __pycache__ recursively
        for pycache in self.base_dir.rglob("__pycache__"):
            if pycache not in targets:
                targets.append(pycache)

        for target in targets:
            resolved = (self.base_dir / target).resolve() if not target.is_absolute() else target
            if resolved.is_dir():
                count = sum(1 for _ in resolved.rglob("*") if _.is_file())
                shutil.rmtree(resolved, ignore_errors=True)
                files_removed += count
                dirs_removed += 1
                logger.info("Cache cleanup: removed dir %s (%d files)", resolved, count)
            elif resolved.is_file():
                resolved.unlink()
                files_removed += 1

        # Clean .pyc files
        for pyc in self.base_dir.rglob("*.pyc"):
            try:
                pyc.unlink()
                files_removed += 1
            except OSError:
                pass

        logger.info("CacheCleanup complete: %d files, %d dirs removed", files_removed, dirs_removed)
        return {"files_removed": files_removed, "dirs_removed": dirs_removed}
