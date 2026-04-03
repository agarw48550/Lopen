"""Desktop organizer tool: categorise and move Desktop files into folders."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

_CATEGORY_MAP: dict[str, list[str]] = {
    "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg", ".heic", ".tiff"],
    "Videos": [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".m4v"],
    "Audio": [".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"],
    "Documents": [".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt", ".pages", ".md"],
    "Spreadsheets": [".xls", ".xlsx", ".csv", ".numbers", ".ods"],
    "Presentations": [".ppt", ".pptx", ".key", ".odp"],
    "Archives": [".zip", ".tar", ".gz", ".rar", ".7z", ".bz2"],
    "Code": [".py", ".js", ".ts", ".html", ".css", ".java", ".c", ".cpp", ".rs", ".go", ".sh"],
    "Data": [".json", ".yaml", ".yml", ".xml", ".sql", ".db"],
}


class DesktopOrganizer(BaseTool):
    """Organises Desktop (or any directory) files into category subfolders."""

    name = "desktop_organizer"
    description = "Organises Desktop files into folders by type (Images, Documents, Code, etc.)."
    requires_permission = True

    def run(self, query: str, **kwargs: Any) -> str:
        target_dir = kwargs.get("directory") or self._detect_target(query)
        dry_run: bool = kwargs.get("dry_run", False)

        target_path = Path(target_dir).expanduser().resolve()
        if not target_path.is_dir():
            return f"Directory not found: {target_path}"

        moved: list[str] = []
        skipped: list[str] = []

        for item in target_path.iterdir():
            if item.is_dir() or item.name.startswith("."):
                continue
            category = self._categorise(item)
            if category is None:
                skipped.append(item.name)
                continue
            dest_dir = target_path / category
            if not dry_run:
                dest_dir.mkdir(exist_ok=True)
                shutil.move(str(item), str(dest_dir / item.name))
            moved.append(f"{item.name} -> {category}/")

        summary = (
            f"{'[DRY RUN] ' if dry_run else ''}Organised {len(moved)} files "
            f"({len(skipped)} skipped) in '{target_path}'.\n"
        )
        if moved:
            summary += "Moved:\n" + "\n".join(f"  {m}" for m in moved[:20])
            if len(moved) > 20:
                summary += f"\n  … and {len(moved) - 20} more."

        logger.info("DesktopOrganizer: %s", summary[:120])

        # Optional AppleScript notification on macOS
        if platform.system() == "Darwin" and not dry_run and moved:
            self._notify_macos(f"Organised {len(moved)} files on your Desktop.")

        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_target(self, query: str) -> str:
        lower = query.lower()
        if "desktop" in lower:
            return str(Path.home() / "Desktop")
        if "downloads" in lower:
            return str(Path.home() / "Downloads")
        if "documents" in lower:
            return str(Path.home() / "Documents")
        return str(Path.home() / "Desktop")

    def _categorise(self, path: Path) -> str | None:
        ext = path.suffix.lower()
        for category, extensions in _CATEGORY_MAP.items():
            if ext in extensions:
                return category
        return "Misc"

    @staticmethod
    def _notify_macos(message: str) -> None:
        try:
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "Lopen"'],
                timeout=5,
                capture_output=True,
            )
        except Exception:
            pass
