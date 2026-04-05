"""Autonomous background task scheduler for Lopen.

Runs background tasks based on config without user prompting.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "autonomy": {
        "enabled": True,
        "tasks": {
            "organize_desktop": {
                "enabled": False,
                "interval_seconds": 3600,
            },
            "check_deadlines": {
                "enabled": False,
                "interval_seconds": 1800,
            },
            "notion_sync": {
                "enabled": False,
                "interval_seconds": 900,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# AutonomousTaskScheduler
# ---------------------------------------------------------------------------

class AutonomousTaskScheduler:
    """Background task scheduler that runs periodic autonomous tasks.

    Uses a single daemon thread with a polling loop.  Each task has its
    own configured interval and can be individually enabled/disabled via
    the config dict passed at construction time.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise the scheduler with the given config.

        Args:
            config: Configuration dict.  If *None*, ``DEFAULT_CONFIG`` is used.
                    Expected structure mirrors ``DEFAULT_CONFIG``.
        """
        cfg = config or DEFAULT_CONFIG
        autonomy_cfg = cfg.get("autonomy", DEFAULT_CONFIG["autonomy"])
        self._enabled: bool = autonomy_cfg.get("enabled", True)
        tasks_cfg: dict[str, Any] = autonomy_cfg.get("tasks", {})

        self._task_configs: dict[str, dict[str, Any]] = {
            name: DEFAULT_CONFIG["autonomy"]["tasks"].get(name, {}) | tasks_cfg.get(name, {})
            for name in DEFAULT_CONFIG["autonomy"]["tasks"]
        }
        # Last run timestamp per task
        self._last_run: dict[str, float] = {name: 0.0 for name in self._task_configs}

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running: bool = False

        # Map task names to handler methods
        self._handlers: dict[str, Callable[[], None]] = {
            "organize_desktop": self._task_organize_desktop,
            "check_deadlines": self._task_check_deadlines,
            "notion_sync": self._task_notion_sync,
        }

        logger.info(
            "AutonomousTaskScheduler initialised (enabled=%s, tasks=%s)",
            self._enabled,
            list(self._task_configs.keys()),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._running:
            logger.debug("AutonomousTaskScheduler already running")
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="autonomous-scheduler",
        )
        self._thread.start()
        logger.info("AutonomousTaskScheduler started")

    def stop(self) -> None:
        """Stop the scheduler and wait for the thread to finish."""
        self._running = False
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("AutonomousTaskScheduler stopped")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return the current scheduler status.

        Returns:
            Dict with ``running``, ``enabled``, and per-task status info.
        """
        task_status: dict[str, Any] = {}
        for name, cfg in self._task_configs.items():
            last = self._last_run.get(name, 0.0)
            task_status[name] = {
                "enabled": cfg.get("enabled", False),
                "interval_seconds": cfg.get("interval_seconds", 0),
                "last_run": last,
                "last_run_ago": round(time.monotonic() - last, 1) if last > 0 else None,
            }
        return {
            "running": self._running,
            "enabled": self._enabled,
            "tasks": task_status,
        }

    # ------------------------------------------------------------------
    # Scheduler loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main loop: poll every 10 s and dispatch due tasks."""
        while not self._stop_event.is_set():
            if self._enabled:
                now = time.monotonic()
                for name, cfg in self._task_configs.items():
                    if not cfg.get("enabled", False):
                        continue
                    interval = cfg.get("interval_seconds", 3600)
                    if now - self._last_run.get(name, 0.0) >= interval:
                        handler = self._handlers.get(name)
                        if handler:
                            try:
                                logger.info("Running autonomous task: %s", name)
                                handler()
                                self._last_run[name] = time.monotonic()
                            except Exception as exc:
                                logger.error("Autonomous task %s failed: %s", name, exc)
            self._stop_event.wait(timeout=10)

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------

    def _task_organize_desktop(self) -> None:
        """Organise Desktop files into folders by file type."""
        logger.info("Task: organize_desktop — scanning Desktop")
        # Actual implementation would use file_ops tool / DesktopOrganizer
        # Placeholder: log only so task runs cleanly without side-effects
        logger.info("Task: organize_desktop — complete (no-op in base implementation)")

    def _task_check_deadlines(self) -> None:
        """Check calendars and Notion for upcoming deadlines."""
        logger.info("Task: check_deadlines — checking sources")
        logger.info("Task: check_deadlines — complete (no-op in base implementation)")

    def _task_notion_sync(self) -> None:
        """Sync latest data from Notion databases."""
        logger.info("Task: notion_sync — syncing")
        logger.info("Task: notion_sync — complete (no-op in base implementation)")
