"""Tests for AutonomousTaskScheduler."""

from __future__ import annotations

import time

import pytest
from agent_core.autonomous_tasks import AutonomousTaskScheduler, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_config(self) -> None:
        scheduler = AutonomousTaskScheduler()
        status = scheduler.get_status()
        assert "running" in status
        assert "enabled" in status
        assert "tasks" in status

    def test_custom_config(self) -> None:
        cfg = {
            "autonomy": {
                "enabled": False,
                "tasks": {
                    "organize_desktop": {"enabled": False, "interval_seconds": 60},
                    "check_deadlines": {"enabled": False, "interval_seconds": 60},
                    "notion_sync": {"enabled": False, "interval_seconds": 60},
                },
            }
        }
        scheduler = AutonomousTaskScheduler(cfg)
        assert scheduler.get_status()["enabled"] is False

    def test_none_config_uses_defaults(self) -> None:
        scheduler = AutonomousTaskScheduler(None)
        status = scheduler.get_status()
        assert isinstance(status["enabled"], bool)

    def test_all_default_tasks_present(self) -> None:
        scheduler = AutonomousTaskScheduler()
        task_names = set(scheduler.get_status()["tasks"].keys())
        assert "organize_desktop" in task_names
        assert "check_deadlines" in task_names
        assert "notion_sync" in task_names


# ---------------------------------------------------------------------------
# Start / Stop
# ---------------------------------------------------------------------------

class TestStartStop:
    def test_not_running_before_start(self) -> None:
        scheduler = AutonomousTaskScheduler()
        assert scheduler.get_status()["running"] is False

    def test_running_after_start(self) -> None:
        scheduler = AutonomousTaskScheduler()
        scheduler.start()
        try:
            assert scheduler.get_status()["running"] is True
        finally:
            scheduler.stop()

    def test_not_running_after_stop(self) -> None:
        scheduler = AutonomousTaskScheduler()
        scheduler.start()
        scheduler.stop()
        assert scheduler.get_status()["running"] is False

    def test_double_start_is_safe(self) -> None:
        scheduler = AutonomousTaskScheduler()
        scheduler.start()
        scheduler.start()  # Should not raise or create a second thread
        try:
            assert scheduler.get_status()["running"] is True
        finally:
            scheduler.stop()

    def test_stop_without_start_is_safe(self) -> None:
        scheduler = AutonomousTaskScheduler()
        scheduler.stop()  # Should not raise
        assert scheduler.get_status()["running"] is False


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_returns_dict(self) -> None:
        scheduler = AutonomousTaskScheduler()
        assert isinstance(scheduler.get_status(), dict)

    def test_task_status_has_required_keys(self) -> None:
        scheduler = AutonomousTaskScheduler()
        status = scheduler.get_status()
        for task_info in status["tasks"].values():
            assert "enabled" in task_info
            assert "interval_seconds" in task_info
            assert "last_run" in task_info

    def test_last_run_none_before_any_run(self) -> None:
        scheduler = AutonomousTaskScheduler()
        status = scheduler.get_status()
        for task_info in status["tasks"].values():
            assert task_info["last_run_ago"] is None
