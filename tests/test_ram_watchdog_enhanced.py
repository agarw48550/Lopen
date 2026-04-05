"""Tests for the enhanced RamWatchdog."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from system_health.ram_watchdog import RamWatchdog, _RESTART_BACKOFF, _PSUTIL_AVAILABLE


# ---------------------------------------------------------------------------
# Tests that work in both mock and live modes
# ---------------------------------------------------------------------------

class TestRamWatchdogThresholds:
    """Test the three-level threshold system."""

    def test_check_returns_dict_with_all_keys(self) -> None:
        watchdog = RamWatchdog(warning_gb=3.2, critical_gb=3.6, halt_gb=3.8)
        result = watchdog.check()
        assert "current_gb" in result
        assert "warning_gb" in result
        assert "critical_gb" in result
        assert "halt_gb" in result

    def test_thresholds_match_config(self) -> None:
        watchdog = RamWatchdog(warning_gb=1.0, critical_gb=2.0, halt_gb=3.0)
        result = watchdog.check()
        assert result["warning_gb"] == 1.0
        assert result["critical_gb"] == 2.0
        assert result["halt_gb"] == 3.0

    def test_mock_mode_when_psutil_unavailable(self) -> None:
        with patch("system_health.ram_watchdog._PSUTIL_AVAILABLE", False):
            watchdog = RamWatchdog()
            result = watchdog.check()
            assert result.get("mock") is True
            assert result["current_gb"] == 0.0

    @pytest.mark.skipif(not _PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_warning_callback_fires_at_warning_threshold(self) -> None:
        fired: list[bool] = []
        watchdog = RamWatchdog(
            warning_gb=0.001,
            critical_gb=999.0,
            halt_gb=9999.0,
            on_warning=lambda: fired.append(True),
        )
        watchdog.check()
        assert len(fired) == 1

    @pytest.mark.skipif(not _PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_warning_callback_fires_only_once_per_crossing(self) -> None:
        fired: list[bool] = []
        watchdog = RamWatchdog(
            warning_gb=0.001,
            critical_gb=999.0,
            halt_gb=9999.0,
            on_warning=lambda: fired.append(True),
        )
        watchdog.check()
        watchdog.check()  # second check — already at warning level → no re-fire
        assert len(fired) == 1

    @pytest.mark.skipif(not _PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_critical_callback_fires_at_critical_threshold(self) -> None:
        fired: list[bool] = []
        watchdog = RamWatchdog(
            warning_gb=0.0,
            critical_gb=0.001,
            halt_gb=9999.0,
            on_critical=lambda: fired.append(True),
        )
        watchdog.check()
        assert len(fired) == 1

    @pytest.mark.skipif(not _PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_halt_callback_fires_at_halt_threshold(self) -> None:
        fired: list[bool] = []
        watchdog = RamWatchdog(
            warning_gb=0.0,
            critical_gb=0.0,
            halt_gb=0.001,
            on_halt=lambda: fired.append(True),
        )
        watchdog.check()
        assert len(fired) == 1

    def test_no_callbacks_fire_when_under_warning(self) -> None:
        fired: list[bool] = []
        watchdog = RamWatchdog(
            warning_gb=9999.0,
            critical_gb=99999.0,
            halt_gb=999999.0,
            on_warning=lambda: fired.append(True),
            on_critical=lambda: fired.append(True),
            on_halt=lambda: fired.append(True),
        )
        watchdog.check()
        assert fired == []

    def test_system_ram_gb_returns_nonnegative(self) -> None:
        result = RamWatchdog.system_ram_gb()
        assert result >= 0.0


class TestRamWatchdogBackoff:
    """Test the restart backoff mechanism."""

    def test_restart_attempts_start_at_zero(self) -> None:
        watchdog = RamWatchdog()
        assert watchdog._restart_attempts == 0

    def test_reset_backoff_resets_counter(self) -> None:
        watchdog = RamWatchdog()
        watchdog._restart_attempts = 2
        watchdog.reset_backoff()
        assert watchdog._restart_attempts == 0

    def test_backoff_increments_on_each_fire_with_backoff(self) -> None:
        called: list[int] = []
        watchdog = RamWatchdog(on_critical=lambda: called.append(1))
        # Patch sleep to speed up test
        with patch("system_health.ram_watchdog.time") as mock_time:
            mock_time.sleep = lambda s: None
            watchdog._fire_with_backoff(watchdog.on_critical)
            assert watchdog._restart_attempts == 1
            watchdog._fire_with_backoff(watchdog.on_critical)
            assert watchdog._restart_attempts == 2
        assert len(called) == 2

    def test_fire_with_backoff_stops_after_max_attempts(self) -> None:
        called: list[int] = []
        watchdog = RamWatchdog(on_critical=lambda: called.append(1))
        watchdog._restart_attempts = len(_RESTART_BACKOFF)
        with patch("system_health.ram_watchdog.time") as mock_time:
            mock_time.sleep = lambda s: None
            watchdog._fire_with_backoff(watchdog.on_critical)
        # callback should NOT be called once max attempts exceeded
        assert len(called) == 0

    def test_backoff_constant_count(self) -> None:
        """Ensure backoff sequence has exactly 3 steps as per spec."""
        assert len(_RESTART_BACKOFF) == 3

    def test_backoff_values_are_ascending(self) -> None:
        """Backoff delays must increase to prevent rapid restart loops."""
        for i in range(len(_RESTART_BACKOFF) - 1):
            assert _RESTART_BACKOFF[i] <= _RESTART_BACKOFF[i + 1]
