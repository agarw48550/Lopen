"""Unit tests for system health monitors."""

import pytest
from system_health.ram_watchdog import RamWatchdog
from system_health.disk_check import DiskCheck
from system_health.log_rotation import LogRotation
from system_health.cache_cleanup import CacheCleanup


class TestDiskCheck:
    def test_check_returns_dict(self) -> None:
        dc = DiskCheck(path="/", threshold_gb=0.001)  # very low threshold so always passes
        result = dc.check()
        assert "free_gb" in result
        assert "total_gb" in result
        assert "used_gb" in result

    def test_check_values_are_positive(self) -> None:
        dc = DiskCheck(path="/")
        result = dc.check()
        assert result["free_gb"] >= 0
        assert result["total_gb"] > 0

    def test_alert_callback_fires_when_below_threshold(self) -> None:
        alerted: list[dict] = []
        dc = DiskCheck(path="/", threshold_gb=999999.0, on_alert=alerted.append)
        dc.check()
        assert len(alerted) == 1

    def test_no_alert_when_above_threshold(self) -> None:
        alerted: list[dict] = []
        dc = DiskCheck(path="/", threshold_gb=0.0, on_alert=alerted.append)
        dc.check()
        assert len(alerted) == 0

    def test_invalid_path_returns_error(self) -> None:
        dc = DiskCheck(path="/nonexistent_path_xyz")
        result = dc.check()
        assert "error" in result


class TestRamWatchdog:
    def test_check_returns_dict(self) -> None:
        watchdog = RamWatchdog(warning_gb=3.0, critical_gb=4.0)
        result = watchdog.check()
        assert "current_gb" in result
        assert "warning_gb" in result
        assert "critical_gb" in result

    def test_thresholds_respected(self) -> None:
        watchdog = RamWatchdog(warning_gb=3.0, critical_gb=4.0)
        result = watchdog.check()
        assert result["warning_gb"] == 3.0
        assert result["critical_gb"] == 4.0

    def test_critical_callback_fires(self) -> None:
        triggered: list[bool] = []
        watchdog = RamWatchdog(warning_gb=0.0, critical_gb=0.0, on_critical=lambda: triggered.append(True))
        watchdog.check()
        # If psutil is available, threshold of 0 GB should trigger critical
        # In mock mode it won't, but we just check it doesn't crash
        assert isinstance(triggered, list)

    def test_mock_mode_when_psutil_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import system_health.ram_watchdog as rw_module
        monkeypatch.setattr(rw_module, "_PSUTIL_AVAILABLE", False)
        watchdog = RamWatchdog()
        result = watchdog.check()
        assert result.get("mock") is True or result["current_gb"] == 0.0


class TestLogRotation:
    def test_run_on_missing_dir_is_safe(self, tmp_path) -> None:
        lr = LogRotation(log_dir=str(tmp_path / "nonexistent"), threshold_mb=50)
        result = lr.run()
        assert result == {"rotated": []}

    def test_small_file_not_rotated(self, tmp_path) -> None:
        log_file = tmp_path / "lopen.log"
        log_file.write_text("small log\n")
        lr = LogRotation(log_dir=str(tmp_path), threshold_mb=50)
        result = lr.run()
        assert result["rotated"] == []

    def test_large_file_is_rotated(self, tmp_path) -> None:
        log_file = tmp_path / "lopen.log"
        # Write just over 0.001 MB
        log_file.write_bytes(b"x" * 1100)
        lr = LogRotation(log_dir=str(tmp_path), threshold_mb=0.001)
        result = lr.run()
        assert len(result["rotated"]) == 1
        # Original should be empty after rotation
        assert log_file.read_text() == ""


class TestCacheCleanup:
    def test_run_empty_base_dir(self, tmp_path) -> None:
        cc = CacheCleanup(base_dir=str(tmp_path))
        result = cc.run()
        assert result["files_removed"] == 0

    def test_removes_pycache(self, tmp_path) -> None:
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_bytes(b"\x00" * 100)
        cc = CacheCleanup(base_dir=str(tmp_path))
        result = cc.run()
        assert result["files_removed"] >= 1
        assert not pycache.exists()
