"""RAM watchdog: monitor RSS of all Lopen processes and trigger restart if needed."""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_PSUTIL_AVAILABLE = False
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    pass


class RamWatchdog:
    """Monitors total RSS of Lopen processes and takes action at configurable thresholds."""

    def __init__(
        self,
        warning_gb: float = 3.0,
        critical_gb: float = 4.0,
        process_name_filter: str = "lopen",
        on_critical: Optional[Callable[[], None]] = None,
    ) -> None:
        self.warning_gb = warning_gb
        self.critical_gb = critical_gb
        self.process_name_filter = process_name_filter.lower()
        self.on_critical = on_critical
        self._mock_mode = not _PSUTIL_AVAILABLE
        if self._mock_mode:
            logger.warning("psutil not available — RamWatchdog in mock mode")

    def check(self) -> dict[str, float]:
        """Run a RAM check; returns dict with current_gb, warning_gb, critical_gb."""
        if self._mock_mode:
            return {"current_gb": 0.0, "warning_gb": self.warning_gb, "critical_gb": self.critical_gb, "mock": True}

        total_rss = self._measure_rss()
        current_gb = total_rss / (1024 ** 3)
        result = {
            "current_gb": round(current_gb, 3),
            "warning_gb": self.warning_gb,
            "critical_gb": self.critical_gb,
        }

        if current_gb >= self.critical_gb:
            logger.critical(
                "RAM usage %.2f GB exceeds critical threshold %.1f GB — triggering recovery",
                current_gb,
                self.critical_gb,
            )
            if self.on_critical:
                try:
                    self.on_critical()
                except Exception as exc:
                    logger.error("on_critical callback failed: %s", exc)
        elif current_gb >= self.warning_gb:
            logger.warning("RAM usage %.2f GB approaching warning threshold %.1f GB", current_gb, self.warning_gb)
        else:
            logger.debug("RAM usage OK: %.2f GB", current_gb)

        return result

    def _measure_rss(self) -> int:
        """Return total RSS bytes across current process and its children."""
        total = 0
        try:
            proc = psutil.Process(os.getpid())
            total += proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as exc:
            logger.error("RSS measurement failed: %s", exc)
        return total

    @staticmethod
    def system_ram_gb() -> float:
        """Return total system RAM in GB."""
        if not _PSUTIL_AVAILABLE:
            return 0.0
        return psutil.virtual_memory().total / (1024 ** 3)
