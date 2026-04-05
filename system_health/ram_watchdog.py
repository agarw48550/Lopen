"""RAM watchdog: monitor RSS of all Lopen processes and trigger restart if needed.

Thresholds for 4 GB target hardware (2017 Intel MacBook Pro):
  warning_gb  = 3.2 — unload LLM model to free memory
  critical_gb = 3.6 — trigger emergency halt / service restart
  halt_gb     = 3.8 — absolute upper bound; force-kill heavy processes

Restart backoff: 1s → 5s → 30s → give up (prevents tight crash loops).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_PSUTIL_AVAILABLE = False
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    pass

# Backoff delays in seconds for automatic service restart
_RESTART_BACKOFF = (1, 5, 30)


class RamWatchdog:
    """Monitors total RSS of Lopen processes and takes action at configurable thresholds.

    Three escalating actions:
      1. **warning**  — log a warning and call ``on_warning`` (e.g. unload LLM).
      2. **critical** — call ``on_critical`` (e.g. restart a service).
      3. **halt**     — call ``on_halt`` (emergency stop; prevents OOM crash).

    Each callback is only invoked once per threshold crossing (hysteresis:
    must drop 10 % below threshold before re-triggering).
    """

    def __init__(
        self,
        warning_gb: float = 3.2,
        critical_gb: float = 3.6,
        halt_gb: float = 3.8,
        process_name_filter: str = "lopen",
        on_warning: Optional[Callable[[], None]] = None,
        on_critical: Optional[Callable[[], None]] = None,
        on_halt: Optional[Callable[[], None]] = None,
    ) -> None:
        self.warning_gb = warning_gb
        self.critical_gb = critical_gb
        self.halt_gb = halt_gb
        self.process_name_filter = process_name_filter.lower()
        self.on_warning = on_warning
        self.on_critical = on_critical
        self.on_halt = on_halt
        self._mock_mode = not _PSUTIL_AVAILABLE
        # Hysteresis state — track which level we've already fired
        self._level: str = "ok"   # "ok" | "warning" | "critical" | "halt"
        # Restart attempt counter for backoff
        self._restart_attempts: int = 0
        if self._mock_mode:
            logger.warning("psutil not available — RamWatchdog in mock mode")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> dict[str, float]:
        """Run a RAM check; returns dict with current_gb and threshold values."""
        if self._mock_mode:
            return {
                "current_gb": 0.0,
                "warning_gb": self.warning_gb,
                "critical_gb": self.critical_gb,
                "halt_gb": self.halt_gb,
                "mock": True,
            }

        total_rss = self._measure_rss()
        current_gb = total_rss / (1024 ** 3)
        result: dict[str, float] = {
            "current_gb": round(current_gb, 3),
            "warning_gb": self.warning_gb,
            "critical_gb": self.critical_gb,
            "halt_gb": self.halt_gb,
        }

        # Hysteresis: require 10% clearance before resetting level
        clearance = 0.10

        if current_gb >= self.halt_gb:
            if self._level != "halt":
                self._level = "halt"
                logger.critical(
                    "RAM %.2f GB >= halt threshold %.1f GB — emergency stop",
                    current_gb, self.halt_gb,
                )
                self._fire(self.on_halt)
        elif current_gb >= self.critical_gb:
            if self._level not in ("critical", "halt"):
                self._level = "critical"
                logger.critical(
                    "RAM %.2f GB exceeds critical threshold %.1f GB — triggering recovery",
                    current_gb, self.critical_gb,
                )
                self._fire_with_backoff(self.on_critical)
        elif current_gb >= self.warning_gb:
            if self._level == "ok":
                self._level = "warning"
                logger.warning(
                    "RAM %.2f GB approaching warning threshold %.1f GB",
                    current_gb, self.warning_gb,
                )
                self._fire(self.on_warning)
        else:
            # Below all thresholds with hysteresis
            if current_gb < self.warning_gb * (1 - clearance):
                if self._level != "ok":
                    logger.info("RAM %.2f GB — back below warning threshold, resetting level", current_gb)
                    self._level = "ok"
                    self._restart_attempts = 0
            logger.debug("RAM usage OK: %.2f GB", current_gb)

        return result

    def reset_backoff(self) -> None:
        """Call after a successful service restart to reset the backoff counter."""
        self._restart_attempts = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire(self, callback: Optional[Callable[[], None]]) -> None:
        if callback:
            try:
                callback()
            except Exception as exc:
                logger.error("Watchdog callback failed: %s", exc)

    def _fire_with_backoff(self, callback: Optional[Callable[[], None]]) -> None:
        """Call callback with exponential backoff to prevent rapid restart loops."""
        if not callback:
            return
        attempt = self._restart_attempts
        if attempt >= len(_RESTART_BACKOFF):
            logger.error(
                "Watchdog: exceeded max restart attempts (%d) — giving up", len(_RESTART_BACKOFF)
            )
            return
        delay = _RESTART_BACKOFF[attempt]
        self._restart_attempts += 1
        if delay > 0:
            logger.info("Watchdog: waiting %ds before restart attempt %d", delay, attempt + 1)
            time.sleep(delay)
        self._fire(callback)

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
