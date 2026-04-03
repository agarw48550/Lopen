"""Disk check: monitor free disk space and alert when below threshold."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DiskCheck:
    """Monitors free disk space on a target path."""

    def __init__(
        self,
        path: str = "/",
        threshold_gb: float = 5.0,
        on_alert: Optional[callable] = None,
    ) -> None:
        self.path = path
        self.threshold_gb = threshold_gb
        self.on_alert = on_alert

    def check(self) -> dict[str, float]:
        """Run disk check. Returns usage dict with free_gb, total_gb, used_gb."""
        try:
            usage = shutil.disk_usage(self.path)
        except Exception as exc:
            logger.error("Disk check failed for %r: %s", self.path, exc)
            return {"free_gb": -1, "total_gb": -1, "used_gb": -1, "error": str(exc)}

        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)

        result = {
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "path": self.path,
        }

        if free_gb < self.threshold_gb:
            logger.warning(
                "Low disk space: %.1f GB free (threshold: %.1f GB) on %r",
                free_gb,
                self.threshold_gb,
                self.path,
            )
            if self.on_alert:
                try:
                    self.on_alert(result)
                except Exception as exc:
                    logger.error("Disk alert callback failed: %s", exc)
        else:
            logger.debug("Disk OK: %.1f GB free on %r", free_gb, self.path)

        return result
