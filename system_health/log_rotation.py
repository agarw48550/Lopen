"""Log rotation: rotate log files that exceed a size threshold."""

from __future__ import annotations

import gzip
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = Path("logs")
_DEFAULT_THRESHOLD_MB = 50
_DEFAULT_KEEP = 5


class LogRotation:
    """Rotates log files that exceed a configurable size threshold."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        threshold_mb: float = _DEFAULT_THRESHOLD_MB,
        keep: int = _DEFAULT_KEEP,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
        self.threshold_bytes = int(threshold_mb * 1024 * 1024)
        self.keep = keep

    def run(self) -> dict[str, list[str]]:
        """Check all .log files; rotate those over the threshold. Returns list of rotated files."""
        rotated: list[str] = []

        if not self.log_dir.is_dir():
            logger.debug("Log dir does not exist: %s", self.log_dir)
            return {"rotated": rotated}

        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_size >= self.threshold_bytes:
                result = self._rotate(log_file)
                if result:
                    rotated.append(result)

        logger.info("LogRotation: rotated %d file(s)", len(rotated))
        return {"rotated": rotated}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rotate(self, log_file: Path) -> Optional[str]:
        """Compress and archive a log file; prune old archives."""
        rotated_name = log_file.with_suffix(f".log.1.gz")
        try:
            # Shift existing .N.gz files up
            for i in range(self.keep - 1, 0, -1):
                src = log_file.with_suffix(f".log.{i}.gz")
                dst = log_file.with_suffix(f".log.{i + 1}.gz")
                if src.exists():
                    src.rename(dst)

            # Compress current log to .1.gz
            with open(log_file, "rb") as f_in:
                with gzip.open(str(rotated_name), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Truncate the original
            log_file.write_text("")

            # Prune oldest beyond keep limit
            oldest = log_file.with_suffix(f".log.{self.keep + 1}.gz")
            if oldest.exists():
                oldest.unlink()

            logger.info("Rotated log: %s -> %s", log_file.name, rotated_name.name)
            return str(rotated_name)
        except Exception as exc:
            logger.error("Log rotation failed for %s: %s", log_file, exc)
            return None
