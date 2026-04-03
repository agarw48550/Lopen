"""Service heartbeat: poll services and record health to SQLite."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_HTTPX_AVAILABLE = False
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    pass


class Heartbeat:
    """Checks service health endpoints and records results to the database."""

    def __init__(
        self,
        services: Optional[dict[str, str]] = None,
        db: Any | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.services: dict[str, str] = services or {
            "orchestrator": "http://localhost:8000/health",
            "dashboard": "http://localhost:8080/health",
        }
        self._db = db
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_all(self) -> dict[str, dict[str, Any]]:
        """Check all services; return map of service_name -> status dict."""
        results: dict[str, dict[str, Any]] = {}
        for name, url in self.services.items():
            status = self._check_service(name, url)
            results[name] = status
            if self._db:
                try:
                    self._db.record_heartbeat(name, status["healthy"], status.get("error", ""))
                except Exception as exc:
                    logger.error("Failed to record heartbeat for %s: %s", name, exc)
        return results

    def check_service(self, name: str, url: str) -> dict[str, Any]:
        return self._check_service(name, url)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_service(self, name: str, url: str) -> dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        if not _HTTPX_AVAILABLE:
            logger.warning("httpx not available — heartbeat for %s skipped", name)
            return {"name": name, "healthy": None, "ts": ts, "error": "httpx not installed"}

        try:
            resp = httpx.get(url, timeout=self.timeout)
            healthy = resp.status_code < 400
            logger.info("Heartbeat %s: %s (%d)", name, "OK" if healthy else "FAIL", resp.status_code)
            return {"name": name, "healthy": healthy, "status_code": resp.status_code, "ts": ts}
        except Exception as exc:
            logger.warning("Heartbeat %s FAILED: %s", name, exc)
            return {"name": name, "healthy": False, "error": str(exc), "ts": ts}
