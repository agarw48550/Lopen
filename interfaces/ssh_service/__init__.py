"""SSH service interface for Lopen.

Exposes a lightweight HTTP API on port 8001 that can be reached from
a remote machine via SSH port-forwarding (ssh -L 8001:localhost:8001 user@mac).

This avoids a full SSH daemon; instead:
  - The MacBook hosts the Lopen orchestrator on port 8000 (internal).
  - This service runs on port 8001 and accepts authenticated REST requests.
  - Remote callers on the MacBook Air forward port 8001 via SSH and call
    http://localhost:8001/query with an API-key header.

No extra dependencies: uses only Python stdlib (http.server) with an
optional FastAPI path for richer routing when FastAPI is available.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

_FASTAPI_AVAILABLE = False
try:
    from fastapi import FastAPI, Request, HTTPException, Depends  # type: ignore
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # type: ignore
    _FASTAPI_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Authentication helper
# ---------------------------------------------------------------------------

def _verify_api_key(provided: str, expected: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    if not expected:
        return False
    return hmac.compare_digest(
        hashlib.sha256(provided.encode()).digest(),
        hashlib.sha256(expected.encode()).digest(),
    )


# ---------------------------------------------------------------------------
# FastAPI-based SSH API (preferred when FastAPI is available)
# ---------------------------------------------------------------------------

def create_ssh_app(
    orchestrator_handler: Any = None,
    api_key: str = "",
) -> Any:
    """Create and return a FastAPI app for the SSH API service.

    Args:
        orchestrator_handler: Callable that accepts a query string and returns a
            response string.  If None, returns a mock response.
        api_key: API key for authentication.  If empty, authentication is
            disabled (development mode).
    """
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is not installed — cannot create SSH API app")

    app = FastAPI(title="Lopen SSH API", version="1.0.0")
    _key = api_key or os.environ.get("LOPEN_SSH_API_KEY", "")

    # ------------------------------------------------------------------
    # Auth dependency
    # ------------------------------------------------------------------

    security = HTTPBearer(auto_error=False)

    async def _auth(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    ) -> None:
        if not _key:
            return  # no key configured → open in dev mode
        token = credentials.credentials if credentials else ""
        if not _verify_api_key(token, _key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "lopen-ssh-api", "ts": time.time()}

    @app.post("/query")
    async def query(
        request: Request,
        _auth: None = Depends(_auth),
    ) -> dict[str, Any]:
        body = await request.json()
        text: str = body.get("query", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="'query' field required")

        t0 = time.time()
        try:
            if orchestrator_handler is not None:
                if asyncio.iscoroutinefunction(orchestrator_handler):
                    response = await orchestrator_handler(text)
                else:
                    response = await asyncio.to_thread(orchestrator_handler, text)
            else:
                response = f"[SSH API mock] Received: {text}"
        except Exception as exc:
            logger.error("SSH API query handler error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "query": text,
            "response": response,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }

    @app.get("/status")
    async def status(_auth: None = Depends(_auth)) -> dict[str, Any]:
        """Return a brief status summary useful from the CLI."""
        return {
            "service": "lopen-ssh-api",
            "orchestrator_connected": orchestrator_handler is not None,
            "ts": time.time(),
        }

    return app


# ---------------------------------------------------------------------------
# Stdlib fallback server (no FastAPI required)
# ---------------------------------------------------------------------------

class _SSHApiHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for the SSH API service."""

    _api_key: str = ""
    _handler: Any = None

    # silence request logs by default
    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
        logger.debug("SSH API: " + fmt, *args)

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        key = self.__class__._api_key
        if not key:
            return True
        auth_header = self.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ").strip()
        return _verify_api_key(token, key)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "ok", "service": "lopen-ssh-api"})
        elif parsed.path == "/status":
            if not self._check_auth():
                self._send_json({"error": "unauthorized"}, 401)
                return
            self._send_json({
                "service": "lopen-ssh-api",
                "orchestrator_connected": self.__class__._handler is not None,
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/query":
            self._send_json({"error": "not found"}, 404)
            return
        if not self._check_auth():
            self._send_json({"error": "unauthorized"}, 401)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        text: str = body.get("query", "").strip()
        if not text:
            self._send_json({"error": "'query' field required"}, 400)
            return
        t0 = time.time()
        handler = self.__class__._handler
        try:
            response = handler(text) if handler else f"[SSH API mock] {text}"
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)
            return
        self._send_json({
            "query": text,
            "response": response,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })


class SSHApiServer:
    """Lightweight SSH API server (stdlib-only fallback).

    Prefer create_ssh_app() + uvicorn when FastAPI is available.
    This class provides a zero-dependency fallback for environments
    where FastAPI is not installed.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8001,
        api_key: str = "",
        handler: Any = None,
    ) -> None:
        self.host = host
        self.port = port
        _SSHApiHandler._api_key = api_key or os.environ.get("LOPEN_SSH_API_KEY", "")
        _SSHApiHandler._handler = handler
        self._server: Optional[HTTPServer] = None

    def start(self) -> None:
        self._server = HTTPServer((self.host, self.port), _SSHApiHandler)
        logger.info("SSH API server (stdlib) listening on %s:%d", self.host, self.port)
        self._server.serve_forever()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
