"""Tests for SSH API service (stdlib and FastAPI variants)."""

from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection
from typing import Any

import pytest

from interfaces.ssh_service import (
    SSHApiServer,
    _verify_api_key,
    _SSHApiHandler,
)


# ---------------------------------------------------------------------------
# Auth helper tests
# ---------------------------------------------------------------------------

class TestVerifyApiKey:
    def test_valid_key(self) -> None:
        assert _verify_api_key("secret", "secret") is True

    def test_invalid_key(self) -> None:
        assert _verify_api_key("wrong", "secret") is False

    def test_empty_expected_key(self) -> None:
        # Empty expected key means auth is disabled → should return False
        assert _verify_api_key("any", "") is False

    def test_both_empty(self) -> None:
        assert _verify_api_key("", "") is False


# ---------------------------------------------------------------------------
# SSHApiServer (stdlib) integration tests
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int, api_key: str = "", handler: Any = None) -> SSHApiServer:
    server = SSHApiServer(host="127.0.0.1", port=port, api_key=api_key, handler=handler)
    t = threading.Thread(target=server.start, daemon=True)
    t.start()
    time.sleep(0.1)  # brief settle time
    return server


class TestSSHApiServerStdlib:
    def test_health_endpoint(self) -> None:
        port = _find_free_port()
        server = _start_server(port)
        conn = HTTPConnection("127.0.0.1", port, timeout=3)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        server.stop()
        assert resp.status == 200
        assert body["status"] == "ok"

    def test_unknown_path_returns_404(self) -> None:
        port = _find_free_port()
        server = _start_server(port)
        conn = HTTPConnection("127.0.0.1", port, timeout=3)
        conn.request("GET", "/nonexistent")
        resp = conn.getresponse()
        server.stop()
        assert resp.status == 404

    def test_query_no_auth_required_when_no_key(self) -> None:
        port = _find_free_port()
        called: list[str] = []

        def _handler(text: str) -> str:
            called.append(text)
            return f"echo: {text}"

        server = _start_server(port, api_key="", handler=_handler)
        conn = HTTPConnection("127.0.0.1", port, timeout=3)
        payload = json.dumps({"query": "hello"}).encode()
        conn.request("POST", "/query", body=payload, headers={"Content-Length": str(len(payload))})
        resp = conn.getresponse()
        body = json.loads(resp.read())
        server.stop()
        assert resp.status == 200
        assert "echo: hello" in body["response"]
        assert "hello" in called

    def test_query_with_invalid_key_returns_401(self) -> None:
        port = _find_free_port()
        server = _start_server(port, api_key="correctkey")
        conn = HTTPConnection("127.0.0.1", port, timeout=3)
        payload = json.dumps({"query": "hi"}).encode()
        conn.request(
            "POST",
            "/query",
            body=payload,
            headers={
                "Content-Length": str(len(payload)),
                "Authorization": "Bearer wrongkey",
            },
        )
        resp = conn.getresponse()
        server.stop()
        assert resp.status == 401

    def test_query_with_correct_key_succeeds(self) -> None:
        port = _find_free_port()
        server = _start_server(port, api_key="mykey", handler=lambda q: f"got: {q}")
        conn = HTTPConnection("127.0.0.1", port, timeout=3)
        payload = json.dumps({"query": "test"}).encode()
        conn.request(
            "POST",
            "/query",
            body=payload,
            headers={
                "Content-Length": str(len(payload)),
                "Authorization": "Bearer mykey",
            },
        )
        resp = conn.getresponse()
        body = json.loads(resp.read())
        server.stop()
        assert resp.status == 200
        assert "got: test" in body["response"]

    def test_query_missing_field_returns_400(self) -> None:
        port = _find_free_port()
        server = _start_server(port)
        conn = HTTPConnection("127.0.0.1", port, timeout=3)
        payload = json.dumps({}).encode()
        conn.request("POST", "/query", body=payload, headers={"Content-Length": str(len(payload))})
        resp = conn.getresponse()
        server.stop()
        assert resp.status == 400

    def test_latency_returned_in_response(self) -> None:
        port = _find_free_port()
        server = _start_server(port, handler=lambda q: "ok")
        conn = HTTPConnection("127.0.0.1", port, timeout=3)
        payload = json.dumps({"query": "ping"}).encode()
        conn.request("POST", "/query", body=payload, headers={"Content-Length": str(len(payload))})
        resp = conn.getresponse()
        body = json.loads(resp.read())
        server.stop()
        assert "latency_ms" in body
        assert body["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# FastAPI SSH app tests (when FastAPI is available)
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient
    from interfaces.ssh_service import create_ssh_app
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestSSHFastAPIApp:
    def _make_client(self, api_key: str = "", handler: Any = None) -> "TestClient":
        app = create_ssh_app(orchestrator_handler=handler, api_key=api_key)
        return TestClient(app)

    def test_health_endpoint(self) -> None:
        client = self._make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_query_no_auth(self) -> None:
        client = self._make_client(handler=lambda q: f"resp:{q}")
        resp = client.post("/query", json={"query": "hello"})
        assert resp.status_code == 200
        assert "resp:hello" in resp.json()["response"]

    def test_query_invalid_key(self) -> None:
        client = self._make_client(api_key="secret")
        resp = client.post(
            "/query",
            json={"query": "hi"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

    def test_query_correct_key(self) -> None:
        client = self._make_client(api_key="key123", handler=lambda q: "ok")
        resp = client.post(
            "/query",
            json={"query": "test"},
            headers={"Authorization": "Bearer key123"},
        )
        assert resp.status_code == 200

    def test_query_missing_body(self) -> None:
        client = self._make_client()
        resp = client.post("/query", json={})
        assert resp.status_code == 400

    def test_status_endpoint(self) -> None:
        client = self._make_client()
        resp = client.get("/status")
        assert resp.status_code == 200
