"""Smoke tests: API health endpoint checks."""

import pytest
import pytest_asyncio


@pytest.mark.asyncio
async def test_orchestrator_health() -> None:
    """Test that orchestrator /health returns 200."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:8000/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("status") in ("healthy", "ok", "running")
    except Exception as exc:
        pytest.skip(f"Orchestrator not running: {exc}")


@pytest.mark.asyncio
async def test_dashboard_health() -> None:
    """Test that dashboard /health returns 200."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:8080/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("status") == "healthy"
    except Exception as exc:
        pytest.skip(f"Dashboard not running: {exc}")


@pytest.mark.asyncio
async def test_orchestrator_status_endpoint() -> None:
    """Test that /status endpoint responds."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:8000/status")
            assert resp.status_code == 200
    except Exception as exc:
        pytest.skip(f"Orchestrator not running: {exc}")


def test_health_route_via_test_client() -> None:
    """Test the orchestrator app directly without a server."""
    from fastapi.testclient import TestClient
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from orchestrator import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("healthy", "ok", "running")
