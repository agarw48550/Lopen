"""Web dashboard FastAPI app on port 8080."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_dashboard_app(cors_origins: list[str] | None = None) -> FastAPI:
    """Factory that builds and returns the dashboard FastAPI app."""
    app = FastAPI(
        title="Lopen Dashboard",
        description="Lopen autonomous assistant web dashboard",
        version="1.0.0",
    )

    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Import and mount API routes
    from interfaces.web_dashboard.api import build_router
    api_router = build_router(templates)
    app.include_router(api_router)

    logger.info("Dashboard app created (cors_origins=%s)", origins)
    return app
