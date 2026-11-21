"""ASGI / FastAPI application factory for cloud hosts (uvicorn app:app)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from nicegui import app as nicegui_app
from nicegui import ui
from sqlalchemy import text

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.session import engine
from material_intelligence.app.jobs.worker import start_worker_on_startup
from material_intelligence.app.main import register_pages
from material_intelligence.app.services.bootstrap import bootstrap_application
from material_intelligence.app.utils.logging import setup_logging

_asgi_app: FastAPI | None = None


def create_asgi_app() -> FastAPI:
    """Build FastAPI app with NiceGUI mounted — entrypoint for platforms expecting FastAPI."""
    global _asgi_app
    if _asgi_app is not None:
        return _asgi_app

    setup_logging()
    settings = get_settings()
    bootstrap_application()
    start_worker_on_startup()
    register_pages()

    fastapi_app = FastAPI(
        title=settings.app_name,
        docs_url="/api/docs",
        redoc_url=None,
    )

    @fastapi_app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "service": settings.app_name}

    @fastapi_app.get("/api/ready")
    def ready() -> dict:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready"}

    data_dir = Path(settings.data_dir)
    if data_dir.exists():
        nicegui_app.add_static_files("/data", str(data_dir))

    ui.run_with(
        fastapi_app,
        storage_secret=settings.secret_key,
        title=settings.app_name,
        dark=True,
        mount_path="/",
        show_welcome_message=False,
    )

    _asgi_app = fastapi_app
    return fastapi_app


# Discovered by hosts that import this module as a package path.
app = create_asgi_app()
