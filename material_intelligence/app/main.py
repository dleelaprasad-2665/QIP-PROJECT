"""Application entrypoint — NiceGUI with health endpoints."""

from __future__ import annotations

from pathlib import Path

from nicegui import app as nicegui_app
from nicegui import ui
from sqlalchemy import text

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.session import engine
from material_intelligence.app.jobs.worker import start_worker_on_startup
from material_intelligence.app.services.bootstrap import bootstrap_application
from material_intelligence.app.ui.pages import (
    administration,
    clustering_analysis,
    dashboard,
    datasets,
    evaluation,
    experiment_comparison,
    explainability,
    inspection,
    inventory,
    login,
    preprocessing,
    quality_review,
    reports,
    testing_laboratory,
    training,
)
from material_intelligence.app.utils.logging import setup_logging


def register_pages() -> None:
    login.register()
    dashboard.register()
    inspection.register()
    datasets.register()
    preprocessing.register()
    training.register()
    evaluation.register()
    clustering_analysis.register()
    testing_laboratory.register()
    experiment_comparison.register()
    explainability.register()
    inventory.register()
    quality_review.register()
    reports.register()
    administration.register()


def _register_health_routes() -> None:
    @nicegui_app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "service": get_settings().app_name}

    @nicegui_app.get("/api/ready")
    def ready() -> dict:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready"}


def run() -> None:
    """Console script entrypoint."""
    setup_logging()
    settings = get_settings()
    bootstrap_application()
    start_worker_on_startup()
    register_pages()
    _register_health_routes()

    data_dir = Path(settings.data_dir)
    if data_dir.exists():
        nicegui_app.add_static_files("/data", str(data_dir))

    ui.run(
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        storage_secret=settings.secret_key,
        title=settings.app_name,
        dark=True,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
