"""Alternate FastAPI entrypoint (some hosts look for main:app)."""

from material_intelligence.app.asgi import app

__all__ = ["app"]
