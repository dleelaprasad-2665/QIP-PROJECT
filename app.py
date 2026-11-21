"""
FastAPI entrypoint for cloud / ASGI hosts.

Run locally either:
  python -m material_intelligence.app.main
  uvicorn app:app --host 0.0.0.0 --port 8080
"""

from material_intelligence.app.asgi import app

__all__ = ["app"]
