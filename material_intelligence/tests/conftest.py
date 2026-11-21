"""Shared fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Isolate tests from developer .env database.
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("BOOTSTRAP_ADMIN_PASSWORD", "TestPassword123!")

from material_intelligence.app.config import get_settings  # noqa: E402
from material_intelligence.app.db import models as _models  # noqa: E402, F401
from material_intelligence.app.db.session import Base  # noqa: E402


@pytest.fixture()
def tmp_data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    get_settings.cache_clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "data" / "uploads"))
    monkeypatch.setenv("DATASET_DIR", str(tmp_path / "data" / "datasets"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "data" / "models"))
    monkeypatch.setenv("REPORT_DIR", str(tmp_path / "data" / "reports"))
    monkeypatch.setenv("JOB_DIR", str(tmp_path / "data" / "jobs"))
    get_settings.cache_clear()
    settings = get_settings()
    settings.ensure_directories()
    yield settings
    get_settings.cache_clear()


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def sample_bgr() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
