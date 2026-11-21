"""Startup / import smoke tests (no server bind)."""

from __future__ import annotations


def test_import_application_modules():
    from material_intelligence.app import config, main
    from material_intelligence.app.services import bootstrap
    from material_intelligence.app.ui.pages import dashboard, inspection, training

    assert config.get_settings().app_name
    assert callable(main.run)
    assert callable(bootstrap.bootstrap_application)
    assert callable(dashboard.register)
    assert callable(inspection.register)
    assert callable(training.register)


def test_bootstrap_creates_admin(tmp_data_dirs):
    from sqlalchemy import select

    from material_intelligence.app.db.models import MaterialType, User
    from material_intelligence.app.db.session import SessionLocal
    from material_intelligence.app.services.bootstrap import bootstrap_application

    bootstrap_application()
    with SessionLocal() as db:
        users = db.scalars(select(User)).all()
        materials = db.scalars(select(MaterialType)).all()
        assert any(u.role == "administrator" for u in users)
        assert len(materials) >= 3
