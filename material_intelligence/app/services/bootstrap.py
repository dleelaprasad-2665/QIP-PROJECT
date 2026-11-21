"""Database bootstrap: admin user, default materials, demo site."""

from __future__ import annotations

from sqlalchemy import select

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import DefectType, MaterialType, Site, User
from material_intelligence.app.db.session import SessionLocal, init_db
from material_intelligence.app.domain.enums import (
    DEFAULT_MATERIALS,
    DEFECT_DISPLAY_NAMES,
    HIGH_RISK_CONDITIONS,
    MATERIAL_DISPLAY_NAMES,
    UserRole,
)
from material_intelligence.app.security.authentication import hash_password
from material_intelligence.app.utils.logging import get_logger

logger = get_logger(__name__)


def bootstrap_application() -> None:
    init_db()
    settings = get_settings()
    with SessionLocal() as db:
        admin = db.scalar(select(User).where(User.email == settings.bootstrap_admin_email.lower()))
        if admin is None:
            db.add(
                User(
                    email=settings.bootstrap_admin_email.lower(),
                    full_name=settings.bootstrap_admin_name,
                    password_hash=hash_password(settings.bootstrap_admin_password),
                    role=UserRole.ADMINISTRATOR.value,
                    is_active=True,
                )
            )
            logger.info("Bootstrap administrator created: %s", settings.bootstrap_admin_email)

        for key, defects in DEFAULT_MATERIALS.items():
            material = db.scalar(select(MaterialType).where(MaterialType.key == key))
            if material is None:
                material = MaterialType(
                    key=key,
                    display_name=MATERIAL_DISPLAY_NAMES.get(key, key),
                    description=f"Default material class: {key}",
                )
                db.add(material)
                db.flush()
            for defect_key in defects:
                existing = db.scalar(
                    select(DefectType).where(
                        DefectType.material_id == material.id, DefectType.key == defect_key
                    )
                )
                if existing is None:
                    db.add(
                        DefectType(
                            material_id=material.id,
                            key=defect_key,
                            display_name=DEFECT_DISPLAY_NAMES.get(defect_key, defect_key),
                            is_high_risk=defect_key in HIGH_RISK_CONDITIONS,
                        )
                    )

        site = db.scalar(select(Site).where(Site.code == "DEMO-01"))
        if site is None:
            db.add(
                Site(
                    name="Demonstration Construction Site",
                    code="DEMO-01",
                    location="Local Development",
                    description="Default site for Material Intelligence demos.",
                )
            )
        db.commit()
