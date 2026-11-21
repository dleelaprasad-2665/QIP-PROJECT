"""Inventory quantity consistency tests."""

from __future__ import annotations

from material_intelligence.app.db.models import Site, User
from material_intelligence.app.domain.enums import InspectionStatus, UserRole
from material_intelligence.app.domain.schemas import InventoryBatchCreate
from material_intelligence.app.security.authentication import hash_password
from material_intelligence.app.services.inventory_service import apply_quality_decision, create_batch


def test_inventory_approval_updates_quantities(db_session):
    site = Site(name="Site", code="S1", location="X")
    user = User(
        email="qe@example.com",
        full_name="QE",
        password_hash=hash_password("password123"),
        role=UserRole.QUALITY_ENGINEER.value,
    )
    db_session.add_all([site, user])
    db_session.commit()

    batch = create_batch(
        db_session,
        InventoryBatchCreate(
            site_id=site.id,
            material_key="clay_brick",
            batch_code="B1",
            quantity=100,
        ),
        user,
    )
    assert batch.quantity == 100
    assert batch.approved_quantity == 0

    updated = apply_quality_decision(
        db_session,
        batch_id=batch.id,
        status=InspectionStatus.QUARANTINED,
        quantity=20,
        user=user,
    )
    assert updated.quarantined_quantity == 20

    updated = apply_quality_decision(
        db_session,
        batch_id=batch.id,
        status=InspectionStatus.APPROVED,
        quantity=10,
        user=user,
    )
    assert updated.approved_quantity == 10
    assert updated.quarantined_quantity == 10
