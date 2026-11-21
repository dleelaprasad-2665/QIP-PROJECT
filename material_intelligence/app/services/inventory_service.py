"""Inventory and batch management with transactional stock updates."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from material_intelligence.app.db.models import (
    AuditEvent,
    MaterialBatch,
    StockMovement,
    Supplier,
)
from material_intelligence.app.domain.enums import InspectionStatus, StockMovementType
from material_intelligence.app.domain.schemas import InventoryBatchCreate, InventorySummary
from material_intelligence.app.security.authorization import require_permission


def get_or_create_supplier(db: Session, name: str | None) -> Supplier | None:
    if not name:
        return None
    supplier = db.scalar(select(Supplier).where(Supplier.name == name))
    if supplier is None:
        supplier = Supplier(name=name)
        db.add(supplier)
        db.flush()
    return supplier


def create_batch(db: Session, payload: InventoryBatchCreate, user) -> MaterialBatch:
    require_permission(user, "manage_inventory")
    supplier = get_or_create_supplier(db, payload.supplier_name)
    batch = MaterialBatch(
        site_id=payload.site_id,
        material_key=payload.material_key,
        batch_code=payload.batch_code,
        supplier_id=supplier.id if supplier else None,
        quantity=payload.quantity,
        approved_quantity=0.0,
        quarantined_quantity=0.0,
        rejected_quantity=0.0,
        unit=payload.unit,
        storage_location=payload.storage_location,
        purchase_reference=payload.purchase_reference,
        quality_status=InspectionStatus.HUMAN_INSPECTION_REQUIRED.value,
    )
    db.add(batch)
    db.flush()
    db.add(
        StockMovement(
            batch_id=batch.id,
            movement_type=StockMovementType.RECEIPT.value,
            quantity=payload.quantity,
            reference=payload.purchase_reference,
            notes="Initial receipt",
            actor_id=user.id if user else None,
        )
    )
    db.add(
        AuditEvent(
            actor_id=user.id if user else None,
            action="inventory.batch_created",
            entity_type="material_batch",
            entity_id=None,
            details=payload.model_dump(),
        )
    )
    db.commit()
    db.refresh(batch)
    return batch


def apply_quality_decision(
    db: Session,
    *,
    batch_id: int,
    status: InspectionStatus,
    quantity: float,
    user,
    notes: str | None = None,
) -> MaterialBatch:
    """AI predictions must not call this directly — only authorized review workflows."""
    require_permission(user, "approve")
    batch = db.get(MaterialBatch, batch_id)
    if batch is None:
        raise ValueError("Batch not found.")
    if quantity <= 0:
        raise ValueError("Quantity must be positive.")
    if quantity > batch.quantity - batch.approved_quantity - batch.rejected_quantity:
        # Allow quarantine of remaining unallocated stock.
        remaining = batch.quantity - batch.approved_quantity - batch.rejected_quantity
        if quantity > remaining + batch.quarantined_quantity:
            raise ValueError("Quantity exceeds available batch stock.")

    if status == InspectionStatus.APPROVED:
        batch.approved_quantity += quantity
        batch.quarantined_quantity = max(0.0, batch.quarantined_quantity - quantity)
        movement = StockMovementType.RELEASE
    elif status == InspectionStatus.REJECTED:
        batch.rejected_quantity += quantity
        batch.quarantined_quantity = max(0.0, batch.quarantined_quantity - quantity)
        movement = StockMovementType.REJECT
    elif status == InspectionStatus.QUARANTINED:
        batch.quarantined_quantity += quantity
        movement = StockMovementType.QUARANTINE
    else:
        raise ValueError("Unsupported inventory quality status.")

    batch.quality_status = status.value
    db.add(
        StockMovement(
            batch_id=batch.id,
            movement_type=movement.value,
            quantity=quantity,
            notes=notes,
            actor_id=user.id,
        )
    )
    db.add(
        AuditEvent(
            actor_id=user.id,
            action="inventory.quality_decision",
            entity_type="material_batch",
            entity_id=str(batch.id),
            details={"status": status.value, "quantity": quantity, "notes": notes},
        )
    )
    db.commit()
    db.refresh(batch)
    return batch


def site_inventory_summary(db: Session, site_id: int) -> list[InventorySummary]:
    batches = db.scalars(select(MaterialBatch).where(MaterialBatch.site_id == site_id)).all()
    grouped: dict[str, InventorySummary] = {}
    for batch in batches:
        summary = grouped.get(batch.material_key)
        pending = 1 if batch.quality_status == InspectionStatus.HUMAN_INSPECTION_REQUIRED.value else 0
        if summary is None:
            grouped[batch.material_key] = InventorySummary(
                site_id=site_id,
                material_key=batch.material_key,
                total_quantity=batch.quantity,
                approved_quantity=batch.approved_quantity,
                quarantined_quantity=batch.quarantined_quantity,
                rejected_quantity=batch.rejected_quantity,
                pending_review=pending,
            )
        else:
            summary.total_quantity += batch.quantity
            summary.approved_quantity += batch.approved_quantity
            summary.quarantined_quantity += batch.quarantined_quantity
            summary.rejected_quantity += batch.rejected_quantity
            summary.pending_review += pending
    return list(grouped.values())
