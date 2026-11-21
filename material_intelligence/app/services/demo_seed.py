"""Optional labeled demo-data seeder for dashboard development.

Demo records are always associated with site code DEMO-DATA and carry explicit
non-operational labeling. They must never be presented as real site data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sqlalchemy import select

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import (
    Inspection,
    MaterialBatch,
    Site,
    StockMovement,
    Supplier,
)
from material_intelligence.app.db.session import SessionLocal, init_db
from material_intelligence.app.domain.enums import InspectionStatus, StockMovementType
from material_intelligence.app.services.dashboard_analytics import DEMO_LABEL, DEMO_SITE_CODE
from material_intelligence.app.utils.logging import get_logger

logger = get_logger(__name__)


def _make_placeholder(path: Path, label: str, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (320, 240), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 300, 220), outline=(255, 255, 255), width=2)
    draw.text((30, 100), label[:40], fill=(255, 255, 255))
    img.save(path)


def seed_demo_data(*, force: bool = False) -> dict:
    """Insert explicitly labeled demo records. Idempotent unless force=True."""
    init_db()
    settings = get_settings()
    with SessionLocal() as db:
        site = db.scalar(select(Site).where(Site.code == DEMO_SITE_CODE))
        if site and not force:
            existing = db.scalar(
                select(Inspection).where(Inspection.site_id == site.id).limit(1)
            )
            if existing:
                logger.info("Demo data already present; skipping (use force=True to reseed).")
                return {"status": "exists", "site_id": site.id}

        if site is None:
            site = Site(
                name="[DEMO] Synthetic Construction Site",
                code=DEMO_SITE_CODE,
                location="Synthetic / Non-operational",
                description=DEMO_LABEL,
            )
            db.add(site)
            db.flush()
        else:
            site.description = DEMO_LABEL

        supplier = db.scalar(select(Supplier).where(Supplier.name == "[DEMO] Demo Supplier"))
        if supplier is None:
            supplier = Supplier(name="[DEMO] Demo Supplier", notes=DEMO_LABEL)
            db.add(supplier)
            db.flush()

        # Clear previous demo inspections/batches if force
        if force:
            for insp in db.scalars(select(Inspection).where(Inspection.site_id == site.id)).all():
                db.delete(insp)
            for batch in db.scalars(select(MaterialBatch).where(MaterialBatch.site_id == site.id)).all():
                for move in db.scalars(
                    select(StockMovement).where(StockMovement.batch_id == batch.id)
                ).all():
                    db.delete(move)
                db.delete(batch)
            db.flush()

        materials = [
            ("clay_brick", "cracked", 0.91),
            ("clay_brick", "no_visible_damage", 0.88),
            ("steel_rebar", "surface_corrosion", 0.76),
            ("steel_rebar", "extensive_visible_corrosion", 0.42),
            ("cement_bag", "torn_packaging", 0.83),
            ("cement_bag", "no_visible_damage", 0.67),
        ]
        statuses = [
            InspectionStatus.SCREENING_COMPLETED,
            InspectionStatus.HUMAN_INSPECTION_REQUIRED,
            InspectionStatus.ADDITIONAL_TESTING_REQUIRED,
            InspectionStatus.QUARANTINED,
            InspectionStatus.APPROVED,
            InspectionStatus.REJECTED,
        ]

        demo_dir = settings.upload_dir / "demo"
        demo_dir.mkdir(parents=True, exist_ok=True)
        created_inspections = 0
        created_batches = 0

        for idx, (material, condition, confidence) in enumerate(materials):
            batch = MaterialBatch(
                site_id=site.id,
                material_key=material,
                batch_code=f"DEMO-{material[:4].upper()}-{idx+1:02d}",
                supplier_id=supplier.id,
                quantity=100 + idx * 10,
                approved_quantity=20.0 if idx % 2 == 0 else 5.0,
                quarantined_quantity=15.0 if condition != "no_visible_damage" else 0.0,
                rejected_quantity=5.0 if "corrosion" in condition or "torn" in condition else 0.0,
                unit="units",
                storage_location=f"DEMO-YARD-{idx % 3 + 1}",
                purchase_reference=f"DEMO-PO-{idx+1}",
                quality_status=statuses[idx % len(statuses)].value,
            )
            db.add(batch)
            db.flush()
            created_batches += 1
            db.add(
                StockMovement(
                    batch_id=batch.id,
                    movement_type=StockMovementType.RECEIPT.value,
                    quantity=batch.quantity,
                    reference=batch.purchase_reference,
                    notes=DEMO_LABEL,
                )
            )

            for day_offset in range(3):
                image_path = demo_dir / f"demo_{material}_{idx}_{day_offset}.jpg"
                ann_path = demo_dir / f"demo_{material}_{idx}_{day_offset}_ann.jpg"
                color = tuple(int(c) for c in np.random.default_rng(idx + day_offset).integers(40, 180, size=3))
                _make_placeholder(image_path, f"DEMO {material}", color)
                _make_placeholder(ann_path, f"DEMO BOX {condition}", (20, 80, 120))

                status = statuses[(idx + day_offset) % len(statuses)]
                created = datetime.now(UTC) - timedelta(days=day_offset * 2 + idx)
                insp = Inspection(
                    site_id=site.id,
                    batch_id=batch.id,
                    image_path=str(image_path),
                    annotated_path=str(ann_path),
                    gradcam_path=None,
                    status=status.value,
                    detection_result={
                        "boxes": [
                            {
                                "x1": 40,
                                "y1": 40,
                                "x2": 200,
                                "y2": 180,
                                "class_name": material,
                                "confidence": confidence,
                            }
                        ],
                        "model_version": "demo-yolo",
                        "model_status": "ready",
                        "message": DEMO_LABEL,
                    },
                    classification_result={
                        "items": [
                            {
                                "material_key": material,
                                "condition": condition,
                                "confidence": confidence - day_offset * 0.05,
                                "model_version": "demo-efficientnet",
                                "model_status": "ready",
                                "message": DEMO_LABEL,
                            }
                        ],
                        "notes": [DEMO_LABEL],
                    },
                    processing_ms=120.0 + idx * 5 + day_offset,
                    model_versions={"yolo": "demo-yolo", "classifier": "demo-efficientnet"},
                    notes=DEMO_LABEL,
                    created_at=created,
                )
                db.add(insp)
                created_inspections += 1

        db.commit()
        logger.info(
            "Seeded demo data: site=%s batches=%s inspections=%s",
            DEMO_SITE_CODE,
            created_batches,
            created_inspections,
        )
        return {
            "status": "seeded",
            "site_code": DEMO_SITE_CODE,
            "batches": created_batches,
            "inspections": created_inspections,
            "label": DEMO_LABEL,
        }


def purge_demo_data() -> dict:
    """Delete all DEMO-DATA site records, demo supplier, and demo image files."""
    init_db()
    settings = get_settings()
    with SessionLocal() as db:
        site = db.scalar(select(Site).where(Site.code == DEMO_SITE_CODE))
        deleted = {"inspections": 0, "movements": 0, "batches": 0, "site": 0, "supplier": 0}

        if site is not None:
            inspections = db.scalars(select(Inspection).where(Inspection.site_id == site.id)).all()
            for insp in inspections:
                db.delete(insp)
            deleted["inspections"] = len(inspections)

            batches = db.scalars(select(MaterialBatch).where(MaterialBatch.site_id == site.id)).all()
            for batch in batches:
                moves = db.scalars(
                    select(StockMovement).where(StockMovement.batch_id == batch.id)
                ).all()
                for move in moves:
                    db.delete(move)
                deleted["movements"] += len(moves)
                db.delete(batch)
            deleted["batches"] = len(batches)

            db.delete(site)
            deleted["site"] = 1
            db.flush()

        supplier = db.scalar(select(Supplier).where(Supplier.name == "[DEMO] Demo Supplier"))
        if supplier is not None:
            # Only remove supplier if no remaining batches reference it.
            still_used = db.scalar(
                select(MaterialBatch.id).where(MaterialBatch.supplier_id == supplier.id).limit(1)
            )
            if still_used is None:
                db.delete(supplier)
                deleted["supplier"] = 1

        db.commit()

    demo_dir = settings.upload_dir / "demo"
    files_removed = 0
    if demo_dir.exists():
        for path in demo_dir.rglob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)
                files_removed += 1
        # Remove empty dirs bottom-up
        for path in sorted(demo_dir.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            demo_dir.rmdir()
        except OSError:
            pass

    logger.info("Purged demo data: %s files=%s", deleted, files_removed)
    return {"status": "purged", "deleted": deleted, "files_removed": files_removed}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Seed or purge explicitly labeled DEMO DATA for Material Intelligence."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate demo records for site DEMO-DATA.",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Permanently remove all DEMO-DATA records and demo image files.",
    )
    args = parser.parse_args()
    if args.purge:
        result = purge_demo_data()
    else:
        result = seed_demo_data(force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
