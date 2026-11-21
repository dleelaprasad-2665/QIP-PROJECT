"""Material inspection workflow service."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import AuditEvent, Inspection
from material_intelligence.app.domain.enums import InspectionStatus
from material_intelligence.app.domain.schemas import InspectionPrediction
from material_intelligence.app.ml.inference_pipeline import run_inspection_pipeline, save_image
from material_intelligence.app.ml.preprocessing import load_bgr_image
from material_intelligence.app.security.authorization import require_permission
from material_intelligence.app.utils.files import save_upload


def create_inspection(
    db: Session,
    *,
    site_id: int,
    batch_id: int | None,
    image_bytes: bytes,
    filename: str,
    user,
    generate_gradcam: bool = True,
    expected_material_key: str | None = None,
) -> tuple[Inspection, InspectionPrediction]:
    require_permission(user, "inspect")
    settings = get_settings()
    image_path = save_upload(image_bytes, settings.upload_dir / "raw", filename, image=True)
    image_bgr = load_bgr_image(image_path)

    prediction, annotated, gradcam_path = run_inspection_pipeline(
        db,
        image_bgr,
        generate_gradcam=generate_gradcam,
        expected_material_key=expected_material_key,
        source_filename=filename,
    )
    annotated_path = settings.upload_dir / "annotated" / f"insp_{image_path.stem}.jpg"
    save_image(annotated_path, annotated)

    summary = prediction.notes[0] if prediction.notes else None
    inspection = Inspection(
        site_id=site_id,
        batch_id=batch_id,
        image_path=str(image_path),
        annotated_path=str(annotated_path),
        gradcam_path=str(gradcam_path) if gradcam_path else None,
        status=prediction.recommended_status.value,
        detection_result=prediction.detection.model_dump(),
        classification_result={
            "items": [c.model_dump() for c in prediction.classifications],
            "notes": prediction.notes,
            "summary": summary,
        },
        processing_ms=prediction.processing_ms,
        model_versions={
            "yolo": prediction.detection.model_version,
            "classifier": next(
                (c.model_version for c in prediction.classifications if c.model_version), None
            ),
        },
        created_by=user.id if user else None,
    )
    db.add(inspection)
    db.add(
        AuditEvent(
            actor_id=user.id if user else None,
            action="inspection.created",
            entity_type="inspection",
            entity_id=None,
            details={"site_id": site_id, "status": inspection.status},
        )
    )
    db.commit()
    db.refresh(inspection)
    return inspection, prediction


def review_inspection(
    db: Session,
    *,
    inspection_id: int,
    new_status: InspectionStatus,
    user,
    notes: str | None = None,
) -> Inspection:
    require_permission(user, "approve")
    inspection = db.get(Inspection, inspection_id)
    if inspection is None:
        raise ValueError("Inspection not found.")
    allowed = {
        InspectionStatus.APPROVED,
        InspectionStatus.REJECTED,
        InspectionStatus.QUARANTINED,
        InspectionStatus.ADDITIONAL_TESTING_REQUIRED,
        InspectionStatus.HUMAN_INSPECTION_REQUIRED,
    }
    if new_status not in allowed:
        raise ValueError(f"Invalid review status: {new_status}")

    inspection.status = new_status.value
    inspection.reviewed_by = user.id
    inspection.reviewed_at = datetime.now(UTC)
    if notes:
        inspection.notes = notes
    db.add(
        AuditEvent(
            actor_id=user.id,
            action="inspection.reviewed",
            entity_type="inspection",
            entity_id=str(inspection.id),
            details={"status": new_status.value, "notes": notes},
        )
    )
    db.commit()
    db.refresh(inspection)
    return inspection


def load_inspection_image(path: str) -> bytes | None:
    p = Path(path)
    if not p.exists():
        return None
    return p.read_bytes()
