"""Dataset import and validation service."""

from __future__ import annotations

import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import AuditEvent, Dataset
from material_intelligence.app.domain.enums import DatasetStatus
from material_intelligence.app.domain.schemas import DatasetImportResult
from material_intelligence.app.ml.dataset_loader import (
    import_archive_to_dataset_dir,
    list_yolo_label_preview,
    normalize_flat_image_archive,
    validate_dataset,
)
from material_intelligence.app.utils.files import save_upload


def import_zip_dataset(
    db: Session,
    *,
    zip_bytes: bytes,
    filename: str,
    name: str,
    version: str = "v1",
    source: str | None = None,
    annotator: str | None = None,
    user_id: int | None = None,
) -> DatasetImportResult:
    settings = get_settings()
    zip_path = save_upload(zip_bytes, settings.upload_dir / "archives", filename, image=False)
    dest = settings.dataset_dir / f"{name}_{version}".replace(" ", "_")
    root = import_archive_to_dataset_dir(zip_path, dest)
    root = normalize_flat_image_archive(root)
    report = validate_dataset(root)

    dataset = Dataset(
        name=name,
        version=version,
        format=report.format.value,
        status=report.status.value,
        root_path=str(root),
        image_count=report.image_count,
        annotation_count=report.annotation_count,
        class_names=report.class_names,
        source=source,
        annotator=annotator,
        validation_report=report.to_dict(),
        class_distribution=report.class_distribution,
        created_by=user_id,
    )
    db.add(dataset)
    db.add(
        AuditEvent(
            actor_id=user_id,
            action="dataset.import",
            entity_type="dataset",
            entity_id=None,
            details={"name": name, "version": version, "status": report.status.value},
        )
    )
    db.commit()
    db.refresh(dataset)
    return DatasetImportResult(
        dataset_id=dataset.id,
        name=dataset.name,
        format=report.format,
        status=report.status,
        image_count=report.image_count,
        annotation_count=report.annotation_count,
        class_names=report.class_names,
        warnings=report.warnings,
        errors=report.errors,
    )


def import_classification_images(
    db: Session,
    *,
    files: list[tuple[str, bytes]],
    class_name: str,
    dataset_name: str,
    version: str = "v1",
    user_id: int | None = None,
) -> DatasetImportResult:
    settings = get_settings()
    root = settings.dataset_dir / f"{dataset_name}_{version}".replace(" ", "_")
    class_dir = root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for original_name, data in files:
        save_upload(data, class_dir, original_name, image=True)

    report = validate_dataset(root)
    existing = db.scalar(
        select(Dataset).where(Dataset.name == dataset_name, Dataset.version == version)
    )
    if existing:
        existing.root_path = str(root)
        existing.image_count = report.image_count
        existing.annotation_count = report.annotation_count
        existing.class_names = report.class_names
        existing.class_distribution = report.class_distribution
        existing.validation_report = report.to_dict()
        existing.status = report.status.value
        existing.format = report.format.value
        dataset = existing
    else:
        dataset = Dataset(
            name=dataset_name,
            version=version,
            format=report.format.value,
            status=report.status.value,
            root_path=str(root),
            image_count=report.image_count,
            annotation_count=report.annotation_count,
            class_names=report.class_names,
            validation_report=report.to_dict(),
            class_distribution=report.class_distribution,
            created_by=user_id,
        )
        db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return DatasetImportResult(
        dataset_id=dataset.id,
        name=dataset.name,
        format=report.format,
        status=DatasetStatus(dataset.status),
        image_count=dataset.image_count,
        annotation_count=dataset.annotation_count,
        class_names=dataset.class_names or [],
        warnings=report.warnings,
        errors=report.errors,
    )


def get_dataset_preview(dataset: Dataset, limit: int = 12) -> list[dict]:
    root = Path(dataset.root_path)
    if not root.exists():
        return []
    return list_yolo_label_preview(root, limit=limit)


def export_dataset_copy(dataset: Dataset, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{dataset.name}_{dataset.version}"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(dataset.root_path, target)
    return target
