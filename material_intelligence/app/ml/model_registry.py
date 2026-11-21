"""Registry for locating active inference model artifacts."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import ModelArtifact
from material_intelligence.app.domain.enums import ModelStatus, ModelType
from material_intelligence.app.ml.efficientnet_classifier import EfficientNetClassifier
from material_intelligence.app.ml.yolo_detector import YOLODetector


def get_active_artifact(db: Session, model_type: ModelType) -> ModelArtifact | None:
    return db.scalar(
        select(ModelArtifact)
        .where(
            ModelArtifact.model_type == model_type.value,
            ModelArtifact.is_active_inference.is_(True),
            ModelArtifact.status == ModelStatus.READY.value,
        )
        .order_by(ModelArtifact.updated_at.desc())
    )


def _starter_yolo_path(settings) -> Path | None:
    candidates = [
        settings.model_dir / "starter" / "yolo_runs" / "starter_detect" / "weights" / "best.pt",
        settings.model_dir / "starter" / "yolov8n_starter.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _starter_classifier_path(settings) -> Path | None:
    candidates = [
        settings.model_dir / "starter" / "efficientnet_b0_starter.pt",
        settings.model_dir / "efficientnet_b0" / "exp_cpu_001" / "exp_cpu_001.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_active_yolo(db: Session) -> YOLODetector:
    settings = get_settings()
    artifact = get_active_artifact(db, ModelType.YOLOV8)
    if artifact and artifact.artifact_path and Path(artifact.artifact_path).exists():
        return YOLODetector(
            artifact.artifact_path,
            variant=settings.yolo_variant,
            class_names=list((artifact.class_mapping or {}).keys()),
            conf_threshold=0.35,
            device=settings.default_device,
        )
    starter = _starter_yolo_path(settings)
    if starter is not None:
        return YOLODetector(
            starter,
            variant=settings.yolo_variant,
            conf_threshold=0.35,
            device=settings.default_device,
        )
    return YOLODetector(variant=settings.yolo_variant, device=settings.default_device)


def load_active_classifier(db: Session, material_key: str | None = None) -> EfficientNetClassifier:
    settings = get_settings()
    # Prefer a classifier whose material_key matches (e.g. field rebar), else any active,
    # else material-agnostic starter.
    artifacts = db.scalars(
        select(ModelArtifact)
        .where(
            ModelArtifact.model_type == ModelType.EFFICIENTNET_B0.value,
            ModelArtifact.status == ModelStatus.READY.value,
        )
        .order_by(ModelArtifact.is_active_inference.desc(), ModelArtifact.updated_at.desc())
    ).all()

    chosen = None
    if material_key:
        for art in artifacts:
            meta_path = Path(art.artifact_path or "")
            if not meta_path.exists():
                continue
            # Peek material_key from sidecar JSON when present
            side = meta_path.with_suffix(".json")
            mk = None
            if side.exists():
                try:
                    import json

                    mk = json.loads(side.read_text(encoding="utf-8")).get("material_key")
                except Exception:  # noqa: BLE001
                    mk = None
            if mk == material_key:
                chosen = art
                break
    if chosen is None:
        for art in artifacts:
            if art.is_active_inference and art.artifact_path and Path(art.artifact_path).exists():
                # Skip material-specific heads when scanning a different material
                side = Path(art.artifact_path).with_suffix(".json")
                mk = None
                if side.exists():
                    try:
                        import json

                        mk = json.loads(side.read_text(encoding="utf-8")).get("material_key")
                    except Exception:  # noqa: BLE001
                        mk = None
                if material_key and mk and mk != material_key:
                    continue
                chosen = art
                break
    if chosen is None:
        for art in artifacts:
            if art.artifact_path and Path(art.artifact_path).exists():
                side = Path(art.artifact_path).with_suffix(".json")
                mk = None
                if side.exists():
                    try:
                        import json

                        mk = json.loads(side.read_text(encoding="utf-8")).get("material_key")
                    except Exception:  # noqa: BLE001
                        mk = None
                if mk is None:
                    chosen = art
                    break

    if chosen and chosen.artifact_path and Path(chosen.artifact_path).exists():
        return EfficientNetClassifier(
            chosen.artifact_path,
            device=settings.default_device,
            image_size=settings.efficientnet_image_size,
        )
    starter = _starter_classifier_path(settings)
    if starter is not None:
        return EfficientNetClassifier(
            starter,
            device=settings.default_device,
            image_size=settings.efficientnet_image_size,
        )
    return EfficientNetClassifier(device=settings.default_device)


def set_active_model(db: Session, artifact_id: int) -> ModelArtifact:
    artifact = db.get(ModelArtifact, artifact_id)
    if artifact is None:
        raise ValueError("Model artifact not found.")
    siblings = db.scalars(
        select(ModelArtifact).where(ModelArtifact.model_type == artifact.model_type)
    ).all()
    for sibling in siblings:
        sibling.is_active_inference = sibling.id == artifact.id
    db.commit()
    db.refresh(artifact)
    return artifact
