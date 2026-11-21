"""Fine-tune EfficientNet on real field rebar photos and activate for inference."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from sqlalchemy import select

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import Dataset, ModelArtifact
from material_intelligence.app.db.session import SessionLocal, init_db
from material_intelligence.app.domain.enums import (
    DatasetFormat,
    DatasetStatus,
    ModelStatus,
    ModelType,
)
from material_intelligence.app.ml.efficientnet_classifier import EfficientNetClassifier
from material_intelligence.app.ml.model_registry import set_active_model


def main() -> None:
    init_db()
    settings = get_settings()
    root = settings.dataset_dir / "field_rebar_v1"
    if not root.exists():
        raise SystemExit(f"Missing dataset: {root}")

    good_dir = root / "no_visible_damage"
    bad_dir = root / "surface_corrosion"
    good_files = [p for p in good_dir.glob("*") if p.is_file()]
    bad_files = [p for p in bad_dir.glob("*") if p.is_file()]
    if good_files and bad_files:
        target = max(len(bad_files) // 2, len(good_files) * 4)
        i = 0
        while len([p for p in good_dir.glob("*") if p.is_file()]) < target:
            src = good_files[i % len(good_files)]
            dst = good_dir / f"aug_{i}_{src.name}"
            if not dst.exists():
                shutil.copy2(src, dst)
            i += 1

    out = settings.model_dir / "field" / "efficientnet_rebar_field.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    clf = EfficientNetClassifier(
        device=settings.default_device,
        image_size=160,
        material_key="steel_rebar",
    )
    metrics = clf.train_from_folder(
        root,
        epochs=8,
        batch_size=8,
        learning_rate=1e-3,
        image_size=160,
        seed=42,
        early_stopping_patience=3,
        output_path=out,
        version="efficientnet_rebar_field_v1",
        material_key="steel_rebar",
    )

    with SessionLocal() as db:
        dataset = db.scalar(
            select(Dataset).where(Dataset.name == "field_rebar", Dataset.version == "v1")
        )
        if dataset is None:
            dataset = Dataset(
                name="field_rebar",
                version="v1",
                format=DatasetFormat.IMAGE_FOLDER_CLASSIFICATION.value,
                status=DatasetStatus.READY_FOR_TRAINING.value,
                root_path=str(root),
                image_count=sum(1 for _ in root.rglob("*.jpg")),
                annotation_count=sum(1 for _ in root.rglob("*.jpg")),
                class_names=["no_visible_damage", "surface_corrosion"],
                source="field GPS photos Eluru",
            )
            db.add(dataset)
            db.flush()
        # Deactivate older classifiers
        for art in db.scalars(
            select(ModelArtifact).where(ModelArtifact.model_type == ModelType.EFFICIENTNET_B0.value)
        ).all():
            art.is_active_inference = False
        artifact = ModelArtifact(
            name="EfficientNet Rebar Field",
            version="efficientnet_rebar_field_v1",
            model_type=ModelType.EFFICIENTNET_B0.value,
            status=ModelStatus.READY.value,
            artifact_path=str(out),
            class_mapping=clf.class_mapping,
            preprocess_config={"image_size": 160},
            metrics=metrics,
            dataset_id=dataset.id,
            is_active_inference=True,
            notes="Fine-tuned on real rebar site photos (Good vs surface rust).",
        )
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        set_active_model(db, artifact.id)
        print(
            json.dumps(
                {"artifact_id": artifact.id, "path": str(out), "metrics": metrics},
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
