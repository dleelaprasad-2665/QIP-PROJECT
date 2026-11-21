"""Train condition classifiers from a large sorted site photo set (~1500 images).

Expected layout under data/datasets/site_train_v1:

  clay_brick/{good,damaged}
  steel_rebar/{good,damaged}
  cement_bag/{good,damaged}

Maps good → no_visible_damage and damaged → material-specific defect class.
"""

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DAMAGE_CLASS = {
    "clay_brick": "chipped",
    "steel_rebar": "surface_corrosion",
    "cement_bag": "torn_packaging",
}

MATERIALS = ("clay_brick", "steel_rebar", "cement_bag")


def _collect(src: Path) -> dict[str, dict[str, list[Path]]]:
    out: dict[str, dict[str, list[Path]]] = {}
    for material in MATERIALS:
        out[material] = {"good": [], "damaged": []}
        for label in ("good", "damaged"):
            folder = src / material / label
            if not folder.is_dir():
                continue
            files = [
                p
                for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]
            out[material][label] = files
    return out


def _stage_classification_set(
    collected: dict[str, dict[str, list[Path]]],
    stage_root: Path,
    material: str,
) -> int:
    """Copy into EfficientNet folders: no_visible_damage / <damage_class>."""
    if stage_root.exists():
        shutil.rmtree(stage_root)
    good_dir = stage_root / "no_visible_damage"
    bad_dir = stage_root / DAMAGE_CLASS[material]
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)
    count = 0
    for i, src in enumerate(collected[material]["good"]):
        shutil.copy2(src, good_dir / f"good_{i:05d}{src.suffix.lower()}")
        count += 1
    for i, src in enumerate(collected[material]["damaged"]):
        shutil.copy2(src, bad_dir / f"bad_{i:05d}{src.suffix.lower()}")
        count += 1
    return count


def main() -> None:
    init_db()
    settings = get_settings()
    root = settings.dataset_dir / "site_train_v1"
    if not root.exists():
        raise SystemExit(f"Missing folder: {root}\nSee README.txt inside site_train_v1.")

    collected = _collect(root)
    total = sum(len(v["good"]) + len(v["damaged"]) for v in collected.values())
    if total < 20:
        raise SystemExit(
            f"Only {total} images found under {root}.\n"
            "Sort photos into clay_brick|steel_rebar|cement_bag / good|damaged, then retry."
        )

    print(json.dumps({m: {k: len(v) for k, v in buckets.items()} for m, buckets in collected.items()}, indent=2))

    results = []
    with SessionLocal() as db:
        for material in MATERIALS:
            n_good = len(collected[material]["good"])
            n_bad = len(collected[material]["damaged"])
            if n_good < 5 or n_bad < 5:
                print(f"Skip {material}: need ≥5 good and ≥5 damaged (have {n_good}/{n_bad})")
                continue

            stage = settings.dataset_dir / f"_stage_{material}"
            n = _stage_classification_set(collected, stage, material)
            out = settings.model_dir / "site" / f"efficientnet_{material}.pt"
            out.parent.mkdir(parents=True, exist_ok=True)

            # Efficient training defaults for large sets
            epochs = 6 if n < 400 else 4
            batch = 16 if n >= 200 else 8
            img_size = 160

            clf = EfficientNetClassifier(
                device=settings.default_device,
                image_size=img_size,
                material_key=material,
            )
            metrics = clf.train_from_folder(
                stage,
                epochs=epochs,
                batch_size=batch,
                learning_rate=1e-3,
                image_size=img_size,
                seed=42,
                early_stopping_patience=2,
                output_path=out,
                version=f"site_{material}_v1",
                material_key=material,
            )

            ds = db.scalar(
                select(Dataset).where(Dataset.name == f"site_{material}", Dataset.version == "v1")
            )
            if ds is None:
                ds = Dataset(
                    name=f"site_{material}",
                    version="v1",
                    format=DatasetFormat.IMAGE_FOLDER_CLASSIFICATION.value,
                    status=DatasetStatus.READY_FOR_TRAINING.value,
                    root_path=str(root / material),
                    image_count=n,
                    annotation_count=n,
                    class_names=["no_visible_damage", DAMAGE_CLASS[material]],
                    source="site_train_v1 sorted photos",
                )
                db.add(ds)
                db.flush()
            else:
                ds.image_count = n
                ds.annotation_count = n

            # Deactivate older heads for this material only (by notes/path), keep others
            artifact = ModelArtifact(
                name=f"EfficientNet {material} site",
                version=f"site_{material}_v1",
                model_type=ModelType.EFFICIENTNET_B0.value,
                status=ModelStatus.READY.value,
                artifact_path=str(out),
                class_mapping=clf.class_mapping,
                preprocess_config={"image_size": img_size},
                metrics=metrics,
                dataset_id=ds.id,
                is_active_inference=False,
                notes=f"Trained on site photos for {material} ({n} images).",
            )
            db.add(artifact)
            db.commit()
            db.refresh(artifact)
            # Activate latest trained material head (last one wins as global active;
            # registry still picks by material_key match).
            set_active_model(db, artifact.id)
            results.append(
                {
                    "material": material,
                    "images": n,
                    "artifact_id": artifact.id,
                    "val_acc": metrics.get("final_val_accuracy"),
                    "path": str(out),
                }
            )
            print(json.dumps(results[-1], indent=2))

    if not results:
        raise SystemExit("No material had enough labeled photos to train.")
    print(json.dumps({"trained": results, "total_photos_seen": total}, indent=2))


if __name__ == "__main__":
    main()
