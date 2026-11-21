"""Train and activate starter construction-material models for local inference.

Creates a visually distinct labeled dataset, trains EfficientNet-B0 for condition
classification, optionally trains a small YOLOv8n detector, then registers both
as active inference models.

This produces real trained checkpoints — not fabricated metrics.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
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
from material_intelligence.app.ml.yolo_detector import YOLODetector
from material_intelligence.app.utils.logging import get_logger

logger = get_logger(__name__)

# Unified condition classes used for starter classifier (easy end-to-end demos).
CONDITION_CLASSES = [
    "no_visible_damage",
    "cracked",
    "chipped",
    "broken",
    "surface_corrosion",
    "torn_packaging",
]

MATERIAL_DETECT_CLASSES = ["clay_brick", "steel_rebar", "cement_bag"]


def _draw_condition(condition: str, size: int = 224, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    # Base material-like textures with class-specific cues the network can learn.
    if condition == "no_visible_damage":
        base = (180, 120, 90)  # clean brick-ish
        img = Image.new("RGB", (size, size), base)
        draw = ImageDraw.Draw(img)
        for y in range(0, size, 28):
            draw.line((0, y, size, y), fill=(150, 95, 70), width=2)
        for x in range(0, size, 56):
            draw.line((x, 0, x, size), fill=(150, 95, 70), width=2)
    elif condition == "cracked":
        base = (170, 110, 85)
        img = Image.new("RGB", (size, size), base)
        draw = ImageDraw.Draw(img)
        for _ in range(4):
            x0, y0 = int(rng.integers(10, size // 3)), int(rng.integers(10, size - 10))
            x1, y1 = int(rng.integers(size // 2, size - 10)), int(rng.integers(10, size - 10))
            draw.line((x0, y0, x1, y1), fill=(30, 20, 15), width=3)
    elif condition == "chipped":
        base = (175, 115, 88)
        img = Image.new("RGB", (size, size), base)
        draw = ImageDraw.Draw(img)
        for _ in range(5):
            x, y = int(rng.integers(20, size - 40)), int(rng.integers(20, size - 40))
            draw.ellipse((x, y, x + 28, y + 22), fill=(210, 200, 190))
    elif condition == "broken":
        base = (140, 90, 70)
        img = Image.new("RGB", (size, size), base)
        draw = ImageDraw.Draw(img)
        draw.polygon([(20, 20), (size - 20, 40), (size // 2, size - 20)], fill=(90, 60, 45))
        draw.line((30, 30, size - 30, size - 30), fill=(20, 10, 5), width=5)
    elif condition == "surface_corrosion":
        base = (120, 125, 130)
        img = Image.new("RGB", (size, size), base)
        draw = ImageDraw.Draw(img)
        for _ in range(40):
            x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
            r = int(rng.integers(3, 12))
            draw.ellipse((x, y, x + r, y + r), fill=(160, 70, 30))
    else:  # torn_packaging
        base = (230, 220, 200)
        img = Image.new("RGB", (size, size), base)
        draw = ImageDraw.Draw(img)
        draw.rectangle((30, 30, size - 30, size - 30), outline=(40, 40, 40), width=4)
        draw.line((40, 50, size // 2, size - 40), fill=(20, 20, 20), width=4)
        draw.polygon([(size // 2, 60), (size - 40, 80), (size - 50, 140)], fill=(250, 250, 245))

    # Mild noise / blur variation
    if seed % 3 == 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    arr = np.array(img).astype(np.int16)
    noise = rng.integers(-8, 9, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_classification_dataset(root: Path, images_per_class: int = 24) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for condition in CONDITION_CLASSES:
        class_dir = root / condition
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            img = _draw_condition(condition, size=224, seed=hash((condition, i)) % 10_000)
            img.save(class_dir / f"{condition}_{i:03d}.jpg", quality=92)
    return root


def build_yolo_dataset(root: Path, images_per_class: int = 16) -> Path:
    """Minimal YOLO detection dataset: material objects with bounding boxes."""
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    colors = {
        "clay_brick": (180, 110, 80),
        "steel_rebar": (100, 110, 120),
        "cement_bag": (220, 210, 190),
    }
    idx = 0
    for class_id, material in enumerate(MATERIAL_DETECT_CLASSES):
        for i in range(images_per_class):
            canvas = Image.new("RGB", (640, 640), (35, 40, 48))
            draw = ImageDraw.Draw(canvas)
            rng = random.Random(class_id * 100 + i)
            w = rng.randint(160, 280)
            h = rng.randint(120, 240)
            x1 = rng.randint(40, 640 - w - 40)
            y1 = rng.randint(40, 640 - h - 40)
            x2, y2 = x1 + w, y1 + h
            draw.rectangle((x1, y1, x2, y2), fill=colors[material], outline=(255, 255, 255), width=2)
            if material == "steel_rebar":
                for yy in range(y1 + 10, y2 - 10, 18):
                    draw.line((x1 + 8, yy, x2 - 8, yy), fill=(70, 75, 80), width=3)
            elif material == "cement_bag":
                draw.text((x1 + 20, y1 + 20), "CEMENT", fill=(40, 40, 40))

            name = f"{material}_{i:03d}"
            canvas.save(images / f"{name}.jpg", quality=90)
            # YOLO normalized cx, cy, w, h
            cx = ((x1 + x2) / 2) / 640
            cy = ((y1 + y2) / 2) / 640
            bw = w / 640
            bh = h / 640
            (labels / f"{name}.txt").write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
            idx += 1

    # Simple split files for ultralytics
    all_imgs = sorted(images.glob("*.jpg"))
    split = max(1, int(len(all_imgs) * 0.8))
    train_list = all_imgs[:split]
    val_list = all_imgs[split:] or all_imgs[-1:]
    (root / "train.txt").write_text("\n".join(str(p.resolve()) for p in train_list) + "\n", encoding="utf-8")
    (root / "val.txt").write_text("\n".join(str(p.resolve()) for p in val_list) + "\n", encoding="utf-8")
    yaml_text = (
        f"path: {root.resolve().as_posix()}\n"
        f"train: train.txt\n"
        f"val: val.txt\n"
        f"names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(MATERIAL_DETECT_CLASSES))
    )
    (root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    return root


def _register_dataset(db, *, name: str, version: str, root: Path, fmt: DatasetFormat, classes: list[str]) -> Dataset:
    existing = db.scalar(select(Dataset).where(Dataset.name == name, Dataset.version == version))
    if existing:
        existing.root_path = str(root)
        existing.status = DatasetStatus.READY_FOR_TRAINING.value
        existing.format = fmt.value
        existing.class_names = classes
        existing.image_count = sum(1 for _ in root.rglob("*.jpg"))
        db.commit()
        db.refresh(existing)
        return existing
    ds = Dataset(
        name=name,
        version=version,
        format=fmt.value,
        status=DatasetStatus.READY_FOR_TRAINING.value,
        root_path=str(root),
        image_count=sum(1 for _ in root.rglob("*.jpg")),
        annotation_count=sum(1 for _ in root.rglob("*.jpg")),
        class_names=classes,
        source="starter_synthetic_labeled",
        annotator="train_starter_models",
        validation_report={"note": "Synthetic starter dataset for local CPU training"},
        class_distribution={c: 1 for c in classes},
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


def _register_artifact(
    db,
    *,
    name: str,
    version: str,
    model_type: ModelType,
    path: Path,
    class_mapping: dict,
    metrics: dict,
    dataset_id: int | None,
    preprocess_config: dict | None,
) -> ModelArtifact:
    artifact = ModelArtifact(
        name=name,
        version=version,
        model_type=model_type.value,
        status=ModelStatus.READY.value,
        artifact_path=str(path),
        class_mapping=class_mapping,
        preprocess_config=preprocess_config,
        metrics=metrics,
        dataset_id=dataset_id,
        is_active_inference=False,
        notes="Starter model trained for local inspection demos",
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    set_active_model(db, artifact.id)
    return artifact


def train_and_activate(*, epochs_cls: int = 4, epochs_yolo: int = 3, skip_yolo: bool = False) -> dict:
    init_db()
    settings = get_settings()
    cls_root = settings.dataset_dir / "starter_conditions_v1"
    yolo_root = settings.dataset_dir / "starter_detection_v1"
    model_dir = settings.model_dir / "starter"

    logger.info("Building classification dataset at %s", cls_root)
    build_classification_dataset(cls_root, images_per_class=20)

    clf = EfficientNetClassifier(device=settings.default_device, image_size=160)
    clf_path = model_dir / "efficientnet_b0_starter.pt"
    logger.info("Training EfficientNet-B0 (%s epochs)...", epochs_cls)
    metrics = clf.train_from_folder(
        cls_root,
        epochs=epochs_cls,
        batch_size=4,
        learning_rate=1e-3,
        image_size=160,
        seed=42,
        early_stopping_patience=3,
        output_path=clf_path,
        version="efficientnet_b0_starter_v1",
        material_key=None,  # accept any uploaded material for condition screening
        val_ratio=0.2,
    )
    logger.info("Classifier trained. val_acc=%s", metrics.get("final_val_accuracy"))

    yolo_result = None
    if not skip_yolo:
        logger.info("Building YOLO dataset at %s", yolo_root)
        build_yolo_dataset(yolo_root, images_per_class=12)
        detector = YOLODetector(variant=settings.yolo_variant, device=settings.default_device)
        logger.info("Training YOLOv8n (%s epochs)...", epochs_yolo)
        yolo_result = detector.train(
            yolo_root / "data.yaml",
            epochs=epochs_yolo,
            imgsz=320,
            batch=4,
            lr0=1e-3,
            project=model_dir / "yolo_runs",
            name="starter_detect",
            device=settings.default_device,
        )
        logger.info("YOLO training finished: %s", yolo_result.get("artifact_path"))

    with SessionLocal() as db:
        cls_ds = _register_dataset(
            db,
            name="starter_conditions",
            version="v1",
            root=cls_root,
            fmt=DatasetFormat.IMAGE_FOLDER_CLASSIFICATION,
            classes=CONDITION_CLASSES,
        )
        clf_art = _register_artifact(
            db,
            name="EfficientNet-B0 Starter",
            version="efficientnet_b0_starter_v1",
            model_type=ModelType.EFFICIENTNET_B0,
            path=clf_path,
            class_mapping=clf.class_mapping,
            metrics=metrics,
            dataset_id=cls_ds.id,
            preprocess_config=clf.preprocess_config,
        )
        out = {
            "classifier": {
                "artifact_id": clf_art.id,
                "path": str(clf_path),
                "classes": clf.class_mapping,
                "val_accuracy": metrics.get("final_val_accuracy"),
                "active": True,
            }
        }
        if yolo_result and yolo_result.get("artifact_path"):
            yolo_ds = _register_dataset(
                db,
                name="starter_detection",
                version="v1",
                root=yolo_root,
                fmt=DatasetFormat.YOLO_DETECTION,
                classes=MATERIAL_DETECT_CLASSES,
            )
            yolo_path = Path(yolo_result["artifact_path"])
            yolo_art = _register_artifact(
                db,
                name="YOLOv8n Starter",
                version="yolov8n_starter_v1",
                model_type=ModelType.YOLOV8,
                path=yolo_path,
                class_mapping={name: i for i, name in enumerate(MATERIAL_DETECT_CLASSES)},
                metrics=yolo_result.get("metrics") or {},
                dataset_id=yolo_ds.id,
                preprocess_config={"imgsz": 320},
            )
            out["detector"] = {
                "artifact_id": yolo_art.id,
                "path": str(yolo_path),
                "active": True,
            }
        else:
            out["detector"] = {"active": False, "message": "YOLO skipped or failed"}
    return out


def smoke_predict(image_path: Path | None = None) -> dict:
    """Quick inference check using the active classifier."""
    settings = get_settings()
    with SessionLocal() as db:
        clf = __import__(
            "material_intelligence.app.ml.model_registry", fromlist=["load_active_classifier"]
        ).load_active_classifier(db)
        detector = __import__(
            "material_intelligence.app.ml.model_registry", fromlist=["load_active_yolo"]
        ).load_active_yolo(db)

    if image_path is None:
        # Generate a cracked sample
        sample = settings.upload_dir / "smoke_cracked.jpg"
        _draw_condition("cracked", seed=7).save(sample)
        image_path = sample
    bgr = cv2.imread(str(image_path))
    det = detector.predict(bgr)
    clf_result = clf.predict(bgr)
    return {
        "image": str(image_path),
        "detection_status": det.model_status.value,
        "boxes": [b.model_dump() for b in det.boxes],
        "condition": clf_result.condition,
        "confidence": clf_result.confidence,
        "classifier_status": clf_result.model_status.value,
        "probabilities": clf_result.class_probabilities,
        "message": clf_result.message,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train starter MI models for local demos")
    parser.add_argument("--epochs-cls", type=int, default=4)
    parser.add_argument("--epochs-yolo", type=int, default=3)
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    if args.predict_only:
        print(json.dumps(smoke_predict(), indent=2))
        return
    result = train_and_activate(
        epochs_cls=args.epochs_cls,
        epochs_yolo=args.epochs_yolo,
        skip_yolo=args.skip_yolo,
    )
    print(json.dumps(result, indent=2))
    print(json.dumps(smoke_predict(), indent=2))


if __name__ == "__main__":
    main()
