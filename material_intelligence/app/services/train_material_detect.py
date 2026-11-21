"""Efficient YOLO retrain with clearer brick / bag / rebar shapes."""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
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
from material_intelligence.app.ml.model_registry import set_active_model
from material_intelligence.app.ml.yolo_detector import YOLODetector

MATERIALS = ["clay_brick", "steel_rebar", "cement_bag"]


def _draw_brick(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, rng: random.Random) -> None:
    draw.rectangle((x1, y1, x2, y2), fill=(170, 95, 65), outline=(90, 50, 35), width=2)
    # Mortar lines
    for yy in range(y1 + 12, y2 - 8, 16):
        draw.line((x1 + 2, yy, x2 - 2, yy), fill=(200, 180, 160), width=2)
    for xx in range(x1 + 20, x2 - 10, 40):
        draw.line((xx, y1 + 2, xx, y2 - 2), fill=(200, 180, 160), width=1)
    # Occasional chip on edge
    if rng.random() < 0.4:
        draw.polygon(
            [(x1, y1), (x1 + 18, y1), (x1, y1 + 14)],
            fill=(40, 40, 40),
        )


def _draw_rebar(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, rng: random.Random) -> None:
    draw.rectangle((x1, y1, x2, y2), fill=(110, 115, 120), outline=(60, 60, 65), width=1)
    for yy in range(y1 + 6, y2 - 6, 10):
        draw.line((x1 + 4, yy, x2 - 4, yy), fill=(70, 75, 80), width=2)
        if rng.random() < 0.5:
            draw.ellipse((x1 + 8, yy - 3, x1 + 18, yy + 3), fill=(150, 70, 30))


def _draw_bag(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, rng: random.Random) -> None:
    draw.rectangle((x1, y1, x2, y2), fill=(235, 230, 215), outline=(50, 50, 50), width=3)
    # Label panel
    mx, my = x1 + 20, y1 + 25
    draw.rectangle((mx, my, x2 - 20, y1 + 90), outline=(30, 30, 30), width=2)
    draw.text((mx + 10, my + 10), "CEMENT", fill=(20, 20, 20))
    draw.text((mx + 10, my + 35), "50 kg", fill=(60, 60, 60))
    if rng.random() < 0.35:
        # Tear
        draw.line((x1 + 10, y2 - 40, x2 - 30, y2 - 10), fill=(15, 15, 15), width=4)


def build_dataset(root: Path, per_class: int = 40) -> Path:
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for p in list(images.glob("*.jpg")) + list(labels.glob("*.txt")):
        p.unlink()

    drawers = {
        "clay_brick": _draw_brick,
        "steel_rebar": _draw_rebar,
        "cement_bag": _draw_bag,
    }
    for class_id, material in enumerate(MATERIALS):
        for i in range(per_class):
            rng = random.Random(class_id * 1000 + i)
            canvas = Image.new("RGB", (640, 640), (rng.randint(30, 55), rng.randint(35, 60), rng.randint(40, 65)))
            draw = ImageDraw.Draw(canvas)
            # Dirt speckles (background — model must not treat as bag)
            for _ in range(80):
                sx, sy = rng.randint(0, 630), rng.randint(0, 630)
                canvas.putpixel((sx, sy), (rng.randint(40, 90), rng.randint(40, 80), rng.randint(30, 70)))

            w = rng.randint(140, 300)
            h = rng.randint(100, 260)
            x1 = rng.randint(30, 640 - w - 30)
            y1 = rng.randint(30, 640 - h - 30)
            x2, y2 = x1 + w, y1 + h
            drawers[material](draw, x1, y1, x2, y2, rng)

            name = f"{material}_{i:03d}"
            canvas.save(images / f"{name}.jpg", quality=90)
            cx = ((x1 + x2) / 2) / 640
            cy = ((y1 + y2) / 2) / 640
            (labels / f"{name}.txt").write_text(
                f"{class_id} {cx:.6f} {cy:.6f} {w / 640:.6f} {h / 640:.6f}\n",
                encoding="utf-8",
            )

    all_imgs = list(images.glob("*.jpg"))
    random.Random(42).shuffle(all_imgs)
    # Stratified-ish: keep some of each class in val
    by_class: dict[str, list[Path]] = {m: [] for m in MATERIALS}
    for p in all_imgs:
        for m in MATERIALS:
            if p.name.startswith(m):
                by_class[m].append(p)
                break
    train_list: list[Path] = []
    val_list: list[Path] = []
    for m, items in by_class.items():
        n_val = max(2, int(len(items) * 0.15))
        val_list.extend(items[:n_val])
        train_list.extend(items[n_val:])
    (root / "train.txt").write_text("\n".join(str(p.resolve()) for p in train_list) + "\n", encoding="utf-8")
    (root / "val.txt").write_text("\n".join(str(p.resolve()) for p in val_list) + "\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        f"path: {root.resolve().as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(MATERIALS)),
        encoding="utf-8",
    )
    return root


def main() -> None:
    init_db()
    settings = get_settings()
    root = settings.dataset_dir / "material_detect_v2"
    build_dataset(root, per_class=36)
    out_project = settings.model_dir / "field" / "yolo_runs"
    detector = YOLODetector(variant=settings.yolo_variant, device=settings.default_device)
    result = detector.train(
        root / "data.yaml",
        epochs=5,
        imgsz=320,
        batch=8,
        lr0=1e-3,
        project=out_project,
        name="material_detect_v2",
        device=settings.default_device,
    )
    path = Path(result["artifact_path"]) if result.get("artifact_path") else None
    if not path or not path.exists():
        raise SystemExit(f"YOLO train failed: {result}")

    with SessionLocal() as db:
        ds = db.scalar(select(Dataset).where(Dataset.name == "material_detect", Dataset.version == "v2"))
        if ds is None:
            ds = Dataset(
                name="material_detect",
                version="v2",
                format=DatasetFormat.YOLO_DETECTION.value,
                status=DatasetStatus.READY_FOR_TRAINING.value,
                root_path=str(root),
                image_count=sum(1 for _ in root.rglob("*.jpg")),
                annotation_count=sum(1 for _ in (root / "labels").glob("*.txt")),
                class_names=MATERIALS,
                source="improved synthetic material shapes",
            )
            db.add(ds)
            db.flush()
        for art in db.scalars(
            select(ModelArtifact).where(ModelArtifact.model_type == ModelType.YOLOV8.value)
        ).all():
            art.is_active_inference = False
        artifact = ModelArtifact(
            name="YOLOv8 Material Detect v2",
            version="yolov8_material_detect_v2",
            model_type=ModelType.YOLOV8.value,
            status=ModelStatus.READY.value,
            artifact_path=str(path),
            class_mapping={n: i for i, n in enumerate(MATERIALS)},
            preprocess_config={"imgsz": 320},
            metrics=result.get("metrics") or {},
            dataset_id=ds.id,
            is_active_inference=True,
            notes="Clearer brick mortar / bag label / rebar rib shapes for auto-detect.",
        )
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        set_active_model(db, artifact.id)
        print(json.dumps({"artifact_id": artifact.id, "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
