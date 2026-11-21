"""Dataset import, parsing, and validation for YOLO / folder / COCO formats."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from material_intelligence.app.domain.enums import DatasetFormat, DatasetStatus
from material_intelligence.app.utils.files import FileValidationError, safe_extract_zip

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DatasetValidationReport:
    format: DatasetFormat
    image_count: int = 0
    annotation_count: int = 0
    class_names: list[str] = field(default_factory=list)
    class_distribution: dict[str, int] = field(default_factory=dict)
    missing_annotations: list[str] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    duplicate_images: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    imbalance_flag: bool = False
    status: DatasetStatus = DatasetStatus.VALIDATING

    def to_dict(self) -> dict:
        return {
            "format": self.format.value,
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "class_names": self.class_names,
            "class_distribution": self.class_distribution,
            "missing_annotations": self.missing_annotations,
            "corrupt_images": self.corrupt_images,
            "duplicate_images": self.duplicate_images,
            "warnings": self.warnings,
            "errors": self.errors,
            "imbalance_flag": self.imbalance_flag,
            "status": self.status.value,
        }


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def normalize_flat_image_archive(root: Path) -> Path:
    """If ZIP is a pile of loose images, group them under unlabeled/ for classification import."""
    try:
        detect_dataset_format(root)
        return root
    except FileValidationError:
        pass

    images = [p for p in root.rglob("*") if p.is_file() and _is_image(p)]
    if not images:
        raise FileValidationError(
            "Unable to detect a supported dataset format. "
            "Use YOLO (images+labels / data.yaml), COCO JSON, class folders, or a ZIP of images."
        )

    unlabeled = root / "unlabeled"
    unlabeled.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        if unlabeled in image_path.parents or image_path.parent == unlabeled:
            continue
        target = unlabeled / image_path.name
        if target.exists():
            target = unlabeled / f"{image_path.stem}_{image_path.stat().st_size}{image_path.suffix}"
        shutil.move(str(image_path), str(target))
    return root


def detect_dataset_format(root: Path) -> DatasetFormat:
    if (root / "data.yaml").exists() or (root / "dataset.yaml").exists():
        return DatasetFormat.ROBOFLOW_YOLO
    if (root / "annotations" / "instances_default.json").exists() or list(root.rglob("instances_*.json")):
        return DatasetFormat.COCO
    # YOLO-style: images/ + labels/
    if (root / "images").exists() and (root / "labels").exists():
        return DatasetFormat.YOLO_DETECTION
    # Nested train/valid with images/labels
    for split in ("train", "valid", "val", "test"):
        if (root / split / "images").exists() and (root / split / "labels").exists():
            return DatasetFormat.CVAT_YOLO
    # Classification folders
    class_dirs = [p for p in root.iterdir() if p.is_dir() and any(_is_image(f) for f in p.rglob("*"))]
    if class_dirs:
        return DatasetFormat.IMAGE_FOLDER_CLASSIFICATION
    # Loose images in root
    if any(p.is_file() and _is_image(p) for p in root.iterdir()):
        return DatasetFormat.IMAGE_FOLDER_CLASSIFICATION
    raise FileValidationError("Unable to detect a supported dataset format.")


def _read_yolo_yaml_classes(root: Path) -> list[str]:
    for name in ("data.yaml", "dataset.yaml"):
        path = root / name
        if not path.exists():
            continue
        names: list[str] = []
        in_names = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                remainder = stripped[len("names:") :].strip()
                if remainder.startswith("["):
                    # inline list
                    items = remainder.strip("[]").split(",")
                    return [item.strip().strip("'\"") for item in items if item.strip()]
                in_names = True
                continue
            if in_names:
                if ":" in stripped:
                    # mapping style: 0: brick
                    _, value = stripped.split(":", 1)
                    names.append(value.strip().strip("'\""))
                elif stripped.startswith("-"):
                    names.append(stripped.lstrip("- ").strip().strip("'\""))
                elif not stripped:
                    break
                else:
                    break
        if names:
            return names
    return []


def _image_fingerprint(path: Path) -> str | None:
    try:
        with Image.open(path) as img:
            img.load()
            return f"{img.size}-{img.mode}-{path.stat().st_size}"
    except (UnidentifiedImageError, OSError):
        return None


def _check_imbalance(distribution: dict[str, int]) -> bool:
    if len(distribution) < 2:
        return False
    counts = list(distribution.values())
    return max(counts) >= 3 * max(1, min(counts))


def _collect_split_image_label_pairs(root: Path) -> list[tuple[Path, Path | None]]:
    pairs: list[tuple[Path, Path | None]] = []
    candidate_roots = [root]
    for split in ("train", "valid", "val", "test"):
        if (root / split).exists():
            candidate_roots.append(root / split)

    seen_images: set[Path] = set()
    for base in candidate_roots:
        images_dir = base / "images" if (base / "images").exists() else base
        labels_dir = base / "labels" if (base / "labels").exists() else base / "labels"
        for image_path in images_dir.rglob("*"):
            if not image_path.is_file() or not _is_image(image_path):
                continue
            if image_path in seen_images:
                continue
            seen_images.add(image_path)
            label_path = labels_dir / f"{image_path.stem}.txt"
            pairs.append((image_path, label_path if label_path.exists() else None))
    return pairs


def validate_yolo_detection(root: Path, fmt: DatasetFormat) -> DatasetValidationReport:
    report = DatasetValidationReport(format=fmt)
    class_names = _read_yolo_yaml_classes(root)
    class_counter: Counter[str] = Counter()
    fingerprints: dict[str, str] = {}

    pairs = _collect_split_image_label_pairs(root)
    report.image_count = len(pairs)

    for image_path, label_path in pairs:
        fingerprint = _image_fingerprint(image_path)
        if fingerprint is None:
            report.corrupt_images.append(str(image_path.relative_to(root)))
            continue
        if fingerprint in fingerprints:
            report.duplicate_images.append(str(image_path.relative_to(root)))
        else:
            fingerprints[fingerprint] = str(image_path)

        if label_path is None:
            report.missing_annotations.append(str(image_path.relative_to(root)))
            continue

        lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        report.annotation_count += len(lines)
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                report.errors.append(f"Malformed label in {label_path.name}: '{line}'")
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                report.errors.append(f"Non-integer class id in {label_path.name}")
                continue
            if class_names and 0 <= class_id < len(class_names):
                class_counter[class_names[class_id]] += 1
            else:
                class_counter[str(class_id)] += 1

    report.class_names = class_names or sorted(class_counter.keys())
    report.class_distribution = dict(class_counter)
    report.imbalance_flag = _check_imbalance(report.class_distribution)
    if report.imbalance_flag:
        report.warnings.append("Class imbalance detected (majority class ≥ 3× minority).")
    if report.corrupt_images:
        report.warnings.append(f"{len(report.corrupt_images)} corrupt image(s) found.")
    if report.missing_annotations:
        report.warnings.append(f"{len(report.missing_annotations)} image(s) missing annotations.")
    if report.errors:
        report.status = DatasetStatus.INVALID
    elif report.image_count == 0:
        report.errors.append("No images found.")
        report.status = DatasetStatus.INVALID
    else:
        report.status = DatasetStatus.READY_FOR_TRAINING
    return report


def validate_classification_folders(root: Path) -> DatasetValidationReport:
    report = DatasetValidationReport(format=DatasetFormat.IMAGE_FOLDER_CLASSIFICATION)
    class_counter: Counter[str] = Counter()
    fingerprints: dict[str, str] = {}

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    # Include root-level loose images as "unlabeled"
    root_images = [p for p in root.iterdir() if p.is_file() and _is_image(p)]
    scan_targets: list[tuple[str, list[Path]]] = [(d.name, list(d.rglob("*"))) for d in class_dirs]
    if root_images:
        scan_targets.append(("unlabeled", root_images))

    for class_name, paths in scan_targets:
        if class_name not in report.class_names:
            report.class_names.append(class_name)
        for image_path in paths:
            if not image_path.is_file() or not _is_image(image_path):
                continue
            report.image_count += 1
            fingerprint = _image_fingerprint(image_path)
            if fingerprint is None:
                try:
                    rel = str(image_path.relative_to(root))
                except ValueError:
                    rel = str(image_path)
                report.corrupt_images.append(rel)
                continue
            if fingerprint in fingerprints:
                try:
                    report.duplicate_images.append(str(image_path.relative_to(root)))
                except ValueError:
                    report.duplicate_images.append(str(image_path))
            else:
                fingerprints[fingerprint] = str(image_path)
            class_counter[class_name] += 1
            report.annotation_count += 1

    report.class_distribution = dict(class_counter)
    report.imbalance_flag = _check_imbalance(report.class_distribution)
    if report.imbalance_flag:
        report.warnings.append("Class imbalance detected.")
    if report.image_count == 0:
        report.errors.append("No classification images found.")
        report.status = DatasetStatus.INVALID
    else:
        report.status = DatasetStatus.READY_FOR_TRAINING
        if "unlabeled" in report.class_names:
            report.warnings.append(
                "Imported as unlabeled image set. Organize into class folders for supervised training."
            )
    return report


def validate_coco(root: Path) -> DatasetValidationReport:
    report = DatasetValidationReport(format=DatasetFormat.COCO)
    annotation_files = list(root.rglob("instances_*.json"))
    if not annotation_files and (root / "annotations.json").exists():
        annotation_files = [root / "annotations.json"]
    if not annotation_files:
        report.errors.append("No COCO annotation JSON found.")
        report.status = DatasetStatus.INVALID
        return report

    ann_path = annotation_files[0]
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    report.class_names = [categories[i] for i in sorted(categories)]
    images = {img["id"]: img for img in data.get("images", [])}
    report.image_count = len(images)
    class_counter: Counter[str] = Counter()
    for ann in data.get("annotations", []):
        report.annotation_count += 1
        cat = categories.get(ann.get("category_id"), str(ann.get("category_id")))
        class_counter[cat] += 1
    report.class_distribution = dict(class_counter)
    report.imbalance_flag = _check_imbalance(report.class_distribution)
    report.status = DatasetStatus.READY_FOR_TRAINING if report.image_count else DatasetStatus.INVALID
    if report.image_count == 0:
        report.errors.append("COCO file contains no images.")
    return report


def validate_dataset(root: Path) -> DatasetValidationReport:
    fmt = detect_dataset_format(root)
    if fmt in {
        DatasetFormat.YOLO_DETECTION,
        DatasetFormat.ROBOFLOW_YOLO,
        DatasetFormat.CVAT_YOLO,
    }:
        return validate_yolo_detection(root, fmt)
    if fmt == DatasetFormat.IMAGE_FOLDER_CLASSIFICATION:
        return validate_classification_folders(root)
    if fmt == DatasetFormat.COCO:
        return validate_coco(root)
    report = DatasetValidationReport(format=fmt)
    report.errors.append(f"Unsupported format: {fmt}")
    report.status = DatasetStatus.INVALID
    return report


def import_archive_to_dataset_dir(zip_path: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    safe_extract_zip(zip_path, destination)
    # If archive contained a single top-level folder, use that as root.
    children = [p for p in destination.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return destination


def list_yolo_label_preview(root: Path, limit: int = 20) -> list[dict]:
    """Return sample image/label pairs for bounding-box preview."""
    samples: list[dict] = []
    for image_path, label_path in _collect_split_image_label_pairs(root)[:limit]:
        boxes = []
        if label_path and label_path.exists():
            with Image.open(image_path) as img:
                width, height = img.size
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id, xc, yc, w, h = parts[:5]
                xc, yc, w, h = map(float, (xc, yc, w, h))
                x1 = (xc - w / 2) * width
                y1 = (yc - h / 2) * height
                x2 = (xc + w / 2) * width
                y2 = (yc + h / 2) * height
                boxes.append(
                    {
                        "class_id": int(class_id),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
        samples.append({"image": str(image_path), "boxes": boxes})
    return samples
