"""Train/test leakage heuristics for evaluation datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _file_hash(path: Path, *, chunk: int = 1024 * 64) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                data = fh.read(chunk)
                if not data:
                    break
                h.update(data)
        return h.hexdigest()
    except OSError:
        return None


def collect_image_hashes(root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not root.exists():
        return mapping
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            digest = _file_hash(path)
            if digest:
                mapping[str(path.resolve())] = digest
    return mapping


def detect_hash_overlap(
    train_root: Path | None,
    test_root: Path,
) -> dict:
    """Flag identical file contents between train and evaluation sets."""
    test_hashes = collect_image_hashes(test_root)
    if train_root is None or not train_root.exists():
        return {
            "status": "unchecked",
            "overlap_count": 0,
            "overlap_fraction": 0.0,
            "message": "Training dataset path unavailable — leakage check skipped.",
            "sample_overlap_paths": [],
        }
    train_hashes = collect_image_hashes(train_root)
    train_by_hash = {v: k for k, v in train_hashes.items()}
    overlaps = []
    for path, digest in test_hashes.items():
        if digest in train_by_hash:
            overlaps.append({"test": path, "train": train_by_hash[digest], "hash": digest})
    n_test = max(len(test_hashes), 1)
    fraction = len(overlaps) / n_test
    status = "clean"
    if overlaps:
        status = "flagged" if fraction >= 0.01 else "minor_overlap"
    return {
        "status": status,
        "overlap_count": len(overlaps),
        "overlap_fraction": float(fraction),
        "test_image_count": len(test_hashes),
        "train_image_count": len(train_hashes),
        "sample_overlap_paths": overlaps[:20],
        "message": (
            f"Detected {len(overlaps)} identical file hash(es) between train and eval sets."
            if overlaps
            else "No identical file hashes detected between train and eval sets."
        ),
    }


def validate_class_mapping_compatibility(
    model_classes: list[str],
    dataset_classes: list[str],
) -> dict:
    model_set = {c.strip().lower() for c in model_classes}
    data_set = {c.strip().lower() for c in dataset_classes if c.strip()}
    missing_in_data = sorted(model_set - data_set)
    extra_in_data = sorted(data_set - model_set)
    compatible = len(model_set & data_set) > 0 and not (
        model_set and data_set and model_set.isdisjoint(data_set)
    )
    return {
        "compatible": compatible,
        "model_classes": sorted(model_set),
        "dataset_classes": sorted(data_set),
        "missing_in_dataset": missing_in_data,
        "extra_in_dataset": extra_in_data,
        "message": (
            "Class mappings are compatible for evaluation."
            if compatible
            else "Class mappings are incompatible — refuse silent evaluation."
        ),
    }
