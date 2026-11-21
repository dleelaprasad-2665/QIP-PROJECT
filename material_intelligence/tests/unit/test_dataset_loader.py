"""Dataset validation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from material_intelligence.app.domain.enums import DatasetStatus
from material_intelligence.app.ml.dataset_loader import validate_dataset


def test_classification_folder_validation(tmp_path: Path):
    for cls in ("cracked", "no_visible_damage"):
        d = tmp_path / cls
        d.mkdir()
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(d / "a.jpg")
        Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8) * 40).save(d / "b.jpg")
    report = validate_dataset(tmp_path)
    assert report.status == DatasetStatus.READY_FOR_TRAINING
    assert report.image_count == 4
    assert set(report.class_names) == {"cracked", "no_visible_damage"}


def test_yolo_dataset_validation(tmp_path: Path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(images / "img1.jpg")
    (labels / "img1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text(
        "names:\n  0: clay_brick\n",
        encoding="utf-8",
    )
    report = validate_dataset(tmp_path)
    assert report.image_count == 1
    assert report.annotation_count == 1
    assert report.status == DatasetStatus.READY_FOR_TRAINING
