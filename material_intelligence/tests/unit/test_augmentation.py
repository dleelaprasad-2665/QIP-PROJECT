"""Augmentation bounding-box consistency tests."""

from __future__ import annotations

import numpy as np

from material_intelligence.app.ml.augmentation import (
    augment_detection_sample,
    build_detection_augmentation,
)


def test_detection_augmentation_keeps_box_count(sample_bgr):
    boxes = [(10.0, 10.0, 60.0, 60.0), (40.0, 40.0, 100.0, 110.0)]
    labels = ["clay_brick", "clay_brick"]
    transform = build_detection_augmentation(image_size=128, seed=0)
    result = augment_detection_sample(sample_bgr, boxes, labels, transform=transform)
    assert result["image"].shape[0] == 128
    assert result["image"].shape[1] == 128
    assert len(result["bboxes"]) == len(result["class_labels"])
    for box in result["bboxes"]:
        x1, y1, x2, y2 = box
        assert x2 >= x1
        assert y2 >= y1
