"""Training-time augmentation with bounding-box consistency."""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np


def build_detection_augmentation(
    *,
    image_size: int = 640,
    seed: int | None = 42,
) -> A.Compose:
    """Stochastic transforms for detection training only."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.Resize(image_size, image_size),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
        seed=seed,
    )


def build_classification_augmentation(
    *,
    image_size: int = 224,
    seed: int | None = 42,
) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=12, border_mode=0, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.25),
            A.GaussianBlur(blur_limit=(3, 3), p=0.15),
            A.Resize(image_size, image_size),
        ],
        seed=seed,
    )


def augment_detection_sample(
    image_bgr: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    class_labels: list[str],
    transform: A.Compose | None = None,
) -> dict[str, Any]:
    """
    boxes: Pascal VOC [x_min, y_min, x_max, y_max] in pixel coordinates.
    Returns transformed image and boxes with the same label order.
    """
    transform = transform or build_detection_augmentation()
    result = transform(image=image_bgr, bboxes=boxes, class_labels=class_labels)
    return {
        "image": result["image"],
        "bboxes": [tuple(map(float, box)) for box in result["bboxes"]],
        "class_labels": list(result["class_labels"]),
    }


def augment_classification_sample(
    image_bgr: np.ndarray,
    transform: A.Compose | None = None,
) -> np.ndarray:
    transform = transform or build_classification_augmentation()
    return transform(image=image_bgr)["image"]
