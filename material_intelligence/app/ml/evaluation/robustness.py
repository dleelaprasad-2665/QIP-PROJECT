"""Controlled robustness transforms and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from material_intelligence.app.ml.efficientnet_classifier import EfficientNetClassifier
from material_intelligence.app.ml.evaluation.classification import compute_classification_metrics
from material_intelligence.app.ml.preprocessing import load_bgr_image


TransformFn = Callable[[np.ndarray], np.ndarray]


def brightness(image: np.ndarray, delta: int) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.0, beta=delta)


def contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image, (k, k), 0)


def compress_jpeg(image: np.ndarray, quality: int = 40) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def rotate(image: np.ndarray, angle: float = 10.0) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def occlude(image: np.ndarray, fraction: float = 0.15) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    bh, bw = int(h * fraction), int(w * fraction)
    y0, x0 = h // 4, w // 4
    out[y0 : y0 + bh, x0 : x0 + bw] = 0
    return out


DEFAULT_TRANSFORMS: dict[str, TransformFn] = {
    "brightness_down": lambda im: brightness(im, -40),
    "brightness_up": lambda im: brightness(im, 40),
    "contrast_down": lambda im: contrast(im, 0.7),
    "contrast_up": lambda im: contrast(im, 1.4),
    "blur_mild": lambda im: blur(im, 5),
    "jpeg_q40": lambda im: compress_jpeg(im, 40),
    "rotate_10": lambda im: rotate(im, 10),
    "occlusion_partial": lambda im: occlude(im, 0.15),
}


@dataclass
class RobustnessResult:
    baseline_accuracy: float | None
    by_transform: dict[str, dict[str, Any]]
    notes: list[str]


def evaluate_classifier_robustness(
    classifier: EfficientNetClassifier,
    image_paths: list[str],
    true_labels: list[str],
    *,
    transforms: dict[str, TransformFn] | None = None,
    limit: int = 40,
) -> RobustnessResult:
    """Compare original vs transformed predictions on a labeled image list."""
    notes = [
        "Robustness results are conditional on the provided labeled samples.",
        "CLAHE is not assumed to improve every image — compare experiments explicitly.",
    ]
    if not classifier.is_ready():
        return RobustnessResult(
            baseline_accuracy=None,
            by_transform={},
            notes=notes + ["Classifier not ready."],
        )
    class_names = [classifier.idx_to_class[i] for i in range(len(classifier.class_mapping))]
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    pairs = [
        (p, t)
        for p, t in zip(image_paths, true_labels, strict=True)
        if t in name_to_idx
    ][:limit]
    if not pairs:
        return RobustnessResult(
            baseline_accuracy=None,
            by_transform={},
            notes=notes + ["No labeled samples matched classifier classes."],
        )

    def _predict_many(images: list[np.ndarray]) -> tuple[list[int], list[int]]:
        y_true, y_pred = [], []
        for (path, label), image in zip(pairs, images, strict=True):
            result = classifier.predict(image)
            y_true.append(name_to_idx[label])
            if result.condition in name_to_idx:
                y_pred.append(name_to_idx[result.condition])
            else:
                y_pred.append(-1)
        # Replace unknown preds with majority wrong index for metric safety
        y_pred = [p if p >= 0 else 0 for p in y_pred]
        return y_true, y_pred

    originals = []
    for path, _ in pairs:
        img = load_bgr_image(path)
        originals.append(img)
    y_true, y_pred = _predict_many(originals)
    baseline = compute_classification_metrics(
        y_true, y_pred, class_names=class_names, model_version=classifier.version or "unknown"
    )
    by_transform: dict[str, dict[str, Any]] = {}
    for name, fn in (transforms or DEFAULT_TRANSFORMS).items():
        altered = [fn(im) for im in originals]
        yt, yp = _predict_many(altered)
        metrics = compute_classification_metrics(
            yt, yp, class_names=class_names, model_version=classifier.version or "unknown"
        )
        by_transform[name] = {
            "accuracy": metrics.metrics.get("accuracy"),
            "macro_f1": metrics.metrics.get("macro_f1"),
            "delta_accuracy": (
                None
                if metrics.metrics.get("accuracy") is None
                or baseline.metrics.get("accuracy") is None
                else float(metrics.metrics["accuracy"]) - float(baseline.metrics["accuracy"])
            ),
            "n": metrics.metrics.get("n_samples"),
        }
    return RobustnessResult(
        baseline_accuracy=baseline.metrics.get("accuracy"),
        by_transform=by_transform,
        notes=notes,
    )
