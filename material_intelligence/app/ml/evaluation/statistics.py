"""Statistical helpers for research-grade evaluation reporting."""

from __future__ import annotations

from typing import Any

import numpy as np


def bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(sample.mean()))
    means_arr = np.sort(np.asarray(means))
    lo = float(np.quantile(means_arr, alpha / 2))
    hi = float(np.quantile(means_arr, 1 - alpha / 2))
    return {
        "mean": float(arr.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(arr.size),
        "alpha": alpha,
        "n_boot": n_boot,
        "seed": seed,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
) -> float | None:
    """ECE for multi-class using max predicted probability."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim != 2 or len(y_true) == 0:
        return None
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = float((predictions[mask] == y_true[mask]).mean())
        conf = float(confidences[mask].mean())
        ece += (mask.sum() / total) * abs(acc - conf)
    return float(ece)


def brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float | None:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if len(y_true) == 0 or y_proba.ndim != 2:
        return None
    one_hot = np.zeros_like(y_proba)
    for i, label in enumerate(y_true):
        if 0 <= int(label) < n_classes:
            one_hot[i, int(label)] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def safe_metric(value: Any) -> Any:
    """Convert NaN/Inf to None for JSON persistence."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(f) or np.isinf(f):
        return None
    return f
