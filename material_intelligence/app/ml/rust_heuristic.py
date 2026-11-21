"""Fast visual rust heuristic for steel rebar field photos."""

from __future__ import annotations

import cv2
import numpy as np


def rust_ratio(image_bgr: np.ndarray) -> float:
    """Fraction of pixels that look orange/brown (typical surface rust)."""
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Broad rust band in HSV (orange–brown).
    lower1 = np.array([5, 40, 40], dtype=np.uint8)
    upper1 = np.array([25, 255, 255], dtype=np.uint8)
    lower2 = np.array([0, 50, 30], dtype=np.uint8)
    upper2 = np.array([10, 255, 200], dtype=np.uint8)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    # Ignore very dark ground / shadows a bit by requiring some value.
    value = hsv[:, :, 2]
    usable = value > 35
    denom = int(np.count_nonzero(usable)) or 1
    return float(np.count_nonzero(mask & usable) / denom)


def suggest_steel_condition(image_bgr: np.ndarray) -> tuple[str, float, str]:
    """
    Returns (condition_key, confidence, note) for steel rebar photos.
    Used to correct weak synthetic-model guesses on real field images.
    """
    ratio = rust_ratio(image_bgr)
    if ratio >= 0.18:
        conf = min(0.92, 0.55 + ratio)
        return (
            "surface_corrosion" if ratio < 0.35 else "extensive_visible_corrosion",
            conf,
            f"Field rust check: ~{ratio:.0%} orange/brown surface.",
        )
    if ratio >= 0.08:
        return (
            "surface_corrosion",
            min(0.75, 0.50 + ratio),
            f"Field rust check: light rust ~{ratio:.0%}.",
        )
    return (
        "no_visible_damage",
        min(0.85, 0.60 + (0.08 - ratio)),
        f"Field rust check: low rust ~{ratio:.0%}.",
    )
