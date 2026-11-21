"""Brick QA: prioritize edges — chips, breaks, uneven silhouette."""

from __future__ import annotations

import cv2
import numpy as np


def _edge_bands(gray: np.ndarray, band: float = 0.14) -> np.ndarray:
    """Boolean mask of the outer edge band of the image (where brick edges usually sit)."""
    h, w = gray.shape[:2]
    by, bx = max(4, int(h * band)), max(4, int(w * band))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:by, :] = 1
    mask[-by:, :] = 1
    mask[:, :bx] = 1
    mask[:, -bx:] = 1
    return mask


def suggest_brick_condition(image_bgr: np.ndarray) -> tuple[str, float, str]:
    """
    Inspect brick edges first.
    Returns (condition_key, confidence, note).
    """
    if image_bgr is None or image_bgr.size == 0:
        return "no_visible_damage", 0.4, "Brick check: empty image."

    h, w = image_bgr.shape[:2]
    scale = 480 / max(h, w)
    if scale < 1:
        img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    else:
        img = image_bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    band = _edge_bands(gray, 0.16)
    edges = cv2.Canny(gray, 50, 150)
    edge_in_band = edges[band == 1]
    edge_density = float(np.count_nonzero(edge_in_band)) / float(max(1, edge_in_band.size))

    # Contour of largest reddish/brown blob ≈ brick body
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clay = cv2.inRange(hsv, (0, 35, 40), (25, 255, 230)) | cv2.inRange(
        hsv, (160, 35, 40), (180, 255, 230)
    )
    clay = cv2.morphologyEx(clay, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(clay, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    uneven = False
    chip_score = 0.0
    broken_score = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area > 200:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(1, bh)
            # Ideal brick aspect roughly 1.5–3.0 (or stack)
            if aspect < 0.55 or (1.0 < aspect < 1.15) or aspect > 4.5:
                uneven = True
            rect_area = float(bw * bh)
            fill = area / max(1.0, rect_area)
            # Jagged silhouette / missing corners → chips/breaks
            if fill < 0.72:
                chip_score = 0.55 + (0.72 - fill)
            if len(approx) >= 8 or fill < 0.55:
                broken_score = 0.65 + (0.55 - min(fill, 0.55))
            # Local dark notches on the perimeter (chips)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.88:
                    chip_score = max(chip_score, 0.6 + (0.88 - solidity))

    # Dense irregular edges in the border band also signal damage
    if edge_density > 0.18:
        chip_score = max(chip_score, 0.55 + edge_density)

    if broken_score >= 0.7:
        return (
            "broken",
            min(0.9, broken_score),
            "Brick edges: broken / missing sections detected.",
        )
    if chip_score >= 0.6:
        return (
            "chipped",
            min(0.88, chip_score),
            "Brick edges: chips or broken corners on the edge band.",
        )
    if uneven:
        return (
            "broken",
            0.72,
            "Brick edges: uneven size / irregular outline vs a normal brick.",
        )
    if edge_density > 0.12 and chip_score >= 0.45:
        return (
            "cracked",
            0.68,
            "Brick edges: crack-like lines near the edges.",
        )
    return (
        "no_visible_damage",
        0.7,
        "Brick edges: outline looks intact in this photo.",
    )
