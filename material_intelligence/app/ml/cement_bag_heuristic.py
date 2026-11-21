"""Cement bag QA: focus on bag body + printed labels, ignore background."""

from __future__ import annotations

import cv2
import numpy as np

from material_intelligence.app.domain.schemas import BoundingBox


def filter_bag_boxes(
    boxes: list[BoundingBox],
    image_shape: tuple[int, ...],
) -> list[BoundingBox]:
    """
    Keep detections that look like bags (not full-frame background).
    Prefer mid-size rectangles covering 8–70% of the frame.
    """
    if not boxes:
        return []
    h, w = image_shape[:2]
    area = float(h * w) or 1.0
    kept: list[BoundingBox] = []
    for box in boxes:
        bw = max(0.0, box.x2 - box.x1)
        bh = max(0.0, box.y2 - box.y1)
        frac = (bw * bh) / area
        # Reject near-full-frame "background" boxes and tiny noise
        if frac < 0.05 or frac > 0.78:
            continue
        aspect = bw / max(1.0, bh)
        if aspect < 0.35 or aspect > 3.5:
            continue
        kept.append(box)
    if not kept:
        # Fall back to best mid-size box of any class
        ranked = sorted(
            boxes,
            key=lambda b: abs(((b.x2 - b.x1) * (b.y2 - b.y1) / area) - 0.35),
        )
        for box in ranked:
            frac = ((box.x2 - box.x1) * (box.y2 - box.y1)) / area
            if 0.08 <= frac <= 0.75:
                return [box]
        return []
    return kept


def bag_roi(image_bgr: np.ndarray) -> np.ndarray:
    """
    Estimate the bag region (light printed area) so background dirt/sky is ignored.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    light = cv2.inRange(hsv, (0, 0, 110), (180, 90, 255))
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Center crop — bags are usually framed in the middle
        m = int(min(h, w) * 0.08)
        return image_bgr[m : h - m, m : w - m]
    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    if bw * bh < 0.08 * h * w:
        m = int(min(h, w) * 0.08)
        return image_bgr[m : h - m, m : w - m]
    pad = 8
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    return image_bgr[y1:y2, x1:x2]


def suggest_cement_condition(image_bgr: np.ndarray) -> tuple[str, float, str]:
    """
    Check bag packaging / labels region for tears and stains — not the yard background.
    """
    roi = bag_roi(image_bgr)
    if roi is None or roi.size == 0:
        return "no_visible_damage", 0.4, "Cement bag: could not isolate bag area."

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    # Dark irregular tears / holes
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))
    dark_ratio = float(np.count_nonzero(dark)) / float(dark.size)

    # Moisture / staining (brown-grey blotches on light bag)
    stain = cv2.inRange(hsv, (5, 20, 40), (30, 120, 160))
    stain_ratio = float(np.count_nonzero(stain)) / float(stain.size)

    # Torn packaging: strong irregular dark edges inside bag ROI
    if dark_ratio > 0.12 and edge_density > 0.10:
        return (
            "torn_packaging",
            min(0.9, 0.55 + dark_ratio + edge_density),
            "Cement bag: tear / open packaging on the bag body (labels area).",
        )
    if dark_ratio > 0.08 and edge_density > 0.08:
        return (
            "damaged_packaging",
            min(0.85, 0.5 + dark_ratio),
            "Cement bag: damaged packaging near printed label area.",
        )
    if stain_ratio > 0.14:
        return (
            "visible_moisture_staining",
            min(0.85, 0.5 + stain_ratio),
            "Cement bag: moisture / stain marks on the bag surface.",
        )
    return (
        "no_visible_damage",
        0.72,
        "Cement bag: packaging and label area look intact (background ignored).",
    )
