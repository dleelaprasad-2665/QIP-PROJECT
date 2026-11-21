"""Fast visual cues to tell brick / cement bag / steel rebar apart (auto-detect)."""

from __future__ import annotations

import cv2
import numpy as np

from material_intelligence.app.ml.rust_heuristic import rust_ratio as field_rust_ratio


def _resize(image_bgr: np.ndarray, max_side: int = 320) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image_bgr
    return cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def material_scores(image_bgr: np.ndarray) -> dict[str, float]:
    if image_bgr is None or image_bgr.size == 0:
        return {"clay_brick": 0.0, "steel_rebar": 0.0, "cement_bag": 0.0}

    img = _resize(image_bgr)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n = float(hsv.shape[0] * hsv.shape[1])

    # Saturated red clay (brick body) — higher sat than light rust wash
    brick_mask = cv2.inRange(hsv, (0, 80, 80), (16, 255, 230))
    brick_mask2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 230))
    brick_ratio = float(np.count_nonzero(brick_mask | brick_mask2)) / n

    # Pale mortar strips
    mortar = cv2.inRange(hsv, (0, 0, 150), (40, 50, 255))
    mortar_ratio = float(np.count_nonzero(mortar)) / n

    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    horiz = float(np.mean(np.abs(sy)))
    vert = float(np.mean(np.abs(sx))) + 1e-3
    course = horiz / vert

    brick_raw = brick_ratio * 4.0 + mortar_ratio * 1.2
    if brick_ratio > 0.05 and 1.05 < course < 2.4:
        brick_raw += 0.35

    rust = field_rust_ratio(img)
    grey_metal = cv2.inRange(hsv, (0, 0, 40), (180, 45, 180))
    metal_ratio = float(np.count_nonzero(grey_metal)) / n
    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(np.count_nonzero(edges)) / n

    # Rebar: rust + metal + dense thin edges (ribs), without strong mortar grid
    rebar_raw = 0.0
    if rust > 0.06:
        rebar_raw += rust * 2.0
    if metal_ratio > 0.15:
        rebar_raw += metal_ratio * 1.2
    if edge_density > 0.12 and rust > 0.08:
        rebar_raw += 0.45
    # Only boost rebar over brick when metal/ribs present (not clay blocks)
    if brick_ratio > 0.10 and metal_ratio < 0.12 and mortar_ratio > 0.02:
        rebar_raw *= 0.2
    if rust > 0.15 and metal_ratio > 0.18 and brick_ratio < 0.12:
        rebar_raw += 0.6

    # Bag: large light printed region (needs text-like print, not shiny steel)
    light = cv2.inRange(hsv, (0, 0, 155), (180, 55, 255))
    light_ratio = float(np.count_nonzero(light)) / n
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    textish = float(np.mean(cv2.absdiff(gray, blur) > 18))
    bag_raw = light_ratio * 1.6 + textish * 2.4
    if light_ratio > 0.25 and textish > 0.08:
        bag_raw += 0.55
    # Shiny metallic grey with ribs is rebar, not a bag
    if metal_ratio > 0.25 and edge_density > 0.10 and textish < 0.12:
        rebar_raw += 0.55
        bag_raw *= 0.35
    bag_raw -= brick_ratio * 2.0 + rust * 1.0
    bag_raw = max(0.0, bag_raw)

    raw = np.array(
        [max(0.02, brick_raw), max(0.02, rebar_raw), max(0.02, bag_raw)],
        dtype=np.float32,
    )
    exp = np.exp(np.clip(raw, 0, 8) * 2.0)
    probs = exp / float(np.sum(exp))
    return {
        "clay_brick": float(probs[0]),
        "steel_rebar": float(probs[1]),
        "cement_bag": float(probs[2]),
    }


def guess_material_from_image(image_bgr: np.ndarray) -> tuple[str | None, float, dict[str, float]]:
    scores = material_scores(image_bgr)
    best = max(scores, key=scores.get)
    conf = scores[best]
    second = sorted(scores.values(), reverse=True)[1]
    if conf < 0.40 or (conf - second) < 0.06:
        return None, conf, scores
    return best, conf, scores
