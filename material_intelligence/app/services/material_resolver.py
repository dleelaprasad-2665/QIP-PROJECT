"""Resolve material type from user hint, filename, visual cues, detections, and condition."""

from __future__ import annotations

from pathlib import Path

from material_intelligence.app.domain.enums import DEFAULT_MATERIALS
from material_intelligence.app.domain.schemas import BoundingBox, ClassificationResult

# Filename / path keywords → material key
_FILENAME_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("rebar", "steel", "rod", "reinforc", "corrosion"), "steel_rebar"),
    (("cement", "bag", "sack", "packaging", "moisture"), "cement_bag"),
    (("brick", "tile", "clay", "masonry"), "clay_brick"),
    (("wood", "timber", "lumber", "plywood", "board"), "wood"),
]

_CONDITION_TO_MATERIAL: dict[str, str] = {
    "cracked": "clay_brick",
    "chipped": "clay_brick",
    "broken": "clay_brick",
    "surface_corrosion": "steel_rebar",
    "extensive_visible_corrosion": "steel_rebar",
    "deformation": "steel_rebar",
    "torn_packaging": "cement_bag",
    "visible_moisture_staining": "cement_bag",
    "damaged_packaging": "cement_bag",
}

KNOWN_MATERIALS = set(DEFAULT_MATERIALS.keys())
YOLO_TRUST_CONF = 0.55  # v2 detector is clearer; visual still breaks ties


def guess_material_from_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    text = Path(filename).stem.lower().replace("-", " ").replace("_", " ")
    for keywords, material in _FILENAME_HINTS:
        if any(k in text for k in keywords):
            return material
    return None


def guess_material_from_condition(condition: str | None) -> str | None:
    if not condition:
        return None
    return _CONDITION_TO_MATERIAL.get(condition)


def best_detection_material(boxes: list[BoundingBox]) -> tuple[str | None, float]:
    """Return highest-confidence material-like detection."""
    best_name: str | None = None
    best_conf = -1.0
    for box in boxes:
        name = (box.class_name or "").strip().lower().replace(" ", "_")
        conf = float(box.confidence or 0.0)
        if name not in KNOWN_MATERIALS:
            continue
        if conf > best_conf:
            best_conf = conf
            best_name = name
    if best_name is None:
        return None, 0.0
    return best_name, best_conf


def resolve_material(
    *,
    expected_material_key: str | None = None,
    filename: str | None = None,
    boxes: list[BoundingBox] | None = None,
    classifications: list[ClassificationResult] | None = None,
    visual_material: str | None = None,
    visual_confidence: float = 0.0,
) -> tuple[str | None, str]:
    """
    Resolve the material to show/use.

    Priority:
    1. Explicit user selection
    2. Filename hint
    3. Strong visual cue (brick vs bag vs rebar appearance)
    4. High-confidence YOLO that agrees with visual (or no visual)
    5. Condition→material (only if it does not fight a clear visual)
    6. Weak YOLO / classifier fallback
    """
    if expected_material_key and expected_material_key in KNOWN_MATERIALS:
        return expected_material_key, "user selection"

    from_name = guess_material_from_filename(filename)
    if from_name:
        return from_name, "filename hint"

    det_name, det_conf = best_detection_material(boxes or [])

    # Strong detector wins when clear.
    if det_name and det_conf >= 0.65:
        # Visual can veto obvious brick↔bag swaps at mid confidence.
        if (
            visual_material in KNOWN_MATERIALS
            and visual_confidence >= 0.50
            and visual_material != det_name
            and det_conf < 0.80
            and {visual_material, det_name} == {"clay_brick", "cement_bag"}
        ):
            return visual_material, f"visual veto ({visual_confidence:.0%})"
        return det_name, f"detector ({det_conf:.0%})"

    if visual_material in KNOWN_MATERIALS and visual_confidence >= 0.45:
        return visual_material, f"visual ({visual_confidence:.0%})"

    if det_name and det_conf >= YOLO_TRUST_CONF:
        return det_name, f"detector ({det_conf:.0%})"

    primary = None
    if classifications:
        primary = max(
            classifications,
            key=lambda c: c.confidence if c.confidence is not None else -1.0,
        )
    if primary and primary.condition:
        from_cond = guess_material_from_condition(primary.condition)
        # Do not let packaging conditions invent cement_bag when visual said brick/rebar
        if from_cond and visual_material and from_cond != visual_material and visual_confidence >= 0.35:
            pass
        elif from_cond:
            return from_cond, f"condition ({primary.condition})"
        if primary.material_key in KNOWN_MATERIALS:
            return primary.material_key, "classifier"

    if det_name:
        return det_name, f"detector weak ({det_conf:.0%})"

    if visual_material in KNOWN_MATERIALS:
        return visual_material, f"visual weak ({visual_confidence:.0%})"

    return None, "unknown"


def apply_resolved_material(
    classifications: list[ClassificationResult],
    material_key: str | None,
) -> list[ClassificationResult]:
    if not material_key:
        return classifications
    updated: list[ClassificationResult] = []
    for item in classifications:
        if item.material_key != material_key:
            updated.append(item.model_copy(update={"material_key": material_key}))
        else:
            updated.append(item)
    return updated
