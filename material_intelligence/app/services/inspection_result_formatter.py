"""Human-readable inspection summaries for UI and exports."""

from __future__ import annotations

from material_intelligence.app.domain.enums import (
    DEFECT_DISPLAY_NAMES,
    HIGH_RISK_CONDITIONS,
    MATERIAL_DISPLAY_NAMES,
    InspectionStatus,
)
from material_intelligence.app.domain.schemas import ClassificationResult, InspectionPrediction


def display_material(key: str | None) -> str:
    if not key:
        return "Unknown material"
    return MATERIAL_DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def display_defect(key: str | None) -> str:
    if not key:
        return "Condition not determined"
    return DEFECT_DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def display_status(status: str | InspectionStatus) -> str:
    value = status.value if isinstance(status, InspectionStatus) else str(status)
    return value.replace("_", " ").title()


def primary_classification(prediction: InspectionPrediction) -> ClassificationResult | None:
    if not prediction.classifications:
        return None
    ranked = sorted(
        prediction.classifications,
        key=lambda c: c.confidence if c.confidence is not None else -1.0,
        reverse=True,
    )
    return ranked[0]


def detected_materials(prediction: InspectionPrediction) -> list[str]:
    names: list[str] = []
    for box in prediction.detection.boxes:
        label = display_material(box.class_name)
        if label not in names:
            names.append(label)
    return names


def format_inspection_summary(
    prediction: InspectionPrediction,
    *,
    expected_material: str | None = None,
) -> str:
    primary = primary_classification(prediction)
    materials = detected_materials(prediction)
    material_text = (
        ", ".join(materials)
        if materials
        else display_material(expected_material or (primary.material_key if primary else None))
    )

    if primary and primary.condition:
        condition = display_defect(primary.condition)
        conf = f"{(primary.confidence or 0) * 100:.0f}%"
        damage_line = f"Primary finding: {condition} ({conf} confidence)."
        if primary.condition in HIGH_RISK_CONDITIONS:
            damage_line += " This is a high-risk visual defect — engineer review required."
        elif primary.condition == "no_visible_damage":
            damage_line += " No obvious surface damage was predicted; confirm with site inspection."
    else:
        msg = primary.message if primary and primary.message else "Classifier could not determine condition."
        damage_line = f"Primary finding: unavailable. {msg}"

    det_count = len(prediction.detection.boxes)
    if prediction.detection.model_status.value != "ready":
        det_line = f"Detector status: {prediction.detection.model_status.value}. Full-frame classification used."
    elif det_count:
        det_line = f"Detected {det_count} object region(s)."
    else:
        det_line = (
            prediction.detection.message
            or "No localized objects above threshold — scanned the full image for condition."
        )

    status_line = f"Recommended workflow: {display_status(prediction.recommended_status)}."
    return f"Material focus: {material_text}. {damage_line} {det_line} {status_line}"
