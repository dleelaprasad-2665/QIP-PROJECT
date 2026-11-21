"""Field-facing scan results aligned to the app flow.

4. Material Condition: Good | Damaged | Inspection Required
5. Preliminary Recommendation: Routine checks / Further inspection / Hold for review
"""

from __future__ import annotations

from dataclasses import dataclass

from material_intelligence.app.domain.enums import HIGH_RISK_CONDITIONS, MATERIAL_DISPLAY_NAMES
from material_intelligence.app.domain.schemas import ClassificationResult, InspectionPrediction
from material_intelligence.app.services.inspection_result_formatter import primary_classification


@dataclass
class SimpleScanResult:
    material_name: str
    condition_label: str  # Good | Damaged | Inspection Required
    condition_detail: str
    confidence_pct: int
    recommendation: str
    color: str  # success | danger | warning
    pipeline_steps: dict[str, str]  # step label → status text


def _material_name(prediction: InspectionPrediction, fallback: str | None = None) -> str:
    primary = primary_classification(prediction)
    key = None
    if prediction.detection.boxes:
        key = prediction.detection.boxes[0].class_name
    if primary and primary.material_key:
        key = primary.material_key
    if not key:
        key = fallback
    if not key:
        return "Unknown material"
    return MATERIAL_DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def _detail_from_condition(condition: str | None) -> str:
    mapping = {
        "no_visible_damage": "No clear surface damage seen in the photo.",
        "surface_corrosion": "Surface rust / corrosion visible on the steel.",
        "extensive_visible_corrosion": "Heavy rust / corrosion — needs engineer review.",
        "deformation": "Bars look bent or deformed.",
        "cracked": "Cracks are visible on the material.",
        "chipped": "Chips or broken edges are visible.",
        "broken": "Material appears broken or uneven.",
        "torn_packaging": "Torn packaging on the cement bag (label area).",
        "visible_moisture_staining": "Moisture staining is visible.",
        "damaged_packaging": "Damaged packaging on the cement bag.",
    }
    if not condition:
        return "Condition could not be determined clearly."
    return mapping.get(condition, condition.replace("_", " ").title())


def simplify_prediction(
    prediction: InspectionPrediction,
    *,
    expected_material: str | None = None,
) -> SimpleScanResult:
    primary: ClassificationResult | None = primary_classification(prediction)
    material = _material_name(prediction, expected_material)
    conf = int(round((primary.confidence or 0) * 100)) if primary else 0
    condition = primary.condition if primary else None

    n_boxes = len(prediction.detection.boxes) if prediction.detection else 0
    yolo_txt = (
        f"YOLOv8 · {n_boxes} region(s) detected"
        if n_boxes
        else "YOLOv8 · full-frame scan"
    )
    eff_txt = (
        f"EfficientNet-B0 · {condition.replace('_', ' ') if condition else 'uncertain'}"
        f" ({conf}%)"
    )

    pipeline = {
        "1. Upload": material,
        "2. Preprocessing": "CLAHE · Resize · Normalize",
        "3. AI Analysis": f"{yolo_txt} → {eff_txt}",
        "4. Condition": "",
        "5. Recommendation": "",
        "6. Inventory": "Result stored for dashboard & reports",
    }

    if primary is None or not condition:
        label, color = "Inspection Required", "warning"
        rec = "Hold for review"
        detail = "AI could not classify this photo confidently."
    elif conf < 45 or (primary.message and "uncertain" in (primary.message or "").lower()):
        label, color = "Inspection Required", "warning"
        rec = "Further inspection"
        detail = "Photo is unclear or AI confidence is low."
    elif condition == "no_visible_damage":
        label, color = "Good", "success"
        rec = "Routine checks"
        detail = _detail_from_condition(condition)
    elif condition in HIGH_RISK_CONDITIONS or condition in {
        "surface_corrosion",
        "extensive_visible_corrosion",
        "deformation",
        "cracked",
        "chipped",
        "broken",
        "torn_packaging",
        "damaged_packaging",
        "visible_moisture_staining",
    }:
        label, color = "Damaged", "danger"
        rec = "Further inspection" if condition in HIGH_RISK_CONDITIONS else "Hold for review"
        if condition in {"extensive_visible_corrosion", "broken", "torn_packaging"}:
            rec = "Hold for review"
        detail = _detail_from_condition(condition)
    else:
        label, color = "Inspection Required", "warning"
        rec = "Further inspection"
        detail = _detail_from_condition(condition)

    pipeline["4. Condition"] = label
    pipeline["5. Recommendation"] = rec

    return SimpleScanResult(
        material_name=material,
        condition_label=label,
        condition_detail=detail,
        confidence_pct=conf,
        recommendation=rec,
        color=color,
        pipeline_steps=pipeline,
    )
