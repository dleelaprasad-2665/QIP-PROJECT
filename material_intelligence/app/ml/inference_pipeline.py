"""Combined YOLO detection + EfficientNet classification inspection pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from material_intelligence.app.config import get_settings
from material_intelligence.app.domain.enums import HIGH_RISK_CONDITIONS, InspectionStatus, ModelStatus
from material_intelligence.app.domain.schemas import (
    ClassificationResult,
    InspectionPrediction,
)
from material_intelligence.app.ml.brick_heuristic import suggest_brick_condition
from material_intelligence.app.ml.cement_bag_heuristic import filter_bag_boxes, suggest_cement_condition
from material_intelligence.app.ml.gradcam import GradCAMExplainer
from material_intelligence.app.ml.material_visual import guess_material_from_image
from material_intelligence.app.ml.model_registry import load_active_classifier, load_active_yolo
from material_intelligence.app.ml.preprocessing import preprocess_image, profile_from_dict
from material_intelligence.app.ml.rust_heuristic import rust_ratio, suggest_steel_condition
from material_intelligence.app.ml.yolo_detector import draw_detections
from material_intelligence.app.services.inspection_result_formatter import format_inspection_summary
from material_intelligence.app.services.material_resolver import (
    apply_resolved_material,
    resolve_material,
)


def recommend_status(classifications: list[ClassificationResult]) -> InspectionStatus:
    if not classifications:
        return InspectionStatus.HUMAN_INSPECTION_REQUIRED
    for result in classifications:
        if result.model_status != ModelStatus.READY:
            return InspectionStatus.HUMAN_INSPECTION_REQUIRED
        if result.message and "mismatch" in result.message.lower():
            return InspectionStatus.HUMAN_INSPECTION_REQUIRED
        if result.condition in HIGH_RISK_CONDITIONS:
            return InspectionStatus.ADDITIONAL_TESTING_REQUIRED
        if result.confidence is not None and result.confidence < 0.55:
            return InspectionStatus.HUMAN_INSPECTION_REQUIRED
        if result.message and "uncertain" in result.message.lower():
            return InspectionStatus.HUMAN_INSPECTION_REQUIRED
    return InspectionStatus.SCREENING_COMPLETED


def _blend_condition(
    classifications: list[ClassificationResult],
    *,
    material_key: str,
    condition: str,
    confidence: float,
    note: str,
) -> list[ClassificationResult]:
    """Always apply material-specific field check (brick edges / bag body / rebar rust)."""
    if not classifications:
        return [
            ClassificationResult(
                material_key=material_key,
                condition=condition,
                confidence=confidence,
                model_status=ModelStatus.READY,
                message=note,
            )
        ]
    primary = classifications[0]
    classifications[0] = primary.model_copy(
        update={
            "material_key": material_key,
            "condition": condition,
            "confidence": max(primary.confidence or 0.0, confidence),
            "message": note,
        }
    )
    return classifications


def run_inspection_pipeline(
    db: Session,
    image_bgr: np.ndarray,
    *,
    generate_gradcam: bool = True,
    expected_material_key: str | None = None,
    source_filename: str | None = None,
    preprocess_config: dict | None = None,
    output_dir: Path | None = None,
) -> tuple[InspectionPrediction, np.ndarray, Path | None]:
    """
    Returns prediction payload, annotated image, and optional Grad-CAM path.

    AI screening never equates to structural safety approval.
    """
    settings = get_settings()
    output_dir = output_dir or settings.upload_dir / "inspections"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    notes = [
        "Visual AI screening does not certify structural safety.",
        "Final approval requires an authorized engineer.",
    ]

    profile = profile_from_dict(preprocess_config)
    profile.resize_to = None
    processed = preprocess_image(image_bgr, profile).processed_bgr
    scan_image = processed if processed.size else image_bgr

    # Visual auto-detect early (fixes brick ↔ cement_bag confusion).
    visual_key, visual_conf, visual_scores = guess_material_from_image(scan_image)
    notes.append(
        "Visual scores: "
        + ", ".join(f"{k}={v:.0%}" for k, v in sorted(visual_scores.items(), key=lambda x: -x[1]))
    )

    detector = load_active_yolo(db)
    detection = detector.predict(scan_image)

    # Cement bags: keep bag-shaped boxes only (drop full-frame background).
    work_boxes = list(detection.boxes)
    if (expected_material_key == "cement_bag" or visual_key == "cement_bag") and work_boxes:
        filtered = filter_bag_boxes(work_boxes, scan_image.shape)
        if filtered:
            work_boxes = filtered
            detection = detection.model_copy(update={"boxes": work_boxes})
            notes.append("Cement bag scan: focused on bag body, not background.")

    annotated = draw_detections(scan_image, detection.boxes)
    if not detection.boxes:
        cv2.putText(
            annotated,
            "Full-frame scan (no boxes)",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    classifications: list[ClassificationResult] = []
    gradcam_path: Path | None = None
    # Prefer material-specific classifier when we already know the material.
    early_material = expected_material_key or visual_key
    classifier = load_active_classifier(db, early_material)

    if detection.model_status != ModelStatus.READY:
        notes.append(detection.message or "Detector not ready — using full-frame classification.")

    crops: list[tuple[str | None, np.ndarray]] = []
    if detection.boxes:
        h, w = scan_image.shape[:2]
        for box in detection.boxes:
            x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = scan_image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # Prefer visual/user material over wrong YOLO class name for the crop hint
            hint = early_material or box.class_name
            crops.append((hint, crop))
    if not crops:
        crops.append((early_material, scan_image))

    for material_hint, crop in crops:
        material_key = expected_material_key or material_hint
        result = classifier.predict(crop, expected_material_key=None)
        # Ignore material-mismatch gates from material-specific heads; we set material ourselves.
        if result.material_key is None and material_key:
            result = result.model_copy(update={"material_key": material_key})
        elif material_key and result.material_key and result.material_key != material_key:
            result = result.model_copy(update={"material_key": material_key, "message": None})
        classifications.append(result)
        if generate_gradcam and classifier.is_ready() and gradcam_path is None:
            explainer = GradCAMExplainer(classifier)
            explanation = explainer.explain(crop)
            gradcam_path = explainer.save_visualization(
                explanation["overlay_bgr"],
                output_dir / f"gradcam_{int(time.time())}.jpg",
            )
            notes.append(explanation["note"])

    if detection.boxes and classifier.is_ready():
        full = classifier.predict(scan_image, expected_material_key=None)
        if full.material_key is None and early_material:
            full = full.model_copy(update={"material_key": early_material})
        if full.condition and (
            not classifications
            or (full.confidence or 0) >= max((c.confidence or 0) for c in classifications)
        ):
            classifications.insert(0, full)

    material_key, material_source = resolve_material(
        expected_material_key=expected_material_key,
        filename=source_filename,
        boxes=detection.boxes,
        classifications=classifications,
        visual_material=visual_key,
        visual_confidence=visual_conf,
    )
    field_rust = rust_ratio(scan_image)
    if material_key is None and field_rust >= 0.12:
        material_key, material_source = "steel_rebar", "field rust pattern"
    # Only override brick→rebar when rust is heavy and detector did not say brick strongly.
    det_brick = any(
        (b.class_name or "").replace(" ", "_") == "clay_brick" and (b.confidence or 0) >= 0.6
        for b in detection.boxes
    )
    if (
        expected_material_key is None
        and field_rust >= 0.18
        and material_key == "clay_brick"
        and not det_brick
    ):
        material_key, material_source = "steel_rebar", "heavy rust pattern"

    classifications = apply_resolved_material(classifications, material_key)
    if material_key:
        notes.append(f"Material resolved as {material_key} via {material_source}.")

    # Material-specific field checks (easy, reliable for site photos).
    if material_key == "steel_rebar":
        rust_cond, rust_conf, rust_note = suggest_steel_condition(scan_image)
        use_rust = field_rust >= 0.10 or (classifications and (classifications[0].confidence or 0) < 0.55)
        if use_rust:
            classifications = _blend_condition(
                classifications,
                material_key="steel_rebar",
                condition=rust_cond,
                confidence=rust_conf,
                note=rust_note,
            )
            notes.append(rust_note)

    elif material_key == "clay_brick":
        b_cond, b_conf, b_note = suggest_brick_condition(scan_image)
        classifications = _blend_condition(
            classifications,
            material_key="clay_brick",
            condition=b_cond,
            confidence=b_conf,
            note=b_note,
        )
        notes.append(b_note)

    elif material_key == "cement_bag":
        c_cond, c_conf, c_note = suggest_cement_condition(scan_image)
        classifications = _blend_condition(
            classifications,
            material_key="cement_bag",
            condition=c_cond,
            confidence=c_conf,
            note=c_note,
        )
        notes.append(c_note)

    status = recommend_status(classifications)
    elapsed = (time.perf_counter() - start) * 1000
    prediction = InspectionPrediction(
        detection=detection,
        classifications=classifications,
        recommended_status=status,
        processing_ms=elapsed,
        gradcam_path=str(gradcam_path) if gradcam_path else None,
        notes=notes,
    )
    summary = format_inspection_summary(
        prediction, expected_material=material_key or expected_material_key
    )
    prediction.notes = [summary, *notes]
    return prediction, annotated, gradcam_path


def save_image(path: Path, image_bgr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)
    return path
