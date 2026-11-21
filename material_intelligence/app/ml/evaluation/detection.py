"""YOLOv8 localization evaluation adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from material_intelligence.app.domain.enums import ModelType
from material_intelligence.app.domain.schemas import EvaluationMetrics
from material_intelligence.app.ml.evaluation.statistics import safe_metric
from material_intelligence.app.ml.yolo_detector import YOLODetector


def yolo_metrics_from_ultralytics(
    val_result: dict[str, Any],
    *,
    model_version: str,
    split: str = "test",
) -> EvaluationMetrics:
    if val_result.get("status") == "not_evaluated":
        return EvaluationMetrics(
            model_type=ModelType.YOLOV8,
            model_version=model_version,
            split=split,
            metrics={"status": "not_evaluated"},
            notes=[val_result.get("message", "Model Not Trained")],
        )
    metrics = {
        "status": "evaluated",
        "map50": safe_metric(val_result.get("map50")),
        "map50_95": safe_metric(val_result.get("map50_95")),
        "precision": safe_metric(val_result.get("precision")),
        "recall": safe_metric(val_result.get("recall")),
        "protocol": val_result.get(
            "protocol",
            "Ultralytics val() default IoU schedule; confidence threshold as configured in YOLO settings.",
        ),
    }
    return EvaluationMetrics(
        model_type=ModelType.YOLOV8,
        model_version=model_version,
        split=split,
        metrics=metrics,
        per_class=val_result.get("per_class", {}),
        class_names=val_result.get("class_names", []),
        notes=val_result.get("notes", []),
    )


def evaluate_yolo_checkpoint(
    weights_path: str | Path,
    data_yaml: str | Path,
    *,
    model_version: str | None = None,
    conf: float = 0.25,
    iou: float = 0.5,
    device: str = "cpu",
    split: str = "test",
) -> EvaluationMetrics:
    detector = YOLODetector(weights_path, conf_threshold=conf, device=device)
    if not detector.is_ready():
        return yolo_metrics_from_ultralytics(
            {"status": "not_evaluated", "message": "Model Not Trained"},
            model_version=model_version or Path(weights_path).stem,
            split=split,
        )
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        return EvaluationMetrics(
            model_type=ModelType.YOLOV8,
            model_version=model_version or detector.version or "unknown",
            split=split,
            metrics={"status": "not_evaluated"},
            notes=[f"YOLO data yaml not found: {yaml_path}"],
        )
    # Ultralytics val uses its own IoU schedule for mAP@50:95; conf affects predictions.
    assert detector._model is not None
    metrics_obj = detector._model.val(
        data=str(yaml_path),
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    out: dict[str, Any] = {
        "status": "evaluated",
        "protocol": f"Ultralytics val conf={conf}, iou={iou}",
        "notes": [
            "mAP is an object-detection metric; do not substitute classification accuracy."
        ],
    }
    box = getattr(metrics_obj, "box", None)
    if box is not None:
        out["map50"] = float(getattr(box, "map50", float("nan")))
        out["map50_95"] = float(getattr(box, "map", float("nan")))
        out["precision"] = float(getattr(box, "mp", float("nan")))
        out["recall"] = float(getattr(box, "mr", float("nan")))
        names = getattr(metrics_obj, "names", None) or {}
        if isinstance(names, dict):
            out["class_names"] = [names[i] for i in sorted(names)]
        # Per-class AP when available
        ap50 = getattr(box, "ap50", None)
        if ap50 is not None:
            try:
                values = list(ap50)
                out["per_class"] = {
                    (out.get("class_names") or [str(i)])[i]
                    if i < len(out.get("class_names") or [])
                    else str(i): {"ap50": safe_metric(float(values[i]))}
                    for i in range(len(values))
                }
            except TypeError:
                pass
    return yolo_metrics_from_ultralytics(
        out, model_version=model_version or detector.version or "unknown", split=split
    )
