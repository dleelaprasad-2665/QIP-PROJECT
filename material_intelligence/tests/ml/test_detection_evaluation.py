"""Detection evaluation adapter tests."""

from __future__ import annotations

from material_intelligence.app.ml.evaluation.detection import yolo_metrics_from_ultralytics


def test_yolo_not_evaluated_passthrough():
    metrics = yolo_metrics_from_ultralytics(
        {"status": "not_evaluated", "message": "Model Not Trained"},
        model_version="x",
    )
    assert metrics.metrics["status"] == "not_evaluated"
    assert "Model Not Trained" in (metrics.notes[0] if metrics.notes else "")


def test_yolo_evaluated_maps_fields():
    metrics = yolo_metrics_from_ultralytics(
        {
            "status": "evaluated",
            "map50": 0.5,
            "map50_95": 0.3,
            "precision": 0.4,
            "recall": 0.6,
            "protocol": "test-protocol",
        },
        model_version="yolo-v",
    )
    assert metrics.metrics["map50"] == 0.5
    assert metrics.metrics["map50_95"] == 0.3
    assert metrics.metrics["protocol"] == "test-protocol"
