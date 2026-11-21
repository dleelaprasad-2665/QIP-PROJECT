"""Metric calculation tests with known examples."""

from __future__ import annotations

from material_intelligence.app.ml.evaluation import classification_metrics


def test_classification_metrics_known_example():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    metrics = classification_metrics(
        y_true,
        y_pred,
        class_names=["ok", "defect"],
        model_version="unit",
    )
    assert metrics.metrics["accuracy"] == 0.75
    assert metrics.confusion_matrix is not None
    assert metrics.confusion_matrix[0][0] == 1
    assert "ok" in metrics.per_class
