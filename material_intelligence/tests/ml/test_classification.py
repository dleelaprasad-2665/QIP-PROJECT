"""Classification metric tests with synthetic fixtures."""

from __future__ import annotations

import numpy as np

from material_intelligence.app.ml.evaluation.classification import (
    build_error_buckets,
    compute_classification_metrics,
)
from material_intelligence.app.ml.evaluation.classification import ClassificationSampleResult
from material_intelligence.app.ml.evaluation.leakage import validate_class_mapping_compatibility


def test_perfect_accuracy_and_confusion_order():
    names = ["a", "b"]
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred, class_names=names)
    assert metrics.metrics["accuracy"] == 1.0
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]
    assert metrics.class_names == names


def test_absent_class_zero_division_safe():
    names = ["a", "b", "c"]
    y_true = [0, 0, 1]
    y_pred = [0, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred, class_names=names)
    assert metrics.per_class["c"]["support"] == 0
    assert metrics.metrics["macro_f1"] is not None


def test_empty_predictions_not_evaluated():
    metrics = compute_classification_metrics([], [], class_names=["a"])
    assert metrics.metrics["status"] == "not_evaluated"


def test_mapping_incompatible():
    check = validate_class_mapping_compatibility(["cracked"], ["steel_only"])
    assert check["compatible"] is False


def test_error_buckets_prioritize_damage_as_good():
    samples = [
        ClassificationSampleResult(
            path="x.jpg",
            y_true=0,
            y_pred=1,
            true_label="cracked",
            pred_label="no_visible_damage",
            confidence=0.9,
            probabilities={},
            correct=False,
        )
    ]
    buckets = build_error_buckets(samples)
    assert len(buckets["false_negative_damage"]) == 1
    assert len(buckets["high_confidence_incorrect"]) == 1
