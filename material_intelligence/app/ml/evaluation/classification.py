"""Supervised classification evaluation with real metrics and error analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from material_intelligence.app.domain.enums import ModelType
from material_intelligence.app.domain.schemas import EvaluationMetrics
from material_intelligence.app.ml.efficientnet_classifier import (
    EfficientNetClassifier,
    FolderClassificationDataset,
    default_transforms,
)
from material_intelligence.app.ml.evaluation.leakage import (
    detect_hash_overlap,
    validate_class_mapping_compatibility,
)
from material_intelligence.app.ml.evaluation.statistics import (
    bootstrap_ci,
    brier_score_multiclass,
    expected_calibration_error,
    safe_metric,
)


@dataclass
class ClassificationSampleResult:
    path: str
    y_true: int
    y_pred: int
    true_label: str
    pred_label: str
    confidence: float
    probabilities: dict[str, float]
    correct: bool


@dataclass
class ClassificationEvalResult:
    metrics: EvaluationMetrics
    samples: list[ClassificationSampleResult] = field(default_factory=list)
    leakage: dict[str, Any] = field(default_factory=dict)
    mapping_check: dict[str, Any] = field(default_factory=dict)
    curves: dict[str, Any] = field(default_factory=dict)
    error_buckets: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    *,
    class_names: list[str],
    y_proba: np.ndarray | None = None,
    model_version: str = "unknown",
    split: str = "test",
) -> EvaluationMetrics:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    if len(y_true_arr) == 0:
        return EvaluationMetrics(
            model_type=ModelType.EFFICIENTNET_B0,
            model_version=model_version,
            split=split,
            metrics={"status": "not_evaluated"},
            class_names=class_names,
            notes=["No predictions available."],
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )
    per_class = {
        class_names[i]: {
            "precision": safe_metric(precision[i]),
            "recall": safe_metric(recall[i]),
            "f1": safe_metric(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }
    correct = (y_true_arr == y_pred_arr).astype(float)
    metrics: dict[str, Any] = {
        "status": "evaluated",
        "n_samples": int(len(y_true_arr)),
        "accuracy": safe_metric(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": safe_metric(float(np.mean(precision))),
        "macro_recall": safe_metric(float(np.mean(recall))),
        "macro_f1": safe_metric(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": safe_metric(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "false_negative_rate": _false_negative_rate(y_true_arr, y_pred_arr, class_names),
        "accuracy_ci": bootstrap_ci(correct.tolist()),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }
    notes: list[str] = []
    if y_proba is not None:
        metrics["ece"] = safe_metric(expected_calibration_error(y_true_arr, y_proba))
        metrics["brier_score"] = safe_metric(
            brier_score_multiclass(y_true_arr, y_proba, len(class_names))
        )
        if len(class_names) == 2:
            try:
                metrics["roc_auc"] = safe_metric(roc_auc_score(y_true_arr, y_proba[:, 1]))
            except ValueError as exc:
                notes.append(f"ROC-AUC unavailable: {exc}")
            try:
                metrics["average_precision"] = safe_metric(
                    average_precision_score(y_true_arr, y_proba[:, 1])
                )
            except ValueError as exc:
                notes.append(f"Average precision unavailable: {exc}")

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    cm_norm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels, normalize="true")
    return EvaluationMetrics(
        model_type=ModelType.EFFICIENTNET_B0,
        model_version=model_version,
        split=split,
        metrics={**metrics, "confusion_matrix_normalized": cm_norm.tolist()},
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        class_names=class_names,
        notes=notes,
    )


def _false_negative_rate(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> float | None:
    """Rate of damaged/defect classes predicted as no_visible_damage."""
    if "no_visible_damage" not in class_names:
        return None
    good_idx = class_names.index("no_visible_damage")
    damaged_mask = y_true != good_idx
    if not np.any(damaged_mask):
        return None
    return float(np.mean(y_pred[damaged_mask] == good_idx))


def build_error_buckets(samples: list[ClassificationSampleResult]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {
        "correct": [],
        "false_positive_good": [],
        "false_negative_damage": [],
        "low_confidence": [],
        "high_confidence_incorrect": [],
        "all_errors": [],
    }
    for s in samples:
        row = {
            "path": s.path,
            "true_label": s.true_label,
            "pred_label": s.pred_label,
            "confidence": s.confidence,
            "correct": s.correct,
        }
        if s.correct:
            buckets["correct"].append(row)
        else:
            buckets["all_errors"].append(row)
            if s.pred_label == "no_visible_damage" and s.true_label != "no_visible_damage":
                buckets["false_negative_damage"].append(row)
            if s.true_label == "no_visible_damage" and s.pred_label != "no_visible_damage":
                buckets["false_positive_good"].append(row)
            if s.confidence >= 0.8:
                buckets["high_confidence_incorrect"].append(row)
        if s.confidence < 0.45:
            buckets["low_confidence"].append(row)
    return buckets


def _curve_payloads(
    y_true: np.ndarray, y_proba: np.ndarray | None, class_names: list[str]
) -> dict[str, Any]:
    if y_proba is None or len(class_names) != 2:
        return {}
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        return {
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr": {"precision": precision.tolist(), "recall": recall.tolist()},
        }
    except ValueError:
        return {}


def evaluate_classifier_on_folder(
    classifier: EfficientNetClassifier,
    dataset_root: Path,
    *,
    train_root: Path | None = None,
    split: str = "test",
    batch_size: int = 8,
    require_compatible_mapping: bool = True,
) -> ClassificationEvalResult:
    if not classifier.is_ready() or classifier.model is None:
        empty = compute_classification_metrics([], [], class_names=[], model_version="n/a", split=split)
        return ClassificationEvalResult(
            metrics=empty,
            mapping_check={"compatible": False, "message": "Classifier not ready."},
        )

    class_names = [classifier.idx_to_class[i] for i in range(len(classifier.class_mapping))]
    dataset_classes = [
        p.name for p in sorted(dataset_root.iterdir()) if p.is_dir()
    ] if dataset_root.exists() else []
    mapping_check = validate_class_mapping_compatibility(class_names, dataset_classes)
    if require_compatible_mapping and not mapping_check["compatible"]:
        metrics = compute_classification_metrics(
            [], [], class_names=class_names, model_version=classifier.version or "unknown", split=split
        )
        metrics.notes.append(mapping_check["message"])
        metrics.metrics["status"] = "refused_incompatible_mapping"
        return ClassificationEvalResult(metrics=metrics, mapping_check=mapping_check)

    leakage = detect_hash_overlap(train_root, dataset_root)
    ds = FolderClassificationDataset(
        dataset_root,
        classifier.class_mapping,
        transform=default_transforms(classifier.image_size, train=False),
    )
    if len(ds) == 0:
        metrics = compute_classification_metrics(
            [], [], class_names=class_names, model_version=classifier.version or "unknown", split=split
        )
        metrics.notes.append("No evaluation images found for model class folders.")
        return ClassificationEvalResult(metrics=metrics, leakage=leakage, mapping_check=mapping_check)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba_batches: list[np.ndarray] = []
    samples: list[ClassificationSampleResult] = []
    sample_offset = 0

    classifier.model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            outputs = classifier.model(images.to(classifier.device))
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            batch_true = labels.tolist()
            batch_pred = preds.cpu().tolist()
            batch_proba = probs.cpu().numpy()
            y_true.extend(batch_true)
            y_pred.extend(batch_pred)
            y_proba_batches.append(batch_proba)
            for i, (yt, yp) in enumerate(zip(batch_true, batch_pred, strict=True)):
                path = str(ds.samples[sample_offset + i][0])
                conf = float(batch_proba[i, yp])
                samples.append(
                    ClassificationSampleResult(
                        path=path,
                        y_true=int(yt),
                        y_pred=int(yp),
                        true_label=class_names[int(yt)],
                        pred_label=class_names[int(yp)],
                        confidence=conf,
                        probabilities={
                            class_names[j]: float(batch_proba[i, j]) for j in range(len(class_names))
                        },
                        correct=int(yt) == int(yp),
                    )
                )
            sample_offset += len(batch_true)

    proba = np.concatenate(y_proba_batches, axis=0) if y_proba_batches else None
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        class_names=class_names,
        y_proba=proba,
        model_version=classifier.version or "unknown",
        split=split,
    )
    if leakage.get("status") == "flagged":
        metrics.notes.append(leakage["message"])
    return ClassificationEvalResult(
        metrics=metrics,
        samples=samples,
        leakage=leakage,
        mapping_check=mapping_check,
        curves=_curve_payloads(np.asarray(y_true), proba, class_names),
        error_buckets=build_error_buckets(samples),
    )


# Backwards-compatible alias used by older imports.
classification_metrics = compute_classification_metrics
