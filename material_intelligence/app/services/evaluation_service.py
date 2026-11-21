"""Orchestration service for research evaluation experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from material_intelligence.app.db.models import Dataset, ModelArtifact
from material_intelligence.app.domain.enums import ModelType
from material_intelligence.app.ml.efficientnet_classifier import EfficientNetClassifier
from material_intelligence.app.ml.evaluation.classification import evaluate_classifier_on_folder
from material_intelligence.app.ml.evaluation.clustering import run_dbscan, run_kmeans
from material_intelligence.app.ml.evaluation.detection import evaluate_yolo_checkpoint
from material_intelligence.app.ml.evaluation.experiment_manager import create_experiment
from material_intelligence.app.ml.evaluation.recommendations import recommend_from_prediction
from material_intelligence.app.ml.evaluation.robustness import evaluate_classifier_robustness
from material_intelligence.app.ml.features.efficientnet_embeddings import (
    embeddings_matrix,
    extract_embeddings_for_folder,
)
from material_intelligence.app.ml.inference_pipeline import run_inspection_pipeline
from material_intelligence.app.ml.preprocessing import load_bgr_image


def run_classification_experiment(
    db: Session,
    *,
    model_id: int,
    dataset_id: int,
    researcher_id: int | None = None,
    split: str = "test",
    name: str | None = None,
) -> dict[str, Any]:
    artifact = db.get(ModelArtifact, model_id)
    dataset = db.get(Dataset, dataset_id)
    if artifact is None or dataset is None:
        raise ValueError("Model or dataset not found.")
    if artifact.model_type != ModelType.EFFICIENTNET_B0.value:
        raise ValueError("Classification evaluation requires an EfficientNet-B0 artifact.")
    if not artifact.artifact_path or not Path(artifact.artifact_path).exists():
        raise ValueError("Model checkpoint missing on disk.")

    clf = EfficientNetClassifier(artifact.artifact_path)
    train_root = None
    if artifact.dataset_id:
        train_ds = db.get(Dataset, artifact.dataset_id)
        if train_ds:
            train_root = Path(train_ds.root_path)

    started = time.perf_counter()
    result = evaluate_classifier_on_folder(
        clf,
        Path(dataset.root_path),
        train_root=train_root,
        split=split,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    metrics_payload = result.metrics.model_dump()
    metrics_payload["metrics"]["avg_inference_latency_ms"] = latency_ms / max(
        result.metrics.metrics.get("n_samples") or 1, 1
    )

    exp = create_experiment(
        db,
        experiment_type="classification",
        name=name or f"clf_{artifact.version}_{dataset.version}",
        model_id=artifact.id,
        dataset_id=dataset.id,
        dataset_version=dataset.version,
        model_version=artifact.version,
        checkpoint_path=artifact.artifact_path,
        preprocess_config=artifact.preprocess_config,
        model_config={"split": split, "task": "condition_classification"},
        metrics={
            **metrics_payload,
            "leakage": result.leakage,
            "mapping_check": result.mapping_check,
            "curves": result.curves,
            "error_bucket_counts": {k: len(v) for k, v in result.error_buckets.items()},
        },
        artifacts={
            "error_buckets": {
                k: v[:30] for k, v in result.error_buckets.items()
            },
            "sample_preview": [
                {
                    "path": s.path,
                    "true_label": s.true_label,
                    "pred_label": s.pred_label,
                    "confidence": s.confidence,
                    "correct": s.correct,
                }
                for s in result.samples[:50]
            ],
        },
        ground_truth_source=f"dataset folder labels @ {dataset.root_path}",
        n_images=int(result.metrics.metrics.get("n_samples") or 0),
        researcher_id=researcher_id,
        notes="; ".join(result.metrics.notes) if result.metrics.notes else None,
    )
    return {"experiment_id": exp.id, "metrics": metrics_payload, "leakage": result.leakage}


def run_detection_experiment(
    db: Session,
    *,
    model_id: int,
    dataset_id: int,
    conf: float = 0.25,
    iou: float = 0.5,
    researcher_id: int | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    artifact = db.get(ModelArtifact, model_id)
    dataset = db.get(Dataset, dataset_id)
    if artifact is None or dataset is None:
        raise ValueError("Model or dataset not found.")
    if artifact.model_type != ModelType.YOLOV8.value:
        raise ValueError("Detection evaluation requires a YOLOv8 artifact.")
    root = Path(dataset.root_path)
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        data_yaml = root / "dataset.yaml"
    metrics = evaluate_yolo_checkpoint(
        artifact.artifact_path or "",
        data_yaml,
        model_version=artifact.version,
        conf=conf,
        iou=iou,
    )
    exp = create_experiment(
        db,
        experiment_type="detection",
        name=name or f"yolo_{artifact.version}_{dataset.version}",
        model_id=artifact.id,
        dataset_id=dataset.id,
        dataset_version=dataset.version,
        model_version=artifact.version,
        checkpoint_path=artifact.artifact_path,
        preprocess_config=artifact.preprocess_config,
        model_config={"conf": conf, "iou": iou, "split": "test"},
        metrics=metrics.model_dump(),
        ground_truth_source=str(data_yaml),
        n_images=dataset.image_count or 0,
        researcher_id=researcher_id,
        notes="; ".join(metrics.notes) if metrics.notes else None,
    )
    return {"experiment_id": exp.id, "metrics": metrics.model_dump()}


def run_clustering_experiment(
    db: Session,
    *,
    model_id: int,
    dataset_id: int,
    algorithm: str = "kmeans",
    n_clusters: int = 3,
    eps: float = 0.8,
    min_samples: int = 5,
    material_filter: str | None = None,
    researcher_id: int | None = None,
    name: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    artifact = db.get(ModelArtifact, model_id)
    dataset = db.get(Dataset, dataset_id)
    if artifact is None or dataset is None:
        raise ValueError("Model or dataset not found.")
    clf = EfficientNetClassifier(artifact.artifact_path)
    if not clf.is_ready():
        raise ValueError("Classifier not ready for embedding extraction.")
    records = extract_embeddings_for_folder(
        clf,
        Path(dataset.root_path),
        material_type=material_filter,
        limit=limit,
    )
    matrix = embeddings_matrix(records)
    paths = [r.path for r in records]
    materials = [r.material_type for r in records]
    expert = []
    for r in records:
        # Parent folder as exploratory expert/condition label when present
        expert.append(Path(r.path).parent.name)

    if algorithm == "dbscan":
        result = run_dbscan(
            matrix,
            eps=eps,
            min_samples=min_samples,
            paths=paths,
            materials=materials,
            expert_labels=expert,
        )
    else:
        result = run_kmeans(
            matrix,
            n_clusters=n_clusters,
            paths=paths,
            materials=materials,
            expert_labels=expert,
        )

    exp = create_experiment(
        db,
        experiment_type="clustering",
        name=name or f"cluster_{algorithm}_{artifact.version}_{dataset.version}",
        model_id=artifact.id,
        dataset_id=dataset.id,
        dataset_version=dataset.version,
        model_version=artifact.version,
        checkpoint_path=artifact.artifact_path,
        preprocess_config={"embedding": "efficientnet_b0_pooled", "material_filter": material_filter},
        model_config={
            "algorithm": algorithm,
            "n_clusters": n_clusters,
            "eps": eps,
            "min_samples": min_samples,
        },
        metrics=result.metrics,
        artifacts={
            "projection_2d": result.projection_2d,
            "labels": result.labels,
            "paths": paths,
            "materials": materials,
            "expert_labels": expert,
            "cluster_summaries": result.cluster_summaries,
            "notes": result.notes,
            "preprocessing": result.preprocessing,
        },
        ground_truth_source="folder names (exploratory external comparison only)",
        n_images=len(records),
        researcher_id=researcher_id,
        notes="; ".join(result.notes),
    )
    return {"experiment_id": exp.id, "metrics": result.metrics, "summaries": result.cluster_summaries}


def run_robustness_experiment(
    db: Session,
    *,
    model_id: int,
    dataset_id: int,
    researcher_id: int | None = None,
    limit: int = 40,
) -> dict[str, Any]:
    artifact = db.get(ModelArtifact, model_id)
    dataset = db.get(Dataset, dataset_id)
    if artifact is None or dataset is None:
        raise ValueError("Model or dataset not found.")
    clf = EfficientNetClassifier(artifact.artifact_path)
    root = Path(dataset.root_path)
    paths: list[str] = []
    labels: list[str] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in class_dir.rglob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                paths.append(str(img))
                labels.append(class_dir.name)
    result = evaluate_classifier_robustness(clf, paths, labels, limit=limit)
    exp = create_experiment(
        db,
        experiment_type="robustness",
        name=f"robust_{artifact.version}_{dataset.version}",
        model_id=artifact.id,
        dataset_id=dataset.id,
        dataset_version=dataset.version,
        model_version=artifact.version,
        checkpoint_path=artifact.artifact_path,
        preprocess_config=artifact.preprocess_config,
        model_config={"limit": limit},
        metrics={
            "baseline_accuracy": result.baseline_accuracy,
            "by_transform": result.by_transform,
        },
        artifacts={"notes": result.notes},
        ground_truth_source="dataset folder labels",
        n_images=min(limit, len(paths)),
        researcher_id=researcher_id,
        notes="; ".join(result.notes),
    )
    return {"experiment_id": exp.id, "metrics": {"baseline_accuracy": result.baseline_accuracy, "by_transform": result.by_transform}}


def run_live_laboratory(
    db: Session,
    *,
    image_path: str,
    generate_gradcam: bool = True,
    expected_material_key: str | None = None,
    ground_truth_label: str | None = None,
) -> dict[str, Any]:
    """End-to-end workflow validation for a single image."""
    image = load_bgr_image(image_path)
    stages: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    stages.append({"step": 1, "name": "load_image", "status": "ok", "ms": 0, "output": image_path})

    prediction, annotated, gradcam_path = run_inspection_pipeline(
        db,
        image,
        generate_gradcam=generate_gradcam,
        expected_material_key=expected_material_key,
        source_filename=Path(image_path).name,
    )
    pipeline_ms = (time.perf_counter() - t0) * 1000
    stages.append(
        {
            "step": "2-8",
            "name": "yolo_plus_efficientnet_pipeline",
            "status": "ok",
            "ms": pipeline_ms,
            "output": {
                "detections": len(prediction.detection.boxes),
                "classifications": [c.model_dump() for c in prediction.classifications],
                "detector_status": prediction.detection.model_status.value,
            },
        }
    )

    rec = recommend_from_prediction(prediction)
    stages.append(
        {
            "step": 11,
            "name": "recommendation",
            "status": "ok",
            "ms": 0,
            "output": {
                "recommendation": rec.recommendation,
                "visual_condition": rec.visual_condition,
                "reason": rec.reason,
            },
        }
    )

    correctness = None
    if ground_truth_label and prediction.classifications:
        pred = prediction.classifications[0].condition
        correctness = {
            "ground_truth": ground_truth_label,
            "prediction": pred,
            "correct": pred == ground_truth_label,
        }
        stages.append(
            {
                "step": 12,
                "name": "expert_comparison",
                "status": "ok",
                "ms": 0,
                "output": correctness,
            }
        )
    else:
        stages.append(
            {
                "step": 12,
                "name": "expert_comparison",
                "status": "skipped",
                "ms": 0,
                "output": "Ground truth unavailable — inference only, no correctness score.",
            }
        )

    return {
        "stages": stages,
        "prediction": prediction.model_dump(),
        "recommendation": {
            "code": rec.recommendation,
            "visual_condition": rec.visual_condition,
            "reason": rec.reason,
            "requires_engineer": rec.requires_engineer,
        },
        "annotated_note": "Annotated image generated in-memory; persist via inspection workflow if needed.",
        "gradcam_path": str(gradcam_path) if gradcam_path else None,
        "correctness": correctness,
        "disclaimer": "AI screening does not certify structural safety.",
        "detection_boxes": [b.model_dump() for b in prediction.detection.boxes],
    }
