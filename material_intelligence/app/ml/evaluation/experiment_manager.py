"""Persist and compare evaluation experiments."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import EvaluationExperiment, EvaluationRecord, ModelArtifact


def create_experiment(
    db: Session,
    *,
    experiment_type: str,
    name: str,
    model_id: int | None,
    dataset_id: int | None,
    dataset_version: str | None,
    model_version: str | None,
    checkpoint_path: str | None,
    preprocess_config: dict | None,
    model_config: dict | None,
    metrics: dict,
    artifacts: dict | None = None,
    ground_truth_source: str | None = None,
    n_images: int = 0,
    seed: int | None = 42,
    status: str = "completed",
    researcher_id: int | None = None,
    notes: str | None = None,
) -> EvaluationExperiment:
    exp = EvaluationExperiment(
        name=name,
        experiment_type=experiment_type,
        status=status,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        model_id=model_id,
        model_version=model_version,
        checkpoint_path=checkpoint_path,
        preprocess_config=preprocess_config or {},
        model_config=model_config or {},
        seed=seed,
        n_images=n_images,
        ground_truth_source=ground_truth_source,
        metrics=metrics,
        artifacts=artifacts or {},
        researcher_id=researcher_id,
        notes=notes,
        finished_at=datetime.now(UTC) if status == "completed" else None,
    )
    db.add(exp)
    if model_id and experiment_type in {"classification", "detection"}:
        record = EvaluationRecord(
            model_id=model_id,
            split=str((model_config or {}).get("split", "test")),
            metrics=metrics,
            per_class=(metrics.get("per_class") if isinstance(metrics, dict) else None),
            confusion_matrix=(
                metrics.get("confusion_matrix") if isinstance(metrics, dict) else None
            ),
            class_names=(metrics.get("class_names") if isinstance(metrics, dict) else None),
            notes=notes,
        )
        # Flatten nested EvaluationMetrics-style payloads
        if "metrics" in metrics and isinstance(metrics["metrics"], dict):
            record.metrics = metrics["metrics"]
            record.per_class = metrics.get("per_class")
            record.confusion_matrix = metrics.get("confusion_matrix")
            record.class_names = metrics.get("class_names")
        db.add(record)
        artifact = db.get(ModelArtifact, model_id)
        if artifact is not None:
            artifact.metrics = {
                **(artifact.metrics or {}),
                "last_evaluation": record.metrics,
                "last_experiment_id": None,
            }
    db.commit()
    db.refresh(exp)
    if model_id:
        artifact = db.get(ModelArtifact, model_id)
        if artifact is not None:
            artifact.metrics = {**(artifact.metrics or {}), "last_experiment_id": exp.id}
            db.commit()
    # Persist JSON sidecar for reproducibility
    settings = get_settings()
    out_dir = settings.report_dir / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = out_dir / f"experiment_{exp.id}.json"
    sidecar.write_text(
        json.dumps(
            {
                "id": exp.id,
                "name": exp.name,
                "experiment_type": exp.experiment_type,
                "dataset_version": exp.dataset_version,
                "model_version": exp.model_version,
                "checkpoint_path": exp.checkpoint_path,
                "seed": exp.seed,
                "metrics": exp.metrics,
                "preprocess_config": exp.preprocess_config,
                "model_config": exp.model_config,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return exp


def list_experiments(
    db: Session,
    *,
    experiment_type: str | None = None,
    limit: int = 50,
) -> list[EvaluationExperiment]:
    stmt = select(EvaluationExperiment).order_by(EvaluationExperiment.created_at.desc())
    if experiment_type:
        stmt = stmt.where(EvaluationExperiment.experiment_type == experiment_type)
    return list(db.scalars(stmt.limit(limit)).all())


def compare_experiments(db: Session, experiment_ids: list[int]) -> dict[str, Any]:
    rows = []
    for exp_id in experiment_ids:
        exp = db.get(EvaluationExperiment, exp_id)
        if exp is None:
            continue
        rows.append(
            {
                "id": exp.id,
                "name": exp.name,
                "type": exp.experiment_type,
                "model_version": exp.model_version,
                "dataset_version": exp.dataset_version,
                "n_images": exp.n_images,
                "metrics": exp.metrics,
                "seed": exp.seed,
            }
        )
    return {
        "experiments": rows,
        "warning": "Only compare compatible metrics for the same task type.",
    }


def latest_kpi_snapshot(db: Session) -> dict[str, Any]:
    """Dashboard KPIs from saved experiments — never fabricate values."""
    def _latest(exp_type: str) -> EvaluationExperiment | None:
        return db.scalar(
            select(EvaluationExperiment)
            .where(
                EvaluationExperiment.experiment_type == exp_type,
                EvaluationExperiment.status == "completed",
            )
            .order_by(EvaluationExperiment.created_at.desc())
        )

    clf = _latest("classification")
    det = _latest("detection")
    clus = _latest("clustering")

    def metric(exp: EvaluationExperiment | None, *keys: str) -> Any:
        if exp is None:
            return "Not Evaluated"
        payload = exp.metrics or {}
        # Support nested {"metrics": {...}}
        cursor: Any = payload.get("metrics", payload)
        for key in keys:
            if not isinstance(cursor, dict) or key not in cursor:
                return "Not Evaluated"
            cursor = cursor[key]
        return cursor if cursor is not None else "Not Evaluated"

    n_images = 0
    for exp in (clf, det, clus):
        if exp and exp.n_images:
            n_images = max(n_images, exp.n_images)

    return {
        "classification_accuracy": metric(clf, "accuracy"),
        "macro_f1": metric(clf, "macro_f1"),
        "yolo_map50": metric(det, "map50"),
        "yolo_map50_95": metric(det, "map50_95"),
        "silhouette": metric(clus, "silhouette") if clus else "Not Evaluated",
        "n_evaluated_images": n_images or "Not Evaluated",
        "false_negative_rate": metric(clf, "false_negative_rate"),
        "avg_inference_latency_ms": metric(clf, "avg_inference_latency_ms"),
        "sources": {
            "classification_experiment_id": clf.id if clf else None,
            "detection_experiment_id": det.id if det else None,
            "clustering_experiment_id": clus.id if clus else None,
        },
    }
