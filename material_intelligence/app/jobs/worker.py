"""Simple persistent job queue using a background thread (local development)."""

from __future__ import annotations

import json
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from sqlalchemy import select

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import Dataset, ModelArtifact, TrainingJob
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.domain.enums import JobStatus, ModelStatus, ModelType
from material_intelligence.app.domain.schemas import TrainingJobConfig
from material_intelligence.app.ml.training import run_training
from material_intelligence.app.utils.logging import get_logger

logger = get_logger(__name__)

_worker_started = False
_lock = threading.Lock()


def enqueue_training_job(job_id: int) -> None:
    """Mark job queued; the worker loop will pick it up."""
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if job is None:
            return
        job.status = JobStatus.QUEUED.value
        job.message = "Queued"
        db.commit()
    _ensure_worker()


def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if _worker_started:
            return
        thread = threading.Thread(target=_worker_loop, name="mi-training-worker", daemon=True)
        thread.start()
        _worker_started = True
        logger.info("Training worker started")


def _worker_loop() -> None:
    import time

    while True:
        job_id = _claim_next_job()
        if job_id is None:
            time.sleep(2.0)
            continue
        _execute_job(job_id)


def _claim_next_job() -> int | None:
    with SessionLocal() as db:
        job = db.scalar(
            select(TrainingJob)
            .where(TrainingJob.status == JobStatus.QUEUED.value)
            .order_by(TrainingJob.created_at.asc())
        )
        if job is None:
            return None
        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.now(UTC)
        job.progress = 0.05
        job.message = "Running"
        db.commit()
        return job.id


def _execute_job(job_id: int) -> None:
    settings = get_settings()
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if job is None:
            return
        dataset = db.get(Dataset, job.dataset_id)
        if dataset is None:
            job.status = JobStatus.FAILED.value
            job.message = "Dataset not found"
            job.finished_at = datetime.now(UTC)
            db.commit()
            return

        log_path = settings.job_dir / f"job_{job_id}.log"
        job.log_path = str(log_path)
        db.commit()
        config = TrainingJobConfig.model_validate(job.config or {})

        def progress(value: float, message: str) -> None:
            with SessionLocal() as progress_db:
                row = progress_db.get(TrainingJob, job_id)
                if row is None:
                    return
                row.progress = value
                row.message = message
                progress_db.commit()

        try:
            progress(0.1, "Preparing training")
            result = run_training(
                config,
                dataset_root=Path(dataset.root_path),
                output_dir=settings.model_dir / config.model_type.value / config.experiment_name,
            )
            artifact = ModelArtifact(
                name=config.experiment_name,
                version=result.get("version", config.experiment_name),
                model_type=config.model_type.value,
                status=ModelStatus.READY.value,
                artifact_path=result.get("artifact_path"),
                class_mapping=result.get("class_mapping"),
                preprocess_config=result.get("preprocess_config"),
                metrics=result.get("metrics"),
                dataset_id=dataset.id,
                is_active_inference=False,
                notes="Trained via Material Intelligence job worker",
            )
            db.add(artifact)
            db.flush()
            job.artifact_id = artifact.id
            job.metrics = result.get("metrics")
            job.status = JobStatus.COMPLETED.value
            job.progress = 1.0
            job.message = "Completed"
            job.finished_at = datetime.now(UTC)
            db.commit()
            log_path.write_text(json.dumps(result.get("metrics", {}), indent=2), encoding="utf-8")
            logger.info("Training job %s completed", job_id)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log_path.write_text(tb, encoding="utf-8")
            job.status = JobStatus.FAILED.value
            job.message = str(exc)
            job.finished_at = datetime.now(UTC)
            job.progress = 1.0
            db.commit()
            logger.exception("Training job %s failed", job_id)


def start_worker_on_startup() -> None:
    _ensure_worker()
