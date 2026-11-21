"""Helpers for creating training jobs from services/API."""

from __future__ import annotations

from sqlalchemy.orm import Session

from material_intelligence.app.db.models import TrainingJob
from material_intelligence.app.domain.enums import JobStatus
from material_intelligence.app.domain.schemas import TrainingJobConfig
from material_intelligence.app.jobs.worker import enqueue_training_job


def create_and_enqueue(
    db: Session,
    config: TrainingJobConfig,
    *,
    user_id: int | None = None,
) -> TrainingJob:
    job = TrainingJob(
        experiment_name=config.experiment_name,
        model_type=config.model_type.value,
        dataset_id=config.dataset_id,
        status=JobStatus.QUEUED.value,
        progress=0.0,
        config=config.model_dump(),
        message="Queued",
        created_by=user_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    enqueue_training_job(job.id)
    return job
