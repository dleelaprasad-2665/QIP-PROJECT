"""Model Training Control Center."""

from __future__ import annotations

from nicegui import ui
from sqlalchemy import select

from material_intelligence.app.db.models import Dataset, ModelArtifact, TrainingJob
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.domain.enums import JobStatus, ModelType
from material_intelligence.app.domain.schemas import PreprocessProfile, TrainingJobConfig
from material_intelligence.app.jobs.worker import enqueue_training_job
from material_intelligence.app.ml.model_registry import set_active_model
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.ui.state import current_user


def register() -> None:
    @ui.page("/training")
    def training_page() -> None:
        def content() -> None:
            user = current_user()
            jobs_box = ui.column().classes("w-full")

            def refresh_jobs() -> None:
                jobs_box.clear()
                with SessionLocal() as db:
                    jobs = db.scalars(
                        select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(20)
                    ).all()
                    artifacts = db.scalars(
                        select(ModelArtifact).order_by(ModelArtifact.created_at.desc()).limit(20)
                    ).all()
                with jobs_box:
                    with ui.element("div").classes("mi-card p-4 w-full"):
                        ui.label("Training jobs").classes("font-semibold mb-2")
                        if not jobs:
                            with ui.element("div").classes("mi-empty"):
                                ui.label("No jobs queued")
                        else:
                            ui.table(
                                columns=[
                                    {"name": "id", "label": "ID", "field": "id"},
                                    {"name": "name", "label": "Experiment", "field": "name"},
                                    {"name": "type", "label": "Type", "field": "type"},
                                    {"name": "status", "label": "Status", "field": "status"},
                                    {"name": "progress", "label": "Progress", "field": "progress"},
                                    {"name": "message", "label": "Message", "field": "message"},
                                ],
                                rows=[
                                    {
                                        "id": j.id,
                                        "name": j.experiment_name,
                                        "type": j.model_type,
                                        "status": j.status,
                                        "progress": f"{j.progress:.0%}",
                                        "message": (j.message or "")[:80],
                                    }
                                    for j in jobs
                                ],
                                row_key="id",
                            ).classes("w-full")

                    with ui.element("div").classes("mi-card p-4 w-full"):
                        ui.label("Checkpoints").classes("font-semibold mb-2")
                        if not artifacts:
                            with ui.element("div").classes("mi-empty"):
                                ui.label("Model Not Trained")
                                ui.label(
                                    "Train a task-specific model before enabling inference."
                                ).classes("text-sm")
                        else:
                            options = {
                                a.id: f"{a.name} · {a.model_type} · {a.status}"
                                + (" · ACTIVE" if a.is_active_inference else "")
                                for a in artifacts
                            }
                            selected = ui.select(options, label="Select checkpoint").classes(
                                "w-full max-w-xl"
                            )

                            def activate() -> None:
                                if not selected.value:
                                    return
                                with SessionLocal() as db:
                                    set_active_model(db, int(selected.value))
                                ui.notify("Active inference model updated", type="positive")
                                refresh_jobs()

                            ui.button("Set active for inference", on_click=activate)

            with SessionLocal() as db:
                datasets = db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).all()
            dataset_options = {
                d.id: f"{d.name} ({d.version}) · {d.format} · {d.status}" for d in datasets
            }

            with ui.element("div").classes("mi-card p-4 w-full"):
                ui.label("Configure training job").classes("font-semibold mb-2")
                ui.label(
                    "Jobs run asynchronously in a background worker. Long training is never executed in UI callbacks."
                ).classes("text-sm mi-muted mb-3")

                if not dataset_options:
                    with ui.element("div").classes("mi-empty"):
                        ui.label("Import a validated dataset before training.")
                else:
                    dataset_select = ui.select(dataset_options, label="Dataset").classes("w-full")
                    model_type = ui.select(
                        {
                            ModelType.EFFICIENTNET_B0.value: "EfficientNet-B0 (classification)",
                            ModelType.YOLOV8.value: "YOLOv8 (detection)",
                        },
                        value=ModelType.EFFICIENTNET_B0.value,
                        label="Model type",
                    ).classes("w-full")
                    experiment = ui.input("Experiment name", value="exp_cpu_001").classes("w-full")
                    epochs = ui.number("Epochs", value=5, min=1, max=200)
                    batch = ui.number("Batch size", value=4, min=1, max=64)
                    lr = ui.number("Learning rate", value=0.001, min=1e-6, max=1.0, step=0.0001)
                    image_size = ui.number("Image size", value=224, min=64, max=1280)
                    device = ui.select({"cpu": "CPU", "cuda": "CUDA (if available)"}, value="cpu")
                    yolo_variant = ui.select(
                        {"yolov8n": "YOLOv8n", "yolov8s": "YOLOv8s"}, value="yolov8n"
                    )
                    material_key = ui.input("Material key (classifier)", value="clay_brick").classes(
                        "w-full"
                    )
                    status = ui.label("").classes("text-sm")

                    def start_job() -> None:
                        if not dataset_select.value:
                            status.text = "Select a dataset."
                            return
                        config = TrainingJobConfig(
                            model_type=ModelType(model_type.value),
                            dataset_id=int(dataset_select.value),
                            experiment_name=experiment.value or "experiment",
                            epochs=int(epochs.value or 5),
                            batch_size=int(batch.value or 4),
                            learning_rate=float(lr.value or 1e-3),
                            image_size=int(image_size.value or 224),
                            device=device.value or "cpu",
                            yolo_variant=yolo_variant.value or "yolov8n",
                            preprocess_profile=PreprocessProfile(),
                            extra={"material_key": material_key.value},
                        )
                        with SessionLocal() as db:
                            job = TrainingJob(
                                experiment_name=config.experiment_name,
                                model_type=config.model_type.value,
                                dataset_id=config.dataset_id,
                                status=JobStatus.QUEUED.value,
                                progress=0.0,
                                config=config.model_dump(),
                                message="Queued",
                                created_by=user.id if user else None,
                            )
                            db.add(job)
                            db.commit()
                            db.refresh(job)
                            job_id = job.id
                        enqueue_training_job(job_id)
                        status.text = f"Queued job #{job_id}"
                        refresh_jobs()

                    ui.button("Start training", on_click=start_job).props("unelevated")
                    ui.button("Refresh status", on_click=refresh_jobs).props("ml-2")

            refresh_jobs()

        page_frame("Model Training", content)
