"""Explainable AI — Grad-CAM workspace."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from nicegui import events, ui
from sqlalchemy import select

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import ModelArtifact
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.domain.enums import ModelStatus, ModelType
from material_intelligence.app.ml.efficientnet_classifier import EfficientNetClassifier
from material_intelligence.app.ml.gradcam import GradCAMExplainer
from material_intelligence.app.ml.model_registry import load_active_yolo
from material_intelligence.app.ml.yolo_detector import draw_detections
from material_intelligence.app.ui.components.image_viewer import show_image
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.utils.files import save_upload


def register() -> None:
    @ui.page("/explainability")
    def explainability_page() -> None:
        def content() -> None:
            with ui.element("div").classes("mi-card p-4 w-full"):
                ui.label("Grad-CAM Interpretability").classes("font-semibold mb-2")
                ui.label(
                    "Grad-CAM highlights image regions influencing the EfficientNet-B0 classifier, "
                    "but it is not proof that the material is safe or defective."
                ).classes("text-sm mi-muted mb-3")

                with SessionLocal() as db:
                    artifacts = db.scalars(
                        select(ModelArtifact).where(
                            ModelArtifact.model_type == ModelType.EFFICIENTNET_B0.value,
                            ModelArtifact.status == ModelStatus.READY.value,
                        )
                    ).all()
                if not artifacts:
                    with ui.element("div").classes("mi-empty"):
                        ui.label("Model Not Trained")
                        ui.label(
                            "Train and register an EfficientNet-B0 construction-material classifier first."
                        ).classes("text-sm")
                    return

                model_select = ui.select(
                    {a.id: f"{a.name} ({a.version})" for a in artifacts},
                    label="Classifier checkpoint",
                ).classes("w-full max-w-xl")
                opacity = ui.slider(min=0.1, max=0.8, value=0.45, step=0.05).props("label")
                ui.label("Heatmap opacity")
                panels = ui.row().classes("w-full gap-4 flex-wrap")
                status = ui.label("").classes("text-sm")
                download_path: dict[str, str] = {}

                async def on_upload(e: events.UploadEventArguments) -> None:
                    if not model_select.value:
                        status.text = "Select a classifier."
                        return
                    data = await e.file.read()
                    settings = get_settings()
                    path = save_upload(data, settings.upload_dir / "xai", e.file.name or "img.jpg")
                    image = cv2.imread(str(path))
                    if image is None:
                        status.text = "Unable to decode image."
                        return
                    with SessionLocal() as db:
                        artifact = db.get(ModelArtifact, int(model_select.value))
                        if artifact is None or not artifact.artifact_path:
                            status.text = "Checkpoint missing."
                            return
                        clf = EfficientNetClassifier(artifact.artifact_path)
                        if not clf.is_ready():
                            status.text = "Classifier not ready."
                            return
                        detector = load_active_yolo(db)
                        detection = detector.predict(image)
                        annotated = draw_detections(image, detection.boxes)
                        explainer = GradCAMExplainer(clf)
                        explanation = explainer.explain(image, opacity=float(opacity.value or 0.45))
                        out = settings.upload_dir / "xai" / f"gradcam_{int(time.time())}.jpg"
                        explainer.save_visualization(explanation["overlay_bgr"], out)
                        ann_path = settings.upload_dir / "xai" / f"det_{int(time.time())}.jpg"
                        cv2.imwrite(str(ann_path), annotated)
                        download_path["path"] = str(out)

                    panels.clear()
                    with panels:
                        show_image(path, caption="Original Image")
                        show_image(ann_path, caption="YOLO Detection")
                        show_image(out, caption="EfficientNet Grad-CAM")
                    status.text = (
                        f"Predicted: {explanation['condition']} "
                        f"({explanation['confidence']:.3f}) · {explanation['note']}"
                    )

                ui.upload(on_upload=on_upload, auto_upload=True, max_files=1).props(
                    'accept="image/*" label="Upload material crop or image"'
                )

                def download() -> None:
                    path = download_path.get("path")
                    if path and Path(path).exists():
                        ui.download(path)
                    else:
                        ui.notify("No Grad-CAM visualization yet", type="warning")

                ui.button("Download Grad-CAM", on_click=download)
                panels

        page_frame("Explainable AI", content)
