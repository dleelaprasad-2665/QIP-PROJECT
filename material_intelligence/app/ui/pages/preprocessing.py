"""Preprocessing Laboratory — CLAHE and augmentation comparison."""

from __future__ import annotations

import cv2
from nicegui import events, ui

from material_intelligence.app.config import get_settings
from material_intelligence.app.domain.schemas import PreprocessProfile
from material_intelligence.app.ml.augmentation import augment_classification_sample
from material_intelligence.app.ml.preprocessing import preprocess_image
from material_intelligence.app.ui.components.image_viewer import show_image
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.utils.files import save_upload


def register() -> None:
    @ui.page("/preprocessing")
    def preprocessing_page() -> None:
        def content() -> None:
            with ui.element("div").classes("mi-card p-4 w-full"):
                ui.label("CLAHE & Augmentation Laboratory").classes("font-semibold mb-2")
                ui.label(
                    "CLAHE is applied on the LAB luminance channel. Stochastic augmentation is for training only."
                ).classes("text-sm mi-muted mb-4")

                enable_clahe = ui.switch("Enable CLAHE", value=True)
                clip = ui.number("CLAHE clip limit", value=2.0, min=0.5, max=10.0, step=0.5)
                tile = ui.number("Tile grid size", value=8, min=2, max=16, step=1)
                brightness = ui.number("Brightness", value=0, min=-50, max=50, step=1)
                contrast = ui.number("Contrast", value=1.0, min=0.5, max=2.0, step=0.05)
                preview = ui.row().classes("w-full gap-4 flex-wrap")
                status = ui.label("").classes("text-sm")

                async def on_upload(e: events.UploadEventArguments) -> None:
                    data = await e.file.read()
                    settings = get_settings()
                    path = save_upload(data, settings.upload_dir / "preprocess", e.file.name or "img.jpg")
                    image = cv2.imread(str(path))
                    if image is None:
                        status.text = "Unable to decode image."
                        return
                    profile = PreprocessProfile(
                        enable_clahe=enable_clahe.value,
                        clahe_clip_limit=float(clip.value or 2.0),
                        clahe_tile_grid=(int(tile.value or 8), int(tile.value or 8)),
                        brightness=float(brightness.value or 0),
                        contrast=float(contrast.value or 1.0),
                        resize_to=None,
                    )
                    result = preprocess_image(image, profile)
                    augmented = augment_classification_sample(result.processed_bgr)

                    out_dir = settings.upload_dir / "preprocess" / "compare"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    original_p = out_dir / "original.jpg"
                    clahe_p = out_dir / "clahe.jpg"
                    aug_p = out_dir / "augmented.jpg"
                    cv2.imwrite(str(original_p), result.original_bgr)
                    cv2.imwrite(str(clahe_p), result.processed_bgr)
                    cv2.imwrite(str(aug_p), augmented)

                    preview.clear()
                    with preview:
                        show_image(original_p, caption="Original Image")
                        show_image(clahe_p, caption="CLAHE Image")
                        show_image(aug_p, caption="Augmented Training Example")
                    status.text = "Comparison generated. Augmentation is not applied to validation/test data."

                ui.upload(on_upload=on_upload, auto_upload=True, max_files=1).props(
                    'accept="image/*" label="Upload sample image"'
                )
                preview

                with ui.element("div").classes("mi-empty mt-4"):
                    ui.label("Ablation note")
                    ui.label(
                        "Do not assume CLAHE always improves accuracy. Run paired training experiments "
                        "with and without CLAHE and compare validation metrics in Model Evaluation."
                    ).classes("text-sm")

        page_frame("Preprocessing Laboratory", content)
