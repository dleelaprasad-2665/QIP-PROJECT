"""Live AI Testing Laboratory — end-to-end workflow validation."""

from __future__ import annotations

from nicegui import events, ui

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.domain.enums import MATERIAL_DISPLAY_NAMES
from material_intelligence.app.ml.evaluation.recommendations import RECOMMENDATION_DISPLAY
from material_intelligence.app.services.evaluation_service import run_live_laboratory
from material_intelligence.app.services.inspection_result_formatter import display_defect
from material_intelligence.app.ui.components.image_viewer import show_image
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.utils.files import save_upload


def register() -> None:
    @ui.page("/testing-laboratory")
    def testing_lab() -> None:
        def content() -> None:
            result_area = ui.column().classes("w-full gap-3")
            with ui.element("div").classes("mi-card p-4 w-full"):
                ui.label("Live AI Testing Laboratory").classes("font-semibold")
                ui.label(
                    "Upload an image, optionally provide independent ground truth, "
                    "and inspect the full pipeline without fabricating outputs."
                ).classes("text-sm mi-muted mb-3")
                material = ui.select(
                    {"__auto__": "Auto", **MATERIAL_DISPLAY_NAMES},
                    value="__auto__",
                    label="Expected material",
                ).classes("w-full max-w-md")
                gt = ui.input(
                    "Ground-truth condition label (optional)",
                    placeholder="e.g. cracked / no_visible_damage",
                ).classes("w-full max-w-md")
                gradcam = ui.switch("Grad-CAM", value=True)
                status = ui.label("").classes("text-sm")

                async def on_upload(e: events.UploadEventArguments) -> None:
                    status.text = "Running integrated pipeline…"
                    data = await e.file.read()
                    settings = get_settings()
                    path = save_upload(
                        data,
                        settings.upload_dir / "lab",
                        e.file.name or "lab.jpg",
                        image=True,
                    )
                    try:
                        with SessionLocal() as db:
                            out = run_live_laboratory(
                                db,
                                image_path=str(path),
                                generate_gradcam=gradcam.value,
                                expected_material_key=(
                                    None if material.value == "__auto__" else material.value
                                ),
                                ground_truth_label=(gt.value or None),
                            )
                        result_area.clear()
                        with result_area:
                            with ui.element("div").classes("mi-card p-4"):
                                ui.label("Workflow stages").classes("font-semibold")
                                for stage in out["stages"]:
                                    ui.label(
                                        f"Step {stage['step']}: {stage['name']} · "
                                        f"{stage['status']} · {stage.get('ms', 0):.0f} ms"
                                    ).classes("text-sm")
                                    ui.label(str(stage.get("output"))[:240]).classes(
                                        "text-xs mi-muted"
                                    )
                            rec = out["recommendation"]
                            with ui.element("div").classes("mi-card p-4"):
                                ui.label("Preliminary recommendation").classes("font-semibold")
                                ui.label(
                                    RECOMMENDATION_DISPLAY.get(
                                        rec["code"], rec["code"]
                                    )
                                ).classes("mi-metric-value")
                                ui.label(rec["reason"]).classes("text-sm")
                                ui.label(out["disclaimer"]).classes("text-xs mi-muted mt-2")
                            pred = out.get("prediction") or {}
                            items = (pred.get("classifications") or [])
                            if items:
                                item = items[0]
                                ui.label(
                                    f"Prediction: {display_defect(item.get('condition'))} "
                                    f"({(item.get('confidence') or 0):.0%})"
                                )
                            if out.get("correctness"):
                                c = out["correctness"]
                                ui.label(
                                    f"Ground truth: {c['ground_truth']} · "
                                    f"correct={c['correct']}"
                                ).classes("text-sm")
                            else:
                                ui.label(
                                    "No ground truth provided — inference result only."
                                ).classes("text-sm mi-muted")
                            with ui.row().classes("w-full gap-3"):
                                show_image(str(path), caption="Original", height="260px")
                                show_image(
                                    out.get("gradcam_path"),
                                    caption="Grad-CAM",
                                    height="260px",
                                )
                        status.text = "Pipeline complete."
                    except Exception as exc:  # noqa: BLE001
                        status.text = f"Error: {exc}"

                ui.upload(on_upload=on_upload, auto_upload=True, max_files=1).props(
                    'accept="image/*" label="Upload test image"'
                )
            result_area
            ui.button(
                "Back to Evaluation", on_click=lambda: ui.navigate.to("/evaluation")
            ).props("flat")

        page_frame("Testing Laboratory", content)
