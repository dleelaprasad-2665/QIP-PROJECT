"""Dataset Studio page."""

from __future__ import annotations

from nicegui import events, ui
from sqlalchemy import select

from material_intelligence.app.db.models import Dataset
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.domain.enums import MATERIAL_DISPLAY_NAMES
from material_intelligence.app.services.dataset_service import (
    get_dataset_preview,
    import_classification_images,
    import_zip_dataset,
)
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.ui.state import current_user


def register() -> None:
    @ui.page("/datasets")
    def datasets_page() -> None:
        def content() -> None:
            user = current_user()
            refresh_box = ui.column().classes("w-full gap-4")

            def render_list() -> None:
                refresh_box.clear()
                with refresh_box:
                    with SessionLocal() as db:
                        datasets = db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).all()
                    if not datasets:
                        with ui.element("div").classes("mi-empty"):
                            ui.label("No datasets imported")
                            ui.label(
                                "Import Roboflow/CVAT YOLO ZIP exports or classification folders."
                            ).classes("text-sm")
                        return
                    rows = [
                        {
                            "id": d.id,
                            "name": d.name,
                            "version": d.version,
                            "format": d.format,
                            "status": d.status,
                            "images": d.image_count,
                            "annotations": d.annotation_count,
                            "classes": ", ".join(d.class_names or [])[:80],
                        }
                        for d in datasets
                    ]
                    ui.table(
                        columns=[
                            {"name": "id", "label": "ID", "field": "id"},
                            {"name": "name", "label": "Name", "field": "name"},
                            {"name": "version", "label": "Version", "field": "version"},
                            {"name": "format", "label": "Format", "field": "format"},
                            {"name": "status", "label": "Status", "field": "status"},
                            {"name": "images", "label": "Images", "field": "images"},
                            {"name": "annotations", "label": "Annotations", "field": "annotations"},
                            {"name": "classes", "label": "Classes", "field": "classes"},
                        ],
                        rows=rows,
                        row_key="id",
                    ).classes("w-full")

                    selected = ui.select(
                        {d.id: f"{d.name} ({d.version})" for d in datasets},
                        label="Inspect dataset",
                    ).classes("w-full max-w-md")
                    preview_area = ui.column().classes("w-full")

                    def show_preview() -> None:
                        preview_area.clear()
                        if not selected.value:
                            return
                        with SessionLocal() as db:
                            dataset = db.get(Dataset, selected.value)
                            if dataset is None:
                                return
                            report = dataset.validation_report or {}
                            with preview_area:
                                with ui.element("div").classes("mi-card p-4"):
                                    ui.label("Validation report").classes("font-semibold")
                                    ui.label(
                                        f"Warnings: {len(report.get('warnings', []))} · "
                                        f"Errors: {len(report.get('errors', []))} · "
                                        f"Imbalance: {report.get('imbalance_flag')}"
                                    ).classes("text-sm mi-muted")
                                    dist = dataset.class_distribution or {}
                                    if dist:
                                        ui.label("Class distribution").classes("mt-2 font-medium")
                                        for k, v in dist.items():
                                            ui.label(f"{k}: {v}").classes("text-sm")
                                    samples = get_dataset_preview(dataset, limit=6)
                                    if samples:
                                        ui.label("Annotation preview (paths)").classes(
                                            "mt-2 font-medium"
                                        )
                                        for sample in samples:
                                            ui.label(
                                                f"{sample['image']} · boxes={len(sample['boxes'])}"
                                            ).classes("text-xs mi-muted")

                    ui.button("Load validation details", on_click=show_preview)
                    preview_area

            with ui.row().classes("w-full gap-4 flex-wrap"):
                with ui.element("div").classes("mi-card p-4 flex-1 min-w-[320px]"):
                    ui.label("Import Roboflow / CVAT ZIP").classes("font-semibold mb-2")
                    ui.label(
                        "External annotation tools are integrated via export formats only. "
                        "No external account API is connected."
                    ).classes("text-sm mi-muted mb-3")
                    name_input = ui.input("Dataset name", value="construction_materials").classes(
                        "w-full"
                    )
                    version_input = ui.input("Version", value="v1").classes("w-full")
                    source_input = ui.input("Source", value="Roboflow/CVAT export").classes("w-full")
                    annotator_input = ui.input("Annotator").classes("w-full")
                    zip_status = ui.label("").classes("text-sm")

                    zip_details = ui.column().classes("w-full gap-1")

                    async def on_zip(e: events.UploadEventArguments) -> None:
                        zip_status.text = f"Importing {e.file.name or 'dataset.zip'}…"
                        zip_details.clear()
                        data = await e.file.read()
                        try:
                            with SessionLocal() as db:
                                result = import_zip_dataset(
                                    db,
                                    zip_bytes=data,
                                    filename=e.file.name or "dataset.zip",
                                    name=name_input.value or "dataset",
                                    version=version_input.value or "v1",
                                    source=source_input.value,
                                    annotator=annotator_input.value,
                                    user_id=user.id if user else None,
                                )
                            zip_status.text = (
                                f"Imported dataset #{result.dataset_id} · "
                                f"status={result.status} · format={result.format} · "
                                f"{result.image_count} images · "
                                f"{result.annotation_count} annotations"
                            )
                            with zip_details:
                                if result.class_names:
                                    ui.label(
                                        "Classes: " + ", ".join(result.class_names[:20])
                                    ).classes("text-xs")
                                for warning in (result.warnings or [])[:5]:
                                    ui.label(f"Warning: {warning}").classes(
                                        "text-xs"
                                    ).style("color: var(--mi-warning);")
                                for error in (result.errors or [])[:5]:
                                    ui.label(f"Error: {error}").classes(
                                        "text-xs"
                                    ).style("color: var(--mi-danger);")
                                if not result.errors:
                                    ui.label(
                                        "Open the table below and click "
                                        "'Load validation details' to preview."
                                    ).classes("text-xs mi-muted")
                            render_list()
                        except Exception as exc:  # noqa: BLE001
                            zip_status.text = f"Import failed: {exc}"
                            with zip_details:
                                ui.label(
                                    "Tip: ZIP may contain YOLO images+labels, class folders, "
                                    "or loose images (imported as unlabeled)."
                                ).classes("text-xs mi-muted")

                    ui.upload(on_upload=on_zip, auto_upload=True, max_files=1).props(
                        'accept=".zip" label="Upload dataset ZIP"'
                    )
                    zip_details

                with ui.element("div").classes("mi-card p-4 flex-1 min-w-[320px]"):
                    ui.label("Import classification images").classes("font-semibold mb-2")
                    clf_name = ui.input("Dataset name", value="condition_cls").classes("w-full")
                    clf_version = ui.input("Version", value="v1").classes("w-full")
                    class_name = ui.select(
                        {
                            **{k: v for k, v in MATERIAL_DISPLAY_NAMES.items()},
                            "cracked": "Cracked",
                            "chipped": "Chipped",
                            "broken": "Broken",
                            "no_visible_damage": "No Visible Damage",
                            "surface_corrosion": "Surface Corrosion",
                            "extensive_visible_corrosion": "Extensive Visible Corrosion",
                            "deformation": "Deformation",
                            "torn_packaging": "Torn Packaging",
                            "visible_moisture_staining": "Visible Moisture Staining",
                            "damaged_packaging": "Damaged Packaging",
                        },
                        label="Class folder",
                        value="no_visible_damage",
                    ).classes("w-full")
                    clf_status = ui.label("").classes("text-sm")

                    async def on_images(e: events.UploadEventArguments) -> None:
                        data = await e.file.read()
                        try:
                            with SessionLocal() as db:
                                result = import_classification_images(
                                    db,
                                    files=[(e.file.name or "image.jpg", data)],
                                    class_name=class_name.value or "unknown",
                                    dataset_name=clf_name.value or "condition_cls",
                                    version=clf_version.value or "v1",
                                    user_id=user.id if user else None,
                                )
                            clf_status.text = (
                                f"Updated dataset #{result.dataset_id} · "
                                f"{result.image_count} images · {result.status}"
                            )
                            render_list()
                        except Exception as exc:  # noqa: BLE001
                            clf_status.text = f"Upload failed: {exc}"

                    ui.upload(on_upload=on_images, auto_upload=True, multiple=True).props(
                        'accept="image/*" label="Upload class images"'
                    )

            render_list()

        page_frame("Dataset Studio", content)
