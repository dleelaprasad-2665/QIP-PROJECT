"""Inspection flow: Upload → Preprocess → AI → Condition → Recommendation → Inventory."""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from nicegui import events, ui
from sqlalchemy import select

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import Inspection, MaterialBatch, Site
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.domain.enums import MATERIAL_DISPLAY_NAMES
from material_intelligence.app.ml.inference_pipeline import run_inspection_pipeline
from material_intelligence.app.services.dashboard_analytics import DEMO_SITE_CODE
from material_intelligence.app.services.inspection_service import create_inspection
from material_intelligence.app.services.simple_result import simplify_prediction
from material_intelligence.app.ui.components.image_viewer import show_image
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.ui.state import current_user, get_app_state, save_app_state

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

FLOW_STEPS = [
    ("1", "Upload", "Brick / Steel / Cement / Wood"),
    ("2", "Preprocessing", "CLAHE · Resize · Normalize"),
    ("3", "AI Analysis", "YOLOv8 + EfficientNet-B0"),
    ("4", "Condition", "Good · Damaged · Inspection Required"),
    ("5", "Recommendation", "Routine / Further / Hold"),
    ("6", "Inventory", "Store · Track · Reports"),
]


def _expected_from_select(value) -> str | None:
    if value in (None, "__auto__"):
        return None
    return value


def _status_color(token: str) -> str:
    return {
        "success": "var(--mi-success)",
        "danger": "var(--mi-danger)",
        "warning": "var(--mi-warning)",
    }.get(token, "var(--mi-accent)")


def _render_flow_banner(active: int = 1) -> None:
    with ui.row().classes("w-full gap-1 flex-wrap mb-2"):
        for i, (num, title, sub) in enumerate(FLOW_STEPS, start=1):
            on = i <= active
            with ui.element("div").classes("mi-kpi flex-1 min-w-[110px]").style(
                f"border-color: {'var(--mi-accent)' if on else 'var(--mi-border)'};"
                f"opacity: {'1' if on else '0.55'};"
            ):
                ui.label(f"{num}. {title}").classes("text-xs font-semibold").style(
                    f"color: {'var(--mi-accent)' if i == active else 'inherit'};"
                )
                ui.label(sub).classes("text-[10px] mi-muted")


def _render_flow_result(prediction, inspection=None, *, title: str = "4–5. Result") -> None:
    simple = simplify_prediction(prediction)
    with ui.element("div").classes("mi-card p-4 w-full"):
        ui.label(title).classes("font-semibold text-lg")
        with ui.row().classes("w-full gap-3 flex-wrap mt-2"):
            with ui.element("div").classes("mi-kpi flex-1 min-w-[140px]"):
                ui.label("1. Material").classes("mi-kpi-label")
                ui.label(simple.material_name).classes("mi-metric-value").style(
                    "font-size: 1.35rem;"
                )
            with ui.element("div").classes("mi-kpi flex-1 min-w-[140px]"):
                ui.label("4. Condition").classes("mi-kpi-label")
                ui.label(simple.condition_label).classes("mi-metric-value").style(
                    f"font-size: 1.45rem; color: {_status_color(simple.color)};"
                )
                ui.label(f"{simple.confidence_pct}% confidence").classes("text-xs mi-muted")

        ui.separator().classes("my-3")
        ui.label("2. Preprocessing").classes("text-xs font-semibold mi-muted")
        ui.label(simple.pipeline_steps.get("2. Preprocessing", "")).classes("text-sm mb-2")
        ui.label("3. AI Model Analysis").classes("text-xs font-semibold mi-muted")
        ui.label(simple.pipeline_steps.get("3. AI Analysis", "")).classes("text-sm mb-2")
        ui.label("Details").classes("text-xs font-semibold mi-muted")
        ui.label(simple.condition_detail).classes("text-sm mb-2")
        ui.label("5. Preliminary Recommendation").classes("text-xs font-semibold mi-muted")
        ui.label(simple.recommendation).classes("text-base font-semibold").style(
            f"color: {_status_color(simple.color)};"
        )
        ui.label("6. Inventory & Dashboard").classes("text-xs font-semibold mi-muted mt-2")
        ui.label(
            "Result stored — view Inventory, Dashboard, and Reports."
        ).classes("text-sm")
        if inspection is not None:
            ui.label(
                f"Saved as scan #{inspection.id} · {inspection.processing_ms or 0:.0f} ms"
            ).classes("text-xs mi-muted mt-2")


def _batch_summary(rows: list[dict]) -> str:
    good = sum(1 for r in rows if r.get("condition") == "Good")
    damaged = sum(1 for r in rows if r.get("condition") == "Damaged")
    check = sum(1 for r in rows if r.get("condition") == "Inspection Required")
    return f"{len(rows)} photos · Good {good} · Damaged {damaged} · Inspection Required {check}"


def _save_batch_csv(rows: list[dict]) -> Path:
    settings = get_settings()
    out_dir = Path(getattr(settings, "report_dir", None) or (settings.upload_dir / "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"batch_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "material",
                "condition",
                "recommendation",
                "conf",
                "details",
                "id",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _encode_jpeg_b64(image_bgr: np.ndarray, max_side: int = 720) -> str:
    h, w = image_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def register() -> None:
    @ui.page("/inspection/{inspection_id}")
    def inspection_detail(inspection_id: int) -> None:
        def content() -> None:
            with SessionLocal() as db:
                insp = db.get(Inspection, inspection_id)
                if insp is None:
                    with ui.element("div").classes("mi-empty"):
                        ui.label("Inspection not found")
                    return
            from material_intelligence.app.domain.enums import ModelStatus
            from material_intelligence.app.domain.schemas import (
                ClassificationResult,
                DetectionResult,
                InspectionPrediction,
                InspectionStatus,
            )

            det = insp.detection_result or {}
            items = (insp.classification_result or {}).get("items") or []
            classifications = [ClassificationResult(**item) for item in items]
            prediction = InspectionPrediction(
                detection=DetectionResult(
                    boxes=[],
                    model_status=ModelStatus(det.get("model_status", "ready")),
                    message=det.get("message"),
                ),
                classifications=classifications,
                recommended_status=InspectionStatus(insp.status)
                if insp.status in InspectionStatus._value2member_map_
                else InspectionStatus.HUMAN_INSPECTION_REQUIRED,
                processing_ms=insp.processing_ms or 0,
                notes=(insp.classification_result or {}).get("notes") or [],
            )
            with ui.column().classes("w-full gap-3"):
                _render_flow_banner(6)
                _render_flow_result(prediction, insp)
                with ui.row().classes("w-full gap-3"):
                    show_image(insp.image_path, caption="Uploaded image", height="45vh")
                    show_image(insp.annotated_path, caption="AI detection", height="45vh")
                ui.button("New scan", on_click=lambda: ui.navigate.to("/inspection")).props(
                    "unelevated"
                )

        page_frame("Inspection Flow", content, lock_scroll=True)

    @ui.page("/inspection")
    def inspection_page() -> None:
        def content() -> None:
            state = get_app_state()
            user = current_user()
            hosts: dict = {}
            live_state: dict = {"running": False, "cap": None, "timer": None}

            with SessionLocal() as db:
                batches = []
                if state.current_site_id:
                    batches = db.scalars(
                        select(MaterialBatch).where(MaterialBatch.site_id == state.current_site_id)
                    ).all()
                batch_options = {b.id: f"{b.batch_code} · {b.material_key}" for b in batches}
                sites = db.scalars(select(Site).where(Site.is_active.is_(True))).all()
                if state.current_site_id is None and sites:
                    operational = next((s for s in sites if s.code != DEMO_SITE_CODE), sites[0])
                    state.current_site_id = operational.id
                    save_app_state(state)

            with ui.column().classes("w-full h-full min-h-0 gap-1"):
                hosts["banner"] = ui.column().classes("w-full")
                with hosts["banner"]:
                    _render_flow_banner(1)

                with ui.row().classes("w-full gap-3 flex-nowrap min-h-0 flex-grow"):
                    with ui.column().classes("w-[360px] shrink-0 gap-2 overflow-auto h-full"):
                        with ui.element("div").classes("mi-card p-3"):
                            ui.label("1. Upload Construction Material Image").classes(
                                "font-semibold text-base"
                            )
                            ui.label("Brick / Steel / Cement / Wood").classes(
                                "text-sm mi-muted mb-2"
                            )

                            material_select = (
                                ui.select(
                                    {
                                        "__auto__": "Auto-detect",
                                        **MATERIAL_DISPLAY_NAMES,
                                    },
                                    value="__auto__",
                                    label="Material type",
                                )
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            batch_select = (
                                ui.select(
                                    batch_options,
                                    label="Batch (optional)",
                                    with_input=True,
                                )
                                .props("dense outlined")
                                .classes("w-full")
                            )
                            status = ui.label("Ready — upload to start the flow").classes(
                                "text-sm mi-muted"
                            )

                            async def run_one(raw: bytes, filename: str):
                                with SessionLocal() as db:
                                    from material_intelligence.app.db.models import User

                                    db_user = db.get(User, user.id) if user else None
                                    return create_inspection(
                                        db,
                                        site_id=state.current_site_id,
                                        batch_id=batch_select.value,
                                        image_bytes=raw,
                                        filename=filename,
                                        user=db_user,
                                        generate_gradcam=False,
                                        expected_material_key=_expected_from_select(
                                            material_select.value
                                        ),
                                    )

                            def show_single(inspection, prediction) -> None:
                                hosts["banner"].clear()
                                with hosts["banner"]:
                                    _render_flow_banner(6)
                                hosts["result"].clear()
                                hosts["images"].clear()
                                with hosts["result"]:
                                    _render_flow_result(prediction, inspection)
                                with hosts["images"]:
                                    show_image(
                                        inspection.image_path,
                                        caption="1. Uploaded image",
                                        height="52vh",
                                    )
                                    show_image(
                                        inspection.annotated_path,
                                        caption="3. YOLOv8 detection",
                                        height="52vh",
                                    )

                            def show_batch_table(rows: list[dict], last_insp=None) -> None:
                                hosts["banner"].clear()
                                with hosts["banner"]:
                                    _render_flow_banner(6)
                                hosts["result"].clear()
                                hosts["images"].clear()
                                csv_path = _save_batch_csv(rows)
                                with hosts["result"]:
                                    with ui.element("div").classes("mi-card p-3"):
                                        ui.label("4–6. Batch results").classes(
                                            "font-semibold text-lg"
                                        )
                                        ui.label(_batch_summary(rows)).classes(
                                            "text-sm mi-muted mb-2"
                                        )
                                        ui.table(
                                            columns=[
                                                {"name": "file", "label": "File", "field": "file"},
                                                {
                                                    "name": "material",
                                                    "label": "Material",
                                                    "field": "material",
                                                },
                                                {
                                                    "name": "condition",
                                                    "label": "Condition",
                                                    "field": "condition",
                                                },
                                                {
                                                    "name": "recommendation",
                                                    "label": "Recommendation",
                                                    "field": "recommendation",
                                                },
                                                {"name": "conf", "label": "Conf", "field": "conf"},
                                            ],
                                            rows=rows,
                                            row_key="id",
                                            pagination=12,
                                        ).classes("w-full")
                                        ui.label(
                                            f"6. CSV saved for reports: {csv_path}"
                                        ).classes("text-xs mi-muted mt-2")
                                if last_insp is not None:
                                    with hosts["images"]:
                                        show_image(
                                            last_insp.image_path,
                                            caption="Last photo in batch",
                                            height="50vh",
                                        )

                            async def scan_members(members: list[tuple[str, bytes]]) -> None:
                                rows = []
                                last = None
                                total = len(members)
                                for idx, (fname, raw) in enumerate(members):
                                    status.text = f"2–3. Scanning {idx + 1}/{total}…"
                                    await asyncio.sleep(0)
                                    inspection, prediction = await run_one(raw, fname)
                                    last = inspection
                                    simple = simplify_prediction(prediction)
                                    rows.append(
                                        {
                                            "file": fname,
                                            "material": simple.material_name,
                                            "condition": simple.condition_label,
                                            "recommendation": simple.recommendation,
                                            "conf": f"{simple.confidence_pct}%",
                                            "details": simple.condition_detail,
                                            "id": inspection.id,
                                        }
                                    )
                                show_batch_table(rows, last)
                                status.text = f"Done · {_batch_summary(rows)}"

                            async def on_upload(e: events.UploadEventArguments) -> None:
                                if state.current_site_id is None:
                                    status.text = "Select a site in the top bar first."
                                    return
                                name = e.file.name or "upload.bin"
                                data = await e.file.read()
                                ext = Path(name).suffix.lower()
                                try:
                                    if ext == ".zip":
                                        status.text = "1. Reading ZIP batch…"
                                        members = []
                                        with zipfile.ZipFile(io.BytesIO(data)) as zf:
                                            for info in zf.infolist():
                                                if info.is_dir():
                                                    continue
                                                p = Path(info.filename)
                                                if p.suffix.lower() in IMAGE_EXTS:
                                                    members.append((p.name, zf.read(info)))
                                        if not members:
                                            status.text = "No photos found in ZIP."
                                            return
                                        await scan_members(members)
                                        return

                                    if ext not in IMAGE_EXTS:
                                        status.text = "Upload a photo (jpg/png) or ZIP."
                                        return
                                    status.text = "2–3. Preprocessing + AI analysis…"
                                    inspection, prediction = await run_one(data, name)
                                    show_single(inspection, prediction)
                                    simple = simplify_prediction(prediction)
                                    status.text = (
                                        f"4. {simple.condition_label} · "
                                        f"5. {simple.recommendation}"
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    status.text = f"Error: {exc}"

                            ui.upload(
                                on_upload=on_upload, auto_upload=True, max_files=1
                            ).props(
                                'accept=".jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff,.zip" '
                                'label="Upload photo or ZIP"'
                            ).classes("w-full")

                            ui.separator().classes("my-2")
                            ui.label("Or scan a folder on this PC").classes(
                                "text-xs font-semibold"
                            )
                            folder_input = (
                                ui.input(label="Folder path", placeholder=r"C:\path\to\photos")
                                .props("dense outlined")
                                .classes("w-full")
                            )

                            async def scan_folder() -> None:
                                folder = Path((folder_input.value or "").strip())
                                if not folder.is_dir():
                                    status.text = "Folder not found."
                                    return
                                members = [
                                    (p.name, p.read_bytes())
                                    for p in sorted(folder.rglob("*"))
                                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                                ]
                                if not members:
                                    status.text = "No images in folder."
                                    return
                                await scan_members(members)

                            ui.button("Scan folder", on_click=scan_folder).props(
                                "outline dense"
                            ).classes("w-full")

                            ui.separator().classes("my-2")
                            ui.label("Live camera (same 1–6 flow)").classes(
                                "text-xs font-semibold"
                            )
                            live_status = ui.label("Camera off").classes("text-xs mi-muted")

                            def stop_live() -> None:
                                live_state["running"] = False
                                if live_state.get("timer") is not None:
                                    live_state["timer"].active = False
                                    live_state["timer"] = None
                                cap = live_state.get("cap")
                                if cap is not None:
                                    try:
                                        cap.release()
                                    except Exception:  # noqa: BLE001
                                        pass
                                    live_state["cap"] = None
                                live_status.text = "Camera off"

                            def live_tick() -> None:
                                if not live_state["running"]:
                                    return
                                cap = live_state.get("cap")
                                if cap is None:
                                    return
                                ok, frame = cap.read()
                                if not ok or frame is None:
                                    live_status.text = "No camera frame"
                                    return
                                try:
                                    with SessionLocal() as db:
                                        prediction, annotated, _ = run_inspection_pipeline(
                                            db,
                                            frame,
                                            generate_gradcam=False,
                                            expected_material_key=_expected_from_select(
                                                material_select.value
                                            ),
                                            source_filename="live_camera.jpg",
                                        )
                                    simple = simplify_prediction(prediction)
                                    hosts["banner"].clear()
                                    with hosts["banner"]:
                                        _render_flow_banner(5)
                                    hosts["result"].clear()
                                    with hosts["result"]:
                                        _render_flow_result(
                                            prediction, title="Live · steps 1–5"
                                        )
                                    hosts["images"].clear()
                                    with hosts["images"]:
                                        ui.image(_encode_jpeg_b64(annotated)).classes(
                                            "w-full"
                                        ).style(
                                            "max-height: 70vh; object-fit: contain; "
                                            "background:#0d0d0d; border-radius:4px;"
                                        )
                                    live_status.text = (
                                        f"{simple.material_name} → {simple.condition_label} · "
                                        f"{simple.recommendation}"
                                    )
                                    status.text = live_status.text
                                except Exception as exc:  # noqa: BLE001
                                    live_status.text = f"Live error: {exc}"

                            def start_live() -> None:
                                stop_live()
                                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                                if not cap.isOpened():
                                    cap = cv2.VideoCapture(0)
                                if not cap.isOpened():
                                    live_status.text = "Cannot open webcam."
                                    return
                                live_state["cap"] = cap
                                live_state["running"] = True
                                live_state["timer"] = ui.timer(1.2, live_tick)
                                live_status.text = "Live · running flow…"

                            with ui.row().classes("w-full gap-2"):
                                ui.button("Start live", on_click=start_live).props(
                                    "unelevated dense color=primary"
                                ).classes("flex-grow")
                                ui.button("Stop", on_click=stop_live).props(
                                    "outline dense"
                                ).classes("flex-grow")

                        hosts["result"] = ui.column().classes("w-full gap-2")

                    with ui.column().classes("flex-grow min-w-0 h-full overflow-hidden gap-1"):
                        hosts["images"] = ui.column().classes(
                            "w-full h-full gap-2 min-w-0 overflow-auto"
                        )

        page_frame("Inspection Flow", content, lock_scroll=True)
