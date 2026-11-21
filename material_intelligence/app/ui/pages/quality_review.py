"""Quality review and approval workflow."""

from __future__ import annotations

from nicegui import ui
from sqlalchemy import select

from material_intelligence.app.db.models import Inspection, MaterialBatch
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.domain.enums import InspectionStatus
from material_intelligence.app.services.inspection_service import review_inspection
from material_intelligence.app.services.inventory_service import apply_quality_decision
from material_intelligence.app.ui.components.image_viewer import show_image
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.ui.state import current_user, get_app_state


def register() -> None:
    @ui.page("/quality-review")
    def quality_review_page() -> None:
        def content() -> None:
            user = current_user()
            state = get_app_state()
            detail = ui.column().classes("w-full gap-4")

            with SessionLocal() as db:
                query = select(Inspection).order_by(Inspection.created_at.desc()).limit(50)
                if state.current_site_id:
                    query = query.where(Inspection.site_id == state.current_site_id)
                inspections = db.scalars(query).all()
                batches = []
                if state.current_site_id:
                    batches = db.scalars(
                        select(MaterialBatch).where(MaterialBatch.site_id == state.current_site_id)
                    ).all()

            with ui.element("div").classes("mi-card p-4 w-full"):
                ui.label("Engineering review queue").classes("font-semibold mb-2")
                ui.label(
                    "Final approval is auditable. Visual AI classification alone must not approve structural materials."
                ).classes("text-sm mi-muted mb-3")
                if not inspections:
                    with ui.element("div").classes("mi-empty"):
                        ui.label("No inspections awaiting review")
                    return

                options = {
                    i.id: f"#{i.id} · {i.status} · batch={i.batch_id or '—'}" for i in inspections
                }
                selected = ui.select(options, label="Inspection").classes("w-full max-w-xl")
                decision = ui.select(
                    {
                        InspectionStatus.APPROVED.value: "Approved by authorized engineer",
                        InspectionStatus.REJECTED.value: "Rejected by authorized engineer",
                        InspectionStatus.QUARANTINED.value: "Quarantined",
                        InspectionStatus.ADDITIONAL_TESTING_REQUIRED.value: "Additional testing required",
                        InspectionStatus.HUMAN_INSPECTION_REQUIRED.value: "Human inspection required",
                    },
                    label="Decision",
                ).classes("w-full max-w-xl")
                notes = ui.textarea("Review notes").classes("w-full max-w-xl")
                apply_inventory = ui.switch("Also update linked batch inventory", value=False)
                qty = ui.number("Inventory quantity affected", value=1, min=0.1)
                status = ui.label("").classes("text-sm")

                def load_detail() -> None:
                    detail.clear()
                    if not selected.value:
                        return
                    with SessionLocal() as db:
                        insp = db.get(Inspection, int(selected.value))
                        if insp is None:
                            return
                        with detail:
                            with ui.row().classes("w-full gap-4 flex-wrap"):
                                show_image(insp.image_path, caption="Original")
                                show_image(insp.annotated_path, caption="Annotated")
                                show_image(insp.gradcam_path, caption="Grad-CAM")
                            ui.label(f"Current status: {insp.status}").classes("text-sm")
                            ui.label(f"Model versions: {insp.model_versions}").classes(
                                "text-xs mi-muted"
                            )

                def submit() -> None:
                    if not selected.value or not decision.value:
                        status.text = "Select inspection and decision."
                        return
                    try:
                        with SessionLocal() as db:
                            from material_intelligence.app.db.models import User

                            db_user = db.get(User, user.id) if user else None
                            insp = review_inspection(
                                db,
                                inspection_id=int(selected.value),
                                new_status=InspectionStatus(decision.value),
                                user=db_user,
                                notes=notes.value,
                            )
                            if apply_inventory.value and insp.batch_id:
                                apply_quality_decision(
                                    db,
                                    batch_id=insp.batch_id,
                                    status=InspectionStatus(decision.value),
                                    quantity=float(qty.value or 0),
                                    user=db_user,
                                    notes=notes.value,
                                )
                        status.text = "Review recorded in audit log."
                        load_detail()
                    except Exception as exc:  # noqa: BLE001
                        status.text = f"Failed: {exc}"

                ui.button("Load inspection", on_click=load_detail)
                ui.button("Submit review", on_click=submit).props("unelevated")
            detail

        page_frame("Quality Review", content)
