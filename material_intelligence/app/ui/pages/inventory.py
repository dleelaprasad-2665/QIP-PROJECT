"""Material inventory management."""

from __future__ import annotations

from nicegui import ui
from sqlalchemy import select

from material_intelligence.app.db.models import MaterialBatch
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.domain.enums import MATERIAL_DISPLAY_NAMES
from material_intelligence.app.domain.schemas import InventoryBatchCreate
from material_intelligence.app.services.inventory_service import create_batch, site_inventory_summary
from material_intelligence.app.ui.components.metric_card import metric_card
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.ui.state import current_user, get_app_state


def register() -> None:
    @ui.page("/inventory")
    def inventory_page() -> None:
        def content() -> None:
            state = get_app_state()
            user = current_user()
            list_box = ui.column().classes("w-full gap-4")

            def refresh() -> None:
                list_box.clear()
                if state.current_site_id is None:
                    with list_box:
                        with ui.element("div").classes("mi-empty"):
                            ui.label("Select a site")
                    return
                with SessionLocal() as db:
                    summaries = site_inventory_summary(db, state.current_site_id)
                    batches = db.scalars(
                        select(MaterialBatch)
                        .where(MaterialBatch.site_id == state.current_site_id)
                        .order_by(MaterialBatch.created_at.desc())
                    ).all()
                with list_box:
                    with ui.row().classes("w-full gap-4 flex-wrap"):
                        for s in summaries:
                            metric_card(
                                MATERIAL_DISPLAY_NAMES.get(s.material_key, s.material_key),
                                f"{s.approved_quantity:.0f}/{s.total_quantity:.0f}",
                                hint=(
                                    f"Quarantine {s.quarantined_quantity:.0f} · "
                                    f"Rejected {s.rejected_quantity:.0f} · Pending {s.pending_review}"
                                ),
                            )
                        if not summaries:
                            with ui.element("div").classes("mi-empty flex-1"):
                                ui.label("No inventory batches for this site")

                    with ui.element("div").classes("mi-card p-4 w-full"):
                        ui.label("Batches").classes("font-semibold mb-2")
                        if not batches:
                            with ui.element("div").classes("mi-empty"):
                                ui.label("No batches")
                        else:
                            ui.table(
                                columns=[
                                    {"name": "code", "label": "Batch", "field": "code"},
                                    {"name": "material", "label": "Material", "field": "material"},
                                    {"name": "qty", "label": "Qty", "field": "qty"},
                                    {"name": "approved", "label": "Approved", "field": "approved"},
                                    {"name": "quarantine", "label": "Quarantine", "field": "quarantine"},
                                    {"name": "rejected", "label": "Rejected", "field": "rejected"},
                                    {"name": "status", "label": "Status", "field": "status"},
                                    {"name": "location", "label": "Location", "field": "location"},
                                ],
                                rows=[
                                    {
                                        "code": b.batch_code,
                                        "material": b.material_key,
                                        "qty": b.quantity,
                                        "approved": b.approved_quantity,
                                        "quarantine": b.quarantined_quantity,
                                        "rejected": b.rejected_quantity,
                                        "status": b.quality_status,
                                        "location": b.storage_location or "—",
                                    }
                                    for b in batches
                                ],
                                row_key="code",
                            ).classes("w-full")

            with ui.element("div").classes("mi-card p-4 w-full"):
                ui.label("Register material batch").classes("font-semibold mb-2")
                ui.label(
                    "AI predictions do not change approved inventory quantities without the approval workflow."
                ).classes("text-sm mi-muted mb-3")
                material = ui.select(MATERIAL_DISPLAY_NAMES, label="Material", value="clay_brick")
                batch_code = ui.input("Batch code", value="BATCH-001").classes("w-full max-w-md")
                qty = ui.number("Quantity", value=100, min=1)
                unit = ui.input("Unit", value="units").classes("w-full max-w-md")
                supplier = ui.input("Supplier").classes("w-full max-w-md")
                location = ui.input("Storage location").classes("w-full max-w-md")
                purchase_ref = ui.input("Purchase reference").classes("w-full max-w-md")
                status = ui.label("").classes("text-sm")

                def add_batch() -> None:
                    if state.current_site_id is None:
                        status.text = "Select a site first."
                        return
                    payload = InventoryBatchCreate(
                        site_id=state.current_site_id,
                        material_key=material.value or "clay_brick",
                        batch_code=batch_code.value or "BATCH",
                        supplier_name=supplier.value or None,
                        quantity=float(qty.value or 0),
                        unit=unit.value or "units",
                        storage_location=location.value or None,
                        purchase_reference=purchase_ref.value or None,
                    )
                    try:
                        with SessionLocal() as db:
                            from material_intelligence.app.db.models import User

                            db_user = db.get(User, user.id) if user else None
                            create_batch(db, payload, db_user)
                        status.text = "Batch created."
                        refresh()
                    except Exception as exc:  # noqa: BLE001
                        status.text = f"Failed: {exc}"

                ui.button("Create batch", on_click=add_batch).props("unelevated")

            refresh()

        page_frame("Material Inventory", content)
