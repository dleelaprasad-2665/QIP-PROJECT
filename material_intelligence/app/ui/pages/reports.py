"""Reports page."""

from __future__ import annotations

from nicegui import ui

from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.services.reporting_service import (
    evaluation_results_csv,
    generate_site_pdf,
    inspections_csv,
    inventory_csv,
)
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.ui.state import get_app_state


def register() -> None:
    @ui.page("/reports")
    def reports_page() -> None:
        def content() -> None:
            state = get_app_state()
            status = ui.label("").classes("text-sm")

            with ui.element("div").classes("mi-card p-4 w-full"):
                ui.label("Export reports").classes("font-semibold mb-2")
                ui.label(
                    "Reports are generated from persisted database records only. "
                    "Each export includes data-source attribution. "
                    "Model evaluation exports never invent metrics — empty results show not_evaluated."
                ).classes("text-sm mi-muted mb-3")

                def download_insp_csv() -> None:
                    with SessionLocal() as db:
                        csv_text = inspections_csv(db, state.current_site_id)
                    ui.download(csv_text.encode("utf-8"), filename="inspections.csv")
                    status.text = "Inspections CSV ready."

                def download_inv_csv() -> None:
                    with SessionLocal() as db:
                        csv_text = inventory_csv(db, state.current_site_id)
                    ui.download(csv_text.encode("utf-8"), filename="inventory.csv")
                    status.text = "Inventory CSV ready."

                def download_eval_csv() -> None:
                    with SessionLocal() as db:
                        csv_text = evaluation_results_csv(db)
                    ui.download(csv_text.encode("utf-8"), filename="model_evaluation.csv")
                    status.text = "Model evaluation CSV ready."

                def download_pdf() -> None:
                    if state.current_site_id is None:
                        status.text = "Select a site first."
                        return
                    with SessionLocal() as db:
                        path = generate_site_pdf(db, state.current_site_id)
                    ui.download(str(path))
                    status.text = f"PDF generated: {path.name}"

                with ui.row().classes("gap-3 flex-wrap"):
                    ui.button("Inspections CSV", on_click=download_insp_csv)
                    ui.button("Inventory CSV", on_click=download_inv_csv)
                    ui.button("Model evaluation CSV", on_click=download_eval_csv)
                    ui.button("Site PDF report", on_click=download_pdf).props("unelevated")

                ui.label(
                    "For filtered dashboard exports (date range + material/supplier/status), "
                    "use the Overview command-center export buttons."
                ).classes("text-xs mi-muted mt-3")

        page_frame("Reports", content)
