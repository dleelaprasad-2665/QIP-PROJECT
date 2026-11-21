"""Side-by-side experiment comparison page."""

from __future__ import annotations

from nicegui import ui

from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.ml.evaluation.experiment_manager import (
    compare_experiments,
    list_experiments,
)
from material_intelligence.app.ui.layout import page_frame


def register() -> None:
    @ui.page("/experiment-comparison")
    def comparison_page() -> None:
        def content() -> None:
            with SessionLocal() as db:
                experiments = list_experiments(db, limit=100)
            opts = {
                e.id: f"#{e.id} [{e.experiment_type}] {e.name}" for e in experiments
            }
            with ui.element("div").classes("mi-card p-4 w-full"):
                ui.label("Experiment Comparison").classes("font-semibold")
                ui.label(
                    "Compare only compatible metrics for the same task "
                    "(classification vs classification, etc.)."
                ).classes("text-sm mi-muted mb-3")
                pick = ui.select(opts, multiple=True, label="Select experiments").classes(
                    "w-full"
                )
                out = ui.column().classes("w-full gap-3")

                def run() -> None:
                    selected = pick.value or []
                    if len(selected) < 2:
                        ui.notify("Pick at least two experiments")
                        return
                    with SessionLocal() as db:
                        payload = compare_experiments(db, [int(i) for i in selected])
                    out.clear()
                    with out:
                        ui.label(payload["warning"]).classes("text-xs mi-muted")
                        with ui.row().classes("w-full gap-3 flex-wrap"):
                            for exp in payload["experiments"]:
                                with ui.element("div").classes("mi-card p-3 flex-1 min-w-[260px]"):
                                    ui.label(f"#{exp['id']} {exp['name']}").classes(
                                        "font-semibold"
                                    )
                                    ui.label(
                                        f"Type: {exp['type']} · model {exp['model_version']}"
                                    ).classes("text-xs mi-muted")
                                    ui.label(f"Images: {exp['n_images']} · seed {exp['seed']}")
                                    ui.label(str(exp["metrics"])[:800]).classes("text-xs mt-2")

                ui.button("Compare", on_click=run).props("unelevated")
                out
            ui.button("Back", on_click=lambda: ui.navigate.to("/evaluation")).props("flat")

        page_frame("Experiment Comparison", content)
