"""Dedicated clustering analysis view with projection scatter."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from nicegui import ui
from sqlalchemy import select

from material_intelligence.app.db.models import EvaluationExperiment
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.ui.components.chart_panel import chart_panel
from material_intelligence.app.ui.layout import page_frame


def register() -> None:
    @ui.page("/clustering-analysis")
    def clustering_page() -> None:
        def content() -> None:
            with SessionLocal() as db:
                experiments = db.scalars(
                    select(EvaluationExperiment)
                    .where(EvaluationExperiment.experiment_type == "clustering")
                    .order_by(EvaluationExperiment.created_at.desc())
                ).all()

            with ui.element("div").classes("mi-card p-4 w-full"):
                ui.label("Clustering Analysis").classes("font-semibold")
                ui.label(
                    "2D projections are visualization aids and may distort distances. "
                    "Clustering ran on prepared high-dimensional embeddings."
                ).classes("text-sm mi-muted mb-3")
                if not experiments:
                    ui.label("Not Evaluated — run clustering from Model Evaluation.").classes(
                        "mi-muted"
                    )
                    return
                pick = ui.select(
                    {e.id: f"#{e.id} {e.name}" for e in experiments},
                    value=experiments[0].id,
                    label="Clustering experiment",
                ).classes("w-full max-w-xl")
                panel = ui.column().classes("w-full gap-3")

                def render() -> None:
                    panel.clear()
                    with SessionLocal() as db:
                        exp = db.get(EvaluationExperiment, int(pick.value))
                    if exp is None:
                        return
                    arts = exp.artifacts or {}
                    proj = arts.get("projection_2d") or []
                    labels = arts.get("labels") or []
                    paths = arts.get("paths") or []
                    with panel:
                        with ui.element("div").classes("mi-kpi-grid"):
                            for key in (
                                "n_clusters",
                                "silhouette",
                                "davies_bouldin",
                                "calinski_harabasz",
                                "n_noise",
                            ):
                                val = (exp.metrics or {}).get(key)
                                ui.label(f"{key}: {val if val is not None else 'Not Evaluated'}").classes(
                                    "text-sm"
                                )
                        if proj and labels:
                            xs = [p[0] for p in proj]
                            ys = [p[1] if len(p) > 1 else 0 for p in proj]
                            fig = go.Figure(
                                data=[
                                    go.Scatter(
                                        x=xs,
                                        y=ys,
                                        mode="markers",
                                        marker=dict(
                                            color=labels,
                                            colorscale="Viridis",
                                            size=8,
                                            showscale=True,
                                        ),
                                        text=[
                                            f"{Path(p).name} · c={c}"
                                            for p, c in zip(paths, labels, strict=False)
                                        ],
                                        hoverinfo="text",
                                    )
                                ]
                            )
                            fig.update_layout(
                                title="PCA projection (visualization only)",
                                height=420,
                                paper_bgcolor="#141414",
                                plot_bgcolor="#141414",
                                font_color="#f5f3ef",
                            )
                            chart_panel(fig, title="Cluster projection")
                        for note in (arts.get("notes") or []):
                            ui.label(f"• {note}").classes("text-xs mi-muted")
                        for summary in arts.get("cluster_summaries") or []:
                            with ui.element("div").classes("mi-card p-3"):
                                ui.label(
                                    f"Cluster {summary['cluster_id']} · {summary['size']} images"
                                ).classes("font-semibold")
                                ui.label(
                                    "Interpretation must be assigned by researchers — "
                                    "not auto-named Good/Damaged."
                                ).classes("text-xs mi-muted")

                ui.button("Load", on_click=render).props("unelevated")
                render()
            ui.button("Back", on_click=lambda: ui.navigate.to("/evaluation")).props("flat")

        page_frame("Clustering Analysis", content)
