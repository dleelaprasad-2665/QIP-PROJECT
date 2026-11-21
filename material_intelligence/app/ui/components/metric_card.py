"""Metric summary card with optional delta and navigation link."""

from __future__ import annotations

from nicegui import ui

from material_intelligence.app.services.dashboard_analytics import KPIValue


def metric_card(
    label: str,
    value: str,
    *,
    hint: str | None = None,
    delta_pct: float | None = None,
    link: str | None = None,
) -> None:
    with ui.element("div").classes("mi-card mi-kpi"):
        with ui.row().classes("items-start justify-between w-full gap-2"):
            ui.label(label).classes("mi-kpi-label")
            if link:
                ui.link("Open →", link).classes("text-xs").style(
                    "color: var(--mi-accent); text-decoration: none; font-weight: 600;"
                )
        ui.label(value).classes("mi-metric-value")
        if delta_pct is not None:
            cls = "mi-delta-up" if delta_pct >= 0 else "mi-delta-down"
            sign = "+" if delta_pct >= 0 else ""
            arrow = "▲" if delta_pct >= 0 else "▼"
            ui.label(f"{arrow} {sign}{delta_pct}% vs prior period").classes(cls)
        if hint:
            ui.label(hint).classes("text-xs mi-muted mt-1")


def render_kpi(kpi: KPIValue) -> None:
    display = str(kpi.value)
    if kpi.unit and isinstance(kpi.value, (int, float)):
        display = f"{kpi.value} {kpi.unit}"
    elif kpi.unit and display not in {"—", ""}:
        display = f"{display} {kpi.unit}"
    metric_card(
        kpi.label,
        display,
        hint=kpi.hint,
        delta_pct=kpi.delta_pct,
        link=kpi.link,
    )
