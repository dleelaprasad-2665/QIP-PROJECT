"""Plotly chart panel."""

from __future__ import annotations

from typing import Any

from nicegui import ui


def chart_panel(figure: dict[str, Any] | None, *, title: str, empty_message: str = "Not evaluated") -> None:
    with ui.element("div").classes("mi-card p-4 w-full"):
        ui.label(title).classes("text-base font-semibold mb-3")
        if figure is None:
            with ui.element("div").classes("mi-empty"):
                ui.label(empty_message)
            return
        ui.plotly(figure).classes("w-full")
