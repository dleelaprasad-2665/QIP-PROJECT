"""Simple table wrapper."""

from __future__ import annotations

from typing import Any

from nicegui import ui


def data_table(columns: list[dict[str, Any]], rows: list[dict[str, Any]], *, title: str | None = None) -> None:
    with ui.element("div").classes("mi-card p-4 w-full"):
        if title:
            ui.label(title).classes("text-base font-semibold mb-3")
        if not rows:
            with ui.element("div").classes("mi-empty"):
                ui.label("No records yet")
            return
        ui.table(columns=columns, rows=rows, row_key=columns[0]["name"]).classes("w-full")
