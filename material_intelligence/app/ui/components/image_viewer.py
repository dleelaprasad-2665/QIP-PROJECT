"""Image viewer helpers with click-to-enlarge lightbox."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from material_intelligence.app.config import get_settings


def media_url(path: str | Path | None) -> str | None:
    """Map a filesystem path under data_dir to the /data static URL."""
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    data_dir = get_settings().data_dir.resolve()
    try:
        relative = file_path.resolve().relative_to(data_dir)
        return f"/data/{relative.as_posix()}"
    except ValueError:
        return str(file_path)


def show_image(
    path: str | Path | None,
    *,
    caption: str | None = None,
    height: str = "220px",
    large: bool = True,
) -> None:
    url = media_url(path)
    with ui.column().classes("gap-1 w-full min-w-0"):
        if caption:
            ui.label(caption).classes("text-xs mi-muted font-semibold")
        if url:
            img = (
                ui.image(url)
                .classes("w-full border cursor-pointer")
                .style(
                    f"border-color: var(--mi-border); max-height: {height}; "
                    "object-fit: contain; background: #0d0d0d; border-radius: 4px;"
                )
            )
            if large:

                def _open(u=url, c=caption or "Image") -> None:
                    with ui.dialog() as dialog, ui.card().classes("p-3").style(
                        "max-width: 96vw; max-height: 94vh; background: #0d0d0d;"
                    ):
                        ui.label(c).classes("text-sm font-semibold mb-2").style("color:#f5f3ef;")
                        ui.image(u).style(
                            "max-width: 92vw; max-height: 82vh; object-fit: contain;"
                        )
                        ui.button("Close", on_click=dialog.close).props("flat dense")
                    dialog.open()

                img.on("click", _open)
                ui.label("Click to enlarge").classes("text-[10px] mi-muted")
        else:
            with ui.element("div").classes("mi-empty w-full"):
                ui.label("No image available")
