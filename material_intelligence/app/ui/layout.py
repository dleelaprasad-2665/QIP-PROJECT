"""Application shell with collapsible sidebar and top navigation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

from nicegui import ui
from sqlalchemy import select

from material_intelligence.app.db.models import Site
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.security.authorization import has_permission
from material_intelligence.app.ui.state import current_user, get_app_state, save_app_state
from material_intelligence.app.ui.theme import apply_theme

# Clear app flow: Inspect → Inventory → Reports (advanced pages stay registered).
NAV_ITEMS = [
    ("dashboard", "6. Dashboard", "/", "view"),
    ("search", "1–5. Inspection Flow", "/inspection", "inspect"),
    ("inventory_2", "Inventory", "/inventory", "manage_inventory"),
    ("description", "Reports", "/reports", "reports"),
    ("settings", "Administration", "/administration", "admin"),
]


def page_frame(
    title: str,
    builder: Callable[[], None],
    *,
    show_date_filters: bool = False,
    lock_scroll: bool = False,
) -> None:
    state = get_app_state()
    apply_theme(state.theme_mode)
    user = current_user()

    with SessionLocal() as db:
        sites = db.scalars(select(Site).where(Site.is_active.is_(True)).order_by(Site.name)).all()
        site_options = {s.id: f"{s.name} ({s.code})" for s in sites}
        if state.current_site_id is None and sites and title not in {"Overview", "Command Center"}:
            operational = next((s for s in sites if s.code != "DEMO-DATA"), sites[0])
            state.current_site_id = operational.id
            save_app_state(state)

    sidebar_classes = "mi-sidebar h-screen sticky top-0 p-3 gap-1"
    if state.sidebar_collapsed:
        sidebar_classes += " collapsed"

    shell_classes = "mi-shell w-full no-wrap"
    if lock_scroll:
        shell_classes += " mi-lock-scroll"
    with ui.row().classes(shell_classes):
        with ui.column().classes(sidebar_classes) as sidebar:
            with ui.row().classes("items-center gap-2 w-full mb-4 px-1"):
                ui.label("MI").classes("mi-brand-mark")
                with ui.column().classes("gap-0 flex-grow"):
                    ui.label("Material").classes("mi-brand mi-brand-line")
                    ui.label("Intelligence").classes("mi-brand mi-brand-line").style(
                        "margin-top:-2px; opacity:0.85;"
                    )
                    ui.label("BuildShield mode").classes("text-[11px] mi-brand-sub").style(
                        "color: var(--mi-accent); letter-spacing: 0.08em; text-transform: uppercase;"
                    )

                def _toggle_sidebar() -> None:
                    st = get_app_state()
                    st.sidebar_collapsed = not st.sidebar_collapsed
                    save_app_state(st)
                    ui.navigate.reload()

                ui.button(icon="menu", on_click=_toggle_sidebar).props(
                    "flat dense round color=grey-5"
                )

            ui.separator().classes("opacity-20 mb-2")

            for icon, label, path, permission in NAV_ITEMS:
                if user and permission not in {"view", "reports"} and not has_permission(user, permission):
                    if permission in {
                        "admin",
                        "train",
                        "approve",
                        "manage_datasets",
                        "evaluate",
                        "inspect",
                        "manage_inventory",
                    }:
                        if not has_permission(user, permission):
                            continue
                classes = "mi-nav-item px-3 py-2.5 w-full no-underline"
                if label.lower() in title.lower() or (
                    title in {"Overview", "Command Center"} and path == "/"
                ):
                    classes += " active"
                with ui.link(target=path).classes(classes):
                    ui.icon(icon).classes("text-[18px]")
                    ui.label(label).classes("mi-nav-label text-sm")

            ui.element("div").classes("flex-grow")
            with ui.column().classes("gap-0 px-2 mt-4 mi-brand-sub"):
                ui.label("v1 · Construction QA").classes("text-[10px]").style(
                    "color: var(--mi-sidebar-muted); letter-spacing: 0.06em;"
                )

        main_classes = "flex-grow h-screen"
        main_classes += " overflow-hidden" if lock_scroll else " overflow-auto"
        with ui.column().classes(main_classes):
            with ui.row().classes(
                "mi-topbar w-full items-center justify-between px-6 py-2.5 gap-3 flex-wrap shrink-0"
            ):
                with ui.column().classes("gap-0"):
                    ui.label(title).classes("mi-page-title")
                    ui.label("AI visual assessment for construction materials").classes(
                        "text-xs mi-muted"
                    )

                with ui.row().classes("items-center gap-3 flex-wrap"):
                    if site_options:
                        allow_all = title in {"Overview", "Command Center"}

                        def _on_site(e) -> None:  # noqa: ANN001
                            st = get_app_state()
                            st.current_site_id = None if e.value == "__all__" else e.value
                            save_app_state(st)
                            ui.navigate.reload()

                        options = (
                            {"__all__": "All sites", **site_options} if allow_all else site_options
                        )
                        current_value: int | str | None = state.current_site_id
                        if allow_all and state.current_site_id is None:
                            current_value = "__all__"
                        elif state.current_site_id not in site_options and not allow_all:
                            current_value = next(iter(site_options))
                        ui.select(
                            options,
                            value=current_value,
                            label="Project / Site",
                            on_change=_on_site,
                        ).props("dense outlined").classes("w-56")

                    if show_date_filters:
                        today = date.today()
                        default_from = state.date_from or (today - timedelta(days=30)).isoformat()
                        default_to = state.date_to or today.isoformat()

                        def _save_dates() -> None:
                            st = get_app_state()
                            st.date_from = date_from.value
                            st.date_to = date_to.value
                            save_app_state(st)
                            ui.navigate.reload()

                        date_from = (
                            ui.input("From", value=default_from)
                            .props("type=date dense outlined")
                            .classes("w-36")
                        )
                        date_to = (
                            ui.input("To", value=default_to)
                            .props("type=date dense outlined")
                            .classes("w-36")
                        )
                        ui.button(icon="event", on_click=_save_dates).props(
                            "flat dense"
                        ).tooltip("Apply date range")

                    def _toggle_theme() -> None:
                        st = get_app_state()
                        st.theme_mode = "light" if st.theme_mode == "dark" else "dark"
                        save_app_state(st)
                        ui.navigate.reload()

                    ui.button(
                        icon="light_mode" if state.theme_mode == "dark" else "dark_mode",
                        on_click=_toggle_theme,
                    ).props("flat dense round").tooltip("Toggle theme")

                    with ui.row().classes("items-center gap-2 px-2"):
                        ui.element("span").classes("mi-status-dot mi-status-ok")
                        ui.label("Online").classes("text-sm mi-muted")
                    if user:
                        with ui.column().classes("gap-0 items-end"):
                            ui.label(user.full_name).classes("text-sm font-medium")
                            ui.label(user.role).classes("text-[11px] mi-muted").style(
                                "text-transform: uppercase; letter-spacing: 0.05em;"
                            )
                        ui.button("Sign out", on_click=lambda: ui.navigate.to("/logout")).props(
                            "outline dense"
                        )
                    else:
                        ui.button("Sign in", on_click=lambda: ui.navigate.to("/login")).props(
                            "unelevated"
                        )

            body_classes = "p-4 gap-3 w-full max-w-[1520px] mx-auto flex-grow"
            if lock_scroll:
                body_classes += " overflow-hidden min-h-0"
            else:
                body_classes += " overflow-auto"
            with ui.column().classes(body_classes):
                if user is None and title not in {"Sign In"}:
                    with ui.element("div").classes("mi-empty"):
                        ui.label("Authentication required").classes("font-semibold")
                        ui.label("Sign in to access Material Intelligence modules.").classes(
                            "mi-muted"
                        )
                        ui.button("Go to Sign In", on_click=lambda: ui.navigate.to("/login")).props(
                            "unelevated"
                        )
                    return
                builder()
