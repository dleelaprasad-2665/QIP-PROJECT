"""Authentication pages."""

from __future__ import annotations

from nicegui import ui

from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.security.authentication import authenticate_user, create_session, revoke_session
from material_intelligence.app.ui.state import set_session_token
from material_intelligence.app.ui.theme import apply_theme


def register() -> None:
    @ui.page("/login")
    def login_page() -> None:
        apply_theme()
        with ui.column().classes("w-full min-h-screen items-center justify-center p-6"):
            with ui.element("div").classes("mi-card p-8 w-full max-w-md"):
                with ui.row().classes("items-center gap-3 mb-1"):
                    ui.label("MI").classes("mi-brand-mark")
                    with ui.column().classes("gap-0"):
                        ui.label("MATERIAL INTELLIGENCE").classes("mi-brand").style(
                            "color: var(--mi-text);"
                        )
                        ui.label("BuildShield industrial UI").classes(
                            "text-[11px] mi-muted"
                        ).style("letter-spacing: 0.1em; text-transform: uppercase;")
                ui.element("div").classes("mi-hero-rule")
                ui.label("Sign in to continue").classes("text-sm mi-muted mb-4")
                email = ui.input("Email").props("dense outlined").classes("w-full")
                password = (
                    ui.input("Password", password=True, password_toggle_button=True)
                    .props("dense outlined")
                    .classes("w-full")
                )
                status = ui.label("").classes("text-sm")

                def _submit() -> None:
                    with SessionLocal() as db:
                        user = authenticate_user(db, email.value or "", password.value or "")
                        if user is None:
                            status.text = "Invalid credentials or inactive account."
                            status.classes("text-negative")
                            return
                        token = create_session(db, user)
                        set_session_token(token)
                    ui.navigate.to("/")

                ui.button("Sign in", on_click=_submit).props("unelevated").classes("w-full mt-2")
                ui.label(
                    "Default bootstrap admin is configured via environment variables."
                ).classes("text-xs mi-muted mt-4")

    @ui.page("/logout")
    def logout_page() -> None:
        from nicegui import app

        token = app.storage.user.get("session_token")
        if token:
            with SessionLocal() as db:
                revoke_session(db, token)
        set_session_token(None)
        ui.navigate.to("/login")
