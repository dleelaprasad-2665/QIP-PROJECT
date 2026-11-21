"""Shared UI session state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from nicegui import app

from material_intelligence.app.db.models import User
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.security.authentication import get_user_by_session


@dataclass
class AppState:
    current_site_id: int | None = None
    notifications: list[str] = field(default_factory=list)
    theme_mode: str = "dark"  # dark | light
    sidebar_collapsed: bool = False
    date_from: str | None = None
    date_to: str | None = None
    include_demo: bool = False


def get_app_state() -> AppState:
    if "mi_state" not in app.storage.user:
        app.storage.user["mi_state"] = {
            "current_site_id": None,
            "notifications": [],
            "theme_mode": "dark",
            "sidebar_collapsed": False,
            "date_from": None,
            "date_to": None,
            "include_demo": False,
        }
    raw = app.storage.user["mi_state"]
    return AppState(
        current_site_id=raw.get("current_site_id"),
        notifications=list(raw.get("notifications") or []),
        theme_mode=raw.get("theme_mode") or "dark",
        sidebar_collapsed=bool(raw.get("sidebar_collapsed")),
        date_from=raw.get("date_from"),
        date_to=raw.get("date_to"),
        include_demo=raw.get("include_demo", True),
    )


def save_app_state(state: AppState) -> None:
    app.storage.user["mi_state"] = {
        "current_site_id": state.current_site_id,
        "notifications": state.notifications,
        "theme_mode": state.theme_mode,
        "sidebar_collapsed": state.sidebar_collapsed,
        "date_from": state.date_from,
        "date_to": state.date_to,
        "include_demo": state.include_demo,
    }


def parse_state_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def current_user() -> User | None:
    token = app.storage.user.get("session_token")
    with SessionLocal() as db:
        user = get_user_by_session(db, token)
        if user is None:
            return None
        db.expunge(user)
        return user


def set_session_token(token: str | None) -> None:
    if token is None:
        app.storage.user.pop("session_token", None)
    else:
        app.storage.user["session_token"] = token
