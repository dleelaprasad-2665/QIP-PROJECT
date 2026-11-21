"""Administration — users, materials, sites."""

from __future__ import annotations

from nicegui import ui
from sqlalchemy import select

from material_intelligence.app.db.models import AuditEvent, DefectType, MaterialType, Site, User
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.domain.enums import UserRole
from material_intelligence.app.security.authentication import hash_password
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.ui.state import current_user


def register() -> None:
    @ui.page("/administration")
    def admin_page() -> None:
        def content() -> None:
            user = current_user()
            with SessionLocal() as db:
                users = db.scalars(select(User).order_by(User.created_at.desc())).all()
                materials = db.scalars(select(MaterialType).order_by(MaterialType.key)).all()
                sites = db.scalars(select(Site).order_by(Site.name)).all()
                audits = db.scalars(
                    select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(30)
                ).all()

            with ui.row().classes("w-full gap-4 flex-wrap"):
                with ui.element("div").classes("mi-card p-4 flex-1 min-w-[320px]"):
                    ui.label("Create user").classes("font-semibold mb-2")
                    email = ui.input("Email").classes("w-full")
                    full_name = ui.input("Full name").classes("w-full")
                    password = ui.input("Password", password=True).classes("w-full")
                    role = ui.select({r.value: r.value for r in UserRole}, value=UserRole.VIEWER.value)
                    msg = ui.label("").classes("text-sm")

                    def create_user() -> None:
                        try:
                            with SessionLocal() as db:
                                db.add(
                                    User(
                                        email=(email.value or "").lower(),
                                        full_name=full_name.value or "User",
                                        password_hash=hash_password(password.value or "changeme"),
                                        role=role.value,
                                        is_active=True,
                                    )
                                )
                                db.commit()
                            msg.text = "User created."
                        except Exception as exc:  # noqa: BLE001
                            msg.text = f"Failed: {exc}"

                    ui.button("Create", on_click=create_user)

                with ui.element("div").classes("mi-card p-4 flex-1 min-w-[320px]"):
                    ui.label("Add material / defect class").classes("font-semibold mb-2")
                    mat_key = ui.input("Material key", placeholder="e.g. timber_plank").classes(
                        "w-full"
                    )
                    mat_name = ui.input("Material display name").classes("w-full")
                    defect_key = ui.input("Defect key", placeholder="e.g. splintered").classes(
                        "w-full"
                    )
                    defect_name = ui.input("Defect display name").classes("w-full")
                    high_risk = ui.switch("High risk", value=True)
                    mat_msg = ui.label("").classes("text-sm")

                    def add_taxonomy() -> None:
                        key = (mat_key.value or "").strip().lower().replace(" ", "_")
                        dkey = (defect_key.value or "").strip().lower().replace(" ", "_")
                        if not key or not dkey:
                            mat_msg.text = "Material and defect keys are required."
                            return
                        with SessionLocal() as db:
                            material = db.scalar(select(MaterialType).where(MaterialType.key == key))
                            if material is None:
                                material = MaterialType(
                                    key=key,
                                    display_name=mat_name.value or key,
                                )
                                db.add(material)
                                db.flush()
                            db.add(
                                DefectType(
                                    material_id=material.id,
                                    key=dkey,
                                    display_name=defect_name.value or dkey,
                                    is_high_risk=bool(high_risk.value),
                                )
                            )
                            db.commit()
                        mat_msg.text = "Taxonomy updated."

                    ui.button("Save taxonomy", on_click=add_taxonomy)

                with ui.element("div").classes("mi-card p-4 flex-1 min-w-[320px]"):
                    ui.label("Add construction site").classes("font-semibold mb-2")
                    site_name = ui.input("Site name").classes("w-full")
                    site_code = ui.input("Site code").classes("w-full")
                    site_loc = ui.input("Location").classes("w-full")
                    site_msg = ui.label("").classes("text-sm")

                    def add_site() -> None:
                        with SessionLocal() as db:
                            db.add(
                                Site(
                                    name=site_name.value or "Site",
                                    code=(site_code.value or "SITE").upper(),
                                    location=site_loc.value,
                                )
                            )
                            db.commit()
                        site_msg.text = "Site created. Refresh page to see it in the selector."

                    ui.button("Create site", on_click=add_site)

            with ui.element("div").classes("mi-card p-4 w-full"):
                ui.label("Users").classes("font-semibold mb-2")
                ui.table(
                    columns=[
                        {"name": "email", "label": "Email", "field": "email"},
                        {"name": "name", "label": "Name", "field": "name"},
                        {"name": "role", "label": "Role", "field": "role"},
                        {"name": "active", "label": "Active", "field": "active"},
                    ],
                    rows=[
                        {
                            "email": u.email,
                            "name": u.full_name,
                            "role": u.role,
                            "active": u.is_active,
                        }
                        for u in users
                    ],
                    row_key="email",
                ).classes("w-full")

            with ui.row().classes("w-full gap-4 flex-wrap"):
                with ui.element("div").classes("mi-card p-4 flex-1"):
                    ui.label("Materials").classes("font-semibold mb-2")
                    for m in materials:
                        ui.label(f"{m.display_name} ({m.key})").classes("text-sm")
                with ui.element("div").classes("mi-card p-4 flex-1"):
                    ui.label("Sites").classes("font-semibold mb-2")
                    for s in sites:
                        ui.label(f"{s.name} · {s.code}").classes("text-sm")

            with ui.element("div").classes("mi-card p-4 w-full"):
                ui.label("Recent audit events").classes("font-semibold mb-2")
                if not audits:
                    with ui.element("div").classes("mi-empty"):
                        ui.label("No audit events")
                else:
                    ui.table(
                        columns=[
                            {"name": "when", "label": "When", "field": "when"},
                            {"name": "action", "label": "Action", "field": "action"},
                            {"name": "entity", "label": "Entity", "field": "entity"},
                            {"name": "actor", "label": "Actor", "field": "actor"},
                        ],
                        rows=[
                            {
                                "when": str(a.created_at),
                                "action": a.action,
                                "entity": f"{a.entity_type}:{a.entity_id or ''}",
                                "actor": a.actor_id,
                            }
                            for a in audits
                        ],
                        row_key="when",
                    ).classes("w-full")

            ui.label(f"Signed in as {user.email if user else 'unknown'}").classes(
                "text-xs mi-muted"
            )

        page_frame("Administration", content)
