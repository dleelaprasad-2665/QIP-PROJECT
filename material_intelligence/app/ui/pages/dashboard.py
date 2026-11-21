"""Premium intelligent command-center dashboard (default landing page)."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.security.authorization import has_permission
from material_intelligence.app.services.dashboard_analytics import (
    DashboardFilters,
    build_dashboard_snapshot,
    last_activity_stamp,
)
from material_intelligence.app.services.reporting_service import (
    filtered_inspections_csv,
    generate_filtered_inspection_pdf,
    inventory_csv,
)
from material_intelligence.app.ui.components.metric_card import render_kpi
from material_intelligence.app.ui.components.plotly_theme import (
    bar_chart,
    confusion_heatmap,
    grouped_outcomes,
    line_chart,
    pie_chart,
)
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.ui.state import current_user, get_app_state, parse_state_date, save_app_state


def register() -> None:
    @ui.page("/")
    def overview() -> None:
        def content() -> None:
            user = current_user()
            state = get_app_state()
            filter_state: dict = {
                "material_key": None,
                "supplier_id": None,
                "status": None,
                "model_version": None,
                "search": "",
                "page": 1,
                "activity": None,
            }

            banner = ui.column().classes("w-full gap-3")
            filter_bar = ui.row().classes("mi-filter-bar w-full gap-3 flex-wrap items-end")
            body = ui.column().classes("w-full gap-6")
            status_line = ui.label("").classes("text-xs mi-muted")

            def _filters() -> DashboardFilters:
                st = get_app_state()
                return DashboardFilters(
                    site_id=st.current_site_id,
                    material_key=filter_state["material_key"],
                    supplier_id=filter_state["supplier_id"],
                    status=filter_state["status"],
                    model_version=filter_state["model_version"],
                    date_from=parse_state_date(st.date_from),
                    date_to=parse_state_date(st.date_to),
                    search=filter_state["search"] or None,
                    include_demo=st.include_demo,
                    page=filter_state["page"],
                    page_size=10,
                )

            def render() -> None:
                with SessionLocal() as db:
                    snap = build_dashboard_snapshot(db, _filters())
                    filter_state["activity"] = last_activity_stamp(db)

                banner.clear()
                with banner:
                    with ui.element("div").classes("mi-hero w-full"):
                        ui.element("div").classes("mi-hero-stripe")
                        with ui.row().classes(
                            "items-end justify-between w-full flex-wrap gap-4"
                        ).style("position: relative; z-index: 1;"):
                            with ui.column().classes("gap-0"):
                                ui.label("MATERIAL INTELLIGENCE").classes(
                                    "text-[11px] font-semibold"
                                ).style(
                                    "letter-spacing: 0.16em; color: #f5a623; "
                                    "font-family: Oswald, sans-serif;"
                                )
                                ui.html(
                                    '<div class="mi-hero-brand">FIELD COMMAND '
                                    '<span>CENTER</span></div>',
                                    sanitize=False,
                                )
                                ui.element("div").classes("mi-hero-rule")
                                ui.label(
                                    "Construction-grade visual assessment — materials, "
                                    "defects, inventory, and quality gates under one roof."
                                ).classes("text-sm mi-muted max-w-xl")
                                ui.label(
                                    f"Source · {snap.data_source}  ·  "
                                    f"{snap.generated_at.strftime('%Y-%m-%d %H:%M UTC')}"
                                ).classes("text-xs mi-muted mt-2")
                            with ui.row().classes("gap-2 items-center"):
                                if has_permission(user, "reports"):
                                    ui.button("Export CSV", on_click=_export_csv).props(
                                        "outline dense color=orange"
                                    )
                                    ui.button("Export PDF", on_click=_export_pdf).props(
                                        "outline dense color=orange"
                                    )
                                    ui.button("Inventory CSV", on_click=_export_inventory).props(
                                        "outline dense color=orange"
                                    )
                                ui.button("Refresh", icon="refresh", on_click=render).props(
                                    "unelevated dense"
                                )

                    if snap.is_demo_present and state.include_demo:
                        with ui.element("div").classes("mi-card p-3 w-full").style(
                            "border-color: color-mix(in srgb, var(--mi-warning) 55%, var(--mi-border));"
                        ):
                            with ui.row().classes("items-center gap-2"):
                                ui.label("DEMO").classes("mi-chip mi-chip-demo")
                                ui.label("Demo data is visible in KPIs and tables").classes(
                                    "font-semibold text-sm"
                                )
                            ui.label(
                                "Records labeled DEMO DATA are not operational. "
                                "Turn off Include demo data to hide them."
                            ).classes("text-sm mi-muted mt-1")

                filter_bar.clear()
                with filter_bar:
                    opts = snap.filter_options
                    material = (
                        ui.select(
                            {"__all__": "All materials", **opts.get("materials", {})},
                            value=filter_state["material_key"] or "__all__",
                            label="Material",
                        )
                        .props("dense outlined")
                        .classes("w-44")
                    )
                    supplier = (
                        ui.select(
                            {
                                "__all__": "All suppliers",
                                **{str(k): v for k, v in opts.get("suppliers", {}).items()},
                            },
                            value=str(filter_state["supplier_id"])
                            if filter_state["supplier_id"]
                            else "__all__",
                            label="Supplier",
                        )
                        .props("dense outlined")
                        .classes("w-44")
                    )
                    status = (
                        ui.select(
                            {"__all__": "All statuses", **opts.get("statuses", {})},
                            value=filter_state["status"] or "__all__",
                            label="Inspection status",
                        )
                        .props("dense outlined")
                        .classes("w-52")
                    )
                    model_ver = (
                        ui.select(
                            {"__all__": "All model versions", **opts.get("model_versions", {})},
                            value=filter_state["model_version"] or "__all__",
                            label="AI model version",
                        )
                        .props("dense outlined")
                        .classes("w-48")
                    )
                    search = (
                        ui.input("Search inspections", value=filter_state["search"] or "")
                        .props("dense outlined")
                        .classes("w-52")
                    )
                    include_demo = ui.switch("Include demo data", value=state.include_demo)

                    def apply_filters() -> None:
                        filter_state["material_key"] = (
                            None if material.value == "__all__" else material.value
                        )
                        filter_state["supplier_id"] = (
                            None if supplier.value == "__all__" else int(supplier.value)
                        )
                        filter_state["status"] = None if status.value == "__all__" else status.value
                        filter_state["model_version"] = (
                            None if model_ver.value == "__all__" else model_ver.value
                        )
                        filter_state["search"] = search.value or ""
                        filter_state["page"] = 1
                        st = get_app_state()
                        st.include_demo = bool(include_demo.value)
                        save_app_state(st)
                        render()

                    ui.button("Apply filters", on_click=apply_filters).props("unelevated dense")

                body.clear()
                with body:
                    _section_kpis(snap.kpis)
                    _section_charts(snap.charts)
                    _section_inventory(snap.inventory)
                    _section_models(snap.model_performance)
                    _section_recent(snap.recent_inspections, filter_state, render)
                    if has_permission(user, "approve") or has_permission(user, "view"):
                        _section_quality(snap.quality_review, user)
                    _section_reporting_note(snap)

                status_line.text = (
                    f"Filters applied · site={snap.filters.get('site_id') or 'all'} · "
                    f"range={snap.filters.get('date_from') or '—'} → {snap.filters.get('date_to') or '—'} · "
                    f"inspections page {snap.recent_inspections.get('page')}/{snap.recent_inspections.get('pages')}"
                )

            def _export_csv() -> None:
                with SessionLocal() as db:
                    csv_text = filtered_inspections_csv(db, _filters())
                ui.download(csv_text.encode("utf-8"), filename="dashboard_inspections.csv")

            def _export_pdf() -> None:
                with SessionLocal() as db:
                    path = generate_filtered_inspection_pdf(db, _filters())
                ui.download(str(path))

            def _export_inventory() -> None:
                with SessionLocal() as db:
                    csv_text = inventory_csv(db, get_app_state().current_site_id)
                ui.download(csv_text.encode("utf-8"), filename="dashboard_inventory.csv")

            def _poll() -> None:
                with SessionLocal() as db:
                    stamp = last_activity_stamp(db)
                if filter_state["activity"] and stamp != filter_state["activity"]:
                    render()

            render()
            # Lightweight change detection every 20s — only reloads when data stamp changes.
            ui.timer(20.0, _poll)

        page_frame("Overview", content, show_date_filters=True)


def _section_header(title: str, subtitle: str | None = None) -> None:
    with ui.column().classes("mi-section gap-0 w-full"):
        ui.label(title).classes("mi-section-title")
        if subtitle:
            ui.label(subtitle).classes("mi-section-sub")


def _section_kpis(kpis) -> None:
    with ui.column().classes("w-full gap-3"):
        _section_header("Executive overview", "Key performance indicators for the selected filters")
        with ui.element("div").classes("mi-kpi-grid"):
            for kpi in kpis:
                render_kpi(kpi)


def _chart_or_empty(title: str, figure) -> None:
    with ui.element("div").classes("mi-card p-3 w-full"):
        if figure is None:
            ui.label(title).classes("font-medium mb-2")
            with ui.element("div").classes("mi-empty"):
                ui.label("No data for current filters")
            return
        ui.plotly(figure).classes("w-full")


def _section_charts(charts: dict) -> None:
    _section_header("AI inspection analytics", "Trends, severity, confidence, and recurring defects")
    if charts.get("empty"):
        with ui.element("div").classes("mi-empty w-full"):
            ui.label("No inspection analytics available")
            ui.label("Complete inspections or adjust filters to populate charts.").classes("text-sm")
        return

    with ui.element("div").classes("mi-chart-grid"):
        ot = charts.get("inspections_over_time") or {}
        _chart_or_empty(
            "Inspections over time",
            line_chart(ot.get("x", []), ot.get("y", []), title="Inspections over time")
            if ot.get("x")
            else None,
        )
        dm = charts.get("defects_by_material") or {}
        _chart_or_empty(
            "Defects by material type",
            bar_chart(dm.get("x", []), dm.get("y", []), title="Defects by material type")
            if dm.get("x")
            else None,
        )
        sev = charts.get("severity_distribution") or {}
        _chart_or_empty(
            "Defect severity distribution",
            pie_chart(sev.get("labels", []), sev.get("values", []), title="Defect severity")
            if sev.get("labels")
            else None,
        )
        conf = charts.get("confidence_distribution") or {}
        _chart_or_empty(
            "Classification confidence distribution",
            bar_chart(conf.get("x", []), conf.get("y", []), title="Confidence distribution")
            if conf.get("x")
            else None,
        )
        out = charts.get("outcomes_by_site") or {}
        _chart_or_empty(
            "Outcomes by site",
            grouped_outcomes(
                out.get("sites", []),
                out.get("statuses", []),
                out.get("counts", []),
                title="Inspection outcomes by site",
            )
            if out.get("sites")
            else None,
        )
        top = charts.get("top_defects") or {}
        _chart_or_empty(
            "Top recurring defects",
            bar_chart(
                top.get("x", []),
                top.get("y", []),
                title="Top recurring visual defect types",
                orientation="h",
            )
            if top.get("x")
            else None,
        )


def _section_inventory(inventory: dict) -> None:
    _section_header("Material inventory", "Stock levels, batches, suppliers, and storage")
    with ui.row().classes("w-full gap-4 flex-wrap"):
        with ui.element("div").classes("mi-card p-4 flex-1 min-w-[280px]"):
            with ui.row().classes("justify-between items-center"):
                ui.label("Material-wise availability").classes("font-semibold")
                ui.link("Open inventory →", "/inventory").classes("text-xs").style(
                    "color: var(--mi-accent); text-decoration: none; font-weight: 600;"
                )
            rows = inventory.get("material_stock") or []
            if not rows:
                with ui.element("div").classes("mi-empty"):
                    ui.label("No inventory records")
            else:
                ui.table(
                    columns=[
                        {"name": "material_key", "label": "Material", "field": "material_key"},
                        {"name": "unit", "label": "Unit", "field": "unit"},
                        {"name": "approved", "label": "Approved", "field": "approved"},
                        {"name": "quarantined", "label": "Held", "field": "quarantined"},
                        {"name": "rejected", "label": "Rejected", "field": "rejected"},
                        {"name": "quantity", "label": "Total", "field": "quantity"},
                    ],
                    rows=rows,
                    row_key="material_key",
                    pagination=5,
                ).classes("w-full")

        with ui.element("div").classes("mi-card p-4 flex-1 min-w-[280px]"):
            ui.label("Low-stock alerts").classes("font-semibold mb-2")
            alerts = inventory.get("low_stock") or []
            if not alerts:
                with ui.element("div").classes("mi-empty"):
                    ui.label("No low-stock alerts")
            else:
                for alert in alerts:
                    ui.label(
                        f"{alert['material_key']} ({alert['unit']}): "
                        f"approved {alert['approved']} / total {alert['quantity']}"
                    ).classes("text-sm")

    with ui.row().classes("w-full gap-4 flex-wrap"):
        with ui.element("div").classes("mi-card p-4 flex-1 min-w-[280px]"):
            ui.label("Batch tracking").classes("font-semibold mb-2")
            batches = inventory.get("batches") or []
            if not batches:
                with ui.element("div").classes("mi-empty"):
                    ui.label("No batches")
            else:
                ui.table(
                    columns=[
                        {"name": "batch_code", "label": "Batch", "field": "batch_code"},
                        {"name": "material_key", "label": "Material", "field": "material_key"},
                        {"name": "supplier", "label": "Supplier", "field": "supplier"},
                        {"name": "storage_location", "label": "Location", "field": "storage_location"},
                        {"name": "quality_status", "label": "Status", "field": "quality_status"},
                        {"name": "is_demo", "label": "Demo", "field": "is_demo"},
                    ],
                    rows=batches,
                    row_key="id",
                    pagination=8,
                ).classes("w-full")

        with ui.element("div").classes("mi-card p-4 flex-1 min-w-[280px]"):
            ui.label("Supplier comparison / storage / movements").classes("font-semibold mb-2")
            with ui.tabs() as tabs:
                t1 = ui.tab("Suppliers")
                t2 = ui.tab("Storage")
                t3 = ui.tab("Movements")
            with ui.tab_panels(tabs, value=t1).classes("w-full"):
                with ui.tab_panel(t1):
                    ui.table(
                        columns=[
                            {"name": "supplier", "label": "Supplier", "field": "supplier"},
                            {"name": "batches", "label": "Batches", "field": "batches"},
                            {"name": "quarantined", "label": "Quarantined", "field": "quarantined"},
                            {"name": "rejected", "label": "Rejected", "field": "rejected"},
                        ],
                        rows=inventory.get("supplier_comparison") or [],
                        row_key="supplier",
                    ).classes("w-full")
                with ui.tab_panel(t2):
                    ui.table(
                        columns=[
                            {"name": "storage_location", "label": "Location", "field": "storage_location"},
                            {"name": "unit", "label": "Unit", "field": "unit"},
                            {"name": "quantity", "label": "Qty", "field": "quantity"},
                            {"name": "batches", "label": "Batches", "field": "batches"},
                        ],
                        rows=inventory.get("storage_locations") or [],
                        row_key="storage_location",
                    ).classes("w-full")
                with ui.tab_panel(t3):
                    moves = inventory.get("movements") or []
                    if not moves:
                        with ui.element("div").classes("mi-empty"):
                            ui.label("No stock movements")
                    else:
                        ui.table(
                            columns=[
                                {"name": "batch", "label": "Batch", "field": "batch"},
                                {"name": "type", "label": "Type", "field": "type"},
                                {"name": "quantity", "label": "Qty", "field": "quantity"},
                                {"name": "created_at", "label": "When", "field": "created_at"},
                            ],
                            rows=moves,
                            row_key="created_at",
                            pagination=8,
                        ).classes("w-full")


def _model_block(title: str, payload: dict, metric_keys: list[tuple[str, str]]) -> None:
    with ui.element("div").classes("mi-card p-4 flex-1 min-w-[280px]"):
        ui.label(title).classes("font-semibold mb-2")
        status = payload.get("status")
        ui.label(f"Version: {payload.get('version') or '—'}").classes("text-sm mi-muted")
        ui.label(f"Dataset: {payload.get('dataset') or '—'}").classes("text-sm mi-muted mb-2")
        if status == "not_trained":
            with ui.element("div").classes("mi-empty"):
                ui.label("Not trained")
            return
        if status == "not_evaluated":
            with ui.element("div").classes("mi-empty"):
                ui.label("Not evaluated")
            return
        metrics = payload.get("metrics") or {}
        for label, key in metric_keys:
            val = metrics.get(key)
            ui.label(f"{label}: {val if val is not None else '—'}").classes("text-sm")
        cm = payload.get("confusion_matrix")
        names = payload.get("class_names") or []
        if cm and names:
            ui.plotly(
                confusion_heatmap(cm, names, title="Confusion matrix")
            ).classes("w-full mt-2")


def _section_models(perf: dict) -> None:
    _section_header("AI model performance", "Active weights, evaluation metrics, and training history")
    with ui.row().classes("w-full gap-4 flex-wrap"):
        _model_block(
            "YOLOv8",
            perf.get("yolo") or {},
            [
                ("mAP@50", "map50"),
                ("mAP@50:95", "map50_95"),
                ("Precision", "precision"),
                ("Recall", "recall"),
            ],
        )
        _model_block(
            "EfficientNet-B0",
            perf.get("efficientnet") or {},
            [
                ("Accuracy", "accuracy"),
                ("Macro F1", "macro_f1"),
                ("Weighted F1", "weighted_f1"),
            ],
        )
        with ui.element("div").classes("mi-card p-4 flex-1 min-w-[280px]"):
            ui.label("Training & latency").classes("font-semibold mb-2")
            active = perf.get("active_models") or {}
            ui.label(
                f"Active YOLO: {active.get('yolo') or '—'} · "
                f"Active EfficientNet: {active.get('efficientnet') or '—'}"
            ).classes("text-sm mb-2")
            latest = perf.get("latest_job")
            if not latest:
                with ui.element("div").classes("mi-empty"):
                    ui.label("No training jobs")
            else:
                ui.label(
                    f"Latest job #{latest['id']}: {latest['name']} · {latest['status']}"
                ).classes("text-sm")
            latency = perf.get("inference_latency")
            if latency:
                ui.label(
                    f"Inference latency (n={latency['count']}): "
                    f"avg {latency['avg_ms']} ms · p50 {latency['p50_ms']} ms · max {latency['max_ms']} ms"
                ).classes("text-sm mi-muted mt-2")
            else:
                ui.label("Inference latency: not available").classes("text-sm mi-muted mt-2")
            history = perf.get("training_history") or []
            if history:
                ui.table(
                    columns=[
                        {"name": "id", "label": "ID", "field": "id"},
                        {"name": "name", "label": "Experiment", "field": "name"},
                        {"name": "type", "label": "Type", "field": "type"},
                        {"name": "status", "label": "Status", "field": "status"},
                    ],
                    rows=history,
                    row_key="id",
                    pagination=5,
                ).classes("w-full mt-2")


def _section_recent(recent: dict, filter_state: dict, rerender) -> None:
    _section_header("Recent AI inspections", "Latest screened materials with confidence and defect labels")
    with ui.element("div").classes("mi-card p-4 w-full"):
        rows = recent.get("rows") or []
        if not rows:
            with ui.element("div").classes("mi-empty"):
                ui.label("No inspections match the current filters")
            return

        for row in rows:
            with ui.row().classes("mi-insp-row w-full items-center gap-3"):
                thumb = row.get("thumbnail")
                if thumb and Path(thumb).exists():
                    ui.image(thumb).classes("mi-thumb")
                else:
                    with ui.element("div").classes("mi-thumb").style(
                        "display:flex;align-items:center;justify-content:center;"
                    ):
                        ui.icon("image").classes("mi-muted")
                with ui.column().classes("gap-0 flex-grow"):
                    with ui.row().classes("gap-2 items-center"):
                        ui.label(f"#{row['id']}").classes("font-semibold")
                        if row.get("is_demo"):
                            ui.label("DEMO").classes("mi-chip mi-chip-demo")
                        ui.label(row["status"]).classes("mi-chip")
                    ui.label(
                        f"{row['material']} · {row['defect']} · "
                        f"conf={row['confidence'] if row['confidence'] is not None else '—'}"
                    ).classes("text-sm")
                    ui.label(
                        f"{row['site']} · batch {row['batch_code']} · {row['created_at']}"
                    ).classes("text-xs mi-muted")
                ui.button(
                    "View details",
                    on_click=lambda r=row: ui.navigate.to(r["detail_url"]),
                ).props("outline dense")

        with ui.row().classes("items-center justify-between w-full mt-3"):
            ui.label(
                f"Page {recent['page']} / {recent['pages']} · {recent['total']} records"
            ).classes("text-sm mi-muted")

            def prev_page() -> None:
                filter_state["page"] = max(1, filter_state["page"] - 1)
                rerender()

            def next_page() -> None:
                filter_state["page"] = min(recent["pages"], filter_state["page"] + 1)
                rerender()

            with ui.row().classes("gap-2"):
                ui.button("Previous", on_click=prev_page).props("flat dense")
                ui.button("Next", on_click=next_page).props("flat dense")


def _section_quality(quality: dict, user) -> None:
    _section_header(
        "Quality review center",
        "Approvals, low-confidence flags, quarantines, and recent decisions",
    )
    if not has_permission(user, "approve"):
        with ui.element("div").classes("mi-empty"):
            ui.label("Engineering approval panels require Quality Engineer or Administrator role.")
            ui.label("AI cannot directly approve structural materials.").classes("text-sm")
        return

    panels = [
        ("Pending engineering approvals", quality.get("pending_approvals") or [], "id"),
        ("Manual inspection required", quality.get("manual_inspection") or [], "id"),
        ("Low-confidence predictions", quality.get("low_confidence") or [], "id"),
        ("Quarantined batches", quality.get("quarantined_batches") or [], "batch_code"),
        ("Missing testing documents", quality.get("missing_testing_documents") or [], "id"),
        ("Recent quality decisions", quality.get("recent_decisions") or [], "id"),
    ]
    with ui.element("div").classes("mi-chart-grid"):
        for title, rows, key in panels:
            with ui.element("div").classes("mi-card p-4"):
                with ui.row().classes("justify-between items-center mb-2"):
                    ui.label(title).classes("font-semibold")
                    ui.link("Review →", "/quality-review").classes("text-xs").style(
                        "color: var(--mi-accent); text-decoration: none; font-weight: 600;"
                    )
                if not rows:
                    with ui.element("div").classes("mi-empty"):
                        ui.label("None")
                else:
                    for row in rows[:6]:
                        label = row.get(key) or row.get("id")
                        extra = row.get("status") or row.get("material_key") or ""
                        demo = " · DEMO" if row.get("is_demo") else ""
                        if "detail_url" in row:
                            ui.link(
                                f"{label} · {extra}{demo}",
                                row["detail_url"],
                            ).classes("text-sm")
                        else:
                            ui.label(f"{label} · {extra}{demo}").classes("text-sm")
    ui.label(
        "AI screening never certifies structural safety. Final approval requires an authorized engineer."
    ).classes("text-xs mi-muted")


def _section_reporting_note(snap) -> None:
    _section_header("Reporting", "Exports respect the active filters and data-source attribution")
    with ui.element("div").classes("mi-card p-4 w-full"):
        ui.label(
            "Exports use the active dashboard filters and persistent database records. "
            "Each file includes date range, filter metadata, and data-source attribution."
        ).classes("text-sm mi-muted")
        ui.label(f"Current filter snapshot: {snap.filters}").classes("text-xs mi-muted mt-2")
