"""AI Evaluation Command Center — classification, clustering, detection, robustness."""

from __future__ import annotations

from nicegui import ui
from sqlalchemy import select

from material_intelligence.app.db.models import Dataset, EvaluationExperiment, ModelArtifact
from material_intelligence.app.db.session import SessionLocal
from material_intelligence.app.domain.enums import ModelType
from material_intelligence.app.ml.evaluation.experiment_manager import (
    compare_experiments,
    latest_kpi_snapshot,
    list_experiments,
)
from material_intelligence.app.services.evaluation_reporting import (
    experiments_to_csv,
    generate_experiment_pdf,
)
from material_intelligence.app.services.evaluation_service import (
    run_classification_experiment,
    run_clustering_experiment,
    run_detection_experiment,
    run_robustness_experiment,
)
from material_intelligence.app.ui.components.chart_panel import chart_panel
from material_intelligence.app.ui.components.metric_card import metric_card
from material_intelligence.app.ui.components.plotly_theme import bar_chart, confusion_heatmap, line_chart
from material_intelligence.app.ui.layout import page_frame
from material_intelligence.app.ui.state import current_user


def _kpi_value(v) -> str:
    if v is None or v == "Not Evaluated":
        return "Not Evaluated"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def register() -> None:
    @ui.page("/evaluation")
    def evaluation_page() -> None:
        def content() -> None:
            user = current_user()
            with SessionLocal() as db:
                kpis = latest_kpi_snapshot(db)
                artifacts = db.scalars(
                    select(ModelArtifact).order_by(ModelArtifact.created_at.desc())
                ).all()
                datasets = db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).all()
                experiments = list_experiments(db, limit=30)

            clf_models = {
                a.id: f"{a.name} · {a.version}"
                for a in artifacts
                if a.model_type == ModelType.EFFICIENTNET_B0.value
            }
            yolo_models = {
                a.id: f"{a.name} · {a.version}"
                for a in artifacts
                if a.model_type == ModelType.YOLOV8.value
            }
            dataset_opts = {d.id: f"{d.name} ({d.version}) · {d.image_count} imgs" for d in datasets}

            with ui.element("div").classes("mi-hero w-full mb-3"):
                ui.element("div").classes("mi-hero-stripe")
                ui.label("AI EVALUATION").classes("text-[11px] font-semibold").style(
                    "letter-spacing:0.16em;color:#f5a623;font-family:Oswald,sans-serif;"
                )
                ui.html(
                    '<div class="mi-hero-brand">RESEARCH <span>VALIDATION</span></div>',
                    sanitize=False,
                )
                ui.element("div").classes("mi-hero-rule")
                ui.label(
                    "Metrics come only from completed experiments. Unavailable values show Not Evaluated."
                ).classes("text-sm mi-muted")

            with ui.element("div").classes("mi-kpi-grid mb-3"):
                metric_card("Classification Accuracy", _kpi_value(kpis["classification_accuracy"]))
                metric_card("Macro F1", _kpi_value(kpis["macro_f1"]))
                metric_card("YOLO mAP@50", _kpi_value(kpis["yolo_map50"]))
                metric_card("YOLO mAP@50:95", _kpi_value(kpis["yolo_map50_95"]))
                metric_card("Clustering Silhouette", _kpi_value(kpis["silhouette"]))
                metric_card("Evaluated Images", _kpi_value(kpis["n_evaluated_images"]))
                metric_card("False-Negative Rate", _kpi_value(kpis["false_negative_rate"]))
                metric_card("Avg Latency (ms)", _kpi_value(kpis["avg_inference_latency_ms"]))

            with ui.tabs().classes("w-full") as tabs:
                t_overview = ui.tab("Overview")
                t_clf = ui.tab("Classification")
                t_cluster = ui.tab("Clustering")
                t_yolo = ui.tab("YOLOv8")
                t_robust = ui.tab("Robustness")
                t_failed = ui.tab("Failed Predictions")
                t_compare = ui.tab("Experiments")
                t_reports = ui.tab("Reports")

            with ui.tab_panels(tabs, value=t_overview).classes("w-full"):
                with ui.tab_panel(t_overview):
                    with ui.element("div").classes("mi-card p-4"):
                        ui.label("Saved experiments").classes("font-semibold")
                        if not experiments:
                            ui.label("Not Evaluated — run an experiment in the tabs above.").classes(
                                "mi-muted text-sm"
                            )
                        else:
                            ui.table(
                                columns=[
                                    {"name": "id", "label": "ID", "field": "id"},
                                    {"name": "name", "label": "Name", "field": "name"},
                                    {"name": "type", "label": "Type", "field": "type"},
                                    {"name": "n", "label": "Images", "field": "n"},
                                    {"name": "model", "label": "Model", "field": "model"},
                                ],
                                rows=[
                                    {
                                        "id": e.id,
                                        "name": e.name,
                                        "type": e.experiment_type,
                                        "n": e.n_images,
                                        "model": e.model_version,
                                    }
                                    for e in experiments
                                ],
                                row_key="id",
                                pagination=10,
                            ).classes("w-full")
                    with ui.row().classes("gap-2 mt-3"):
                        ui.button(
                            "Live Testing Laboratory",
                            on_click=lambda: ui.navigate.to("/testing-laboratory"),
                        ).props("unelevated")
                        ui.button(
                            "Clustering Analysis",
                            on_click=lambda: ui.navigate.to("/clustering-analysis"),
                        ).props("outline")
                        ui.button(
                            "Experiment Comparison",
                            on_click=lambda: ui.navigate.to("/experiment-comparison"),
                        ).props("outline")

                with ui.tab_panel(t_clf):
                    status = ui.label("").classes("text-sm")
                    result_box = ui.column().classes("w-full gap-3")
                    model_sel = ui.select(clf_models, label="EfficientNet checkpoint").classes(
                        "w-full max-w-xl"
                    )
                    ds_sel = ui.select(dataset_opts, label="Labeled test dataset").classes(
                        "w-full max-w-xl"
                    )

                    def run_clf() -> None:
                        if not model_sel.value or not ds_sel.value:
                            status.text = "Select model and dataset."
                            return
                        status.text = "Running classification evaluation…"
                        try:
                            with SessionLocal() as db:
                                out = run_classification_experiment(
                                    db,
                                    model_id=int(model_sel.value),
                                    dataset_id=int(ds_sel.value),
                                    researcher_id=user.id if user else None,
                                )
                            status.text = f"Saved experiment #{out['experiment_id']}"
                            _render_clf_result(result_box, out)
                        except Exception as exc:  # noqa: BLE001
                            status.text = f"Error: {exc}"

                    ui.button("Run classification evaluation", on_click=run_clf).props(
                        "unelevated"
                    )
                    result_box

                with ui.tab_panel(t_cluster):
                    c_status = ui.label("").classes("text-sm")
                    c_box = ui.column().classes("w-full gap-3")
                    c_model = ui.select(clf_models, label="Embedding backbone (EfficientNet)").classes(
                        "w-full max-w-xl"
                    )
                    c_ds = ui.select(dataset_opts, label="Image dataset").classes("w-full max-w-xl")
                    c_algo = ui.select(
                        {"kmeans": "K-Means", "dbscan": "DBSCAN"}, value="kmeans", label="Algorithm"
                    ).classes("w-48")
                    c_k = ui.number("K (K-Means)", value=3, min=2, max=20).classes("w-40")
                    c_eps = ui.number("eps (DBSCAN)", value=0.8, min=0.05, step=0.05).classes("w-40")

                    def run_cluster() -> None:
                        if not c_model.value or not c_ds.value:
                            c_status.text = "Select model and dataset."
                            return
                        c_status.text = "Extracting embeddings and clustering…"
                        try:
                            with SessionLocal() as db:
                                out = run_clustering_experiment(
                                    db,
                                    model_id=int(c_model.value),
                                    dataset_id=int(c_ds.value),
                                    algorithm=c_algo.value,
                                    n_clusters=int(c_k.value or 3),
                                    eps=float(c_eps.value or 0.8),
                                    researcher_id=user.id if user else None,
                                )
                            c_status.text = f"Saved clustering experiment #{out['experiment_id']}"
                            _render_cluster_result(c_box, out)
                        except Exception as exc:  # noqa: BLE001
                            c_status.text = f"Error: {exc}"

                    ui.button("Run clustering evaluation", on_click=run_cluster).props(
                        "unelevated"
                    )
                    c_box

                with ui.tab_panel(t_yolo):
                    y_status = ui.label("").classes("text-sm")
                    y_box = ui.column().classes("w-full gap-3")
                    y_model = ui.select(yolo_models, label="YOLOv8 checkpoint").classes(
                        "w-full max-w-xl"
                    )
                    y_ds = ui.select(dataset_opts, label="YOLO dataset (data.yaml)").classes(
                        "w-full max-w-xl"
                    )
                    y_conf = ui.number("Confidence", value=0.25, min=0.05, max=0.95, step=0.05)
                    y_iou = ui.number("IoU", value=0.5, min=0.1, max=0.95, step=0.05)

                    def run_yolo() -> None:
                        if not y_model.value or not y_ds.value:
                            y_status.text = "Select YOLO model and dataset."
                            return
                        y_status.text = "Running YOLO validation…"
                        try:
                            with SessionLocal() as db:
                                out = run_detection_experiment(
                                    db,
                                    model_id=int(y_model.value),
                                    dataset_id=int(y_ds.value),
                                    conf=float(y_conf.value or 0.25),
                                    iou=float(y_iou.value or 0.5),
                                    researcher_id=user.id if user else None,
                                )
                            y_status.text = f"Saved detection experiment #{out['experiment_id']}"
                            metrics = (out.get("metrics") or {}).get("metrics") or out.get("metrics") or {}
                            y_box.clear()
                            with y_box:
                                with ui.element("div").classes("mi-kpi-grid"):
                                    metric_card("mAP@50", _kpi_value(metrics.get("map50")))
                                    metric_card("mAP@50:95", _kpi_value(metrics.get("map50_95")))
                                    metric_card("Precision", _kpi_value(metrics.get("precision")))
                                    metric_card("Recall", _kpi_value(metrics.get("recall")))
                                ui.label(
                                    metrics.get("protocol")
                                    or "Object-detection mAP — not classification accuracy."
                                ).classes("text-xs mi-muted")
                        except Exception as exc:  # noqa: BLE001
                            y_status.text = f"Error: {exc}"

                    ui.button("Run YOLO evaluation", on_click=run_yolo).props("unelevated")
                    y_box

                with ui.tab_panel(t_robust):
                    r_status = ui.label("").classes("text-sm")
                    r_box = ui.column().classes("w-full")
                    r_model = ui.select(clf_models, label="Classifier").classes("w-full max-w-xl")
                    r_ds = ui.select(dataset_opts, label="Dataset").classes("w-full max-w-xl")

                    def run_robust() -> None:
                        if not r_model.value or not r_ds.value:
                            r_status.text = "Select model and dataset."
                            return
                        r_status.text = "Running robustness transforms…"
                        try:
                            with SessionLocal() as db:
                                out = run_robustness_experiment(
                                    db,
                                    model_id=int(r_model.value),
                                    dataset_id=int(r_ds.value),
                                    researcher_id=user.id if user else None,
                                )
                            r_status.text = f"Saved experiment #{out['experiment_id']}"
                            metrics = out["metrics"]
                            r_box.clear()
                            with r_box:
                                ui.label(
                                    f"Baseline accuracy: {_kpi_value(metrics.get('baseline_accuracy'))}"
                                )
                                rows = [
                                    {
                                        "transform": k,
                                        "accuracy": v.get("accuracy"),
                                        "delta": v.get("delta_accuracy"),
                                    }
                                    for k, v in (metrics.get("by_transform") or {}).items()
                                ]
                                ui.table(
                                    columns=[
                                        {"name": "transform", "label": "Transform", "field": "transform"},
                                        {"name": "accuracy", "label": "Accuracy", "field": "accuracy"},
                                        {"name": "delta", "label": "Δ vs baseline", "field": "delta"},
                                    ],
                                    rows=rows,
                                    row_key="transform",
                                ).classes("w-full")
                        except Exception as exc:  # noqa: BLE001
                            r_status.text = f"Error: {exc}"

                    ui.button("Run robustness evaluation", on_click=run_robust).props("unelevated")
                    r_box

                with ui.tab_panel(t_failed):
                    with SessionLocal() as db:
                        latest_clf = db.scalar(
                            select(EvaluationExperiment)
                            .where(EvaluationExperiment.experiment_type == "classification")
                            .order_by(EvaluationExperiment.created_at.desc())
                        )
                    if not latest_clf or not (latest_clf.artifacts or {}).get("error_buckets"):
                        chart_panel(
                            None,
                            title="Failed predictions",
                            empty_message="Not Evaluated — run classification first",
                        )
                    else:
                        buckets = latest_clf.artifacts["error_buckets"]
                        for name in [
                            "false_negative_damage",
                            "high_confidence_incorrect",
                            "low_confidence",
                            "all_errors",
                        ]:
                            rows = buckets.get(name) or []
                            ui.label(f"{name.replace('_', ' ').title()} ({len(rows)})").classes(
                                "font-semibold mt-2"
                            )
                            if not rows:
                                ui.label("None").classes("text-xs mi-muted")
                            else:
                                for row in rows[:15]:
                                    ui.label(
                                        f"{row.get('path')} · true={row.get('true_label')} · "
                                        f"pred={row.get('pred_label')} · conf={row.get('confidence')}"
                                    ).classes("text-xs")

                with ui.tab_panel(t_compare):
                    ids = {e.id: f"#{e.id} {e.name}" for e in experiments}
                    pick = ui.select(ids, multiple=True, label="Experiments").classes("w-full")
                    cmp_box = ui.column().classes("w-full")

                    def do_compare() -> None:
                        selected = pick.value or []
                        if len(selected) < 2:
                            ui.notify("Select at least two experiments")
                            return
                        with SessionLocal() as db:
                            payload = compare_experiments(db, [int(x) for x in selected])
                        cmp_box.clear()
                        with cmp_box:
                            ui.label(payload["warning"]).classes("text-xs mi-muted")
                            for exp in payload["experiments"]:
                                with ui.element("div").classes("mi-card p-3 mb-2"):
                                    ui.label(
                                        f"#{exp['id']} · {exp['name']} · {exp['type']}"
                                    ).classes("font-semibold")
                                    ui.label(str(exp["metrics"])[:500]).classes("text-xs")

                    ui.button("Compare", on_click=do_compare).props("outline")
                    cmp_box

                with ui.tab_panel(t_reports):
                    rep_status = ui.label("").classes("text-sm")
                    exp_opts = {e.id: f"#{e.id} {e.name}" for e in experiments}
                    rep_sel = ui.select(exp_opts, label="Experiment").classes("w-full max-w-xl")

                    def download_csv() -> None:
                        with SessionLocal() as db:
                            csv_text = experiments_to_csv(db)
                        ui.download(csv_text.encode("utf-8"), "evaluation_experiments.csv")

                    def download_pdf() -> None:
                        if not rep_sel.value:
                            rep_status.text = "Select an experiment."
                            return
                        with SessionLocal() as db:
                            path = generate_experiment_pdf(db, int(rep_sel.value))
                        ui.download(str(path))
                        rep_status.text = f"PDF ready: {path.name}"

                    ui.button("Export all experiments CSV", on_click=download_csv).props("outline")
                    ui.button("Download experiment PDF", on_click=download_pdf).props("unelevated")

        page_frame("Model Evaluation", content)


def _render_clf_result(container, out: dict) -> None:
    container.clear()
    metrics_wrap = out.get("metrics") or {}
    metrics = metrics_wrap.get("metrics") or metrics_wrap
    cm = metrics_wrap.get("confusion_matrix")
    names = metrics_wrap.get("class_names") or []
    with container:
        with ui.element("div").classes("mi-kpi-grid"):
            metric_card("Accuracy", _kpi_value(metrics.get("accuracy")))
            metric_card("Macro F1", _kpi_value(metrics.get("macro_f1")))
            metric_card("Weighted F1", _kpi_value(metrics.get("weighted_f1")))
            metric_card("ECE", _kpi_value(metrics.get("ece")))
            metric_card("Brier", _kpi_value(metrics.get("brier_score")))
            metric_card("FN rate (damage→good)", _kpi_value(metrics.get("false_negative_rate")))
        leakage = out.get("leakage") or {}
        ui.label(leakage.get("message", "")).classes("text-xs mi-muted")
        if cm and names:
            chart_panel(
                confusion_heatmap(cm, names, title="Confusion matrix (counts)"),
                title="Confusion Matrix",
            )
        per_class = metrics_wrap.get("per_class") or {}
        if per_class:
            labels = list(per_class.keys())
            f1s = [per_class[k].get("f1") or 0 for k in labels]
            chart_panel(bar_chart(labels, f1s, title="Per-class F1"), title="Per-class F1")


def _render_cluster_result(container, out: dict) -> None:
    container.clear()
    metrics = out.get("metrics") or {}
    with container:
        with ui.element("div").classes("mi-kpi-grid"):
            metric_card("Clusters", _kpi_value(metrics.get("n_clusters")))
            metric_card("Silhouette", _kpi_value(metrics.get("silhouette")))
            metric_card("Davies–Bouldin", _kpi_value(metrics.get("davies_bouldin")))
            metric_card("Calinski–Harabasz", _kpi_value(metrics.get("calinski_harabasz")))
            metric_card("Stability ARI", _kpi_value(metrics.get("stability_ari")))
            metric_card("Noise points", _kpi_value(metrics.get("n_noise")))
        for summary in out.get("summaries") or []:
            with ui.element("div").classes("mi-card p-3"):
                ui.label(
                    f"Cluster {summary['cluster_id']} · n={summary['size']} "
                    f"({summary['percentage']:.0%})"
                    + (" · NOISE" if summary.get("is_noise") else "")
                ).classes("font-semibold")
                ui.label(f"Materials: {summary.get('material_distribution')}").classes("text-xs")
                ui.label(f"Labels: {summary.get('expert_label_distribution')}").classes("text-xs")
