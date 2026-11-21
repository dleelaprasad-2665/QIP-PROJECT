"""Centralized dashboard analytics — all KPI and chart aggregates live here."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from material_intelligence.app.db.models import (
    Dataset,
    EvaluationRecord,
    Inspection,
    MaterialBatch,
    ModelArtifact,
    Site,
    StockMovement,
    Supplier,
    TrainingJob,
)
from material_intelligence.app.domain.enums import (
    HIGH_RISK_CONDITIONS,
    InspectionStatus,
    ModelStatus,
    ModelType,
)

DEMO_SITE_CODE = "DEMO-DATA"
DEMO_LABEL = "DEMO DATA — not operational"


@dataclass
class DashboardFilters:
    site_id: int | None = None
    material_key: str | None = None
    supplier_id: int | None = None
    status: str | None = None
    model_version: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    search: str | None = None
    include_demo: bool = True
    page: int = 1
    page_size: int = 10

    def date_from_dt(self) -> datetime | None:
        if self.date_from is None:
            return None
        return datetime.combine(self.date_from, datetime.min.time()).replace(tzinfo=UTC)

    def date_to_dt(self) -> datetime | None:
        if self.date_to is None:
            return None
        return datetime.combine(self.date_to, datetime.max.time()).replace(tzinfo=UTC)


@dataclass
class KPIValue:
    key: str
    label: str
    value: float | int | str
    unit: str | None = None
    hint: str | None = None
    delta_pct: float | None = None
    link: str | None = None
    empty: bool = False


@dataclass
class DashboardSnapshot:
    generated_at: datetime
    filters: dict[str, Any]
    data_source: str
    is_demo_present: bool
    kpis: list[KPIValue]
    charts: dict[str, Any]
    inventory: dict[str, Any]
    model_performance: dict[str, Any]
    recent_inspections: dict[str, Any]
    quality_review: dict[str, Any]
    filter_options: dict[str, Any] = field(default_factory=dict)


def _demo_site_ids(db: Session) -> set[int]:
    rows = db.scalars(select(Site.id).where(Site.code == DEMO_SITE_CODE)).all()
    return set(rows)


def _inspection_filters(filters: DashboardFilters, demo_ids: set[int]):
    clauses = []
    start = filters.date_from_dt()
    end = filters.date_to_dt()
    if start is not None:
        clauses.append(Inspection.created_at >= start)
    if end is not None:
        clauses.append(Inspection.created_at <= end)
    if filters.site_id is not None:
        clauses.append(Inspection.site_id == filters.site_id)
    if filters.status:
        clauses.append(Inspection.status == filters.status)
    if filters.model_version:
        # JSON text contains model version string
        clauses.append(Inspection.model_versions.contains(filters.model_version))
    if not filters.include_demo and demo_ids:
        clauses.append(Inspection.site_id.notin_(demo_ids))
    return clauses


def _batch_filters(filters: DashboardFilters, demo_ids: set[int]):
    clauses = []
    if filters.site_id is not None:
        clauses.append(MaterialBatch.site_id == filters.site_id)
    if filters.material_key:
        clauses.append(MaterialBatch.material_key == filters.material_key)
    if filters.supplier_id is not None:
        clauses.append(MaterialBatch.supplier_id == filters.supplier_id)
    if not filters.include_demo and demo_ids:
        clauses.append(MaterialBatch.site_id.notin_(demo_ids))
    return clauses


def _pct_change(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return round(((current - previous) / previous) * 100.0, 1)


def _period_bounds(filters: DashboardFilters) -> tuple[datetime | None, datetime | None, datetime | None, datetime | None]:
    """Return current and previous comparison window bounds."""
    start = filters.date_from_dt()
    end = filters.date_to_dt() or datetime.now(UTC)
    if start is None:
        return None, end, None, None
    span = end - start
    prev_end = start - timedelta(microseconds=1)
    prev_start = start - span
    return start, end, prev_start, prev_end


def compute_executive_kpis(db: Session, filters: DashboardFilters) -> list[KPIValue]:
    demo_ids = _demo_site_ids(db)
    insp_clauses = _inspection_filters(filters, demo_ids)
    batch_clauses = _batch_filters(filters, demo_ids)

    batch_count = db.scalar(
        select(func.count()).select_from(MaterialBatch).where(and_(*batch_clauses) if batch_clauses else True)
    ) or 0

    insp_base = select(Inspection).where(and_(*insp_clauses) if insp_clauses else True)
    inspections = list(db.scalars(insp_base).all())
    insp_count = len(inspections)

    # Explicit denominators for rates
    screening_pass = sum(
        1 for i in inspections if i.status == InspectionStatus.SCREENING_COMPLETED.value
    )
    flagged = 0
    for insp in inspections:
        clf = insp.classification_result or {}
        items = clf.get("items") or []
        for item in items:
            condition = item.get("condition")
            if condition in HIGH_RISK_CONDITIONS:
                flagged += 1
                break
        else:
            if insp.status in {
                InspectionStatus.ADDITIONAL_TESTING_REQUIRED.value,
                InspectionStatus.QUARANTINED.value,
                InspectionStatus.REJECTED.value,
            }:
                flagged += 1

    awaiting = sum(
        1
        for i in inspections
        if i.status
        in {
            InspectionStatus.HUMAN_INSPECTION_REQUIRED.value,
            InspectionStatus.ADDITIONAL_TESTING_REQUIRED.value,
        }
    )

    batches = list(
        db.scalars(
            select(MaterialBatch).where(and_(*batch_clauses) if batch_clauses else True)
        ).all()
    )
    # Never mix units — report per-unit breakdown and primary "units" total separately.
    by_unit: dict[str, dict[str, float]] = {}
    for b in batches:
        unit = b.unit or "units"
        bucket = by_unit.setdefault(
            unit, {"approved": 0.0, "quarantined": 0.0, "rejected": 0.0, "total": 0.0}
        )
        bucket["approved"] += float(b.approved_quantity or 0)
        bucket["quarantined"] += float(b.quarantined_quantity or 0)
        bucket["rejected"] += float(b.rejected_quantity or 0)
        bucket["total"] += float(b.quantity or 0)

    primary_unit = "units" if "units" in by_unit else (next(iter(by_unit), "units"))
    approved = by_unit.get(primary_unit, {}).get("approved", 0.0)
    quarantined = by_unit.get(primary_unit, {}).get("quarantined", 0.0)
    rejected = by_unit.get(primary_unit, {}).get("rejected", 0.0)

    pass_rate = (screening_pass / insp_count * 100.0) if insp_count else None
    defect_rate = (flagged / insp_count * 100.0) if insp_count else None

    # Comparison period deltas when a date range is set
    start, end, prev_start, prev_end = _period_bounds(filters)
    delta_insp = None
    delta_batches = None
    if start is not None and prev_start is not None and prev_end is not None:
        prev_insp = db.scalar(
            select(func.count())
            .select_from(Inspection)
            .where(
                and_(
                    Inspection.created_at >= prev_start,
                    Inspection.created_at <= prev_end,
                    *([Inspection.site_id == filters.site_id] if filters.site_id else []),
                    *([Inspection.site_id.notin_(demo_ids)] if not filters.include_demo and demo_ids else []),
                )
            )
        ) or 0
        delta_insp = _pct_change(insp_count, prev_insp)
        # Batches are stock snapshots — compare count created in previous window
        prev_batch = db.scalar(
            select(func.count())
            .select_from(MaterialBatch)
            .where(
                and_(
                    MaterialBatch.created_at >= prev_start,
                    MaterialBatch.created_at <= prev_end,
                    *([MaterialBatch.site_id == filters.site_id] if filters.site_id else []),
                    *([MaterialBatch.site_id.notin_(demo_ids)] if not filters.include_demo and demo_ids else []),
                )
            )
        ) or 0
        curr_batch_created = db.scalar(
            select(func.count())
            .select_from(MaterialBatch)
            .where(
                and_(
                    MaterialBatch.created_at >= start,
                    MaterialBatch.created_at <= end,
                    *([MaterialBatch.site_id == filters.site_id] if filters.site_id else []),
                    *([MaterialBatch.site_id.notin_(demo_ids)] if not filters.include_demo and demo_ids else []),
                )
            )
        ) or 0
        delta_batches = _pct_change(curr_batch_created, prev_batch)

    unit_hint = None
    if len(by_unit) > 1:
        unit_hint = (
            "Stock KPIs shown for unit '"
            + primary_unit
            + "'. Other units: "
            + ", ".join(f"{u}" for u in by_unit if u != primary_unit)
            + " (not combined)."
        )

    return [
        KPIValue(
            key="batches",
            label="Total material batches",
            value=batch_count,
            delta_pct=delta_batches,
            link="/inventory",
            hint="Persisted batch records",
        ),
        KPIValue(
            key="inspections",
            label="Total material inspections",
            value=insp_count,
            delta_pct=delta_insp,
            link="/inspection",
            hint=f"Denominator for rates: {insp_count}",
        ),
        KPIValue(
            key="pass_rate",
            label="Visual screening pass rate",
            value=f"{pass_rate:.1f}%" if pass_rate is not None else "—",
            hint=f"{screening_pass}/{insp_count} screening_completed" if insp_count else "No inspections",
            empty=insp_count == 0,
            link="/quality-review",
        ),
        KPIValue(
            key="defect_rate",
            label="Flagged defect rate",
            value=f"{defect_rate:.1f}%" if defect_rate is not None else "—",
            hint=f"{flagged}/{insp_count} flagged" if insp_count else "No inspections",
            empty=insp_count == 0,
            link="/quality-review",
        ),
        KPIValue(
            key="awaiting",
            label="Awaiting engineering review",
            value=awaiting,
            link="/quality-review",
            hint="Human inspection or additional testing",
        ),
        KPIValue(
            key="approved_stock",
            label="Approved available stock",
            value=round(approved, 2),
            unit=primary_unit,
            link="/inventory",
            hint=unit_hint,
        ),
        KPIValue(
            key="quarantined",
            label="Quarantined material quantity",
            value=round(quarantined, 2),
            unit=primary_unit,
            link="/inventory",
            hint=unit_hint,
        ),
        KPIValue(
            key="rejected",
            label="Rejected material quantity",
            value=round(rejected, 2),
            unit=primary_unit,
            link="/inventory",
            hint=unit_hint,
        ),
    ]


def compute_inspection_charts(db: Session, filters: DashboardFilters) -> dict[str, Any]:
    demo_ids = _demo_site_ids(db)
    clauses = _inspection_filters(filters, demo_ids)
    rows = list(db.scalars(select(Inspection).where(and_(*clauses) if clauses else True)).all())
    if not rows:
        return {
            "inspections_over_time": None,
            "defects_by_material": None,
            "severity_distribution": None,
            "confidence_distribution": None,
            "outcomes_by_site": None,
            "top_defects": None,
            "empty": True,
        }

    site_map = {
        s.id: s.name for s in db.scalars(select(Site)).all()
    }
    batch_map = {
        b.id: b for b in db.scalars(select(MaterialBatch)).all()
    }

    records = []
    for insp in rows:
        clf = insp.classification_result or {}
        items = clf.get("items") or []
        primary = items[0] if items else {}
        condition = primary.get("condition")
        confidence = primary.get("confidence")
        material = primary.get("material_key")
        if not material and insp.batch_id and insp.batch_id in batch_map:
            material = batch_map[insp.batch_id].material_key
        if filters.material_key and material != filters.material_key:
            continue
        if filters.supplier_id and insp.batch_id:
            batch = batch_map.get(insp.batch_id)
            if batch is None or batch.supplier_id != filters.supplier_id:
                continue
        records.append(
            {
                "id": insp.id,
                "created_at": insp.created_at,
                "status": insp.status,
                "site": site_map.get(insp.site_id, str(insp.site_id)),
                "material": material or "unknown",
                "condition": condition or "unknown",
                "confidence": confidence,
                "high_risk": condition in HIGH_RISK_CONDITIONS if condition else False,
            }
        )

    if not records:
        return {
            "inspections_over_time": None,
            "defects_by_material": None,
            "severity_distribution": None,
            "confidence_distribution": None,
            "outcomes_by_site": None,
            "top_defects": None,
            "empty": True,
        }

    df = pd.DataFrame(records)
    df["day"] = pd.to_datetime(df["created_at"], utc=True).dt.date.astype(str)

    over_time = df.groupby("day").size().reset_index(name="count")
    defects = (
        df[df["condition"] != "no_visible_damage"]
        .groupby("material")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    severity = df.assign(
        severity=df["condition"].map(
            lambda c: "high_risk" if c in HIGH_RISK_CONDITIONS else (
                "no_visible_damage" if c == "no_visible_damage" else "moderate"
            )
        )
    ).groupby("severity").size().reset_index(name="count")

    conf = df.dropna(subset=["confidence"]).copy()
    if not conf.empty:
        conf["bucket"] = pd.cut(
            conf["confidence"],
            bins=[0, 0.45, 0.7, 0.85, 1.01],
            labels=["<0.45", "0.45–0.70", "0.70–0.85", "0.85–1.00"],
            include_lowest=True,
        )
        conf_dist = conf.groupby("bucket", observed=False).size().reset_index(name="count")
        conf_dist["bucket"] = conf_dist["bucket"].astype(str)
    else:
        conf_dist = pd.DataFrame(columns=["bucket", "count"])

    outcomes = df.groupby(["site", "status"]).size().reset_index(name="count")
    top_defects = (
        df[df["condition"] != "no_visible_damage"]
        .groupby("condition")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(8)
    )

    return {
        "inspections_over_time": {
            "x": over_time["day"].tolist(),
            "y": over_time["count"].tolist(),
        },
        "defects_by_material": {
            "x": defects["material"].tolist(),
            "y": defects["count"].tolist(),
        },
        "severity_distribution": {
            "labels": severity["severity"].tolist(),
            "values": severity["count"].tolist(),
        },
        "confidence_distribution": {
            "x": conf_dist["bucket"].tolist() if not conf_dist.empty else [],
            "y": conf_dist["count"].tolist() if not conf_dist.empty else [],
        },
        "outcomes_by_site": {
            "sites": outcomes["site"].tolist(),
            "statuses": outcomes["status"].tolist(),
            "counts": outcomes["count"].tolist(),
        },
        "top_defects": {
            "x": top_defects["condition"].tolist(),
            "y": top_defects["count"].tolist(),
        },
        "empty": False,
    }


def compute_inventory_panel(db: Session, filters: DashboardFilters) -> dict[str, Any]:
    demo_ids = _demo_site_ids(db)
    clauses = _batch_filters(filters, demo_ids)
    batches = list(
        db.scalars(select(MaterialBatch).where(and_(*clauses) if clauses else True)).all()
    )
    suppliers = {s.id: s.name for s in db.scalars(select(Supplier)).all()}

    low_stock: list[dict[str, Any]] = []
    if not batches:
        return {
            "material_stock": [],
            "batches": [],
            "low_stock": [],
            "supplier_comparison": [],
            "storage_locations": [],
            "movements": [],
            "empty": True,
        }

    rows = []
    for b in batches:
        rows.append(
            {
                "material_key": b.material_key,
                "unit": b.unit or "units",
                "quantity": float(b.quantity or 0),
                "approved": float(b.approved_quantity or 0),
                "quarantined": float(b.quarantined_quantity or 0),
                "rejected": float(b.rejected_quantity or 0),
                "batch_code": b.batch_code,
                "storage_location": b.storage_location or "—",
                "supplier": suppliers.get(b.supplier_id, "—") if b.supplier_id else "—",
                "quality_status": b.quality_status,
                "id": b.id,
                "is_demo": b.site_id in demo_ids,
            }
        )
    df = pd.DataFrame(rows)
    material_stock = (
        df.groupby(["material_key", "unit"], as_index=False)[["quantity", "approved", "quarantined", "rejected"]]
        .sum()
        .to_dict(orient="records")
    )
    for row in material_stock:
        available = row["approved"]
        # Low stock: approved available below 10% of total quantity for that material/unit
        if row["quantity"] > 0 and available / row["quantity"] < 0.1:
            low_stock.append(
                {
                    "material_key": row["material_key"],
                    "unit": row["unit"],
                    "approved": row["approved"],
                    "quantity": row["quantity"],
                    "alert": "Low approved stock (<10% of batch quantity)",
                }
            )

    supplier_comparison = (
        df.groupby("supplier", as_index=False)
        .agg(batches=("id", "count"), quarantined=("quarantined", "sum"), rejected=("rejected", "sum"))
        .to_dict(orient="records")
    )
    storage = (
        df.groupby(["storage_location", "unit"], as_index=False)
        .agg(quantity=("quantity", "sum"), batches=("id", "count"))
        .to_dict(orient="records")
    )

    batch_ids = [b.id for b in batches]
    movements = []
    if batch_ids:
        moves = db.scalars(
            select(StockMovement)
            .where(StockMovement.batch_id.in_(batch_ids))
            .order_by(StockMovement.created_at.desc())
            .limit(40)
        ).all()
        code_map = {b.id: b.batch_code for b in batches}
        movements = [
            {
                "batch": code_map.get(m.batch_id, m.batch_id),
                "type": m.movement_type,
                "quantity": m.quantity,
                "reference": m.reference,
                "created_at": str(m.created_at),
                "notes": m.notes,
            }
            for m in moves
        ]

    return {
        "material_stock": material_stock,
        "batches": rows,
        "low_stock": low_stock,
        "supplier_comparison": supplier_comparison,
        "storage_locations": storage,
        "movements": movements,
        "empty": False,
    }


def compute_model_performance(db: Session) -> dict[str, Any]:
    yolo = db.scalar(
        select(ModelArtifact)
        .where(
            ModelArtifact.model_type == ModelType.YOLOV8.value,
            ModelArtifact.is_active_inference.is_(True),
        )
        .order_by(ModelArtifact.updated_at.desc())
    )
    clf = db.scalar(
        select(ModelArtifact)
        .where(
            ModelArtifact.model_type == ModelType.EFFICIENTNET_B0.value,
            ModelArtifact.is_active_inference.is_(True),
        )
        .order_by(ModelArtifact.updated_at.desc())
    )

    def _eval_for(artifact: ModelArtifact | None) -> dict[str, Any]:
        if artifact is None:
            return {"status": "not_trained", "metrics": None, "version": None, "dataset": None}
        if artifact.status != ModelStatus.READY.value or not artifact.artifact_path:
            return {
                "status": "not_trained",
                "metrics": None,
                "version": artifact.version,
                "dataset": None,
            }
        record = db.scalar(
            select(EvaluationRecord)
            .where(EvaluationRecord.model_id == artifact.id)
            .order_by(EvaluationRecord.created_at.desc())
        )
        dataset_version = None
        if artifact.dataset_id:
            ds = db.get(Dataset, artifact.dataset_id)
            if ds:
                dataset_version = f"{ds.name} ({ds.version})"
        if record is None:
            return {
                "status": "not_evaluated",
                "metrics": None,
                "version": artifact.version,
                "dataset": dataset_version,
                "confusion_matrix": None,
                "class_names": None,
            }
        return {
            "status": "evaluated",
            "metrics": record.metrics,
            "per_class": record.per_class,
            "confusion_matrix": record.confusion_matrix,
            "class_names": record.class_names,
            "version": artifact.version,
            "dataset": dataset_version,
            "split": record.split,
        }

    latest_job = db.scalar(select(TrainingJob).order_by(TrainingJob.created_at.desc()))
    history = list(
        db.scalars(select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(10)).all()
    )

    # Inference latency from recent inspections with processing_ms
    latencies = db.scalars(
        select(Inspection.processing_ms)
        .where(Inspection.processing_ms.is_not(None))
        .order_by(Inspection.created_at.desc())
        .limit(100)
    ).all()
    latency = None
    if latencies:
        vals = [float(v) for v in latencies if v is not None]
        if vals:
            latency = {
                "count": len(vals),
                "avg_ms": round(sum(vals) / len(vals), 2),
                "p50_ms": round(sorted(vals)[len(vals) // 2], 2),
                "max_ms": round(max(vals), 2),
            }

    return {
        "yolo": _eval_for(yolo),
        "efficientnet": _eval_for(clf),
        "active_models": {
            "yolo": yolo.version if yolo and yolo.is_active_inference else None,
            "efficientnet": clf.version if clf and clf.is_active_inference else None,
        },
        "latest_job": (
            {
                "id": latest_job.id,
                "name": latest_job.experiment_name,
                "type": latest_job.model_type,
                "status": latest_job.status,
                "progress": latest_job.progress,
                "message": latest_job.message,
                "finished_at": str(latest_job.finished_at) if latest_job.finished_at else None,
            }
            if latest_job
            else None
        ),
        "training_history": [
            {
                "id": j.id,
                "name": j.experiment_name,
                "type": j.model_type,
                "status": j.status,
                "progress": j.progress,
                "created_at": str(j.created_at),
            }
            for j in history
        ],
        "inference_latency": latency,
    }


def compute_recent_inspections(db: Session, filters: DashboardFilters) -> dict[str, Any]:
    demo_ids = _demo_site_ids(db)
    base_clauses = _inspection_filters(filters, demo_ids)
    needs_batch_join = bool(filters.material_key or filters.supplier_id)

    if filters.search:
        term = filters.search.strip()
        search_clauses = [
            Inspection.status.ilike(f"%{term}%"),
            Inspection.notes.ilike(f"%{term}%"),
        ]
        if term.isdigit():
            search_clauses.append(Inspection.id == int(term))
        base_clauses.append(or_(*search_clauses))

    query = select(Inspection)
    if needs_batch_join:
        query = query.outerjoin(MaterialBatch, Inspection.batch_id == MaterialBatch.id)
        if filters.material_key:
            base_clauses.append(MaterialBatch.material_key == filters.material_key)
        if filters.supplier_id:
            base_clauses.append(MaterialBatch.supplier_id == filters.supplier_id)

    if base_clauses:
        query = query.where(and_(*base_clauses))
    query = query.order_by(Inspection.created_at.desc())

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    page = max(1, filters.page)
    page_size = max(1, min(filters.page_size, 50))
    rows = list(db.scalars(query.offset((page - 1) * page_size).limit(page_size)).all())

    sites = {s.id: s.name for s in db.scalars(select(Site)).all()}
    batches = {b.id: b for b in db.scalars(select(MaterialBatch)).all()}

    items = []
    for insp in rows:
        clf = insp.classification_result or {}
        primary = (clf.get("items") or [{}])[0]
        batch = batches.get(insp.batch_id) if insp.batch_id else None
        items.append(
            {
                "id": insp.id,
                "thumbnail": insp.annotated_path or insp.image_path,
                "material": primary.get("material_key")
                or (batch.material_key if batch else "—"),
                "defect": primary.get("condition") or "—",
                "confidence": primary.get("confidence"),
                "site": sites.get(insp.site_id, "—"),
                "batch_id": insp.batch_id,
                "batch_code": batch.batch_code if batch else "—",
                "created_at": str(insp.created_at),
                "status": insp.status,
                "is_demo": insp.site_id in demo_ids,
                "detail_url": f"/inspection/{insp.id}",
            }
        )

    return {
        "rows": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


def compute_quality_review(db: Session, filters: DashboardFilters) -> dict[str, Any]:
    demo_ids = _demo_site_ids(db)
    clauses = _inspection_filters(filters, demo_ids)
    inspections = list(
        db.scalars(
            select(Inspection)
            .where(and_(*clauses) if clauses else True)
            .order_by(Inspection.created_at.desc())
            .limit(200)
        ).all()
    )
    batch_clauses = _batch_filters(filters, demo_ids)
    batches = list(
        db.scalars(select(MaterialBatch).where(and_(*batch_clauses) if batch_clauses else True)).all()
    )

    pending = []
    manual = []
    low_conf = []
    for insp in inspections:
        item = {
            "id": insp.id,
            "status": insp.status,
            "created_at": str(insp.created_at),
            "detail_url": f"/inspection/{insp.id}",
            "is_demo": insp.site_id in demo_ids,
        }
        if insp.status in {
            InspectionStatus.HUMAN_INSPECTION_REQUIRED.value,
            InspectionStatus.ADDITIONAL_TESTING_REQUIRED.value,
        }:
            pending.append(item)
        if insp.status == InspectionStatus.HUMAN_INSPECTION_REQUIRED.value:
            manual.append(item)
        clf = insp.classification_result or {}
        for c in clf.get("items") or []:
            conf = c.get("confidence")
            if conf is not None and conf < 0.55:
                low_conf.append({**item, "confidence": conf, "condition": c.get("condition")})
                break

    quarantined = [
        {
            "id": b.id,
            "batch_code": b.batch_code,
            "material_key": b.material_key,
            "quarantined_quantity": b.quarantined_quantity,
            "unit": b.unit,
            "is_demo": b.site_id in demo_ids,
        }
        for b in batches
        if (b.quarantined_quantity or 0) > 0
        or b.quality_status == InspectionStatus.QUARANTINED.value
    ]

    # "Missing testing documents" — inspections requiring additional testing without notes/docs
    missing_docs = [
        {
            "id": i.id,
            "status": i.status,
            "created_at": str(i.created_at),
            "detail_url": f"/inspection/{i.id}",
        }
        for i in inspections
        if i.status == InspectionStatus.ADDITIONAL_TESTING_REQUIRED.value
        and not (i.notes and i.notes.strip())
    ]

    recent_decisions = [
        {
            "id": i.id,
            "status": i.status,
            "reviewed_at": str(i.reviewed_at) if i.reviewed_at else None,
            "reviewed_by": i.reviewed_by,
            "notes": i.notes,
            "is_demo": i.site_id in demo_ids,
        }
        for i in inspections
        if i.reviewed_at is not None
    ][:20]

    return {
        "pending_approvals": pending[:30],
        "manual_inspection": manual[:30],
        "low_confidence": low_conf[:30],
        "quarantined_batches": quarantined[:30],
        "missing_testing_documents": missing_docs[:30],
        "recent_decisions": recent_decisions,
    }


def get_filter_options(db: Session) -> dict[str, Any]:
    sites = db.scalars(select(Site).where(Site.is_active.is_(True)).order_by(Site.name)).all()
    materials = db.scalars(
        select(MaterialBatch.material_key).distinct().order_by(MaterialBatch.material_key)
    ).all()
    suppliers = db.scalars(select(Supplier).order_by(Supplier.name)).all()
    versions = set()
    for artifact in db.scalars(select(ModelArtifact)).all():
        if artifact.version:
            versions.add(artifact.version)
    return {
        "sites": {s.id: f"{s.name} ({s.code})" for s in sites},
        "materials": {m: m for m in materials if m},
        "suppliers": {s.id: s.name for s in suppliers},
        "statuses": {s.value: s.value.replace("_", " ") for s in InspectionStatus},
        "model_versions": {v: v for v in sorted(versions)},
        "demo_present": any(s.code == DEMO_SITE_CODE for s in sites),
    }


def last_activity_stamp(db: Session) -> str:
    """Lightweight stamp for change detection / refresh."""
    insp = db.scalar(select(func.max(Inspection.updated_at)))
    batch = db.scalar(select(func.max(MaterialBatch.updated_at)))
    job = db.scalar(select(func.max(TrainingJob.updated_at)))
    ev = db.scalar(select(func.max(EvaluationRecord.created_at)))
    return "|".join(str(x) for x in (insp, batch, job, ev))


def build_dashboard_snapshot(db: Session, filters: DashboardFilters) -> DashboardSnapshot:
    options = get_filter_options(db)
    demo_present = bool(options.get("demo_present"))
    return DashboardSnapshot(
        generated_at=datetime.now(UTC),
        filters={
            "site_id": filters.site_id,
            "material_key": filters.material_key,
            "supplier_id": filters.supplier_id,
            "status": filters.status,
            "model_version": filters.model_version,
            "date_from": str(filters.date_from) if filters.date_from else None,
            "date_to": str(filters.date_to) if filters.date_to else None,
            "include_demo": filters.include_demo,
            "search": filters.search,
            "page": filters.page,
            "page_size": filters.page_size,
        },
        data_source="SQLAlchemy persistent records (Material Intelligence DB)",
        is_demo_present=demo_present,
        kpis=compute_executive_kpis(db, filters),
        charts=compute_inspection_charts(db, filters),
        inventory=compute_inventory_panel(db, filters),
        model_performance=compute_model_performance(db),
        recent_inspections=compute_recent_inspections(db, filters),
        quality_review=compute_quality_review(db, filters),
        filter_options=options,
    )
