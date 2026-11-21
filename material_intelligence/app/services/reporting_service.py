"""CSV and PDF report generation from persisted records."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.orm import Session

from material_intelligence.app.config import get_settings
from material_intelligence.app.db.models import EvaluationRecord, Inspection, MaterialBatch, ModelArtifact, Site
from material_intelligence.app.services.dashboard_analytics import (
    DEMO_LABEL,
    DashboardFilters,
    compute_recent_inspections,
)


def inspections_csv(db: Session, site_id: int | None = None) -> str:
    query = select(Inspection).order_by(Inspection.created_at.desc())
    if site_id is not None:
        query = query.where(Inspection.site_id == site_id)
    rows = db.scalars(query).all()
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "site_id",
            "batch_id",
            "status",
            "processing_ms",
            "created_at",
            "reviewed_at",
            "image_path",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.id,
                row.site_id,
                row.batch_id,
                row.status,
                row.processing_ms,
                row.created_at,
                row.reviewed_at,
                row.image_path,
            ]
        )
    return buffer.getvalue()


def inventory_csv(db: Session, site_id: int | None = None) -> str:
    query = select(MaterialBatch).order_by(MaterialBatch.created_at.desc())
    if site_id is not None:
        query = query.where(MaterialBatch.site_id == site_id)
    rows = db.scalars(query).all()
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "site_id",
            "material_key",
            "batch_code",
            "quantity",
            "approved_quantity",
            "quarantined_quantity",
            "rejected_quantity",
            "quality_status",
            "storage_location",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.id,
                row.site_id,
                row.material_key,
                row.batch_code,
                row.quantity,
                row.approved_quantity,
                row.quarantined_quantity,
                row.rejected_quantity,
                row.quality_status,
                row.storage_location,
            ]
        )
    return buffer.getvalue()


def generate_site_pdf(db: Session, site_id: int) -> Path:
    settings = get_settings()
    site = db.get(Site, site_id)
    if site is None:
        raise ValueError("Site not found.")
    inspections = db.scalars(
        select(Inspection).where(Inspection.site_id == site_id).order_by(Inspection.created_at.desc())
    ).all()
    batches = db.scalars(select(MaterialBatch).where(MaterialBatch.site_id == site_id)).all()

    out = settings.report_dir / f"site_{site.code}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Material Intelligence — Site Quality Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Site: {site.name} ({site.code})", styles["Heading2"]),
        Paragraph(f"Location: {site.location or 'N/A'}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Inventory Batches", styles["Heading2"]),
    ]

    batch_data = [["Batch", "Material", "Qty", "Approved", "Quarantine", "Rejected", "Status"]]
    for b in batches:
        batch_data.append(
            [
                b.batch_code,
                b.material_key,
                f"{b.quantity:.1f}",
                f"{b.approved_quantity:.1f}",
                f"{b.quarantined_quantity:.1f}",
                f"{b.rejected_quantity:.1f}",
                b.quality_status,
            ]
        )
    if len(batch_data) == 1:
        batch_data.append(["—", "No batches", "—", "—", "—", "—", "—"])
    table = Table(batch_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2744")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
            ]
        )
    )
    story.extend([table, Spacer(1, 16), Paragraph("Recent Inspections", styles["Heading2"])])

    insp_data = [["ID", "Status", "Processing ms", "Created"]]
    for insp in inspections[:40]:
        insp_data.append(
            [
                str(insp.id),
                insp.status,
                f"{insp.processing_ms or 0:.1f}",
                str(insp.created_at),
            ]
        )
    if len(insp_data) == 1:
        insp_data.append(["—", "No inspections", "—", "—"])
    insp_table = Table(insp_data, repeatRows=1)
    insp_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2744")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(insp_table)
    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "Note: Visual AI screening does not certify structural material safety. "
            "Approved quantities require authorized engineering review.",
            styles["Italic"],
        )
    )
    doc.build(story)
    return out


def filtered_inspections_csv(db: Session, filters: DashboardFilters) -> str:
    result = compute_recent_inspections(
        db,
        DashboardFilters(
            site_id=filters.site_id,
            material_key=filters.material_key,
            supplier_id=filters.supplier_id,
            status=filters.status,
            model_version=filters.model_version,
            date_from=filters.date_from,
            date_to=filters.date_to,
            search=filters.search,
            include_demo=filters.include_demo,
            page=1,
            page_size=10_000,
        ),
    )
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["# Material Intelligence — Filtered Inspection Export"])
    writer.writerow(["# data_source", "SQLAlchemy persistent records"])
    writer.writerow(["# date_from", filters.date_from])
    writer.writerow(["# date_to", filters.date_to])
    writer.writerow(["# site_id", filters.site_id])
    writer.writerow(["# material_key", filters.material_key])
    writer.writerow(["# supplier_id", filters.supplier_id])
    writer.writerow(["# status", filters.status])
    writer.writerow(["# model_version", filters.model_version])
    writer.writerow(["# include_demo", filters.include_demo])
    writer.writerow([])
    writer.writerow(
        [
            "id",
            "material",
            "defect",
            "confidence",
            "site",
            "batch_code",
            "created_at",
            "status",
            "is_demo",
        ]
    )
    for row in result["rows"]:
        writer.writerow(
            [
                row["id"],
                row["material"],
                row["defect"],
                row["confidence"],
                row["site"],
                row["batch_code"],
                row["created_at"],
                row["status"],
                row["is_demo"],
            ]
        )
    return buffer.getvalue()


def generate_filtered_inspection_pdf(db: Session, filters: DashboardFilters) -> Path:
    settings = get_settings()
    result = compute_recent_inspections(
        db,
        DashboardFilters(
            **{**filters.__dict__, "page": 1, "page_size": 100},
        ),
    )
    out = settings.report_dir / f"inspections_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Material Intelligence — Filtered Inspection Summary", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"Data source: SQLAlchemy persistent records", styles["Normal"]),
        Paragraph(
            f"Date range: {filters.date_from or '—'} → {filters.date_to or '—'}",
            styles["Normal"],
        ),
        Paragraph(
            f"Filters: site={filters.site_id}, material={filters.material_key}, "
            f"supplier={filters.supplier_id}, status={filters.status}, "
            f"model={filters.model_version}, include_demo={filters.include_demo}",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]
    data = [["ID", "Material", "Defect", "Conf", "Site", "Status", "Demo"]]
    for row in result["rows"][:80]:
        data.append(
            [
                str(row["id"]),
                str(row["material"]),
                str(row["defect"]),
                f"{row['confidence']:.2f}" if row["confidence"] is not None else "—",
                str(row["site"])[:18],
                str(row["status"])[:22],
                "YES" if row["is_demo"] else "",
            ]
        )
    if len(data) == 1:
        data.append(["—", "No rows", "—", "—", "—", "—", "—"])
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2744")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 12),
            Paragraph(
                f"Note: Rows marked Demo are {DEMO_LABEL}. "
                "Visual AI screening does not certify structural safety.",
                styles["Italic"],
            ),
        ]
    )
    doc.build(story)
    return out


def evaluation_results_csv(db: Session) -> str:
    rows = db.scalars(select(EvaluationRecord).order_by(EvaluationRecord.created_at.desc())).all()
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["# Material Intelligence — Model Evaluation Export"])
    writer.writerow(["# data_source", "evaluation_records table"])
    writer.writerow([])
    writer.writerow(["id", "model_id", "model_name", "model_type", "version", "split", "metrics", "created_at"])
    for row in rows:
        model = db.get(ModelArtifact, row.model_id)
        writer.writerow(
            [
                row.id,
                row.model_id,
                model.name if model else None,
                model.model_type if model else None,
                model.version if model else None,
                row.split,
                row.metrics,
                row.created_at,
            ]
        )
    if not rows:
        writer.writerow(["", "", "", "", "", "not_evaluated", "", ""])
    return buffer.getvalue()
