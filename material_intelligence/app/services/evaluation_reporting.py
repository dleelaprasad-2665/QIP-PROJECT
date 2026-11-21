"""PDF/CSV research evaluation report generation from saved experiments."""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from sqlalchemy.orm import Session

from material_intelligence.app.config import get_settings
from sqlalchemy import select

from material_intelligence.app.db.models import EvaluationExperiment


def experiments_to_csv(db: Session, experiment_ids: list[int] | None = None) -> str:
    exps = list(
        db.scalars(
            select(EvaluationExperiment).order_by(EvaluationExperiment.created_at.desc())
        ).all()
    )
    if experiment_ids:
        id_set = set(experiment_ids)
        exps = [e for e in exps if e.id in id_set]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "name",
            "type",
            "status",
            "model_version",
            "dataset_version",
            "n_images",
            "seed",
            "created_at",
            "metrics_json",
        ]
    )
    for exp in exps:
        writer.writerow(
            [
                exp.id,
                exp.name,
                exp.experiment_type,
                exp.status,
                exp.model_version,
                exp.dataset_version,
                exp.n_images,
                exp.seed,
                exp.created_at.isoformat() if exp.created_at else "",
                str(exp.metrics)[:2000],
            ]
        )
        rows.append(exp.id)
    return buffer.getvalue()


def generate_experiment_pdf(db: Session, experiment_id: int) -> Path:
    exp = db.get(EvaluationExperiment, experiment_id)
    if exp is None:
        raise ValueError("Experiment not found.")
    settings = get_settings()
    out_dir = settings.report_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"evaluation_report_{exp.id}.pdf"

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    story = []
    story.append(Paragraph("Material Intelligence — Evaluation Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.now(UTC).isoformat()}", styles["Normal"]))
    story.append(Paragraph(f"Experiment ID: {exp.id}", styles["Heading2"]))
    story.append(Paragraph(f"Name: {exp.name}", styles["Normal"]))
    story.append(Paragraph(f"Type: {exp.experiment_type}", styles["Normal"]))
    story.append(Paragraph(f"Status: {exp.status}", styles["Normal"]))
    story.append(Paragraph(f"Model version: {exp.model_version or '—'}", styles["Normal"]))
    story.append(Paragraph(f"Dataset version: {exp.dataset_version or '—'}", styles["Normal"]))
    story.append(Paragraph(f"Checkpoint: {exp.checkpoint_path or '—'}", styles["Normal"]))
    story.append(Paragraph(f"Seed: {exp.seed}", styles["Normal"]))
    story.append(Paragraph(f"Images evaluated: {exp.n_images}", styles["Normal"]))
    story.append(Paragraph(f"Ground-truth source: {exp.ground_truth_source or '—'}", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Metrics (from completed experiment)", styles["Heading2"]))
    story.append(Paragraph(str(exp.metrics)[:4000].replace("<", "&lt;"), styles["Code"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Limitations", styles["Heading2"]))
    story.append(
        Paragraph(
            "Metrics reflect the selected dataset and checkpoint only. "
            "Clustering metrics do not establish structural meaning. "
            "AI screening does not certify structural safety. "
            "No fabricated metrics are included.",
            styles["Normal"],
        )
    )
    if exp.notes:
        story.append(Paragraph("Notes", styles["Heading2"]))
        story.append(Paragraph(exp.notes.replace("<", "&lt;"), styles["Normal"]))
    doc.build(story)
    return path
