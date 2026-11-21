"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from material_intelligence.app.db.session import Base
from material_intelligence.app.domain.enums import (
    DatasetFormat,
    DatasetStatus,
    InspectionStatus,
    JobStatus,
    ModelStatus,
    ModelType,
    NotificationSeverity,
    StockMovementType,
    UserRole,
)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(64), default=UserRole.VIEWER.value)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    sessions: Mapped[list[SessionToken]] = relationship(back_populates="user")
    audit_events: Mapped[list[AuditEvent]] = relationship(back_populates="actor")


class SessionToken(Base):
    __tablename__ = "session_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    token: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="sessions")


class Site(Base, TimestampMixin):
    __tablename__ = "sites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    batches: Mapped[list[MaterialBatch]] = relationship(back_populates="site")
    inspections: Mapped[list[Inspection]] = relationship(back_populates="site")


class MaterialType(Base, TimestampMixin):
    __tablename__ = "material_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    defects: Mapped[list[DefectType]] = relationship(back_populates="material")


class DefectType(Base, TimestampMixin):
    __tablename__ = "defect_types"
    __table_args__ = (UniqueConstraint("material_id", "key", name="uq_defect_per_material"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    material_id: Mapped[int] = mapped_column(ForeignKey("material_types.id", ondelete="CASCADE"))
    key: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str] = mapped_column(String(120))
    is_high_risk: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    material: Mapped[MaterialType] = relationship(back_populates="defects")


class Supplier(Base, TimestampMixin):
    __tablename__ = "suppliers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    batches: Mapped[list[MaterialBatch]] = relationship(back_populates="supplier")


class MaterialBatch(Base, TimestampMixin):
    __tablename__ = "material_batches"
    __table_args__ = (UniqueConstraint("site_id", "batch_code", name="uq_batch_per_site"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    material_key: Mapped[str] = mapped_column(String(64), index=True)
    batch_code: Mapped[str] = mapped_column(String(64))
    supplier_id: Mapped[int | None] = mapped_column(ForeignKey("suppliers.id"), nullable=True)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    approved_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    quarantined_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    rejected_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    unit: Mapped[str] = mapped_column(String(32), default="units")
    storage_location: Mapped[str | None] = mapped_column(String(120), nullable=True)
    purchase_reference: Mapped[str | None] = mapped_column(String(120), nullable=True)
    quality_status: Mapped[str] = mapped_column(
        String(64), default=InspectionStatus.HUMAN_INSPECTION_REQUIRED.value
    )

    site: Mapped[Site] = relationship(back_populates="batches")
    supplier: Mapped[Supplier | None] = relationship(back_populates="batches")
    movements: Mapped[list[StockMovement]] = relationship(back_populates="batch")
    inspections: Mapped[list[Inspection]] = relationship(back_populates="batch")


class StockMovement(Base, TimestampMixin):
    __tablename__ = "stock_movements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("material_batches.id", ondelete="CASCADE"))
    movement_type: Mapped[str] = mapped_column(String(32), default=StockMovementType.RECEIPT.value)
    quantity: Mapped[float] = mapped_column(Float)
    reference: Mapped[str | None] = mapped_column(String(120), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    batch: Mapped[MaterialBatch] = relationship(back_populates="movements")


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(32), default="v1")
    format: Mapped[str] = mapped_column(String(64), default=DatasetFormat.YOLO_DETECTION.value)
    status: Mapped[str] = mapped_column(String(32), default=DatasetStatus.UPLOADED.value)
    root_path: Mapped[str] = mapped_column(String(512))
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    annotation_count: Mapped[int] = mapped_column(Integer, default=0)
    class_names: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str | None] = mapped_column(String(120), nullable=True)
    annotator: Mapped[str | None] = mapped_column(String(120), nullable=True)
    validation_report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    class_distribution: Mapped[dict[str, int] | None] = mapped_column(JSON, nullable=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)


class ModelArtifact(Base, TimestampMixin):
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(64))
    model_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default=ModelStatus.NOT_TRAINED.value)
    artifact_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    class_mapping: Mapped[dict[str, int] | None] = mapped_column(JSON, nullable=True)
    preprocess_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    is_active_inference: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    evaluations: Mapped[list[EvaluationRecord]] = relationship(back_populates="model")


class TrainingJob(Base, TimestampMixin):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_name: Mapped[str] = mapped_column(String(120))
    model_type: Mapped[str] = mapped_column(String(32), default=ModelType.EFFICIENTNET_B0.value)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    status: Mapped[str] = mapped_column(String(32), default=JobStatus.QUEUED.value)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    log_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_id: Mapped[int | None] = mapped_column(ForeignKey("model_artifacts.id"), nullable=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvaluationRecord(Base, TimestampMixin):
    __tablename__ = "evaluation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_artifacts.id", ondelete="CASCADE"))
    split: Mapped[str] = mapped_column(String(32), default="validation")
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
    per_class: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    confusion_matrix: Mapped[list[list[int]] | None] = mapped_column(JSON, nullable=True)
    class_names: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    model: Mapped[ModelArtifact] = relationship(back_populates="evaluations")


class EvaluationExperiment(Base, TimestampMixin):
    """Research-grade evaluation experiment with full reproducibility metadata."""

    __tablename__ = "evaluation_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    experiment_type: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="completed")
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("model_artifacts.id"), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checkpoint_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    preprocess_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    model_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_images: Mapped[int] = mapped_column(Integer, default=0)
    ground_truth_source: Mapped[str | None] = mapped_column(String(160), nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    artifacts: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    researcher_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Inspection(Base, TimestampMixin):
    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id"))
    batch_id: Mapped[int | None] = mapped_column(ForeignKey("material_batches.id"), nullable=True)
    image_path: Mapped[str] = mapped_column(String(512))
    annotated_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    gradcam_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(
        String(64), default=InspectionStatus.SCREENING_COMPLETED.value
    )
    detection_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    classification_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    processing_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_versions: Mapped[dict[str, str] | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    site: Mapped[Site] = relationship(back_populates="inspections")
    batch: Mapped[MaterialBatch | None] = relationship(back_populates="inspections")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(120))
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    actor: Mapped[User | None] = relationship(back_populates="audit_events")


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(160))
    body: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(32), default=NotificationSeverity.INFO.value)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
