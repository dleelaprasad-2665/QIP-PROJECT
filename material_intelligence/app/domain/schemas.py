"""Pydantic schemas shared across services and API layers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from material_intelligence.app.domain.enums import (
    DatasetFormat,
    DatasetStatus,
    InspectionStatus,
    JobStatus,
    ModelStatus,
    ModelType,
    UserRole,
)


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=8, max_length=128)
    role: UserRole = UserRole.VIEWER


class UserOut(ORMModel):
    id: int
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SiteCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    code: str = Field(min_length=2, max_length=32)
    location: str | None = None
    description: str | None = None


class SiteOut(ORMModel):
    id: int
    name: str
    code: str
    location: str | None
    description: str | None
    is_active: bool


class MaterialClassCreate(BaseModel):
    key: str = Field(min_length=2, max_length=64)
    display_name: str = Field(min_length=2, max_length=120)
    description: str | None = None

    @field_validator("key")
    @classmethod
    def _normalize_key(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")


class DefectClassCreate(BaseModel):
    material_key: str
    key: str = Field(min_length=2, max_length=64)
    display_name: str = Field(min_length=2, max_length=120)
    is_high_risk: bool = False

    @field_validator("key")
    @classmethod
    def _normalize_key(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")


class DatasetImportResult(BaseModel):
    dataset_id: int
    name: str
    format: DatasetFormat
    status: DatasetStatus
    image_count: int
    annotation_count: int
    class_names: list[str]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float | None = None


class DetectionResult(BaseModel):
    boxes: list[BoundingBox] = Field(default_factory=list)
    model_version: str | None = None
    model_status: ModelStatus = ModelStatus.NOT_TRAINED
    inference_ms: float | None = None
    message: str | None = None


class ClassificationResult(BaseModel):
    material_key: str | None = None
    condition: str | None = None
    confidence: float | None = None
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    model_version: str | None = None
    model_status: ModelStatus = ModelStatus.NOT_TRAINED
    message: str | None = None


class InspectionPrediction(BaseModel):
    detection: DetectionResult
    classifications: list[ClassificationResult] = Field(default_factory=list)
    recommended_status: InspectionStatus
    processing_ms: float
    gradcam_path: str | None = None
    notes: list[str] = Field(default_factory=list)


class PreprocessProfile(BaseModel):
    name: str = "default"
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    resize_to: tuple[int, int] | None = (640, 640)
    normalize: bool = True
    brightness: float = 0.0
    contrast: float = 1.0


class TrainingJobConfig(BaseModel):
    model_type: ModelType
    dataset_id: int
    experiment_name: str
    epochs: int = Field(default=10, ge=1, le=500)
    batch_size: int = Field(default=8, ge=1, le=128)
    learning_rate: float = Field(default=1e-3, gt=0)
    image_size: int = Field(default=224, ge=64, le=1280)
    device: str = "cpu"
    yolo_variant: str = "yolov8n"
    early_stopping_patience: int = Field(default=5, ge=1, le=50)
    seed: int = 42
    preprocess_profile: PreprocessProfile = Field(default_factory=PreprocessProfile)
    extra: dict[str, Any] = Field(default_factory=dict)


class TrainingJobOut(ORMModel):
    id: int
    experiment_name: str
    model_type: ModelType
    status: JobStatus
    progress: float
    message: str | None
    metrics: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class EvaluationMetrics(BaseModel):
    model_type: ModelType
    model_version: str
    split: str
    metrics: dict[str, Any]
    per_class: dict[str, Any] = Field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = None
    class_names: list[str] = Field(default_factory=list)
    evaluated_at: datetime | None = None
    notes: list[str] = Field(default_factory=list)


class InventoryBatchCreate(BaseModel):
    site_id: int
    material_key: str
    batch_code: str = Field(min_length=2, max_length=64)
    supplier_name: str | None = None
    quantity: float = Field(gt=0)
    unit: str = "units"
    storage_location: str | None = None
    purchase_reference: str | None = None


class InventorySummary(BaseModel):
    site_id: int
    material_key: str
    total_quantity: float
    approved_quantity: float
    quarantined_quantity: float
    rejected_quantity: float
    pending_review: int
