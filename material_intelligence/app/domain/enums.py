"""Domain enumerations for Material Intelligence."""

from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    ADMINISTRATOR = "administrator"
    AI_RESEARCHER = "ai_researcher"
    QUALITY_ENGINEER = "quality_engineer"
    SITE_SUPERVISOR = "site_supervisor"
    VIEWER = "viewer"


class InspectionStatus(StrEnum):
    SCREENING_COMPLETED = "screening_completed"
    HUMAN_INSPECTION_REQUIRED = "human_inspection_required"
    ADDITIONAL_TESTING_REQUIRED = "additional_testing_required"
    QUARANTINED = "quarantined"
    APPROVED = "approved_by_authorized_engineer"
    REJECTED = "rejected_by_authorized_engineer"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(StrEnum):
    YOLOV8 = "yolov8"
    EFFICIENTNET_B0 = "efficientnet_b0"


class ModelStatus(StrEnum):
    NOT_TRAINED = "not_trained"
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


class DatasetFormat(StrEnum):
    YOLO_DETECTION = "yolo_detection"
    IMAGE_FOLDER_CLASSIFICATION = "image_folder_classification"
    COCO = "coco"
    ROBOFLOW_YOLO = "roboflow_yolo"
    CVAT_YOLO = "cvat_yolo"


class DatasetStatus(StrEnum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    VALID = "valid"
    INVALID = "invalid"
    READY_FOR_TRAINING = "ready_for_training"


class StockMovementType(StrEnum):
    RECEIPT = "receipt"
    ISSUE = "issue"
    QUARANTINE = "quarantine"
    RELEASE = "release"
    REJECT = "reject"
    ADJUSTMENT = "adjustment"


class NotificationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


# Default material / defect taxonomies (administrators may extend via DB).
DEFAULT_MATERIALS: dict[str, list[str]] = {
    "clay_brick": [
        "cracked",
        "chipped",
        "broken",
        "no_visible_damage",
    ],
    "steel_rebar": [
        "surface_corrosion",
        "extensive_visible_corrosion",
        "deformation",
        "no_visible_damage",
    ],
    "cement_bag": [
        "torn_packaging",
        "visible_moisture_staining",
        "damaged_packaging",
        "no_visible_damage",
    ],
    "wood": [
        "cracked",
        "broken",
        "visible_moisture_staining",
        "no_visible_damage",
    ],
}

MATERIAL_DISPLAY_NAMES: dict[str, str] = {
    "clay_brick": "Brick",
    "steel_rebar": "Steel",
    "cement_bag": "Cement",
    "wood": "Wood",
}

DEFECT_DISPLAY_NAMES: dict[str, str] = {
    "cracked": "Cracked",
    "chipped": "Chipped",
    "broken": "Broken",
    "no_visible_damage": "No Visible Damage",
    "surface_corrosion": "Surface Corrosion",
    "extensive_visible_corrosion": "Extensive Visible Corrosion",
    "deformation": "Deformation",
    "torn_packaging": "Torn Packaging",
    "visible_moisture_staining": "Visible Moisture Staining",
    "damaged_packaging": "Damaged Packaging",
}

# Conditions that require human review / quarantine screening.
HIGH_RISK_CONDITIONS: set[str] = {
    "cracked",
    "broken",
    "extensive_visible_corrosion",
    "deformation",
    "torn_packaging",
    "visible_moisture_staining",
    "damaged_packaging",
}
