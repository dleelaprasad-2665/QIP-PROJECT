"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Material Intelligence"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8080
    secret_key: str = "dev-only-change-me"
    session_max_age_seconds: int = 86_400

    database_url: str = "sqlite:///./data/material_intelligence.db"

    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    dataset_dir: Path = Path("./data/datasets")
    model_dir: Path = Path("./data/models")
    report_dir: Path = Path("./data/reports")
    job_dir: Path = Path("./data/jobs")

    max_upload_mb: int = 200
    allowed_image_extensions: str = ".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp"
    allowed_archive_extensions: str = ".zip"

    default_device: str = "cpu"
    yolo_variant: str = "yolov8n"
    efficientnet_image_size: int = 224
    training_seed: int = 42

    bootstrap_admin_email: str = "admin@material-intelligence.local"
    bootstrap_admin_password: str = "ChangeMeNow!"
    bootstrap_admin_name: str = "System Administrator"

    log_level: str = "INFO"

    @field_validator(
        "data_dir",
        "upload_dir",
        "dataset_dir",
        "model_dir",
        "report_dir",
        "job_dir",
        mode="before",
    )
    @classmethod
    def _resolve_path(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path

    @property
    def image_extensions(self) -> set[str]:
        return {ext.strip().lower() for ext in self.allowed_image_extensions.split(",") if ext.strip()}

    @property
    def archive_extensions(self) -> set[str]:
        return {
            ext.strip().lower() for ext in self.allowed_archive_extensions.split(",") if ext.strip()
        }

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.upload_dir,
            self.dataset_dir,
            self.model_dir,
            self.report_dir,
            self.job_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
