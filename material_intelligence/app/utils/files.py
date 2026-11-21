"""Safe file upload and archive handling."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from material_intelligence.app.config import get_settings


class FileValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def unique_filename(original_name: str) -> str:
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix.lower()
    digest = hashlib.sha256(f"{stem}-{Path(original_name).stat().st_mtime if False else stem}".encode()).hexdigest()[:12]
    # Use content-independent uniqueness via random-ish hash of name + size placeholder replaced below.
    return f"{stem[:40]}_{digest}{suffix}"


def make_unique_name(original_name: str, content: bytes | None = None) -> str:
    suffix = Path(original_name).suffix.lower()
    stem = Path(original_name).stem[:40]
    payload = content if content is not None else original_name.encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{stem}_{digest}{suffix}"


def validate_extension(filename: str, allowed: set[str]) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise FileValidationError(f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}")
    return ext


def validate_upload_size(size: int) -> None:
    settings = get_settings()
    if size > settings.max_upload_bytes:
        raise FileValidationError(
            f"File exceeds maximum upload size of {settings.max_upload_mb} MB."
        )


def verify_image_bytes(data: bytes) -> Image.Image:
    from io import BytesIO

    try:
        image = Image.open(BytesIO(data))
        image.verify()
    except UnidentifiedImageError as exc:
        raise FileValidationError("File is not a valid image.") from exc
    # Re-open after verify (Pillow requirement).
    image = Image.open(BytesIO(data))
    image.load()
    return image


def save_upload(data: bytes, destination_dir: Path, original_name: str, *, image: bool = True) -> Path:
    settings = get_settings()
    validate_upload_size(len(data))
    if image:
        validate_extension(original_name, settings.image_extensions)
        verify_image_bytes(data)
    else:
        validate_extension(original_name, settings.archive_extensions | settings.image_extensions)

    destination_dir.mkdir(parents=True, exist_ok=True)
    name = make_unique_name(original_name, data)
    path = destination_dir / name
    path.write_bytes(data)
    return path


def safe_extract_zip(zip_path: Path, destination: Path) -> Path:
    """Extract a ZIP archive while blocking path-traversal entries."""
    destination.mkdir(parents=True, exist_ok=True)
    dest_resolved = destination.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not str(member_path).startswith(str(dest_resolved)):
                raise FileValidationError(
                    f"Archive entry escapes destination directory: {member.filename}"
                )
            if member.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
            else:
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, open(member_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return destination
