"""OpenCV-based image preprocessing including CLAHE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from material_intelligence.app.domain.schemas import PreprocessProfile


class ImageValidationError(ValueError):
    """Raised when an image cannot be decoded or fails validation."""


@dataclass
class PreprocessResult:
    original_bgr: np.ndarray
    processed_bgr: np.ndarray
    profile: PreprocessProfile

    def to_metadata(self) -> dict:
        return {"profile": asdict(self.profile) if hasattr(self.profile, "__dataclass_fields__") else self.profile.model_dump()}


def load_bgr_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError(f"Unable to decode image: {path}")
    return image


def validate_image_array(image: np.ndarray) -> None:
    if image is None or not isinstance(image, np.ndarray):
        raise ImageValidationError("Image must be a NumPy array.")
    if image.ndim not in (2, 3):
        raise ImageValidationError(f"Unexpected image dimensions: {image.shape}")
    if image.size == 0:
        raise ImageValidationError("Image is empty.")


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def apply_brightness_contrast(
    image_bgr: np.ndarray, brightness: float = 0.0, contrast: float = 1.0
) -> np.ndarray:
    """brightness in [-100, 100], contrast around 1.0."""
    adjusted = image_bgr.astype(np.float32) * float(contrast) + float(brightness)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_clahe_lab(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE on the L channel in LAB space to reduce color distortion."""
    validate_image_array(image_bgr)
    if image_bgr.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(image_bgr)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid)
    l_eq = clahe.apply(l_channel)
    merged = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def resize_image(image_bgr: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    if size is None:
        return image_bgr
    width, height = size
    return cv2.resize(image_bgr, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def normalize_image(image_bgr: np.ndarray) -> np.ndarray:
    """Return float32 image normalized to [0, 1]. Does not mutate uint8 pipeline outputs."""
    return (image_bgr.astype(np.float32) / 255.0).clip(0.0, 1.0)


def preprocess_image(
    image_bgr: np.ndarray,
    profile: PreprocessProfile | None = None,
) -> PreprocessResult:
    profile = profile or PreprocessProfile()
    validate_image_array(image_bgr)
    working = image_bgr.copy()

    if profile.enable_clahe:
        working = apply_clahe_lab(
            working,
            clip_limit=profile.clahe_clip_limit,
            tile_grid=profile.clahe_tile_grid,
        )

    if profile.brightness != 0.0 or profile.contrast != 1.0:
        working = apply_brightness_contrast(working, profile.brightness, profile.contrast)

    if profile.resize_to is not None:
        working = resize_image(working, profile.resize_to)

    # Normalization is available for model adapters; keep uint8 for visual comparison.
    if profile.normalize:
        _ = normalize_image(working)

    return PreprocessResult(original_bgr=image_bgr, processed_bgr=working, profile=profile)


def preprocess_file(path: str | Path, profile: PreprocessProfile | None = None) -> PreprocessResult:
    return preprocess_image(load_bgr_image(path), profile=profile)


def profile_from_dict(data: dict | None) -> PreprocessProfile:
    if not data:
        return PreprocessProfile()
    tile = data.get("clahe_tile_grid", (8, 8))
    if isinstance(tile, list):
        tile = tuple(tile)
    resize_to = data.get("resize_to")
    if isinstance(resize_to, list):
        resize_to = tuple(resize_to)
    return PreprocessProfile(
        name=data.get("name", "default"),
        enable_clahe=bool(data.get("enable_clahe", True)),
        clahe_clip_limit=float(data.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid=tile,
        resize_to=resize_to,
        normalize=bool(data.get("normalize", True)),
        brightness=float(data.get("brightness", 0.0)),
        contrast=float(data.get("contrast", 1.0)),
    )
