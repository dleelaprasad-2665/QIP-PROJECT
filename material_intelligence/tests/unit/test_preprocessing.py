"""CLAHE and preprocessing tests."""

from __future__ import annotations

import numpy as np

from material_intelligence.app.domain.schemas import PreprocessProfile
from material_intelligence.app.ml.preprocessing import (
    apply_clahe_lab,
    normalize_image,
    preprocess_image,
    validate_image_array,
)


def test_validate_image_array(sample_bgr):
    validate_image_array(sample_bgr)


def test_clahe_changes_luminance(sample_bgr):
    enhanced = apply_clahe_lab(sample_bgr, clip_limit=3.0, tile_grid=(8, 8))
    assert enhanced.shape == sample_bgr.shape
    assert enhanced.dtype == np.uint8
    # CLAHE should alter pixel distribution for non-uniform images.
    assert not np.array_equal(enhanced, sample_bgr) or sample_bgr.std() < 1e-6


def test_preprocess_profile_disable_clahe(sample_bgr):
    profile = PreprocessProfile(enable_clahe=False, resize_to=(64, 64), brightness=0, contrast=1.0)
    result = preprocess_image(sample_bgr, profile)
    assert result.processed_bgr.shape[:2] == (64, 64)


def test_normalize_range(sample_bgr):
    norm = normalize_image(sample_bgr)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
