"""Grad-CAM output checks with a tiny trained classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from material_intelligence.app.ml.efficientnet_classifier import EfficientNetClassifier
from material_intelligence.app.ml.gradcam import GradCAMExplainer


def test_gradcam_finite_and_shaped(tmp_path: Path):
    for cls in ("a", "b"):
        d = tmp_path / cls
        d.mkdir()
        for i in range(3):
            Image.fromarray(
                np.random.default_rng(i).integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
            ).save(d / f"{i}.jpg")

    clf = EfficientNetClassifier(device="cpu", image_size=64)
    ckpt = tmp_path / "clf.pt"
    clf.train_from_folder(
        tmp_path,
        epochs=1,
        batch_size=2,
        image_size=64,
        output_path=ckpt,
        version="gcam",
        material_key="clay_brick",
        val_ratio=0.34,
    )
    loaded = EfficientNetClassifier(ckpt, device="cpu")
    explainer = GradCAMExplainer(loaded)
    image = np.random.default_rng(7).integers(0, 255, size=(80, 90, 3), dtype=np.uint8)
    result = explainer.explain(image, opacity=0.4)
    assert result["heatmap"].shape[:2] == image.shape[:2]
    assert np.isfinite(result["heatmap"]).all()
    assert result["overlay_bgr"].shape == image.shape
