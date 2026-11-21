"""EfficientNet dimension and training smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from material_intelligence.app.ml.efficientnet_classifier import (
    EfficientNetClassifier,
    build_efficientnet_b0,
)


def test_output_dimensions():
    model = build_efficientnet_b0(num_classes=4, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (2, 4)


def test_train_and_infer_tiny_folder(tmp_path: Path):
    for cls in ("cracked", "no_visible_damage"):
        d = tmp_path / cls
        d.mkdir()
        for i in range(4):
            arr = np.random.default_rng(i).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i}.jpg")

    clf = EfficientNetClassifier(device="cpu", image_size=64)
    out = tmp_path / "model.pt"
    metrics = clf.train_from_folder(
        tmp_path,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        image_size=64,
        seed=0,
        early_stopping_patience=2,
        output_path=out,
        version="tiny_test",
        material_key="clay_brick",
        val_ratio=0.25,
    )
    assert out.exists()
    assert "history" in metrics
    loaded = EfficientNetClassifier(out, device="cpu")
    assert loaded.is_ready()
    image = np.random.default_rng(1).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    result = loaded.predict(image, expected_material_key="clay_brick")
    assert result.condition in {"cracked", "no_visible_damage"}
    assert result.confidence is not None
