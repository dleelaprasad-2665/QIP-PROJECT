"""YOLO adapter untrained-state behaviour."""

from __future__ import annotations

import numpy as np

from material_intelligence.app.domain.enums import ModelStatus
from material_intelligence.app.ml.yolo_detector import YOLODetector


def test_untrained_yolo_returns_clear_state(sample_bgr):
    detector = YOLODetector(weights_path=None, device="cpu")
    result = detector.predict(sample_bgr)
    assert result.model_status == ModelStatus.NOT_TRAINED
    assert result.boxes == []
    assert result.message is not None
    assert "Model Not Trained" in result.message
