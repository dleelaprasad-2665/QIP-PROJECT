"""Ultralytics YOLOv8 detection adapter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from material_intelligence.app.domain.enums import ModelStatus
from material_intelligence.app.domain.schemas import BoundingBox, DetectionResult
from material_intelligence.app.utils.logging import get_logger

logger = get_logger(__name__)


class YOLODetector:
    """Wraps Ultralytics YOLO for training and inference with clear untrained states."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        variant: str = "yolov8n",
        class_names: list[str] | None = None,
        conf_threshold: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.variant = variant
        self.weights_path = Path(weights_path) if weights_path else None
        self.class_names = class_names or []
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = None
        self.status = ModelStatus.NOT_TRAINED
        self.version: str | None = None

        if self.weights_path and self.weights_path.exists():
            self._load(self.weights_path)

    def _load(self, path: Path) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(path))
            self.status = ModelStatus.READY
            self.version = path.stem
            names = getattr(self._model, "names", None)
            if isinstance(names, dict):
                self.class_names = [names[i] for i in sorted(names)]
            logger.info("Loaded YOLO weights from %s", path)
        except Exception as exc:  # noqa: BLE001 — surface load failures as NOT_TRAINED
            logger.exception("Failed to load YOLO weights: %s", exc)
            self._model = None
            self.status = ModelStatus.FAILED
            self.version = None

    def is_ready(self) -> bool:
        return self.status == ModelStatus.READY and self._model is not None

    def predict(self, image_bgr: np.ndarray) -> DetectionResult:
        if not self.is_ready():
            return DetectionResult(
                boxes=[],
                model_version=self.version,
                model_status=self.status,
                message="Model Not Trained — register a construction-material YOLO checkpoint.",
            )

        start = time.perf_counter()
        results = self._model.predict(
            source=image_bgr,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        elapsed = (time.perf_counter() - start) * 1000
        boxes: list[BoundingBox] = []
        if results:
            result = results[0]
            names = result.names or {}
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    class_name = names.get(cls_id, str(cls_id))
                    boxes.append(
                        BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                            class_name=str(class_name),
                            confidence=conf,
                        )
                    )

        message = None
        if not boxes:
            message = "No objects detected above the confidence threshold."

        return DetectionResult(
            boxes=boxes,
            model_version=self.version,
            model_status=self.status,
            inference_ms=elapsed,
            message=message,
        )

    def train(
        self,
        data_yaml: str | Path,
        *,
        epochs: int = 10,
        imgsz: int = 640,
        batch: int = 8,
        lr0: float = 1e-3,
        project: str | Path,
        name: str,
        device: str | None = None,
    ) -> dict[str, Any]:
        from ultralytics import YOLO

        base = self.weights_path if self.weights_path and self.weights_path.exists() else f"{self.variant}.pt"
        model = YOLO(str(base))
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            project=str(project),
            name=name,
            device=device or self.device,
            exist_ok=True,
            pretrained=True,
        )
        best = Path(project) / name / "weights" / "best.pt"
        if best.exists():
            self._load(best)
        metrics = {}
        if hasattr(results, "results_dict"):
            metrics = dict(results.results_dict)
        return {"artifact_path": str(best) if best.exists() else None, "metrics": metrics}

    def evaluate(self, data_yaml: str | Path) -> dict[str, Any]:
        if not self.is_ready():
            return {"status": "not_evaluated", "message": "Model Not Trained"}
        metrics = self._model.val(data=str(data_yaml), device=self.device, verbose=False)
        out: dict[str, Any] = {"status": "evaluated"}
        box = getattr(metrics, "box", None)
        if box is not None:
            out["map50"] = float(getattr(box, "map50", float("nan")))
            out["map50_95"] = float(getattr(box, "map", float("nan")))
            out["precision"] = float(getattr(box, "mp", float("nan")))
            out["recall"] = float(getattr(box, "mr", float("nan")))
        return out


def draw_detections(image_bgr: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray:
    import cv2

    canvas = image_bgr.copy()
    for box in boxes:
        pt1 = (int(box.x1), int(box.y1))
        pt2 = (int(box.x2), int(box.y2))
        cv2.rectangle(canvas, pt1, pt2, (0, 200, 255), 2)
        label = box.class_name
        if box.confidence is not None:
            label = f"{label} {box.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (pt1[0], max(15, pt1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas
