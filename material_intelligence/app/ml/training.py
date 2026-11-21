"""Training orchestration helpers shared by job workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from material_intelligence.app.domain.enums import ModelType
from material_intelligence.app.domain.schemas import TrainingJobConfig
from material_intelligence.app.ml.efficientnet_classifier import EfficientNetClassifier
from material_intelligence.app.ml.yolo_detector import YOLODetector
from material_intelligence.app.utils.logging import get_logger

logger = get_logger(__name__)


def run_training(
    config: TrainingJobConfig,
    *,
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.model_type == ModelType.EFFICIENTNET_B0:
        clf = EfficientNetClassifier(device=config.device, image_size=config.image_size)
        version = config.experiment_name
        artifact = output_dir / f"{version}.pt"
        metrics = clf.train_from_folder(
            dataset_root,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            image_size=config.image_size,
            seed=config.seed,
            early_stopping_patience=config.early_stopping_patience,
            output_path=artifact,
            version=version,
            material_key=config.extra.get("material_key"),
        )
        return {
            "artifact_path": str(artifact),
            "metrics": metrics,
            "class_mapping": clf.class_mapping,
            "preprocess_config": clf.preprocess_config,
            "version": version,
        }

    if config.model_type == ModelType.YOLOV8:
        data_yaml = dataset_root / "data.yaml"
        if not data_yaml.exists():
            data_yaml = dataset_root / "dataset.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(
                "YOLO training requires data.yaml (Roboflow/CVAT export) in the dataset root."
            )
        detector = YOLODetector(variant=config.yolo_variant, device=config.device)
        result = detector.train(
            data_yaml,
            epochs=config.epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            lr0=config.learning_rate,
            project=output_dir,
            name=config.experiment_name,
            device=config.device,
        )
        return {
            "artifact_path": result.get("artifact_path"),
            "metrics": result.get("metrics", {}),
            "class_mapping": {name: i for i, name in enumerate(detector.class_names)},
            "preprocess_config": config.preprocess_profile.model_dump(),
            "version": config.experiment_name,
        }

    raise ValueError(f"Unsupported model type: {config.model_type}")
