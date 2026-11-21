"""Research-grade evaluation package."""

from material_intelligence.app.ml.evaluation.classification import (
    classification_metrics,
    compute_classification_metrics,
    evaluate_classifier_on_folder,
)
from material_intelligence.app.ml.evaluation.clustering import run_dbscan, run_kmeans
from material_intelligence.app.ml.evaluation.detection import (
    evaluate_yolo_checkpoint,
    yolo_metrics_from_ultralytics,
)
from material_intelligence.app.ml.evaluation.recommendations import (
    recommend_from_prediction,
    evaluate_recommendations,
)

__all__ = [
    "classification_metrics",
    "compute_classification_metrics",
    "evaluate_classifier_on_folder",
    "run_kmeans",
    "run_dbscan",
    "evaluate_yolo_checkpoint",
    "yolo_metrics_from_ultralytics",
    "recommend_from_prediction",
    "evaluate_recommendations",
]
