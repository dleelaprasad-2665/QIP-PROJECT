"""Genuine Grad-CAM for EfficientNet-B0 using Captum."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from captum.attr import LayerGradCam, LayerAttribution
from PIL import Image

from material_intelligence.app.ml.efficientnet_classifier import EfficientNetClassifier, default_transforms
from material_intelligence.app.utils.logging import get_logger

logger = get_logger(__name__)


class GradCAMExplainer:
    """
    Computes Layer Grad-CAM on EfficientNet-B0's final convolutional features.

    Grad-CAM highlights regions influencing the classifier decision. It is not
    proof that a material is safe or defective.
    """

    def __init__(self, classifier: EfficientNetClassifier) -> None:
        if not classifier.is_ready() or classifier.model is None:
            raise RuntimeError("Classifier must be ready before Grad-CAM.")
        self.classifier = classifier
        self.model = classifier.model
        # Final convolutional feature layer in EfficientNet-B0.
        self.target_layer = self.model.features[-1]
        self.layer_gc = LayerGradCam(self.model, self.target_layer)

    def explain(
        self,
        image_bgr: np.ndarray,
        *,
        target_class: int | None = None,
        opacity: float = 0.45,
    ) -> dict:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = default_transforms(self.classifier.image_size, train=False)(pil)
        tensor = tensor.unsqueeze(0).to(self.classifier.device)

        self.model.eval()
        # Gradients must be enabled for attribution even if inference uses inference_mode.
        with torch.enable_grad():
            tensor = tensor.requires_grad_(True)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            if target_class is None:
                target_class = int(torch.argmax(probs).item())
            attributions = self.layer_gc.attribute(tensor, target=target_class)

        # Upsample attribution map to crop dimensions.
        upsampled = LayerAttribution.interpolate(attributions, (image_bgr.shape[0], image_bgr.shape[1]))
        heatmap = upsampled.squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        else:
            heatmap = np.zeros_like(heatmap)

        if not np.isfinite(heatmap).all():
            raise RuntimeError("Grad-CAM produced non-finite values.")

        overlay = _overlay_heatmap(image_bgr, heatmap, opacity=opacity)
        condition = self.classifier.idx_to_class.get(target_class, str(target_class))
        confidence = float(probs[target_class].item())

        return {
            "heatmap": heatmap,
            "overlay_bgr": overlay,
            "target_class": target_class,
            "condition": condition,
            "confidence": confidence,
            "note": (
                "Grad-CAM highlights image regions influencing the classifier; "
                "it is not proof that the material is safe or defective."
            ),
        }

    def save_visualization(self, overlay_bgr: np.ndarray, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), overlay_bgr)
        return path


def _overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, opacity: float = 0.45) -> np.ndarray:
    heat_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    if colored.shape[:2] != image_bgr.shape[:2]:
        colored = cv2.resize(colored, (image_bgr.shape[1], image_bgr.shape[0]))
    return cv2.addWeighted(image_bgr, 1.0 - opacity, colored, opacity, 0)
