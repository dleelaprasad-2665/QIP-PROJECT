"""EfficientNet-B0 embedding extraction (pooled features, no classification head)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from material_intelligence.app.config import get_settings
from material_intelligence.app.ml.efficientnet_classifier import EfficientNetClassifier, default_transforms


@dataclass
class EmbeddingRecord:
    image_id: str
    path: str
    embedding: list[float]
    material_type: str | None
    model_version: str | None
    preprocess_version: str
    extracted_at: str


class EfficientNetEmbedder:
    """Extract 1280-d pooled features from a ready EfficientNet-B0 classifier."""

    def __init__(self, classifier: EfficientNetClassifier) -> None:
        if not classifier.is_ready() or classifier.model is None:
            raise RuntimeError("Classifier must be ready for embedding extraction.")
        self.classifier = classifier
        self.model = classifier.model
        self.model.eval()
        self.preprocess_version = f"imagenet_resize_{classifier.image_size}"

    @torch.inference_mode()
    def embed_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = default_transforms(self.classifier.image_size, train=False)(pil)
        tensor = tensor.unsqueeze(0).to(self.classifier.device)
        features = self.model.features(tensor)
        pooled = self.model.avgpool(features)
        flat = torch.flatten(pooled, 1)
        return flat.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def embed_path(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unable to read image: {path}")
        return self.embed_bgr(image)


def _cache_key(path: Path, model_version: str | None, preprocess_version: str) -> str:
    payload = f"{path.resolve()}|{path.stat().st_mtime_ns}|{model_version}|{preprocess_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_embeddings_for_folder(
    classifier: EfficientNetClassifier,
    root: Path,
    *,
    material_type: str | None = None,
    use_cache: bool = True,
    limit: int | None = None,
) -> list[EmbeddingRecord]:
    from datetime import UTC, datetime

    embedder = EfficientNetEmbedder(classifier)
    settings = get_settings()
    cache_dir = settings.model_dir / "embeddings_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in suffixes]
    if material_type:
        # Prefer class folder named material_type when present.
        preferred = root / material_type
        if preferred.exists():
            paths = [p for p in paths if preferred in p.parents or p.parent == preferred]
    if limit is not None:
        paths = paths[:limit]

    records: list[EmbeddingRecord] = []
    now = datetime.now(UTC).isoformat()
    for path in paths:
        key = _cache_key(path, classifier.version, embedder.preprocess_version)
        cache_path = cache_dir / f"{key}.npz"
        meta_path = cache_dir / f"{key}.json"
        if use_cache and cache_path.exists():
            vec = np.load(cache_path)
        else:
            vec = embedder.embed_path(path)
            if use_cache:
                np.save(cache_path, vec)
                meta_path.write_text(
                    json.dumps(
                        {
                            "path": str(path),
                            "model_version": classifier.version,
                            "preprocess_version": embedder.preprocess_version,
                        }
                    ),
                    encoding="utf-8",
                )
        # Infer material from parent folder when available.
        parent_name = path.parent.name
        material = material_type or (parent_name if parent_name not in {"images", "train", "val", "test", "valid"} else None)
        records.append(
            EmbeddingRecord(
                image_id=key[:16],
                path=str(path),
                embedding=vec.tolist(),
                material_type=material,
                model_version=classifier.version,
                preprocess_version=embedder.preprocess_version,
                extracted_at=now,
            )
        )
    return records


def embeddings_matrix(records: list[EmbeddingRecord]) -> np.ndarray:
    if not records:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray([r.embedding for r in records], dtype=np.float32)
