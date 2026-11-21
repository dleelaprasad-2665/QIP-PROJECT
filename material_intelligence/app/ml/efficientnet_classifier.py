"""EfficientNet-B0 condition classifier using Torchvision."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from material_intelligence.app.domain.enums import ModelStatus
from material_intelligence.app.domain.schemas import ClassificationResult
from material_intelligence.app.utils.logging import get_logger

logger = get_logger(__name__)


class FolderClassificationDataset(Dataset):
    def __init__(self, root: Path, class_to_idx: dict[str, int], transform=None) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        self.class_to_idx = class_to_idx
        for class_name, idx in class_to_idx.items():
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.rglob("*"):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                    self.samples.append((path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    try:
        model = models.efficientnet_b0(weights=weights)
    except Exception:  # noqa: BLE001 — offline / no weights available
        logger.warning("ImageNet weights unavailable; initializing EfficientNet-B0 randomly.")
        model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def default_transforms(image_size: int = 224, train: bool = False):
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train:
        ops = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    return transforms.Compose(ops)


class EfficientNetClassifier:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        class_mapping: dict[str, int] | None = None,
        device: str = "cpu",
        image_size: int = 224,
        material_key: str | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.image_size = image_size
        self.material_key = material_key
        self.class_mapping = class_mapping or {}
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        self.model: nn.Module | None = None
        self.status = ModelStatus.NOT_TRAINED
        self.version: str | None = None
        self.preprocess_config: dict[str, Any] = {"image_size": image_size}

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load(checkpoint_path)

    def is_ready(self) -> bool:
        return self.status == ModelStatus.READY and self.model is not None and bool(self.class_mapping)

    def load(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
            self.class_mapping = payload.get("class_mapping", {})
            self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
            self.material_key = payload.get("material_key", self.material_key)
            self.preprocess_config = payload.get("preprocess_config", self.preprocess_config)
            self.image_size = int(self.preprocess_config.get("image_size", self.image_size))
            num_classes = len(self.class_mapping)
            if num_classes == 0:
                raise ValueError("Checkpoint missing class_mapping.")
            self.model = build_efficientnet_b0(num_classes, pretrained=False)
            self.model.load_state_dict(payload["state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.status = ModelStatus.READY
            self.version = payload.get("version", path.stem)
            logger.info("Loaded EfficientNet checkpoint %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load EfficientNet checkpoint: %s", exc)
            self.model = None
            self.status = ModelStatus.FAILED

    def save(
        self,
        path: str | Path,
        *,
        version: str,
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        if self.model is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "class_mapping": self.class_mapping,
                "material_key": self.material_key,
                "preprocess_config": self.preprocess_config,
                "version": version,
                "metrics": metrics or {},
            },
            path,
        )
        meta = path.with_suffix(".json")
        meta.write_text(
            json.dumps(
                {
                    "version": version,
                    "class_mapping": self.class_mapping,
                    "material_key": self.material_key,
                    "preprocess_config": self.preprocess_config,
                    "metrics": metrics or {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.version = version
        self.status = ModelStatus.READY
        return path

    def _tensor_from_bgr(self, image_bgr: np.ndarray) -> torch.Tensor:
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = default_transforms(self.image_size, train=False)(pil)
        return tensor.unsqueeze(0).to(self.device)

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        expected_material_key: str | None = None,
    ) -> ClassificationResult:
        if not self.is_ready():
            return ClassificationResult(
                model_status=self.status,
                message="Model Not Trained — train EfficientNet-B0 on construction-material conditions.",
            )

        if (
            expected_material_key
            and self.material_key
            and expected_material_key != self.material_key
        ):
            return ClassificationResult(
                material_key=self.material_key,
                model_version=self.version,
                model_status=self.status,
                message=(
                    f"Material mismatch: detector reported '{expected_material_key}' but "
                    f"classifier is trained for '{self.material_key}'."
                ),
            )

        assert self.model is not None
        tensor = self._tensor_from_bgr(image_bgr)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        _ = (time.perf_counter() - start) * 1000
        confidence, pred_idx = torch.max(probs, dim=0)
        condition = self.idx_to_class.get(int(pred_idx.item()), str(int(pred_idx.item())))
        class_probabilities = {
            self.idx_to_class[i]: float(probs[i].item()) for i in range(len(probs))
        }

        uncertain = float(confidence.item()) < 0.45
        message = "Uncertain prediction — human inspection recommended." if uncertain else None

        return ClassificationResult(
            material_key=self.material_key,
            condition=condition,
            confidence=float(confidence.item()),
            class_probabilities=class_probabilities,
            model_version=self.version,
            model_status=self.status,
            message=message,
        )

    def train_from_folder(
        self,
        dataset_root: Path,
        *,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        image_size: int = 224,
        seed: int = 42,
        early_stopping_patience: int = 3,
        output_path: Path,
        version: str,
        material_key: str | None = None,
        val_ratio: float = 0.2,
    ) -> dict[str, Any]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        class_names = sorted(
            p.name for p in dataset_root.iterdir() if p.is_dir() and any(p.iterdir())
        )
        if len(class_names) < 2:
            raise ValueError("Classification dataset must contain at least two class folders.")

        self.class_mapping = {name: i for i, name in enumerate(class_names)}
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        self.material_key = material_key
        self.image_size = image_size
        self.preprocess_config = {"image_size": image_size, "normalize": "imagenet"}

        full = FolderClassificationDataset(
            dataset_root, self.class_mapping, transform=default_transforms(image_size, train=True)
        )
        if len(full) == 0:
            raise ValueError("No training images found.")

        indices = np.arange(len(full))
        np.random.shuffle(indices)
        split = max(1, int(len(indices) * (1 - val_ratio)))
        train_idx, val_idx = indices[:split], indices[split:]
        if len(val_idx) == 0:
            val_idx = train_idx[:1]

        train_ds = torch.utils.data.Subset(full, train_idx.tolist())
        # Validation uses eval transforms — rebuild subset with shared samples but different transform.
        val_full = FolderClassificationDataset(
            dataset_root, self.class_mapping, transform=default_transforms(image_size, train=False)
        )
        val_ds = torch.utils.data.Subset(val_full, val_idx.tolist())

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        self.model = build_efficientnet_b0(len(class_names), pretrained=True).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val = float("inf")
        patience = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            train_loss = running_loss / max(1, total)
            train_acc = correct / max(1, total)

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_loss /= max(1, val_total)
            val_acc = val_correct / max(1, val_total)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            logger.info(
                "Epoch %s/%s train_loss=%.4f val_loss=%.4f val_acc=%.3f",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                val_acc,
            )

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    logger.info("Early stopping at epoch %s", epoch + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        metrics = {
            "history": history,
            "best_val_loss": best_val,
            "final_val_accuracy": history["val_acc"][-1] if history["val_acc"] else None,
            "num_classes": len(class_names),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
        }
        self.save(output_path, version=version, metrics=metrics)
        return metrics
