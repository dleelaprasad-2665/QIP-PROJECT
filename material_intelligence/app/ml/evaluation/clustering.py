"""Unsupervised clustering evaluation on EfficientNet embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from material_intelligence.app.ml.evaluation.statistics import safe_metric


@dataclass
class ClusteringResult:
    algorithm: str
    labels: list[int]
    metrics: dict[str, Any]
    projection_2d: list[list[float]]
    cluster_summaries: list[dict[str, Any]] = field(default_factory=list)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def prepare_features(
    embeddings: np.ndarray,
    *,
    l2_normalize: bool = True,
    standardize: bool = True,
    pca_dims: int | None = 50,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(embeddings, dtype=np.float64)
    info: dict[str, Any] = {
        "l2_normalize": l2_normalize,
        "standardize": standardize,
        "pca_dims": pca_dims,
        "seed": seed,
        "input_dim": int(x.shape[1]) if x.ndim == 2 else 0,
        "n_samples": int(x.shape[0]) if x.ndim == 2 else 0,
    }
    if x.size == 0:
        return x, info
    if l2_normalize:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x = x / norms
    if standardize and x.shape[0] > 1:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        info["scaler_mean_norm"] = float(np.linalg.norm(scaler.mean_))
    if pca_dims and x.shape[0] > 2 and x.shape[1] > pca_dims:
        n_comp = min(pca_dims, x.shape[0] - 1, x.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        x = pca.fit_transform(x)
        info["pca_explained_variance_ratio_sum"] = float(pca.explained_variance_ratio_.sum())
        info["pca_components"] = int(n_comp)
    info["output_dim"] = int(x.shape[1])
    return x, info


def _internal_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    unique = sorted(set(int(v) for v in labels if int(v) != -1))
    metrics: dict[str, Any] = {
        "n_clusters": len(unique),
        "n_noise": int(np.sum(labels == -1)),
        "n_samples": int(len(labels)),
        "cluster_sizes": {str(c): int(np.sum(labels == c)) for c in unique},
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    # Need ≥2 clusters and ≥2 non-noise points for these scores.
    mask = labels != -1
    valid_labels = labels[mask]
    valid_x = x[mask]
    if len(set(valid_labels.tolist())) >= 2 and len(valid_labels) > len(set(valid_labels.tolist())):
        try:
            metrics["silhouette"] = safe_metric(silhouette_score(valid_x, valid_labels))
        except ValueError:
            metrics["silhouette"] = None
        try:
            metrics["davies_bouldin"] = safe_metric(davies_bouldin_score(valid_x, valid_labels))
        except ValueError:
            metrics["davies_bouldin"] = None
        try:
            metrics["calinski_harabasz"] = safe_metric(calinski_harabasz_score(valid_x, valid_labels))
        except ValueError:
            metrics["calinski_harabasz"] = None
    return metrics


def _external_metrics(labels: np.ndarray, expert_labels: list[str] | None) -> dict[str, Any]:
    if not expert_labels or len(expert_labels) != len(labels):
        return {}
    # Map expert strings to ints
    uniq = sorted(set(expert_labels))
    mapping = {name: i for i, name in enumerate(uniq)}
    y = np.asarray([mapping[v] for v in expert_labels])
    mask = labels != -1
    if mask.sum() < 2 or len(set(y[mask].tolist())) < 1:
        return {"adjusted_rand_index": None, "normalized_mutual_info": None}
    return {
        "adjusted_rand_index": safe_metric(adjusted_rand_score(y[mask], labels[mask])),
        "normalized_mutual_info": safe_metric(
            normalized_mutual_info_score(y[mask], labels[mask])
        ),
        "note": "External metrics compare clusters to expert labels exploratorily; they do not prove structural meaning.",
    }


def project_2d(x: np.ndarray, *, seed: int = 42) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0, 2))
    if x.shape[0] == 1:
        return np.zeros((1, 2))
    n_comp = min(2, x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    proj = pca.fit_transform(x)
    if proj.shape[1] == 1:
        proj = np.hstack([proj, np.zeros((proj.shape[0], 1))])
    return proj


def summarize_clusters(
    labels: np.ndarray,
    *,
    paths: list[str],
    materials: list[str | None] | None = None,
    expert_labels: list[str] | None = None,
    max_examples: int = 6,
) -> list[dict[str, Any]]:
    summaries = []
    for cluster_id in sorted(set(int(v) for v in labels)):
        idxs = [i for i, v in enumerate(labels) if int(v) == cluster_id]
        mat_counts: dict[str, int] = {}
        exp_counts: dict[str, int] = {}
        for i in idxs:
            if materials and materials[i]:
                mat_counts[materials[i]] = mat_counts.get(materials[i], 0) + 1
            if expert_labels and i < len(expert_labels):
                exp_counts[expert_labels[i]] = exp_counts.get(expert_labels[i], 0) + 1
        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "is_noise": cluster_id == -1,
                "size": len(idxs),
                "percentage": float(len(idxs) / max(len(labels), 1)),
                "material_distribution": mat_counts,
                "expert_label_distribution": exp_counts,
                "example_paths": [paths[i] for i in idxs[:max_examples]],
                "interpretation": None,
            }
        )
    return summaries


def run_kmeans(
    embeddings: np.ndarray,
    *,
    n_clusters: int = 3,
    seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    paths: list[str] | None = None,
    materials: list[str | None] | None = None,
    expert_labels: list[str] | None = None,
    l2_normalize: bool = True,
    standardize: bool = True,
    pca_dims: int | None = 50,
) -> ClusteringResult:
    notes = [
        "Clustering explores visual similarity; it does not certify material usability.",
        "Do not name clusters Good/Damaged solely from cluster IDs.",
    ]
    x, prep = prepare_features(
        embeddings,
        l2_normalize=l2_normalize,
        standardize=standardize,
        pca_dims=pca_dims,
        seed=seed,
    )
    if x.shape[0] < max(2, n_clusters):
        notes.append("Insufficient samples for requested K-Means configuration.")
        return ClusteringResult(
            algorithm="kmeans",
            labels=[-1] * int(x.shape[0]),
            metrics={"n_clusters": 0, "silhouette": None},
            projection_2d=[],
            preprocessing=prep,
            notes=notes,
        )
    model = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        init="k-means++",
    )
    labels = model.fit_predict(x)
    metrics = _internal_metrics(x, labels)
    metrics.update(_external_metrics(labels, expert_labels))
    # Stability: rerun once with different seed and measure label agreement via ARI.
    model2 = KMeans(
        n_clusters=n_clusters, random_state=seed + 1, n_init=n_init, max_iter=max_iter
    )
    labels2 = model2.fit_predict(x)
    metrics["stability_ari"] = safe_metric(adjusted_rand_score(labels, labels2))
    proj = project_2d(x, seed=seed)
    path_list = paths or [f"sample_{i}" for i in range(len(labels))]
    return ClusteringResult(
        algorithm="kmeans",
        labels=[int(v) for v in labels],
        metrics=metrics,
        projection_2d=proj.tolist(),
        cluster_summaries=summarize_clusters(
            labels, paths=path_list, materials=materials, expert_labels=expert_labels
        ),
        preprocessing=prep,
        notes=notes,
    )


def run_dbscan(
    embeddings: np.ndarray,
    *,
    eps: float = 0.8,
    min_samples: int = 5,
    metric: str = "euclidean",
    seed: int = 42,
    paths: list[str] | None = None,
    materials: list[str | None] | None = None,
    expert_labels: list[str] | None = None,
    l2_normalize: bool = True,
    standardize: bool = True,
    pca_dims: int | None = 50,
) -> ClusteringResult:
    notes = [
        "DBSCAN may label outliers as noise (-1).",
        "Clustering explores visual similarity; it does not certify material usability.",
    ]
    x, prep = prepare_features(
        embeddings,
        l2_normalize=l2_normalize,
        standardize=standardize,
        pca_dims=pca_dims,
        seed=seed,
    )
    if x.shape[0] < 2:
        return ClusteringResult(
            algorithm="dbscan",
            labels=[-1] * int(x.shape[0]),
            metrics={"n_clusters": 0, "silhouette": None},
            projection_2d=[],
            preprocessing=prep,
            notes=notes + ["Insufficient samples for DBSCAN."],
        )
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(x)
    metrics = _internal_metrics(x, labels)
    metrics.update(_external_metrics(labels, expert_labels))
    proj = project_2d(x, seed=seed)
    path_list = paths or [f"sample_{i}" for i in range(len(labels))]
    return ClusteringResult(
        algorithm="dbscan",
        labels=[int(v) for v in labels],
        metrics=metrics,
        projection_2d=proj.tolist(),
        cluster_summaries=summarize_clusters(
            labels, paths=path_list, materials=materials, expert_labels=expert_labels
        ),
        preprocessing=prep,
        notes=notes,
    )
