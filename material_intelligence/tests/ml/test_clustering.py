"""Clustering evaluation tests."""

from __future__ import annotations

import numpy as np

from material_intelligence.app.ml.evaluation.clustering import run_dbscan, run_kmeans


def test_kmeans_two_blobs_silhouette_defined():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, size=(20, 8))
    b = rng.normal(3, 0.1, size=(20, 8))
    x = np.vstack([a, b])
    result = run_kmeans(x, n_clusters=2, seed=0, pca_dims=None, standardize=False)
    assert result.metrics["n_clusters"] == 2
    assert result.metrics["silhouette"] is not None
    assert result.metrics["silhouette"] > 0.4


def test_dbscan_noise_handling():
    rng = np.random.default_rng(1)
    dense = rng.normal(0, 0.05, size=(30, 4))
    noise = rng.uniform(-5, 5, size=(5, 4))
    x = np.vstack([dense, noise])
    result = run_dbscan(x, eps=0.3, min_samples=5, pca_dims=None, standardize=False)
    assert "n_noise" in result.metrics
    assert result.metrics["silhouette"] is None or isinstance(result.metrics["silhouette"], float)


def test_insufficient_samples_undefined_not_zero():
    x = np.array([[1.0, 2.0]])
    result = run_kmeans(x, n_clusters=3, pca_dims=None)
    assert result.metrics.get("silhouette") is None
