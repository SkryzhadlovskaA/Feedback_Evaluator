"""K-Means clustering of sentence embeddings."""

import numpy as np
from sklearn.cluster import KMeans


def to_matrix(embedding_series):
    return np.vstack(embedding_series.values)


def choose_n_clusters(n_samples: int, max_clusters: int = 4) -> int:
    """Pick a cluster count that fits the amount of feedback available."""
    if n_samples < 6:
        return 1
    if n_samples < 12:
        return 2
    if n_samples < 20:
        return 3
    return max_clusters


def run_kmeans(embeddings, n_clusters=3):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(embeddings)

    return labels, model