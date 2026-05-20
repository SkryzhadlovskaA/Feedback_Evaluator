import numpy as np
from sklearn.cluster import KMeans


def to_matrix(embedding_series):
    return np.vstack(embedding_series.values)


def run_kmeans(embeddings, n_clusters=4):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(embeddings)

    return labels, model
