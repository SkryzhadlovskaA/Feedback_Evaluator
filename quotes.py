import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def parse_embedding(value):
    """
    Converts stored embedding text back into a NumPy array.
    This is needed if embeddings were saved to CSV.
    """
    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, list):
        return np.array(value)

    value = str(value).strip()

    # Remove brackets and split numbers
    value = value.replace("[", "").replace("]", "")
    numbers = [float(x) for x in value.split()]

    return np.array(numbers)


def get_representative_quotes(df, cluster_id, top_n=3):
    subset = df[df["cluster"] == cluster_id].copy()

    if subset.empty:
        return []

    subset["embedding_array"] = subset["embedding"].apply(parse_embedding)

    embeddings = np.vstack(subset["embedding_array"].values)

    # Cluster center = average embedding of all sentences in this cluster
    center = embeddings.mean(axis=0)

    # Compare every sentence to the center
    similarities = cosine_similarity(embeddings, [center]).flatten()

    subset["similarity_to_cluster_center"] = similarities

    subset = subset.sort_values(
        "similarity_to_cluster_center",
        ascending=False
    )

    return subset[["sentence", "similarity_to_cluster_center"]].head(top_n)