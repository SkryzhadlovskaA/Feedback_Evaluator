import pandas as pd

from src.embeddings import Embedder
from src.clustering import run_kmeans, to_matrix, choose_n_clusters

INPUT_PATH = "data/processed/sentences_with_sentiment.csv"

POS_OUTPUT = "data/processed/positive_clusters.csv"
NEG_OUTPUT = "data/processed/negative_clusters.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Keep only clear sentiment
    pos_df = df[df["sentiment"] == "positive"].copy()
    neg_df = df[df["sentiment"] == "negative"].copy()

    print("Positive sentences:", len(pos_df))
    print("Negative sentences:", len(neg_df))

    embedder = Embedder()

    # Generate embeddings
    if not pos_df.empty:
        pos_df["embedding"] = list(
            embedder.encode(pos_df["sentence"].tolist())
        )

    if not neg_df.empty:
        neg_df["embedding"] = list(
            embedder.encode(neg_df["sentence"].tolist())
        )

    n_pos = choose_n_clusters(len(pos_df))
    n_neg = choose_n_clusters(len(neg_df))

    if n_pos == 1 or pos_df.empty:
        if not pos_df.empty:
            pos_df["cluster"] = 0
    else:
        pos_labels, _ = run_kmeans(to_matrix(pos_df["embedding"]), n_clusters=n_pos)
        pos_df["cluster"] = pos_labels

    if n_neg == 1 or neg_df.empty:
        if not neg_df.empty:
            neg_df["cluster"] = 0
    else:
        neg_labels, _ = run_kmeans(to_matrix(neg_df["embedding"]), n_clusters=n_neg)
        neg_df["cluster"] = neg_labels

    print("\nPositive cluster counts:")
    print(pos_df["cluster"].value_counts())

    print("\nNegative cluster counts:")
    print(neg_df["cluster"].value_counts())

    pos_df.to_csv(POS_OUTPUT, index=False)
    neg_df.to_csv(NEG_OUTPUT, index=False)

    print("\nSaved:")
    print(POS_OUTPUT)
    print(NEG_OUTPUT)


if __name__ == "__main__":
    main()
