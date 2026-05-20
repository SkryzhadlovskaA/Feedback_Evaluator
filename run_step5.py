import pandas as pd

from src.embeddings import Embedder
from src.clustering import run_kmeans, to_matrix

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
    pos_df["embedding"] = list(
        embedder.encode(pos_df["sentence"].tolist())
    )

    neg_df["embedding"] = list(
        embedder.encode(neg_df["sentence"].tolist())
    )

    # Convert to matrices
    pos_matrix = to_matrix(pos_df["embedding"])
    neg_matrix = to_matrix(neg_df["embedding"])

    # Run clustering
    pos_labels, pos_model = run_kmeans(pos_matrix, n_clusters=4)
    neg_labels, neg_model = run_kmeans(neg_matrix, n_clusters=4)

    pos_df["cluster"] = pos_labels
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