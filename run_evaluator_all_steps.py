import json
import pandas as pd

from src.preprocess import load_data, preprocess_dataframe
from src.sentiment import get_sentiment
from src.learning_filter import is_learning_sentence
from src.embeddings import Embedder
from src.learningoutcomes_detector import OutcomeMatcher
from src.clustering import run_kmeans, to_matrix
from src.quotes import get_representative_quotes


INPUT_PATH = "data/raw/feedback_test_dataset.csv"

SENTENCES_OUTPUT = "data/processed/sentences.csv"
SENTIMENT_OUTPUT = "data/processed/sentences_with_sentiment.csv"
LEARNING_OUTPUT = "data/processed/sentences_with_learning_flag.csv"
OUTCOME_OUTPUT = "data/processed/learning_outcomes_debug.csv"
POS_CLUSTERS_OUTPUT = "data/processed/positive_clusters.csv"
NEG_CLUSTERS_OUTPUT = "data/processed/negative_clusters.csv"
FINAL_OUTPUT = "outputs/final_structured_output.json"


POSITIVE_CLUSTER_LABELS = {
    0: "Learning outcomes and personal development",
    1: "Mixed logistical experiences",
    2: "Positive learning atmosphere",
    3: "Skills development and future motivation"
}

NEGATIVE_CLUSTER_LABELS = {
    0: "Organization and communication issues",
    1: "Food and meal options",
    2: "False negative / should be reviewed",
    3: "Schedule intensity and session clarity"
}


def is_improvement_question(question: str) -> bool:
    q = str(question).lower()
    return any(phrase in q for phrase in [
        "what could be improved",
        "could be improved",
        "improve",
        "improvement"
    ])


def is_learning_question(question: str) -> bool:
    q = str(question).lower()
    return any(phrase in q for phrase in [
        "what did you learn",
        "learn",
        "learning",
        "gained",
        "take away",
        "takeaway"
    ])


def add_learning_flag(df):
    df = df.copy()

    df["is_learning"] = df.apply(
        lambda row: (
            is_learning_question(row.get("question", ""))
            or (
                is_learning_sentence(row["sentence"])
                and not is_improvement_question(row.get("question", ""))
            )
        ),
        axis=1
    )

    return df


def add_outcome_detection(df, embedder):
    learning_df = df[df["is_learning"] == True].copy()

    if learning_df.empty:
        return learning_df

    matcher = OutcomeMatcher(embedder)

    learning_df["embedding"] = list(
        embedder.encode(learning_df["sentence"].tolist())
    )

    single_results = learning_df["embedding"].apply(matcher.match_single)

    learning_df["outcome_key"] = single_results.apply(lambda x: x[0])
    learning_df["outcome_label"] = single_results.apply(lambda x: x[1])
    learning_df["outcome_score"] = single_results.apply(lambda x: x[2])

    learning_df["multi_matches"] = learning_df["embedding"].apply(
        lambda emb: json.dumps(matcher.match_multi(emb), ensure_ascii=False)
    )

    return learning_df


def cluster_sentiment_group(df, sentiment_label, n_clusters=4):
    subset = df[df["sentiment"] == sentiment_label].copy()

    if len(subset) < n_clusters:
        print(f"Not enough {sentiment_label} sentences for clustering.")
        return subset

    embedder = Embedder()

    subset["embedding"] = list(
        embedder.encode(subset["sentence"].tolist())
    )

    labels, _ = run_kmeans(
        to_matrix(subset["embedding"]),
        n_clusters=n_clusters
    )

    subset["cluster"] = labels

    return subset


def cluster_section(df, label_dict, top_n=3):
    results = []

    if df.empty or "cluster" not in df.columns:
        return results

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_id = int(cluster_id)

        quotes_df = get_representative_quotes(df, cluster_id, top_n=top_n)

        results.append({
            "cluster_id": cluster_id,
            "label": label_dict.get(cluster_id, "Unlabeled theme"),
            "count": int((df["cluster"] == cluster_id).sum()),
            "representative_quotes": quotes_df["sentence"].tolist()
        })

    return results


def outcome_section(df, top_n=3):
    results = []

    if df.empty or "outcome_label" not in df.columns:
        return results

    valid_df = df[df["outcome_label"].notna()].copy()

    for label in sorted(valid_df["outcome_label"].unique()):
        subset = valid_df[valid_df["outcome_label"] == label].copy()
        subset = subset.sort_values("outcome_score", ascending=False)

        results.append({
            "label": label,
            "count": int(len(subset)),
            "representative_quotes": subset["sentence"].head(top_n).tolist()
        })

    return results


def main():
    print("Step 1: preprocessing...")
    df = load_data(INPUT_PATH)
    df = preprocess_dataframe(df)
    df.to_csv(SENTENCES_OUTPUT, index=False)

    print("Step 2: sentiment analysis...")
    df[["sentiment", "sentiment_score"]] = df["sentence"].apply(
        lambda s: pd.Series(get_sentiment(s))
    )
    df.to_csv(SENTIMENT_OUTPUT, index=False)

    print("Step 3: learning sentence detection...")
    df = add_learning_flag(df)
    df.to_csv(LEARNING_OUTPUT, index=False)

    print("Loading embedding model...")
    embedder = Embedder()

    print("Step 4: Erasmus outcome detection...")
    learning_df = add_outcome_detection(df, embedder)
    learning_df.to_csv(OUTCOME_OUTPUT, index=False)

    print("Step 5: clustering strengths and improvements...")
    pos_df = cluster_sentiment_group(df, "positive", n_clusters=4)
    neg_df = cluster_sentiment_group(df, "negative", n_clusters=4)

    pos_df.to_csv(POS_CLUSTERS_OUTPUT, index=False)
    neg_df.to_csv(NEG_CLUSTERS_OUTPUT, index=False)

    print("Step 6–7: building final structured output...")
    final_output = {
        "strengths": cluster_section(pos_df, POSITIVE_CLUSTER_LABELS),
        "improvements": cluster_section(neg_df, NEGATIVE_CLUSTER_LABELS),
        "learning_outcomes": outcome_section(learning_df)
    }

    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Final output saved to: {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()