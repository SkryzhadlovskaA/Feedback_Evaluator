"""End-to-end feedback analysis pipeline (CLI and Streamlit entry point)."""

import json
from pathlib import Path

import pandas as pd

from src.preprocess import load_data, preprocess_dataframe
from src.sentiment import get_sentiment
from src.learning_filter import is_learning_sentence
from src.embeddings import Embedder
from src.learningoutcomes_detector import OutcomeMatcher
from src.clustering import run_kmeans, to_matrix, choose_n_clusters
from src.quotes import get_representative_quotes
from src.taxonomy import PROJECT_TOPICS


def _load_cluster_labels(path="outputs/cluster_labels.json"):
    """Load human-readable theme names for positive/negative cluster IDs."""
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return (
        {int(k): v for k, v in labels["positive"].items()},
        {int(k): v for k, v in labels["negative"].items()},
    )


def is_improvement_question(question: str) -> bool:
    q = str(question).lower()
    return any(phrase in q for phrase in [
        "what could be improved",
        "could be improved",
        "improve",
        "improvement",
    ])


def is_learning_question(question: str) -> bool:
    q = str(question).lower()
    return any(phrase in q for phrase in [
        "what did you learn",
        "learn",
        "learning",
        "gained",
        "take away",
        "takeaway",
    ])


def add_learning_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Mark learning sentences: learning questions OR keyword cues (not on improve Qs)."""
    df = df.copy()
    df["is_learning"] = df.apply(
        lambda row: (
            is_learning_question(row.get("question", ""))
            or (
                is_learning_sentence(row["sentence"])
                and not is_improvement_question(row.get("question", ""))
            )
        ),
        axis=1,
    )
    return df


def get_project_id(df: pd.DataFrame) -> str | None:
    """Read project_id from CSV; used to select PROJECT_TOPICS entry."""
    if "project_id" not in df.columns:
        return None
    ids = df["project_id"].dropna().astype(str).str.strip().unique()
    if len(ids) == 0:
        return None
    return ids[0]


def add_outcome_detection(
    df: pd.DataFrame,
    embedder: Embedder,
    project_id: str | None = None,
) -> pd.DataFrame:
    """Embed and classify learning sentences only."""
    learning_df = df[df["is_learning"] == True].copy()
    if learning_df.empty:
        return learning_df

    matcher = OutcomeMatcher(embedder, project_id=project_id)
    learning_df["embedding"] = list(embedder.encode(learning_df["sentence"].tolist()))
    single_results = learning_df["embedding"].apply(matcher.match_single)
    learning_df["outcome_key"] = single_results.apply(lambda x: x[0])
    learning_df["outcome_label"] = single_results.apply(lambda x: x[1])
    learning_df["outcome_score"] = single_results.apply(lambda x: x[2])
    return learning_df


def cluster_sentiment_group(
    df: pd.DataFrame,
    sentiment_label: str,
    embedder: Embedder,
) -> pd.DataFrame:
    """Cluster positive or negative sentences separately by embedding similarity."""
    subset = df[df["sentiment"] == sentiment_label].copy()
    if len(subset) < 2:
        if not subset.empty:
            subset["cluster"] = 0
        return subset

    n_clusters = choose_n_clusters(len(subset))
    subset["embedding"] = list(embedder.encode(subset["sentence"].tolist()))
    if n_clusters == 1:
        subset["cluster"] = 0
        return subset

    labels, _ = run_kmeans(to_matrix(subset["embedding"]), n_clusters=n_clusters)
    subset["cluster"] = labels
    return subset


def cluster_section(df: pd.DataFrame, label_dict: dict, top_n: int = 3) -> list:
    """Build strengths/improvements JSON section from clustered sentences."""
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
            "representative_quotes": quotes_df["sentence"].tolist(),
        })
    return results


def outcome_section(df: pd.DataFrame, top_n: int = 3) -> list:
    """Build learning_outcomes JSON section grouped by outcome label."""
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
            "representative_quotes": subset["sentence"].head(top_n).tolist(),
        })
    return results


def run_pipeline(input_path: str, on_step=None) -> dict:
    """Run the full feedback analysis pipeline. Returns the final JSON output."""

    def step(msg: str):
        if on_step:
            on_step(msg)

    input_path = str(input_path)
    project_name = Path(input_path).stem.replace(" ", "_")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    positive_labels, negative_labels = _load_cluster_labels()

    step("Loading and preprocessing data...")
    df = load_data(input_path)
    df = preprocess_dataframe(df)
    df.to_csv(f"data/processed/{project_name}_sentences.csv", index=False)

    step("Analyzing sentiment...")
    df[["sentiment", "sentiment_score"]] = df.apply(
        lambda row: pd.Series(get_sentiment(row["sentence"], row.get("question"))),
        axis=1,
    )
    df.to_csv(f"data/processed/{project_name}_sentiment.csv", index=False)

    step("Detecting learning sentences...")
    df = add_learning_flag(df)
    df.to_csv(f"data/processed/{project_name}_learning_flags.csv", index=False)

    step("Loading embedding model (first run may take a minute)...")
    embedder = Embedder()

    step("Matching learning outcomes...")
    project_id = get_project_id(df)
    learning_df = add_outcome_detection(df, embedder, project_id=project_id)
    learning_df.to_csv(f"data/processed/{project_name}_learning_outcomes.csv", index=False)

    step("Clustering strengths and improvements...")
    pos_df = cluster_sentiment_group(df, "positive", embedder)
    neg_df = cluster_sentiment_group(df, "negative", embedder)
    pos_df.to_csv(f"data/processed/{project_name}_positive_clusters.csv", index=False)
    neg_df.to_csv(f"data/processed/{project_name}_negative_clusters.csv", index=False)

    step("Building final report...")
    final_output = {
        "project_name": project_name,
        "input_file": input_path,
        "project_id": project_id,
        "project_topic_category": (
            PROJECT_TOPICS[project_id]["label"]
            if project_id and project_id in PROJECT_TOPICS
            else None
        ),
        "strengths": cluster_section(pos_df, positive_labels),
        "improvements": cluster_section(neg_df, negative_labels),
        "learning_outcomes": outcome_section(learning_df),
    }

    final_path = f"outputs/{project_name}_final_output.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    final_output["_output_path"] = final_path
    return final_output
