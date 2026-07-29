import json
import pandas as pd
from src.quotes import get_representative_quotes

POS_INPUT = "data/processed/positive_clusters.csv"
NEG_INPUT = "data/processed/negative_clusters.csv"
OUTCOME_INPUT = "data/processed/learning_outcomes_debug.csv"

OUTPUT_JSON = "outputs/final_structured_output.json"


def load_cluster_labels(path="outputs/cluster_labels.json"):
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    positive_labels = {
        int(k): v for k, v in labels.get("positive", {}).items()
    }

    negative_labels = {
        int(k): v for k, v in labels.get("negative", {}).items()
    }

    return positive_labels, negative_labels


def cluster_section(df, label_dict, top_n=3):
    results = []

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
    pos_df = pd.read_csv(POS_INPUT)
    neg_df = pd.read_csv(NEG_INPUT)
    outcome_df = pd.read_csv(OUTCOME_INPUT)

    positive_labels, negative_labels = load_cluster_labels()

    final_output = {
        "strengths": cluster_section(pos_df, positive_labels),
        "improvements": cluster_section(neg_df, negative_labels),
        "learning_outcomes": outcome_section(outcome_df)
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(json.dumps(final_output, indent=2, ensure_ascii=False))
    print(f"\nSaved final output to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
