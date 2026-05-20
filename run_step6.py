import json
import pandas as pd

from src.quotes import get_representative_quotes

POS_INPUT = "data/processed/positive_clusters.csv"
NEG_INPUT = "data/processed/negative_clusters.csv"

OUTPUT_JSON = "outputs/representative_quotes.json"


# You can manually edit these labels based on your cluster inspection
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


def extract_quotes(df, label_dict, top_n=3):
    results = {}

    for cluster_id in sorted(df["cluster"].unique()):
        cluster_id = int(cluster_id)

        quotes_df = get_representative_quotes(
            df=df,
            cluster_id=cluster_id,
            top_n=top_n
        )

        results[f"cluster_{cluster_id}"] = {
            "label": label_dict.get(cluster_id, "Unlabeled theme"),
            "count": int((df["cluster"] == cluster_id).sum()),
            "quotes": [
                {
                    "sentence": row["sentence"],
                    "similarity": round(float(row["similarity_to_cluster_center"]), 3)
                }
                for _, row in quotes_df.iterrows()
            ]
        }

    return results


def main():
    pos_df = pd.read_csv(POS_INPUT)
    neg_df = pd.read_csv(NEG_INPUT)

    results = {
        "strengths": extract_quotes(
            pos_df,
            POSITIVE_CLUSTER_LABELS,
            top_n=3
        ),
        "improvements": extract_quotes(
            neg_df,
            NEGATIVE_CLUSTER_LABELS,
            top_n=3
        )
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved representative quotes to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()