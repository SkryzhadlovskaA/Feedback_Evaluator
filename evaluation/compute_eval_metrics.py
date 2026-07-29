"""
Compare pipeline predictions against manual gold labels.

Requires gold_sentiment and gold_is_learning filled in the CSV.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment import get_sentiment
from src.taxonomy import (
    ERASMUS_OUTCOMES,
    PROJECT_TOPICS,
    OTHER_OUTCOME_LABEL,
    get_outcome_labels_for_project,
)

VALID_SENTIMENTS = {"positive", "negative", "neutral"}
VALID_OUTCOMES = set(get_outcome_labels_for_project(None))
for topic in PROJECT_TOPICS.values():
    VALID_OUTCOMES.add(topic["label"])
VALID_OUTCOMES.add(OTHER_OUTCOME_LABEL)


def accuracy(y_true, y_pred) -> float:
    pairs = list(zip(y_true, y_pred))
    if not pairs:
        return 0.0
    return sum(a == b for a, b in pairs) / len(pairs)


def parse_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_labeled(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df["gold_sentiment"].isna().all() or (df["gold_sentiment"] == "").all():
        raise SystemExit(
            f"No gold labels found in {path.name}. "
            "Fill gold_sentiment and gold_is_learning first."
        )

    labeled = df[df["gold_sentiment"].notna() & (df["gold_sentiment"] != "")].copy()
    labeled["gold_sentiment"] = labeled["gold_sentiment"].str.strip().str.lower()
    labeled["gold_is_learning"] = labeled["gold_is_learning"].map(parse_bool)
    labeled["pred_is_learning"] = labeled["pred_is_learning"].map(parse_bool)

    invalid_sentiment = labeled[
        ~labeled["gold_sentiment"].isin(VALID_SENTIMENTS)
    ]
    if not invalid_sentiment.empty:
        ids = invalid_sentiment["eval_id"].tolist() if "eval_id" in invalid_sentiment else "?"
        raise SystemExit(
            f"Invalid gold_sentiment values in rows {ids}. "
            f"Use: {', '.join(sorted(VALID_SENTIMENTS))}"
        )

    return labeled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="eval_full.csv",
        help="CSV file inside evaluation/ folder (default: eval_full.csv)",
    )
    args = parser.parse_args()

    sample_path = PROJECT_ROOT / "evaluation" / args.file
    results_path = PROJECT_ROOT / "evaluation" / f"{Path(args.file).stem}_results.json"

    labeled = load_labeled(sample_path)

    baseline = labeled.apply(
        lambda row: get_sentiment(row["sentence"], question=None)[0],
        axis=1,
    )
    current = labeled.apply(
        lambda row: get_sentiment(row["sentence"], row.get("question"))[0],
        axis=1,
    )

    gold_sentiment = labeled["gold_sentiment"]
    gold_learning = labeled["gold_is_learning"]

    results = {
        "source_file": args.file,
        "n_labeled_sentences": int(len(labeled)),
        "sentiment_accuracy_current": round(accuracy(gold_sentiment, current), 3),
        "sentiment_accuracy_baseline_no_question": round(
            accuracy(gold_sentiment, baseline), 3
        ),
        "learning_detection_accuracy": round(
            accuracy(gold_learning, labeled["pred_is_learning"]), 3
        ),
    }

    if "dataset" in labeled.columns:
        by_dataset = {}
        for dataset in sorted(labeled["dataset"].unique()):
            subset = labeled[labeled["dataset"] == dataset]
            by_dataset[dataset] = {
                "n_sentences": int(len(subset)),
                "sentiment_accuracy_current": round(
                    accuracy(subset["gold_sentiment"], subset.apply(
                        lambda row: get_sentiment(row["sentence"], row.get("question"))[0],
                        axis=1,
                    )),
                    3,
                ),
                "learning_detection_accuracy": round(
                    accuracy(
                        subset["gold_is_learning"],
                        subset["pred_is_learning"],
                    ),
                    3,
                ),
            }
        results["by_dataset"] = by_dataset

    results["sentiment_errors_current"] = labeled[gold_sentiment != current][
        ["eval_id", "dataset", "sentence", "question", "pred_sentiment", "gold_sentiment"]
        if "eval_id" in labeled.columns
        else ["dataset", "sentence", "question", "pred_sentiment", "gold_sentiment"]
    ].to_dict(orient="records")

    outcome_labeled = labeled[
        labeled["gold_is_learning"]
        & labeled["gold_outcome"].notna()
        & (labeled["gold_outcome"] != "")
    ].copy()

    if not outcome_labeled.empty:
        invalid_outcomes = outcome_labeled[
            ~outcome_labeled["gold_outcome"].str.strip().isin(VALID_OUTCOMES)
        ]
        if not invalid_outcomes.empty:
            ids = invalid_outcomes["eval_id"].tolist() if "eval_id" in invalid_outcomes else "?"
            raise SystemExit(
                f"Invalid gold_outcome in rows {ids}. "
                f"See evaluation/outcome_labels.txt"
            )

        gold_outcome = outcome_labeled["gold_outcome"].str.strip()
        pred_outcome = outcome_labeled["pred_outcome"].fillna("").str.strip()

        results["outcome_accuracy"] = round(accuracy(gold_outcome, pred_outcome), 3)
        results["n_outcome_labeled"] = int(len(outcome_labeled))
        results["outcome_coverage"] = round(
            (pred_outcome != "").sum() / len(outcome_labeled), 3
        )
        results["outcome_errors"] = outcome_labeled[gold_outcome != pred_outcome][
            ["eval_id", "sentence", "pred_outcome", "gold_outcome"]
            if "eval_id" in outcome_labeled.columns
            else ["sentence", "pred_outcome", "gold_outcome"]
        ].to_dict(orient="records")

    learning_only = labeled[~labeled["gold_is_learning"]]
    if not learning_only.empty:
        false_learning = learning_only[learning_only["pred_is_learning"]]
        results["learning_false_positives"] = int(len(false_learning))

    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved metrics to {results_path}")


if __name__ == "__main__":
    main()
