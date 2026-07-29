"""
Export ALL sentences from real datasets for manual evaluation.

"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.taxonomy import get_outcome_labels_for_project
OUTPUT = PROJECT_ROOT / "evaluation" / "eval_full.csv"
LABELS_FILE = PROJECT_ROOT / "evaluation" / "outcome_labels.txt"

DATASETS = [
    "project1_real_data",
    "project2_real_data",
]


def load_dataset(name: str) -> pd.DataFrame:
    flags_path = PROJECT_ROOT / "data/processed" / f"{name}_learning_flags.csv"
    outcomes_path = PROJECT_ROOT / "data/processed" / f"{name}_learning_outcomes.csv"

    df = pd.read_csv(flags_path)
    df["dataset"] = name

    outcomes = pd.read_csv(
        outcomes_path,
        usecols=["sentence", "outcome_label", "outcome_score"],
    )
    df = df.merge(outcomes, on="sentence", how="left")

    return df


def main():
    frames = [load_dataset(name) for name in DATASETS]
    df = pd.concat(frames, ignore_index=True)

    df.insert(0, "eval_id", range(1, len(df) + 1))

    export = pd.DataFrame({
        "eval_id": df["eval_id"],
        "dataset": df["dataset"],
        "response_id": df["response_id"],
        "question": df["question"],
        "sentence": df["sentence"],
        "pred_sentiment": df["sentiment"],
        "pred_sentiment_score": df["sentiment_score"],
        "pred_is_learning": df["is_learning"],
        "pred_outcome": df["outcome_label"],
        "pred_outcome_score": df["outcome_score"],
        "gold_sentiment": "",
        "gold_is_learning": "",
        "gold_outcome": "",
        "reviewer_notes": "",
    })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    labels = sorted(set(get_outcome_labels_for_project(None)))
    LABELS_FILE.write_text("\n".join(labels) + "\n", encoding="utf-8")

    print(f"Exported {len(export)} sentences to {OUTPUT}")
    print(f"  project1_real_data: {(export['dataset'] == 'project1_real_data').sum()}")
    print(f"  project2_real_data: {(export['dataset'] == 'project2_real_data').sum()}")
    print(f"Valid outcome labels saved to {LABELS_FILE}")
    print("\nNext steps:")
    print("  1. Open eval_full.csv in Excel")
    print("  2. Fill gold_sentiment, gold_is_learning, gold_outcome")
    print("  3. Run: python evaluation/compute_eval_metrics.py --file eval_full.csv")


if __name__ == "__main__":
    main()
