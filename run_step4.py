import pandas as pd

from src.embeddings import Embedder
from src.learningoutcomes_detector import OutcomeMatcher

INPUT_PATH = "data/processed/sentences_with_learning_flag.csv"
OUTPUT_PATH = "data/processed/learning_outcomes_debug.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    learning_df = df[df["is_learning"] == True].copy()

    print("Total sentences:", len(df))
    print("Learning sentences:", len(learning_df))

    if learning_df.empty:
        print("No learning sentences found. Stop here and check Step 3.")
        return

    embedder = Embedder()

    project_id = None
    if "project_id" in df.columns:
        ids = df["project_id"].dropna().astype(str).str.strip().unique()
        if len(ids) > 0:
            project_id = ids[0]

    matcher = OutcomeMatcher(embedder, project_id=project_id)

    sentence_embeddings = embedder.encode(learning_df["sentence"].tolist())
    learning_df["embedding"] = list(sentence_embeddings)

    single_results = learning_df["embedding"].apply(matcher.match_single)

    learning_df["outcome_key"] = single_results.apply(lambda x: x[0])
    learning_df["outcome_label"] = single_results.apply(lambda x: x[1])
    learning_df["outcome_score"] = single_results.apply(lambda x: x[2])

    print("\nOutcome counts:")
    print(learning_df["outcome_label"].value_counts(dropna=False), "\n")

    print(
        learning_df[
            ["sentence", "outcome_label", "outcome_score"]
        ].head(20)
    )

    learning_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved outcome debug file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()