import pandas as pd
from src.learning_filter import is_learning_sentence

INPUT_PATH = "data/processed/sentences_with_sentiment.csv"
OUTPUT_PATH = "data/processed/sentences_with_learning_flag.csv"


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


def main():
    df = pd.read_csv(INPUT_PATH)

    df["is_learning"] = df.apply(
        lambda row: (
            is_learning_question(row["question"])
            or (
                is_learning_sentence(row["sentence"])
                and not is_improvement_question(row["question"])
            )
        ),
        axis=1
    )

    print("\nLearning sentence counts:")
    print(df["is_learning"].value_counts(), "\n")

    print(df[["question", "sentence", "is_learning"]].head(20))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
