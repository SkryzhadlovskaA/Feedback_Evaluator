import pandas as pd
from src.sentiment import get_sentiment

INPUT_PATH = "data/processed/sentences.csv"
OUTPUT_PATH = "data/processed/sentences_with_sentiment.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    df[["sentiment", "sentiment_score"]] = df.apply(
        lambda row: pd.Series(
            get_sentiment(row["sentence"], row.get("question"))
        ),
        axis=1
    )

    print("Total sentences:", len(df))
    print("\nSentiment counts:")
    print(df["sentiment"].value_counts(), "\n")

    print(df[["sentence", "sentiment", "sentiment_score"]].head(10))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()