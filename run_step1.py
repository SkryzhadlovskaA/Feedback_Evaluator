from src.preprocess import load_data, preprocess_dataframe

INPUT_PATH = "data/raw/feedback_test_dataset.csv"
OUTPUT_PATH = "data/processed/sentences.csv"

def main():
    df = load_data(INPUT_PATH)
    print("Loaded rows:", len(df))
    print(df.head(), "\n")

    processed_df = preprocess_dataframe(df)

    print("Sentence-level rows:", len(processed_df))
    print(processed_df[["project_id", "response_id", "question", "sentence"]].head(10))

    processed_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved processed file to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()