"""Load and clean feedback CSV files, then split responses into sentences."""

import re
import pandas as pd
import spacy
import csv

nlp = spacy.load("en_core_web_sm")


def load_data(path: str) -> pd.DataFrame:
    try:
        # Try normal comma CSV first
        df = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.ParserError:
        # Try semicolon CSV
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", engine="python")

    print("Detected columns:", df.columns.tolist())
    print(df.head())

    required_columns = {"project_id", "response_id", "question", "text"}

    if not required_columns.issubset(set(df.columns)):
        raise ValueError(
            f"CSV must contain columns: {required_columns}. "
            f"Detected columns: {df.columns.tolist()}"
        )

    return df


def clean_text(text: str) -> str:
    """Normalise whitespace; return empty string for missing values."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_sentences(text: str):
    """Use spaCy to split one response into individual sentences."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence (exploded from multi-sentence responses)."""
    df = df.copy()

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""]

    df["sentence_list"] = df["text"].apply(split_sentences)

    df = df.explode("sentence_list").rename(columns={"sentence_list": "sentence"})
    df = df[df["sentence"].notna()]

    df["sentence"] = df["sentence"].str.strip()

    return df.reset_index(drop=True)