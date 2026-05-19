import re
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""]

    df["sentence_list"] = df["text"].apply(split_sentences)

    df = df.explode("sentence_list").rename(columns={"sentence_list": "sentence"})
    df = df[df["sentence"].notna()]

    df["sentence"] = df["sentence"].str.strip()

    return df.reset_index(drop=True)