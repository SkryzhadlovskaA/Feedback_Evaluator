"""Sentence embeddings for outcome matching and clustering."""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Thin wrapper around sentence transformer for consistent encoding."""

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts):
        """Encode a string or list of strings into embedding vectors."""
        return self.model.encode(texts, show_progress_bar=False)