"""Match learning sentences to Erasmus+, project topic, or Other via cosine similarity."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.taxonomy import (
    ERASMUS_OUTCOMES,
    PROJECT_TOPICS,
    OTHER_OUTCOME_LABEL,
)

SINGLE_LABEL_THRESHOLD = 0.40   # minimum similarity to assign a category


def _category_text(item: dict) -> str:
    """Combine description + examples into one text for embedding."""
    return item["description"] + " " + " ".join(item["examples"])


class OutcomeMatcher:
    """Compare sentence embeddings against Erasmus + project topic categories."""

    def __init__(self, embedder, project_id: str | None = None):
        self.embedder = embedder
        self.project_id = project_id

        self.categories = dict(ERASMUS_OUTCOMES)

        if project_id and project_id in PROJECT_TOPICS:
            self.categories["project_topic"] = PROJECT_TOPICS[project_id]
        elif project_id:
            print(
                f"Warning: no PROJECT_TOPICS entry for project_id={project_id!r}. "
                "Matching uses Erasmus outcomes only (+ Other fallback)."
            )

        self.keys = list(self.categories.keys())
        self.labels = [self.categories[k]["label"] for k in self.keys]

        category_texts = [_category_text(self.categories[k]) for k in self.keys]
        # Pre-compute category embeddings once per project run
        self.category_embeddings = self.embedder.encode(category_texts)

    def match_single(self, sentence_embedding):
        """Return (key, label, score) for the best-matching category, or Other."""
        sims = cosine_similarity([sentence_embedding], self.category_embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= SINGLE_LABEL_THRESHOLD:
            return self.keys[best_idx], self.labels[best_idx], best_score

        return "other", OTHER_OUTCOME_LABEL, best_score
