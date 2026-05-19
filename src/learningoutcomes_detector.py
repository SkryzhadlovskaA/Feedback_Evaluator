import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.taxonomy import ERASMUS_OUTCOMES

SINGLE_LABEL_THRESHOLD = 0.40
MULTI_LABEL_THRESHOLD = 0.38


class OutcomeMatcher:
    def __init__(self, embedder):
        self.embedder = embedder

        self.keys = list(ERASMUS_OUTCOMES.keys())
        self.labels = [ERASMUS_OUTCOMES[k]["label"] for k in self.keys]

        category_texts = []
        for key in self.keys:
            item = ERASMUS_OUTCOMES[key]
            combined_text = item["description"] + " " + " ".join(item["examples"])
            category_texts.append(combined_text)

        self.category_embeddings = self.embedder.encode(category_texts)

    def match_single(self, sentence_embedding):
        sims = cosine_similarity([sentence_embedding], self.category_embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= SINGLE_LABEL_THRESHOLD:
            return self.keys[best_idx], self.labels[best_idx], best_score

        return None, None, best_score

    def match_multi(self, sentence_embedding):
        sims = cosine_similarity([sentence_embedding], self.category_embeddings)[0]

        matches = []
        for i, score in enumerate(sims):
            if score >= MULTI_LABEL_THRESHOLD:
                matches.append({
                    "outcome_key": self.keys[i],
                    "outcome_label": self.labels[i],
                    "score": float(score)
                })

        matches = sorted(matches, key=lambda x: x["score"], reverse=True)
        return matches