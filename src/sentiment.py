"""Sentiment classification with VADER, custom rules, and question-aware logic."""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()


def is_improvement_question(question: str) -> bool:
    """True when the survey question asks for suggestions or criticism."""
    q = str(question).lower()

    return any(phrase in q for phrase in [
        "what could be improved",
        "could be improved",
        "improve",
        "improvement"
    ])


# Answers that mean "nothing to improve" on improvement questions
NO_IMPROVEMENT_PATTERNS = [
    "no",
    "nope",
    "none",
    "nothing",
    "don't have",
    "do not have",
    "no i don't",
    "no, i don't",
    "wouldn't add",
    "would not add",
    "no recommendations",
    "nothing to add",
    "all good",
    "all is good",
    "everything was",
    "it was amazing",
    "was amazing",
    "very smooth",
    "no ideas",
    "no ideeas",
]


# Fix phrases like "no negative aspects"
NO_NEGATIVE_PATTERNS = [
    "no negative",
    "no negative aspects",
    "nothing negative",
    "no complaints",
    "nothing bad",
    "not bad",
    "no issues",
    "no problem",
    "no problems",
    "no disadvantages",
    "nothing to improve"
]


NEGATIVE_PATTERNS = [
    "too intense",
    "not enough",
    "could have been better",
    "should have been better",
    "could be improved",
    "needs improvement",
    "was difficult",
    "was confusing",
    "was unclear",
    "not well organized",
    "poorly organized",
    "too much",
    "too many",
    "too few",
    "not clear",
    "not organized"
]

POSITIVE_PATTERNS = [
    "very supportive",
    "well organized",
    "really liked",
    "i liked",
    "i enjoyed",
    "was engaging",
    "helpful",
    "inspiring",
    "motivating",
    "comfortable atmosphere"
]


def get_sentiment(text: str, question: str | None = None):
    """Return (label, score). Rules run before VADER; question context can override."""
    t = text.lower().strip()

    # 0. On improvement questions, treat "no / nothing to improve" as positive
    if question and is_improvement_question(question):
        if t in {"no", "nope", "none", "nothing"}:
            return "positive", 0.5
        if any(pattern in t for pattern in NO_IMPROVEMENT_PATTERNS):
            return "positive", 0.5

    # 1. FIRST check "no negative" type phrases
    if any(pattern in t for pattern in NO_NEGATIVE_PATTERNS):
        return "positive", 0.6

    # 2. Then check negative rules
    for pattern in NEGATIVE_PATTERNS:
        if pattern in t:
            return "negative", -0.6

    # 3. Then check positive rules
    for pattern in POSITIVE_PATTERNS:
        if pattern in t:
            return "positive", 0.6

    # 4. Finally use VADER
    score = sia.polarity_scores(text)["compound"]

    if score >= 0.2:
        label = "positive"
    elif score <= -0.2:
        label = "negative"
    else:
        label = "neutral"

    return label, score