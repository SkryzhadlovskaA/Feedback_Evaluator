import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()



##needed improvemnts, so added some domain rules:
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


def get_sentiment(text: str):
    t = text.lower().strip()

    for pattern in NEGATIVE_PATTERNS:
        if pattern in t:
            return "negative", -0.6

    for pattern in POSITIVE_PATTERNS:
        if pattern in t:
            return "positive", 0.6

    score = sia.polarity_scores(text)["compound"]

    if score >= 0.2:
        label = "positive"
    elif score <= -0.2:
        label = "negative"
    else:
        label = "neutral"

    return label, score