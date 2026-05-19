from typing import List

LEARNING_CUES: List[str] = [
    "learned",
    "learnt",
    "i learned",
    "i have learned",
    "improved",
    "gained",
    "developed",
    "became",
    "i feel more",
    "i am more",
    "i can now",
    "this helped me",
    "this allowed me",
    "i realized",
    "i understood",
    "i now",
    "i am now",
    "i feel",
    "i feel more",
    "i feel confident",
    "i am confident",
    "i am more confident",
    "i am more open",
    "i became more",
    "i learned how to"
]

NON_LEARNING_HINTS: List[str] = [
    "food",
    "hotel",
    "room",
    "schedule",
    "transport",
    "bus",
    "accommodation",
    "organizers",
    "trainer",
    "facilitator",
    "location",
    "venue"
]


def is_learning_sentence(text: str) -> bool:
    t = text.lower()

    # Rule 1: if it's clearly about logistics → not learning
    if any(word in t for word in NON_LEARNING_HINTS):
        return False

    # Rule 2: if it contains learning cues → likely learning
    if any(cue in t for cue in LEARNING_CUES):
        return True

    return False