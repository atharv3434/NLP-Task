"""A classical, rule-based sentiment baseline: a hand-crafted word lexicon
plus simple negation handling, in the spirit of lexicon-based sentiment
tools like VADER (Hutto & Gilberto, 2014) — no training data required.

This exists mainly as a genuine point of comparison for the trained ML
model in train.py: how far does "just count polarity words, with basic
negation handling" get you, versus a model that learns weighted patterns
from labeled examples?
"""

import re

POSITIVE_WORDS = {
    "amazing", "wonderful", "fantastic", "excellent", "great", "brilliant",
    "superb", "outstanding", "delightful", "impressive", "love", "loved",
    "enjoy", "enjoyed", "recommend", "best", "perfect", "happy", "pleased",
}

NEGATIVE_WORDS = {
    "terrible", "awful", "horrible", "disappointing", "poor", "dreadful",
    "mediocre", "bad", "frustrating", "unpleasant", "hate", "hated",
    "dislike", "disliked", "worst", "waste", "boring", "annoying",
}

# If one of these appears within `NEGATION_WINDOW` tokens before a polarity
# word, that word's contribution is flipped — this is what lets "not great"
# register as negative instead of positive.
NEGATION_WORDS = {"not", "no", "never", "n't", "without", "hardly", "isn't", "wasn't", "wouldn't", "won't", "doesn't", "didn't"}
NEGATION_WINDOW = 3


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def score_text(text):
    """Returns a raw polarity score: positive words count +1, negative
    words count -1, each flipped if a negation word appears shortly before
    it in the sentence.
    """
    tokens = tokenize(text)
    score = 0

    for i, tok in enumerate(tokens):
        if tok not in POSITIVE_WORDS and tok not in NEGATIVE_WORDS:
            continue

        window_start = max(0, i - NEGATION_WINDOW)
        negated = any(t in NEGATION_WORDS for t in tokens[window_start:i])

        polarity = 1 if tok in POSITIVE_WORDS else -1
        if negated:
            polarity *= -1
        score += polarity

    return score


def predict_label(text):
    score = score_text(text)
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def predict_batch(texts):
    return [predict_label(t) for t in texts]