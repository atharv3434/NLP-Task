# Sentiment Analysis — Lexicon Baseline vs. ML Model

Comparing a classical hand-crafted lexicon + negation-handling baseline (`src/lexicon_baseline.py`) against the trained ML model (**Naive Bayes**, TF-IDF + scikit-learn) on two test sets.

## Templated test data (same distribution as training)

n = 2400

| Method | Accuracy | Macro F1 |
|---|---|---|
| Lexicon baseline | 0.933 | 0.932 |
| ML model (Naive Bayes) | 1.000 | 1.000 |

**ML model per-class:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 1.000 | 1.000 | 1.000 |
| neutral | 1.000 | 1.000 | 1.000 |
| positive | 1.000 | 1.000 | 1.000 |

## Hand-written out-of-distribution examples
*(harder, more natural phrasing not seen in training templates)*

n = 30

| Method | Accuracy | Macro F1 |
|---|---|---|
| Lexicon baseline | 0.333 | 0.253 |
| ML model (Naive Bayes) | 0.433 | 0.434 |

**ML model per-class:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.167 | 0.222 | 0.190 |
| neutral | 0.636 | 0.700 | 0.667 |
| positive | 0.571 | 0.364 | 0.444 |

## Example disagreements (hard set)

- *"The movie wasn't bad at all, actually quite enjoyable."* — true: **positive**, lexicon: positive, ML: negative
- *"I wouldn't call it terrible, more just forgettable."* — true: **neutral**, lexicon: positive, ML: neutral
- *"Honestly? Not what I expected, but I'm not complaining."* — true: **positive**, lexicon: neutral, ML: negative
- *"A truly captivating experience from start to finish."* — true: **positive**, lexicon: neutral, ML: positive
- *"This gadget is a total lemon, broke within a week."* — true: **negative**, lexicon: neutral, ML: positive
- *"So good I watched it twice in one weekend!"* — true: **positive**, lexicon: neutral, ML: negative
- *"Not the worst meal I've had, but far from memorable."* — true: **neutral**, lexicon: positive, ML: negative
- *"I can't stop thinking about how good this was."* — true: **positive**, lexicon: neutral, ML: negative