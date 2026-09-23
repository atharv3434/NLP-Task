"""
Generate a synthetic labeled sentiment dataset (positive / negative /
neutral) spanning movie, product, and restaurant reviews.

This project ships with a pre-generated dataset already in place
(data/reviews.csv), so you don't need to run this to try the project out.
Run it again for a fresh random sample or a different size.

Design note: negative examples include both direct negative-adjective
sentences ("The movie was terrible") AND negated-positive sentences
("The movie was not great at all") on purpose. Negation is a classic hard
case for bag-of-words models — "not great" contains the very positive word
"great" — so including it lets the evaluation actually show whether a
model learned something beyond individual word polarity.

Usage:python data/generate_data.py [--n-per-class 800] [--seed 42]
"""

import argparse
import os

import numpy as np
import pandas as pd

DOMAIN_NOUNS = {
    "movie": ["movie", "film"],
    "product": ["product", "gadget", "device"],
    "restaurant": ["restaurant", "meal", "dish"],
}

POS_ADJ = ["amazing", "wonderful", "fantastic", "excellent", "great", "brilliant", "superb", "outstanding", "delightful", "impressive"]
NEG_ADJ = ["terrible", "awful", "horrible", "disappointing", "poor", "dreadful", "mediocre", "bad", "frustrating", "unpleasant"]
NEU_ADJ = ["okay", "average", "standard", "typical", "fine", "acceptable", "unremarkable", "ordinary"]

POS_TEMPLATES = [
    "The {noun} was {adj}, I loved every moment.",
    "Absolutely {adj}! Highly recommend this {noun}.",
    "What a {adj} {noun}, truly worth it.",
    "I really enjoyed the {noun}, it was {adj}.",
    "This {noun} exceeded my expectations, {adj} in every way.",
    "Such a {adj} experience, I'll definitely be back.",
]

NEG_TEMPLATES = [
    "The {noun} was {adj}, a complete waste of time.",
    "Absolutely {adj}. Would not recommend this {noun}.",
    "What a {adj} {noun}, truly disappointing.",
    "I really disliked the {noun}, it was {adj}.",
    "This {noun} fell far short of expectations, {adj} in every way.",
    "Such a {adj} experience, I won't be back.",
    "The {noun} was not {pos_adj} at all, honestly {adj}.",
    "I expected a good {noun} but it definitely wasn't {pos_adj}.",
]

NEU_TEMPLATES = [
    "The {noun} was {adj}, nothing special.",
    "It was an {adj} {noun}, neither good nor bad.",
    "The {noun} was {adj}, met my expectations, nothing more.",
    "Just an {adj} {noun}, wouldn't go out of my way for it.",
    "The {noun} was {adj}. It is what it is.",
]


def build_examples(rng, n_per_class):
    domains = list(DOMAIN_NOUNS.keys())
    rows = []

    for _ in range(n_per_class):
        domain = domains[rng.integers(0, len(domains))]
        noun = DOMAIN_NOUNS[domain][rng.integers(0, len(DOMAIN_NOUNS[domain]))]
        template = POS_TEMPLATES[rng.integers(0, len(POS_TEMPLATES))]
        adj = POS_ADJ[rng.integers(0, len(POS_ADJ))]
        text = template.format(noun=noun, adj=adj)
        rows.append({"text": text, "label": "positive", "domain": domain})

    for _ in range(n_per_class):
        domain = domains[rng.integers(0, len(domains))]
        noun = DOMAIN_NOUNS[domain][rng.integers(0, len(DOMAIN_NOUNS[domain]))]
        template = NEG_TEMPLATES[rng.integers(0, len(NEG_TEMPLATES))]
        adj = NEG_ADJ[rng.integers(0, len(NEG_ADJ))]
        pos_adj = POS_ADJ[rng.integers(0, len(POS_ADJ))]
        text = template.format(noun=noun, adj=adj, pos_adj=pos_adj)
        rows.append({"text": text, "label": "negative", "domain": domain})

    for _ in range(n_per_class):
        domain = domains[rng.integers(0, len(domains))]
        noun = DOMAIN_NOUNS[domain][rng.integers(0, len(DOMAIN_NOUNS[domain]))]
        template = NEU_TEMPLATES[rng.integers(0, len(NEU_TEMPLATES))]
        adj = NEU_ADJ[rng.integers(0, len(NEU_ADJ))]
        text = template.format(noun=noun, adj=adj)
        rows.append({"text": text, "label": "neutral", "domain": domain})

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sentiment dataset.")
    parser.add_argument("--n-per-class", type=int, default=800, help="Examples per sentiment class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", default="data/reviews.csv", help="Output CSV path")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = build_examples(rng, args.n_per_class)
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)  # shuffle

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} labeled reviews to {args.out}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()