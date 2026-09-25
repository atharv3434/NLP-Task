"""Predict sentiment for new text using the trained ML model.

Usage:
    python src/predict.py --text "This was a fantastic experience!"
    python src/predict.py --file my_reviews.txt        # one text per line
    python src/predict.py --interactive
    python src/predict.py --text "..." --compare-lexicon   # also show the lexicon baseline's prediction
"""

import argparse
import os
import sys

import joblib

sys.path.append(os.path.dirname(__file__))
from utils import load_config
from lexicon_baseline import predict_label as lexicon_predict


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at '{model_path}'. Run `python src/train.py` first.")
    return joblib.load(model_path)


def predict_texts(pipeline, texts):
    preds = pipeline.predict(texts)
    probs = pipeline.predict_proba(texts) if hasattr(pipeline, "predict_proba") else None
    results = []
    for i, (text, pred) in enumerate(zip(texts, preds)):
        confidence = float(probs[i].max()) if probs is not None else None
        results.append({"text": text, "prediction": pred, "confidence": confidence})
    return results


def print_result(result, show_lexicon):
    conf_str = f" ({result['confidence']:.2f} confidence)" if result["confidence"] is not None else ""
    print(f"[{result['prediction']:>8s}{conf_str}]  {result['text']}")
    if show_lexicon:
        lex_pred = lexicon_predict(result["text"])
        marker = "  <- differs from ML" if lex_pred != result["prediction"] else ""
        print(f"           lexicon baseline: {lex_pred}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment for text.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--text", help="A single piece of text to classify")
    parser.add_argument("--file", help="Path to a file with one text per line")
    parser.add_argument("--interactive", action="store_true", help="Prompt for text interactively")
    parser.add_argument("--compare-lexicon", action="store_true", help="Also show the lexicon baseline's prediction")
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = load_model(config["model_path"])
    pipeline = bundle["pipeline"]
    print(f"Loaded model: {bundle['model_name']}\n")

    if args.text:
        result = predict_texts(pipeline, [args.text])[0]
        print_result(result, args.compare_lexicon)
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        for result in predict_texts(pipeline, texts):
            print_result(result, args.compare_lexicon)
    elif args.interactive:
        print("Interactive mode. Type a sentence and press Enter (Ctrl+C to quit).\n")
        try:
            while True:
                text = input("> ").strip()
                if not text:
                    continue
                result = predict_texts(pipeline, [text])[0]
                print_result(result, args.compare_lexicon)
        except KeyboardInterrupt:
            print("\nGoodbye.")
    else:
        print("Provide --text \"...\", --file <path>, or --interactive.")


if __name__ == "__main__":
    main()
