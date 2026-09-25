"""Compare the classical lexicon baseline against the trained ML model —
on both the same-distribution templated test data and a genuinely
challenging, hand-written out-of-distribution test set.

Usage:
    python src/evaluate.py [--config config.yaml]
"""

import argparse
import os
import sys

import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.append(os.path.dirname(__file__))
from utils import load_config, load_data
from lexicon_baseline import predict_batch as lexicon_predict_batch


def evaluate_method(y_true, y_pred, labels):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "report": report}


def main():
    parser = argparse.ArgumentParser(description="Compare lexicon baseline vs. trained ML model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--hard-examples", default="data/hard_examples.csv", help="Hand-written OOD test set")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = config["labels"]

    bundle = joblib.load(config["model_path"])
    ml_pipeline = bundle["pipeline"]
    print(f"Loaded ML model: {bundle['model_name']}\n")

    df_templated = load_data(config["data_path"])
    df_hard = load_data(args.hard_examples)

    lines = ["# Sentiment Analysis — Lexicon Baseline vs. ML Model", ""]
    lines.append(
        "Comparing a classical hand-crafted lexicon + negation-handling "
        "baseline (`src/lexicon_baseline.py`) against the trained ML model "
        f"(**{bundle['model_name']}**, TF-IDF + scikit-learn) on two test sets."
    )
    lines.append("")

    for name, df, note in [
        ("Templated test data (same distribution as training)", df_templated, None),
        ("Hand-written out-of-distribution examples", df_hard, "harder, more natural phrasing not seen in training templates"),
    ]:
        X = df["text"]
        y_true = df["label"]

        lexicon_preds = lexicon_predict_batch(X.tolist())
        ml_preds = ml_pipeline.predict(X)

        lexicon_metrics = evaluate_method(y_true, lexicon_preds, labels)
        ml_metrics = evaluate_method(y_true, ml_preds, labels)

        print(f"--- {name} ---")
        if note:
            print(f"({note})")
        print(f"  Lexicon baseline:  accuracy={lexicon_metrics['accuracy']:.3f}  macro_f1={lexicon_metrics['macro_f1']:.3f}")
        print(f"  ML model:          accuracy={ml_metrics['accuracy']:.3f}  macro_f1={ml_metrics['macro_f1']:.3f}\n")

        lines.append(f"## {name}")
        if note:
            lines.append(f"*({note})*")
        lines.append("")
        lines.append(f"n = {len(df)}")
        lines.append("")
        lines.append("| Method | Accuracy | Macro F1 |")
        lines.append("|---|---|---|")
        lines.append(f"| Lexicon baseline | {lexicon_metrics['accuracy']:.3f} | {lexicon_metrics['macro_f1']:.3f} |")
        lines.append(f"| ML model ({bundle['model_name']}) | {ml_metrics['accuracy']:.3f} | {ml_metrics['macro_f1']:.3f} |")
        lines.append("")

        lines.append("**ML model per-class:**")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|")
        for label in labels:
            r = ml_metrics["report"][label]
            lines.append(f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} |")
        lines.append("")

    # A few concrete side-by-side examples from the hard set, to make the
    # difference between methods concrete rather than just numbers.
    lines.append("## Example disagreements (hard set)")
    lines.append("")
    X_hard = df_hard["text"]
    y_hard = df_hard["label"]
    lex_hard = lexicon_predict_batch(X_hard.tolist())
    ml_hard = ml_pipeline.predict(X_hard)

    shown = 0
    for text, true_label, lex_pred, ml_pred in zip(X_hard, y_hard, lex_hard, ml_hard):
        if lex_pred != ml_pred and shown < 8:
            lines.append(f"- *\"{text}\"* — true: **{true_label}**, lexicon: {lex_pred}, ML: {ml_pred}")
            shown += 1
    if shown == 0:
        lines.append("(No disagreements between the two methods on this set.)")

    os.makedirs(os.path.dirname(config["metrics_report_path"]), exist_ok=True)
    report_path = os.path.join(os.path.dirname(config["metrics_report_path"]), "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Comparison report saved to {report_path}")


if __name__ == "__main__":
    main()
