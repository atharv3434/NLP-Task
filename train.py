"""
Train and evaluate ML sentiment classifiers, then save the
best-performing model.

Usage:
    python src/train.py [--config config.yaml]

Trains Multinomial Naive Bayes and Logistic Regression (both on TF-IDF
features, bundled into a single scikit-learn Pipeline per model),
evaluates both on a held-out stratified test split, and saves the
higher-macro-F1 model.

"""

import argparse
import os
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from utils import load_config, load_data

MODEL_BUILDERS = {
    "naive_bayes": lambda: MultinomialNB(),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
}


def build_pipeline(model_key, config):
    vectorizer = TfidfVectorizer(
        max_features=config.get("max_features", 3000),
        ngram_range=tuple(config.get("ngram_range", [1, 2])),
    )
    classifier = MODEL_BUILDERS[model_key]()
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def plot_confusion_matrix(y_test, preds, labels, model_name, figures_dir):
    cm = confusion_matrix(y_test, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(figures_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Train sentiment classifiers.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config["figures_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(config["model_path"]), exist_ok=True)

    print(f"Loading data from {config['data_path']} ...")
    df = load_data(config["data_path"])
    labels = config["labels"]

    X = df[config["text_col"]]
    y = df[config["label_col"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42), stratify=y,
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

    results = {}
    pipelines = {}
    for model_key in config.get("models", ["naive_bayes", "logistic_regression"]):
        display_name = model_key.replace("_", " ").title()
        print(f"Training {display_name}...")
        pipeline = build_pipeline(model_key, config)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        report = classification_report(y_test, preds, labels=labels, output_dict=True, zero_division=0)
        macro_f1 = f1_score(y_test, preds, labels=labels, average="macro", zero_division=0)
        accuracy = report["accuracy"]

        results[display_name] = {"report": report, "macro_f1": macro_f1, "accuracy": accuracy, "preds": preds}
        pipelines[display_name] = pipeline

        print(f"  Accuracy: {accuracy:.3f}  |  Macro F1: {macro_f1:.3f}")
        for label in labels:
            r = report[label]
            print(f"    {label:10s} precision={r['precision']:.3f}  recall={r['recall']:.3f}  f1={r['f1-score']:.3f}")
        print()

        plot_confusion_matrix(y_test, preds, labels, display_name, config["figures_dir"])

    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_pipeline = pipelines[best_name]
    print(f"Best model by macro F1: {best_name} ({results[best_name]['macro_f1']:.3f})")

    joblib.dump({"pipeline": best_pipeline, "labels": labels, "model_name": best_name}, config["model_path"])
    print(f"Saved best model to {config['model_path']}")

    # --- Metrics report ---
    lines = ["# Sentiment Classifier — Training Report", ""]
    lines.append(f"Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")
    lines.append("")
    lines.append("| Model | Accuracy | Macro F1 |")
    lines.append("|---|---|---|")
    for name, res in results.items():
        marker = " **(best)**" if name == best_name else ""
        lines.append(f"| {name}{marker} | {res['accuracy']:.3f} | {res['macro_f1']:.3f} |")
    lines.append("")
    for name, res in results.items():
        lines.append(f"## {name} — per-class metrics")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|")
        for label in labels:
            r = res["report"][label]
            lines.append(f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} |")
        lines.append("")
        slug = name.replace(" ", "_")
        lines.append(f"![Confusion Matrix](figures/confusion_matrix_{slug}.png)")
        lines.append("")

    with open(config["metrics_report_path"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Metrics report saved to {config['metrics_report_path']}")


if __name__ == "__main__":
    main()
