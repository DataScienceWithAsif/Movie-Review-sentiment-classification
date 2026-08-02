"""
evaluate.py
-----------
Evaluates the fine-tuned DistilBERT sentiment model on a held-out test set
and reports accuracy, precision, recall, F1-score, and a confusion matrix.

Usage:
    python evaluate.py --data data/test.csv --text-col review --label-col sentiment

Place this file in the project root (same level as main.py / app.py) so the
`from src.model.predictor import Predictor` import resolves correctly.
"""

import argparse
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.model.predictor import Predictor


LABEL_MAP = {
    "positive": 1, "pos": 1, "label_1": 1, "1": 1, 1: 1,
    "negative": 0, "neg": 0, "label_0": 0, "0": 0, 0: 0,
}


def normalize_label(value):
    key = str(value).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"Unrecognized label value: {value!r}")
    return LABEL_MAP[key]


def run_evaluation(data_path: str, text_col: str, label_col: str, out_dir: str = "."):
    df = pd.read_csv(data_path)

    if text_col not in df.columns or label_col not in df.columns:
        print(f"Error: expected columns '{text_col}' and '{label_col}' in {data_path}.")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {data_path}")

    predictor = Predictor()

    y_true, y_pred = [], []

    for i, row in df.iterrows():
        text = str(row[text_col])
        true_label = normalize_label(row[label_col])

        label, confidence = predictor.get_predicted_sentiment_with_confidence(text)
        pred_label = normalize_label(label)

        y_true.append(true_label)
        y_pred.append(pred_label)

        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(df)}")

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    print("\n=== Evaluation Results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path = f"{out_dir}/confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nConfusion matrix saved to: {out_path}")

    metrics_path = f"{out_dir}/metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the sentiment model on a labeled test set.")
    parser.add_argument("--data", default="data/test.csv", help="Path to test CSV file")
    parser.add_argument("--text-col", default="review", help="Column name containing review text")
    parser.add_argument("--label-col", default="sentiment", help="Column name containing true label")
    parser.add_argument("--out-dir", default=".", help="Directory to save outputs")
    args = parser.parse_args()

    run_evaluation(args.data, args.text_col, args.label_col, args.out_dir)