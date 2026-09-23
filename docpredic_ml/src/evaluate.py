"""
DocPredic Evaluation
Comprehensive model evaluation and reporting.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
import json
from pathlib import Path

from .config import METRICS_DIR


def evaluate_model(y_true, y_pred, y_prob=None, class_names=None) -> Dict[str, Any]:
    results = {}
    results["accuracy"] = float(accuracy_score(y_true, y_pred))
    results["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    results["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    results["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    results["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    results["per_class"] = {}
    if class_names:
        for name in class_names:
            if name in report:
                results["per_class"][name] = {
                    "precision": float(report[name]["precision"]),
                    "recall": float(report[name]["recall"]),
                    "f1": float(report[name]["f1-score"]),
                    "support": int(report[name]["support"]),
                }

    results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    if y_prob is not None and y_prob.shape[1] > 2:
        try:
            y_true_onehot = np.eye(y_prob.shape[1])[y_true]
            results["roc_auc_macro"] = float(roc_auc_score(y_true_onehot, y_prob, average="macro", multi_class="ovr"))
            results["roc_auc_weighted"] = float(roc_auc_score(y_true_onehot, y_prob, average="weighted", multi_class="ovr"))
        except Exception:
            results["roc_auc_macro"] = None
            results["roc_auc_weighted"] = None

    return results


def print_evaluation_report(results: Dict[str, Any], model_name: str = "Model") -> None:
    print("\n" + "=" * 70)
    print(f"  EVALUATION REPORT: {model_name}")
    print("=" * 70)

    print(f"\n  Overall Metrics:")
    print(f"    Accuracy:       {results['accuracy']:.4f}")
    print(f"    F1 (macro):     {results['f1_macro']:.4f}")
    print(f"    F1 (weighted):  {results['f1_weighted']:.4f}")
    print(f"    Precision:      {results['precision_macro']:.4f}")
    print(f"    Recall:         {results['recall_macro']:.4f}")

    if results.get("roc_auc_macro"):
        print(f"    ROC-AUC (macro): {results['roc_auc_macro']:.4f}")

    if results.get("per_class"):
        print(f"\n  Per-Class Metrics:")
        print(f"  {'Class':25s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Supp':>6s}")
        print(f"  {'-'*50}")
        for cls, metrics in sorted(results["per_class"].items()):
            print(f"  {cls:25s} {metrics['precision']:6.3f} {metrics['recall']:6.3f} "
                  f"{metrics['f1']:6.3f} {metrics['support']:6d}")

    print("\n" + "=" * 70)


def save_evaluation(results: Dict[str, Any], model_name: str) -> Path:
    path = METRICS_DIR / f"{model_name}_metrics.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Metrics saved to {path}")
    return path


def compare_models(all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for model_name, results in all_results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": results.get("accuracy", 0),
            "F1 (macro)": results.get("f1_macro", 0),
            "F1 (weighted)": results.get("f1_weighted", 0),
            "Precision": results.get("precision_macro", 0),
            "Recall": results.get("recall_macro", 0),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("F1 (macro)", ascending=False)
    return df
