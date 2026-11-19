"""Metric helpers (all JSON-serialisable)."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def _down(a, n=200):
    a = np.asarray(a)
    if len(a) <= n:
        return a.tolist()
    idx = np.linspace(0, len(a) - 1, n).astype(int)
    return a[idx].tolist()


def scalar_metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "avg_precision": float(average_precision_score(y, proba)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }


def full_metrics(y, proba, labels, thr=0.5):
    y = np.asarray(y)
    pred = (proba >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y, proba)
    pr, rc, _ = precision_recall_curve(y, proba)
    sweep = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        p = (proba >= t).astype(int)
        sweep.append({"threshold": float(t),
                      "accuracy": float(accuracy_score(y, p)),
                      "precision": float(precision_score(y, p, zero_division=0)),
                      "recall": float(recall_score(y, p, zero_division=0)),
                      "f1": float(f1_score(y, p, zero_division=0))})
    rep = classification_report(y, pred, target_names=labels, output_dict=True, zero_division=0)
    return {
        **scalar_metrics(y, proba, thr),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "roc": {"fpr": _down(fpr), "tpr": _down(tpr)},
        "pr": {"precision": _down(pr), "recall": _down(rc)},
        "threshold_sweep": sweep,
        "classification_report": rep,
    }
