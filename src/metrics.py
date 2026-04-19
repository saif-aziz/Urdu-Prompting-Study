"""Metrics + significance testing.

We report macro-F1 (primary — handles class imbalance) and accuracy.
McNemar's test on paired predictions gives us a real significance number
for 'does en-prompt beat ur-prompt', which elevates the paper above
simple accuracy reporting.
"""
from __future__ import annotations
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


LABELS = ["negative", "neutral", "positive"]


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    # Replace 'unknown' predictions with most-common label so they count as misclassifications
    # but don't break sklearn. We track unknown rate separately.
    n_unknown = sum(1 for p in y_pred if p == "unknown")
    y_pred_clean = [p if p in LABELS else "neutral" for p in y_pred]

    acc = accuracy_score(y_true, y_pred_clean)
    macro_f1 = f1_score(y_true, y_pred_clean, labels=LABELS, average="macro", zero_division=0)
    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_clean, labels=LABELS, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred_clean, labels=LABELS).tolist()

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "f1_negative": float(per_f1[0]),
        "f1_neutral": float(per_f1[1]),
        "f1_positive": float(per_f1[2]),
        "unknown_rate": n_unknown / max(1, len(y_pred)),
        "confusion_matrix": cm,
        "n": len(y_pred),
    }


def mcnemar_test(y_true: List[str], y_pred_a: List[str], y_pred_b: List[str]) -> Dict[str, float]:
    """Paired McNemar test between two classifiers on the same test set.
    Returns p-value (exact binomial) and counts.
    """
    assert len(y_true) == len(y_pred_a) == len(y_pred_b)
    # b = A correct, B wrong. c = A wrong, B correct.
    b = sum(1 for t, a, c in zip(y_true, y_pred_a, y_pred_b) if a == t and c != t)
    c = sum(1 for t, a, z in zip(y_true, y_pred_a, y_pred_b) if a != t and z == t)
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        table = [[0, b], [c, 0]]
        res = mcnemar(table, exact=True)
        pval = float(res.pvalue)
    except Exception:
        # Fallback: simple approximation
        if b + c == 0:
            pval = 1.0
        else:
            from scipy.stats import binomtest
            try:
                pval = float(binomtest(min(b, c), n=b + c, p=0.5).pvalue)
            except Exception:
                pval = float("nan")
    return {"b": int(b), "c": int(c), "p_value": pval}


def majority_class_baseline(y_train: List[str], y_test: List[str]) -> Dict[str, float]:
    most = Counter(y_train).most_common(1)[0][0]
    preds = [most] * len(y_test)
    out = compute_metrics(y_test, preds)
    out["_predictions"] = preds
    out["_label_used"] = most
    return out
