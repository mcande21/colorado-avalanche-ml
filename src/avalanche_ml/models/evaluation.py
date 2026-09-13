from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)

from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def ordinal_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    within_one = np.abs(y_true.astype(int) - y_pred.astype(int)) <= 1
    return float(within_one.mean())


def danger_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])


def false_negative_rate_elevated(
    y_true_danger: np.ndarray, y_pred_binary: np.ndarray
) -> float:
    elevated_mask = y_true_danger >= 3
    if not elevated_mask.any():
        return 0.0
    missed = ((elevated_mask) & (y_pred_binary == 0)).sum()
    return float(missed / elevated_mask.sum())


def high_danger_detection_rate(
    y_true_danger: np.ndarray, y_pred_binary: np.ndarray
) -> float:
    high_mask = y_true_danger >= 4
    if not high_mask.any():
        return 1.0
    detected = ((high_mask) & (y_pred_binary == 1)).sum()
    return float(detected / high_mask.sum())


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true - y_prob) ** 2))


def class_distribution_report(y: np.ndarray) -> dict:
    counts = Counter(y.tolist())
    total = len(y)
    return {
        "total": total,
        "distribution": dict(counts),
        "percentages": {k: round(v / total * 100, 1) for k, v in counts.items()},
    }


def per_elevation_band_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    elevation_bands: np.ndarray,
) -> dict:
    result = {}
    for band in np.unique(elevation_bands):
        mask = elevation_bands == band
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) == 0:
            continue

        present_labels = sorted(set(yt) | set(yp))
        macro = float(f1_score(yt, yp, labels=present_labels, average="macro", zero_division=0))
        result[band] = {
            "ordinal_accuracy": ordinal_accuracy(yt, yp),
            "macro_f1": macro,
            "n_samples": len(yt),
        }
    return result


def evaluate_stage1(
    y_true: pd.DataFrame, y_pred: pd.DataFrame
) -> dict:
    per_type = {}
    for pt in PROBLEM_TYPE_FLAGS:
        yt = y_true[pt].values
        yp = y_pred[pt].values
        per_type[pt] = compute_binary_metrics(yt, yp)

    exact_match = (y_true.values == y_pred.values).all(axis=1).mean()

    return {
        "per_type": per_type,
        "exact_match_ratio": float(exact_match),
    }


def evaluate_stage2(
    y_true_danger: np.ndarray,
    y_pred_binary: np.ndarray,
    y_pred_danger: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    binary = compute_binary_metrics(
        (y_true_danger >= 3).astype(int), y_pred_binary
    )

    present_labels = sorted(set(y_true_danger) | set(y_pred_danger))
    macro = float(
        f1_score(y_true_danger, y_pred_danger, labels=present_labels, average="macro",
                 zero_division=0)
    )

    y_true_binary = (y_true_danger >= 3).astype(int)

    return {
        "binary_f1": binary["f1"],
        "binary_precision": binary["precision"],
        "binary_recall": binary["recall"],
        "ordinal_accuracy": ordinal_accuracy(y_true_danger, y_pred_danger),
        "confusion_matrix": danger_confusion_matrix(y_true_danger, y_pred_danger),
        "macro_f1": macro,
        "brier_score": brier_score(y_true_binary, y_pred_proba),
        "false_negative_rate": false_negative_rate_elevated(y_true_danger, y_pred_binary),
        "high_danger_detection_rate": high_danger_detection_rate(y_true_danger, y_pred_binary),
    }
