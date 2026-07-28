"""任务二扩展二分类指标。"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from src.common.metrics import binary_classification_metrics


def _ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error（等频分箱）。"""
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob < hi if i < n_bins - 1 else prob <= hi)
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(prob[mask].mean())
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def rich_binary_metrics(
    y: Sequence[Any],
    pred: Sequence[Any],
    prob: Sequence[float] | None = None,
) -> Dict[str, Any]:
    """包装 binary_classification_metrics 并补充 pr_auc / brier / ece / specificity / f1。"""
    y_arr = np.asarray(y).astype(int)
    pred_arr = np.asarray(pred).astype(int)
    base = binary_classification_metrics(y_arr, pred_arr, prob)

    if prob is not None:
        p = np.clip(np.asarray(prob, dtype=float), 1e-15, 1.0 - 1e-15)
        try:
            base["pr_auc"] = float(average_precision_score(y_arr, p))
        except ValueError:
            base["pr_auc"] = float("nan")
        try:
            base["brier"] = float(brier_score_loss(y_arr, p))
        except ValueError:
            base["brier"] = float("nan")
        base["ece"] = _ece(y_arr, p)
        try:
            base["roc_auc"] = float(roc_auc_score(y_arr, p))
        except ValueError:
            base["roc_auc"] = float("nan")
    else:
        base["pr_auc"] = float("nan")
        base["brier"] = float("nan")
        base["ece"] = float("nan")

    tn = int(((y_arr == 0) & (pred_arr == 0)).sum())
    fp = int(((y_arr == 0) & (pred_arr == 1)).sum())
    base["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    base["f1"] = float(
        f1_score(y_arr, pred_arr, average="binary", pos_label=1, zero_division=0)
    )
    return base
