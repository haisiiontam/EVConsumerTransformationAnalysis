from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from src.config import TASK1_LABEL_ORDER

LABEL_TO_ORD = {lab: i for i, lab in enumerate(TASK1_LABEL_ORDER)}


def to_ord(labels: Sequence[Any]) -> np.ndarray:
    arr = pd.Series(labels).astype(str).str.strip()
    mapped = arr.map(
        lambda x: LABEL_TO_ORD.get(
            {"low": "Low", "medium": "Medium", "high": "High"}.get(x.lower(), x), np.nan
        )
    )
    return mapped.to_numpy(dtype=float)


def normalize_label_array(labels: Sequence[Any]) -> np.ndarray:
    arr = pd.Series(labels).astype(str).str.strip()
    return arr.replace({"low": "Low", "medium": "Medium", "high": "High"}).to_numpy()


def proba_matrix(
    prob_low: np.ndarray, prob_med: np.ndarray, prob_high: np.ndarray
) -> np.ndarray:
    p = np.column_stack([prob_low, prob_med, prob_high]).astype(float)
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def evaluate_multiclass(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    proba: np.ndarray | None = None,
) -> Dict[str, Any]:
    yt = normalize_label_array(y_true)
    yp = normalize_label_array(y_pred)
    yt_o = to_ord(yt)
    yp_o = to_ord(yp)
    mask = np.isfinite(yt_o) & np.isfinite(yp_o)
    yt, yp = yt[mask], yp[mask]
    yt_o, yp_o = yt_o[mask].astype(int), yp_o[mask].astype(int)

    acc = float(accuracy_score(yt, yp))
    macro_f1 = float(f1_score(yt, yp, average="macro", labels=TASK1_LABEL_ORDER, zero_division=0))
    weighted_f1 = float(
        f1_score(yt, yp, average="weighted", labels=TASK1_LABEL_ORDER, zero_division=0)
    )
    qwk = float(cohen_kappa_score(yt_o, yp_o, weights="quadratic"))

    prec, rec, f1s, _ = precision_recall_fscore_support(
        yt, yp, labels=TASK1_LABEL_ORDER, zero_division=0
    )
    diff = np.abs(yt_o - yp_o)
    n = max(len(yt), 1)
    adjacent_err = float((diff == 1).sum() / n)
    cross_two = int((diff == 2).sum())
    medium_mask = yt == "Medium"
    medium_recall = float((yp[medium_mask] == "Medium").mean()) if medium_mask.any() else float("nan")

    out: Dict[str, Any] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "qwk": qwk,
        "adjacent_error_rate": adjacent_err,
        "cross_two_level_errors": cross_two,
        "medium_recall": medium_recall,
        "precision_Low": float(prec[0]),
        "recall_Low": float(rec[0]),
        "f1_Low": float(f1s[0]),
        "precision_Medium": float(prec[1]),
        "recall_Medium": float(rec[1]),
        "f1_Medium": float(f1s[1]),
        "precision_High": float(prec[2]),
        "recall_High": float(rec[2]),
        "f1_High": float(f1s[2]),
    }
    if proba is not None:
        p = np.asarray(proba, dtype=float)[mask]
        p = np.clip(p, 1e-15, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        y_idx = yt_o
        try:
            out["logloss"] = float(log_loss(y_idx, p, labels=[0, 1, 2]))
        except ValueError:
            out["logloss"] = float("nan")
    else:
        out["logloss"] = float("nan")
    return out


def classification_report_df(y_true: Sequence[Any], y_pred: Sequence[Any]) -> pd.DataFrame:
    yt = normalize_label_array(y_true)
    yp = normalize_label_array(y_pred)
    rep = classification_report(
        yt, yp, labels=TASK1_LABEL_ORDER, output_dict=True, zero_division=0
    )
    return pd.DataFrame(rep).T.reset_index().rename(columns={"index": "label"})


def fold_summary(fold_metrics: Sequence[Dict[str, Any]], key: str = "accuracy") -> Dict[str, float]:
    vals = np.array([float(r[key]) for r in fold_metrics], dtype=float)
    return {
        f"fold_mean_{key}": float(vals.mean()) if len(vals) else float("nan"),
        f"fold_std_{key}": float(vals.std(ddof=0)) if len(vals) else float("nan"),
    }


def pred_from_proba(proba: np.ndarray) -> np.ndarray:
    idx = np.argmax(proba, axis=1)
    return np.array([TASK1_LABEL_ORDER[i] for i in idx], dtype=object)
