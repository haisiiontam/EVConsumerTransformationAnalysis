from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.metrics import binary_classification_metrics
from src.config import TASK3_THRESHOLD_MAX, TASK3_THRESHOLD_MIN, TASK3_THRESHOLD_STEP, TASK3_TIE_EPS


def search_threshold(p1: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for thr in np.round(np.arange(TASK3_THRESHOLD_MIN, TASK3_THRESHOLD_MAX + 1e-12, TASK3_THRESHOLD_STEP), 10):
        pred = (p1 >= thr).astype(int)
        m = binary_classification_metrics(y, pred, p1)
        rows.append({"threshold": float(thr), **m})
    search = pd.DataFrame(rows)
    max_acc = search["accuracy"].max()
    cand = search[search["accuracy"] >= max_acc - TASK3_TIE_EPS].copy()
    cand["thr_dist"] = (cand["threshold"] - 0.5).abs()
    cand = cand.sort_values(
        by=["accuracy", "balanced_accuracy", "recall_0", "thr_dist"],
        ascending=[False, False, False, True],
    )
    best = cand.iloc[0]
    return search, {
        "threshold": float(best["threshold"]),
        "oof_accuracy": float(best["accuracy"]),
        "oof_roc_auc": float(best["roc_auc"]),
        "oof_balanced_accuracy": float(best["balanced_accuracy"]),
    }


def calibrate_oof(
    p1: np.ndarray, y: np.ndarray, fold_assign: np.ndarray
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    methods = {"none": p1.copy()}
    n_folds = int(fold_assign.max()) + 1

    p_platt = np.zeros_like(p1)
    p_iso = np.zeros_like(p1)
    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        lr = LogisticRegression(max_iter=200)
        lr.fit(p1[tr].reshape(-1, 1), y[tr])
        p_platt[va] = lr.predict_proba(p1[va].reshape(-1, 1))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p1[tr], y[tr])
        p_iso[va] = iso.predict(p1[va])
    methods["platt"] = p_platt
    methods["isotonic"] = p_iso

    rows = []
    for name, p in methods.items():
        m = binary_classification_metrics(y, (p >= 0.5).astype(int), p)
        rows.append({"method": name, **m})
    return methods, pd.DataFrame(rows)


def persist_calibration(output_dir: Path, compare: pd.DataFrame, best_thr: Dict[str, Any], search: pd.DataFrame) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "calibration_comparison.csv", compare)
    atomic_write_csv(output_dir / "threshold_search.csv", search)
    atomic_write_json(output_dir / "threshold_best.json", best_thr)
