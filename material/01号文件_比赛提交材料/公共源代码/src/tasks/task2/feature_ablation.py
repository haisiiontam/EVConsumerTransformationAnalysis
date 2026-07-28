"""任务二特征消融（快速对照，防过完备宣称）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.tasks.task2.features import feature_sets
from src.tasks.task2.metrics_ext import rich_binary_metrics
from src.tasks.task2.preprocessing import fit_tabular, transform_tabular

logger = logging.getLogger(__name__)


def _oof_logistic_acc_auc(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    features: Sequence[str],
    fold_assign: np.ndarray,
    n_folds: int,
) -> Dict[str, Any]:
    y = y_dev.to_numpy().astype(int)
    p1 = np.zeros(len(df_dev), dtype=float)
    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        state = fit_tabular(df_dev.loc[tr], features, scale=True)
        X_tr = transform_tabular(df_dev.loc[tr], state)
        X_va = transform_tabular(df_dev.loc[va], state)
        if X_tr.shape[1] == 0 or len(np.unique(y[tr])) < 2:
            p1[va] = float(y[tr].mean()) if len(y[tr]) else 0.5
            continue
        lr = LogisticRegression(max_iter=400, C=1.0, random_state=42)
        lr.fit(X_tr, y[tr])
        p1[va] = lr.predict_proba(X_va)[:, 1]
    pred = (p1 >= 0.5).astype(int)
    return rich_binary_metrics(y, pred, p1)


def run_feature_ablation(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    fold_assign: np.ndarray,
    output_dir: Path,
    majority_acc: float,
    n_folds: int = 5,
) -> pd.DataFrame:
    out = Path(output_dir) / "features"
    out.mkdir(parents=True, exist_ok=True)
    sets = feature_sets(df_dev)
    keys = [
        "A_official",
        "B_safe_raw",
        "B+F1_charging",
        "B+F2_economy",
        "B+F3_travel",
        "B+F4_vehicle",
        "B+F5_city_demo",
        "B+F6_readiness",
        "B+F7_housing_proxy",
        "B+all_mechanism",
    ]
    rows: List[Dict[str, Any]] = []
    for key in keys:
        feats = sets.get(key)
        if not feats:
            continue
        try:
            m = _oof_logistic_acc_auc(df_dev, y_dev, feats, fold_assign, n_folds)
            rows.append(
                {
                    "feature_set": key,
                    "n_features": len(feats),
                    "accuracy": m["accuracy"],
                    "roc_auc": m["roc_auc"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "brier": m.get("brier"),
                    "lift_vs_majority": m["accuracy"] - majority_acc,
                }
            )
            logger.info("ablation %s acc=%.6f auc=%.4f", key, m["accuracy"], m["roc_auc"])
        except Exception as e:
            logger.warning("ablation %s failed: %s", key, e)
            rows.append({"feature_set": key, "n_features": len(feats), "error": str(e)})

    df = pd.DataFrame(rows)
    atomic_write_csv(out / "feature_ablation.csv", df)
    best = None
    if len(df) and "roc_auc" in df.columns:
        best = str(df.sort_values("roc_auc", ascending=False).iloc[0]["feature_set"])
    atomic_write_json(
        out / "feature_ablation_summary.json",
        {"best_by_auc": best, "note": "Fast logistic OOF ablation; not sole final-lock input."},
    )
    return df
