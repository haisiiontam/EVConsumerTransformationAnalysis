"""二层 Stacking（元特征再次 5 折 OOF）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.common.io_utils import atomic_write_csv, atomic_write_json

logger = logging.getLogger(__name__)


def _meta_matrix(model_p1: Dict[str, np.ndarray], pred_score: np.ndarray | None) -> np.ndarray:
    cols = [model_p1[k] for k in model_p1]
    X = np.column_stack(cols)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    rng = X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True)
    ent = -(X.clip(1e-6, 1 - 1e-6) * np.log(X.clip(1e-6, 1 - 1e-6))).sum(axis=1, keepdims=True)
    parts = [X, mean, std, rng, ent]
    if pred_score is not None:
        parts.append(pred_score.reshape(-1, 1))
        parts.append(np.abs(pred_score - 5.5).reshape(-1, 1))
    return np.hstack(parts)


def stacking_oof(
    model_p1: Dict[str, np.ndarray],
    y: np.ndarray,
    fold_assign: np.ndarray,
    pred_score: np.ndarray | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    X = _meta_matrix(model_p1, pred_score)
    oof = np.zeros(len(y), dtype=float)
    n_folds = int(fold_assign.max()) + 1
    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        clf = LogisticRegression(max_iter=500)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    acc = float(((oof >= 0.5).astype(int) == y).mean())
    info = {"meta_model": "LogisticRegression", "oof_accuracy": acc, "n_meta_features": int(X.shape[1])}
    return oof, info


def persist_stacking(
    output_dir: Path,
    oof_p1: np.ndarray,
    y: np.ndarray,
    sample_index: np.ndarray,
    info: Dict[str, Any],
    compare_rows: List[Dict[str, Any]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(
        output_dir / "stacking_level2_oof.csv",
        pd.DataFrame(
            {
                "sample_index": sample_index,
                "true_label": y,
                "prob_1": oof_p1,
                "prob_0": 1.0 - oof_p1,
                "predicted_label": (oof_p1 >= 0.5).astype(int),
            }
        ),
    )
    atomic_write_csv(output_dir / "stacking_comparison.csv", pd.DataFrame(compare_rows))
    atomic_write_json(output_dir / "stacking_best.json", info)
