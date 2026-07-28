"""M3：有序评分学习（累积边界 + 10 类分布对照）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.metrics import binary_classification_metrics
from src.config import RANDOM_SEED, TASK3_EARLY_STOPPING_ROUNDS, TASK3_N_FOLDS
from src.tasks.task3.features import raw_score
from src.tasks.task3.preprocessing import cat_indices, fit_cat_preprocessor, transform_cat

logger = logging.getLogger(__name__)


def _enforce_monotone_cum(cum: np.ndarray) -> np.ndarray:
    """cum shape (n, 9) for P(score>=2)...P(score>=10); enforce non-increasing."""
    out = cum.copy()
    for i in range(1, out.shape[1]):
        out[:, i] = np.minimum(out[:, i], out[:, i - 1])
    return np.clip(out, 0.0, 1.0)


def _fit_binary_safe(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cats: Sequence[int],
    params: Dict[str, Any],
) -> np.ndarray:
    """折内若标签只有一类，CatBoost Logloss 会崩：改用常数概率。"""
    uniq = np.unique(y_tr)
    if len(uniq) < 2:
        const = float(uniq[0]) if len(uniq) == 1 else 0.5
        logger.warning("ordinal boundary: single-class train (value=%s), use const p=%.3f", uniq, const)
        return np.full(len(X_va), const, dtype=float)

    model = CatBoostClassifier(**params)
    fit_kw: Dict[str, Any] = {"cat_features": list(cats)}
    # eval_set 也需要两类，否则 early stopping / Logloss 可能失败
    if len(np.unique(y_va)) >= 2:
        fit_kw["eval_set"] = (X_va, y_va)
        fit_kw["early_stopping_rounds"] = TASK3_EARLY_STOPPING_ROUNDS
        fit_kw["use_best_model"] = True
    model.fit(X_tr, y_tr, **fit_kw)
    proba = model.predict_proba(X_va)
    classes = [int(c) for c in model.classes_]
    if 1 in classes:
        return proba[:, classes.index(1)]
    if 0 in classes:
        return 1.0 - proba[:, classes.index(0)]
    return proba[:, -1]


def _fit_multiclass_safe(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cats: Sequence[int],
    params: Dict[str, Any],
    n_classes: int = 10,
) -> np.ndarray:
    """返回 (n_va, n_classes)；缺类填 0，单类则 one-hot。"""
    out = np.zeros((len(X_va), n_classes), dtype=float)
    uniq = np.unique(y_tr)
    if len(uniq) < 2:
        c = int(uniq[0]) if len(uniq) == 1 else 0
        c = int(np.clip(c, 0, n_classes - 1))
        logger.warning("ordinal multiclass: single-class train (class=%d), one-hot fallback", c)
        out[:, c] = 1.0
        return out

    mc_params = {
        **params,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        # 不强制 classes_count：折内可能缺某些评分档
    }
    model = CatBoostClassifier(**mc_params)
    fit_kw: Dict[str, Any] = {"cat_features": list(cats)}
    if len(np.unique(y_va)) >= 2:
        fit_kw["eval_set"] = (X_va, y_va)
        fit_kw["early_stopping_rounds"] = TASK3_EARLY_STOPPING_ROUNDS
        fit_kw["use_best_model"] = True
    model.fit(X_tr, y_tr, **fit_kw)
    proba = model.predict_proba(X_va)
    for i, c in enumerate(model.classes_):
        ci = int(c)
        if 0 <= ci < n_classes:
            out[:, ci] = proba[:, i]
    # 数值稳定：若某行全 0（不应发生），均匀分布
    row_sum = out.sum(axis=1, keepdims=True)
    miss = row_sum.ravel() <= 0
    if miss.any():
        out[miss] = 1.0 / n_classes
        row_sum = out.sum(axis=1, keepdims=True)
    out = out / np.clip(row_sum, 1e-12, None)
    return out


def oof_ordinal_score(
    X_dev: pd.DataFrame,
    y_bin: pd.Series,
    features: Sequence[str],
    fold_assign: np.ndarray,
    n_folds: int = TASK3_N_FOLDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    score = raw_score(X_dev).to_numpy().astype(int)
    score = np.clip(score, 1, 10)
    y = y_bin.to_numpy().astype(int)
    sample_index = X_dev.index.to_numpy().astype(int)
    n = len(X_dev)

    # 累积边界：score >= k for k=2..10
    thresholds = list(range(2, 11))
    cum_oof = np.zeros((n, len(thresholds)), dtype=float)
    dist_oof = np.zeros((n, 10), dtype=float)

    params = {
        "loss_function": "Logloss",
        "iterations": 400,
        "depth": 5,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "allow_writing_files": False,
    }

    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        state = fit_cat_preprocessor(X_dev.loc[tr], features)
        X_tr = transform_cat(X_dev.loc[tr], state)
        X_va = transform_cat(X_dev.loc[va], state)
        cats = cat_indices(state)

        for j, thr in enumerate(thresholds):
            y_tr = (score[tr] >= thr).astype(int)
            y_va = (score[va] >= thr).astype(int)
            cum_oof[va, j] = _fit_binary_safe(X_tr, y_tr, X_va, y_va, cats, params)

        # 10-class distribution（标签 0..9）
        y_mc_tr = score[tr] - 1
        y_mc_va = score[va] - 1
        dist_oof[va] = _fit_multiclass_safe(X_tr, y_mc_tr, X_va, y_mc_va, cats, params, n_classes=10)

    cum_oof = _enforce_monotone_cum(cum_oof)
    # P(score > 5) = P(score >= 6)
    p_ge6 = cum_oof[:, thresholds.index(6)]
    p_dist = dist_oof[:, 5:].sum(axis=1)  # scores 6..10

    acc_cum = float(((p_ge6 >= 0.5).astype(int) == y).mean())
    acc_dist = float(((p_dist >= 0.5).astype(int) == y).mean())
    use_dist = acc_dist > acc_cum
    p1 = p_dist if use_dist else p_ge6

    oof = pd.DataFrame(
        {
            "sample_index": sample_index,
            "true_label": y,
            "true_score": score,
            "prob_0": 1.0 - p1,
            "prob_1": p1,
            "prob_cum_ge6": p_ge6,
            "prob_dist_gt5": p_dist,
            "fold": fold_assign,
            "predicted_label": (p1 >= 0.5).astype(int),
        }
    )
    dist_df = pd.DataFrame(
        {"sample_index": sample_index, **{f"p_score_{k}": dist_oof[:, k - 1] for k in range(1, 11)}}
    )
    metrics = {
        "method": "multiclass_dist" if use_dist else "cumulative_boundary",
        "acc_cumulative": acc_cum,
        "acc_distribution": acc_dist,
        **binary_classification_metrics(y, (p1 >= 0.5).astype(int), p1),
    }
    return oof, dist_df, metrics


def persist_m3(oof: pd.DataFrame, dist: pd.DataFrame, metrics: Dict[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "models" / "m3_ordinal_oof.csv", oof)
    atomic_write_csv(output_dir / "models" / "m3_score_distribution_oof.csv", dist)
    atomic_write_json(output_dir / "models" / "m3_metrics.json", metrics)
