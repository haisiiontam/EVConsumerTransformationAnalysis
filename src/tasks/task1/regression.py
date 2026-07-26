"""M3：有序回归 + 阈值映射 + Softmax 伪概率。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.config import RANDOM_SEED, TASK1_EARLY_STOPPING_ROUNDS, TASK1_LABEL_ORDER
from src.tasks.task1.metrics_ext import (
    LABEL_TO_ORD,
    evaluate_multiclass,
    normalize_label_array,
    pred_from_proba,
    to_ord,
)
from src.tasks.task1.preprocess import cat_indices, fit_cat_preprocessor, transform_cat

logger = logging.getLogger(__name__)


def score_to_label(score: np.ndarray, t_lm: float, t_mh: float) -> np.ndarray:
    out = np.empty(len(score), dtype=object)
    out[score < t_lm] = "Low"
    out[(score >= t_lm) & (score < t_mh)] = "Medium"
    out[score >= t_mh] = "High"
    return out


def score_to_soft_proba(score: np.ndarray, temperature: float) -> np.ndarray:
    # logit_k = -|pred - k| / T
    centers = np.array([0.0, 1.0, 2.0])
    logits = -np.abs(score.reshape(-1, 1) - centers.reshape(1, -1)) / max(temperature, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def oof_ordinal_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    tag: str = "M3_reg",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    X_pos = X.reset_index(drop=True)
    y_pos = pd.Series(normalize_label_array(y.reset_index(drop=True)))
    y_ord = to_ord(y_pos).astype(float)
    oof_score = np.zeros(len(X_pos), dtype=float)
    fold_rows: List[Dict[str, Any]] = []
    params = {
        "loss_function": "RMSE",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "allow_writing_files": False,
    }
    for fold_id, (tr, va) in enumerate(folds):
        state = fit_cat_preprocessor(X_pos.iloc[tr], features)
        X_tr, X_va = transform_cat(X_pos.iloc[tr], state), transform_cat(X_pos.iloc[va], state)
        model = CatBoostRegressor(**params)
        model.fit(
            X_tr,
            y_ord[tr],
            cat_features=cat_indices(state),
            eval_set=(X_va, y_ord[va]),
            early_stopping_rounds=TASK1_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        pred = model.predict(X_va)
        oof_score[va] = pred
        # 临时用 0.5/1.5 看折内
        lab = score_to_label(pred, 0.5, 1.5)
        m = evaluate_multiclass(y_pos.iloc[va], lab)
        fold_rows.append({"fold": fold_id, "model": tag, "accuracy@0.5_1.5": m["accuracy"]})
        logger.info("%s fold=%d rough_acc=%.4f", tag, fold_id, m["accuracy"])
    return oof_score, fold_rows


def search_reg_thresholds(
    y: Sequence[Any],
    scores: np.ndarray,
    coarse: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray]:
    y_n = normalize_label_array(y)
    step = 0.05 if coarse else 0.025
    t_lm_grid = np.round(np.arange(0.25, 1.00 + 1e-12, step), 10)
    t_mh_grid = np.round(np.arange(1.00, 1.75 + 1e-12, step), 10)
    best = None
    for t_lm in t_lm_grid:
        for t_mh in t_mh_grid:
            if t_lm >= t_mh:
                continue
            pred = score_to_label(scores, float(t_lm), float(t_mh))
            m = evaluate_multiclass(y_n, pred)
            row = {"t_low_medium": float(t_lm), "t_medium_high": float(t_mh), **m}
            if best is None or m["accuracy"] > best["accuracy"] + 1e-12:
                best = row
            elif abs(m["accuracy"] - best["accuracy"]) <= 1e-12 and m["qwk"] > best["qwk"]:
                best = row
    assert best is not None
    pred = score_to_label(scores, best["t_low_medium"], best["t_medium_high"])
    return best, pred


def search_reg_soft_proba(
    y: Sequence[Any],
    scores: np.ndarray,
    coarse: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray]:
    y_n = normalize_label_array(y)
    temps = [0.25, 0.5, 0.75, 1.0] if coarse else list(np.round(np.arange(0.10, 1.00 + 1e-12, 0.05), 10))
    best = None
    best_p = None
    for T in temps:
        p = score_to_soft_proba(scores, float(T))
        pred = pred_from_proba(p)
        m = evaluate_multiclass(y_n, pred, p)
        row = {"temperature": float(T), **m}
        if best is None or m["accuracy"] > best["accuracy"] + 1e-12:
            best, best_p = row, p
    assert best is not None and best_p is not None
    return best, best_p
