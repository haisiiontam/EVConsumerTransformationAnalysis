from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config import RANDOM_SEED, TASK1_EARLY_STOPPING_ROUNDS, TASK1_LABEL_ORDER
from src.tasks.task1.checkpoints import (
    assemble_oof_from_checkpoints,
    fold_is_complete,
    load_fold_predictions,
    save_fold_checkpoint,
)
from src.tasks.task1.metrics_ext import evaluate_multiclass, normalize_label_array, pred_from_proba
from src.tasks.task1.preprocess import cat_indices, fit_cat_preprocessor, transform_cat

logger = logging.getLogger(__name__)


def _binary_labels(y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    y_n = normalize_label_array(y)
    ge_med = np.array([0 if lab == "Low" else 1 for lab in y_n], dtype=int)
    ge_high = np.array([1 if lab == "High" else 0 for lab in y_n], dtype=int)
    return ge_med, ge_high


def ordinal_probs_from_boundaries(p_ge_med: np.ndarray, p_ge_high: np.ndarray) -> np.ndarray:
    p_ge_high = np.minimum(p_ge_high, p_ge_med)
    p_low = 1.0 - p_ge_med
    p_med = p_ge_med - p_ge_high
    p_high = p_ge_high
    p = np.column_stack([p_low, p_med, p_high])
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return p / s


def oof_ordinal_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    params: Dict[str, Any] | None = None,
    tag: str = "M2_ordinal",
    checkpoint_dir: Optional[Path] = None,
    sample_index: Optional[Sequence[int]] = None,
    resume_folds: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    X_pos = X.reset_index(drop=True)
    y_pos = pd.Series(normalize_label_array(y.reset_index(drop=True)))
    ge_med_all, ge_high_all = _binary_labels(y_pos)
    n = len(X_pos)
    oof_p = np.zeros((n, 3), dtype=float)
    fold_rows: List[Dict[str, Any]] = []
    if sample_index is None:
        sample_index = list(range(n))
    sample_index = np.asarray(sample_index).astype(int)

    base = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "allow_writing_files": False,
    }
    if params:
        base.update(params)

    if checkpoint_dir is not None and resume_folds:
        ckpt = Path(checkpoint_dir)
        if all(fold_is_complete(ckpt, fid, expected_n=len(va)) for fid, (_, va) in enumerate(folds)):
            oof_p, oof_pred, fold_rows, _ = assemble_oof_from_checkpoints(
                ckpt, n, len(folds), sample_index
            )
            return oof_p, oof_pred, fold_rows

    for fold_id, (tr, va) in enumerate(folds):
        if checkpoint_dir is not None and resume_folds and fold_is_complete(
            Path(checkpoint_dir), fold_id, expected_n=len(va)
        ):
            pred_df = load_fold_predictions(Path(checkpoint_dir), fold_id)
            met = json.loads(
                (Path(checkpoint_dir) / f"fold_{fold_id}_metrics.json").read_text(encoding="utf-8")
            )
            idx_to_pos = {int(s): i for i, s in enumerate(sample_index)}
            for _, row in pred_df.iterrows():
                pos = idx_to_pos[int(row["sample_index"])]
                oof_p[pos] = [row["prob_Low"], row["prob_Medium"], row["prob_High"]]
            fold_rows.append(met)
            logger.info("%s fold=%d resumed from checkpoint", tag, fold_id)
            continue

        state = fit_cat_preprocessor(X_pos.iloc[tr], features)
        X_tr, X_va = transform_cat(X_pos.iloc[tr], state), transform_cat(X_pos.iloc[va], state)
        cats = cat_indices(state)

        m_med = CatBoostClassifier(**base)
        m_med.fit(
            X_tr,
            ge_med_all[tr],
            cat_features=cats,
            eval_set=(X_va, ge_med_all[va]),
            early_stopping_rounds=TASK1_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        m_high = CatBoostClassifier(**base)
        m_high.fit(
            X_tr,
            ge_high_all[tr],
            cat_features=cats,
            eval_set=(X_va, ge_high_all[va]),
            early_stopping_rounds=TASK1_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        p_ge_m = m_med.predict_proba(X_va)[:, 1]
        p_ge_h = m_high.predict_proba(X_va)[:, 1]
        proba = ordinal_probs_from_boundaries(p_ge_m, p_ge_h)
        pred = pred_from_proba(proba)
        oof_p[va] = proba
        m = evaluate_multiclass(y_pos.iloc[va], pred, proba)
        fold_rows.append({"fold": fold_id, "model": tag, **m})
        logger.info("%s fold=%d acc=%.4f medium_recall=%.4f", tag, fold_id, m["accuracy"], m["medium_recall"])
        if checkpoint_dir is not None:
            save_fold_checkpoint(
                Path(checkpoint_dir),
                fold_id,
                sample_index=sample_index[va],
                predicted_label=pred,
                proba=proba,
                metrics=m,
                best_iteration=None,
                params=base,
                features=features,
                model_name=tag,
            )

    oof_pred = pred_from_proba(oof_p)
    return oof_p, oof_pred, fold_rows
