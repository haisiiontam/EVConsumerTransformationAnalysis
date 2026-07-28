from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.common.fold_checkpoints import PROB_COLS_BINARY, fold_is_complete, save_fold_checkpoint
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.metrics import binary_classification_metrics
from src.config import RANDOM_SEED, TASK3_EARLY_STOPPING_ROUNDS, TASK3_N_FOLDS
from src.tasks.task3.features import raw_score
from src.tasks.task3.preprocessing import cat_indices, fit_cat_preprocessor, transform_cat

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray, center: float, temperature: float) -> np.ndarray:
    z = (x - center) / max(temperature, 1e-6)
    return 1.0 / (1.0 + np.exp(-z))


def oof_regression_score(
    X_dev: pd.DataFrame,
    y_bin: pd.Series,
    features: Sequence[str],
    fold_assign: np.ndarray,
    ckpt_dir: Path,
    params: Dict[str, Any] | None = None,
    n_folds: int = TASK3_N_FOLDS,
    resume_folds: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """回归预测 range_anxiety_score；X 中不含该列。"""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    score = raw_score(X_dev).to_numpy().astype(float)
    y = y_bin.to_numpy().astype(int)
    sample_index = X_dev.index.to_numpy().astype(int)
    params = params or {
        "loss_function": "RMSE",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "allow_writing_files": False,
    }

    pred_score = np.zeros(len(X_dev), dtype=float)
    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        if resume_folds and fold_is_complete(ckpt_dir, fold_id, int(va.sum()), PROB_COLS_BINARY):
            pred = pd.read_csv(ckpt_dir / f"fold_{fold_id}_predictions.csv")
            idx_map = {int(s): i for i, s in enumerate(sample_index)}
            for _, row in pred.iterrows():
                pred_score[idx_map[int(row["sample_index"])]] = float(row["pred_score"])
            continue
        state = fit_cat_preprocessor(X_dev.loc[tr], features)
        model = CatBoostRegressor(**params)
        model.fit(
            transform_cat(X_dev.loc[tr], state),
            score[tr],
            cat_features=cat_indices(state),
            eval_set=(transform_cat(X_dev.loc[va], state), score[va]),
            early_stopping_rounds=TASK3_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        ps = model.predict(transform_cat(X_dev.loc[va], state))
        pred_score[va] = ps

        p1 = _sigmoid(ps, 5.5, 1.0)
        save_fold_checkpoint(
            ckpt_dir,
            fold_id,
            sample_index=sample_index[va],
            predicted_label=(ps >= 5.5).astype(int),
            proba=np.column_stack([1 - p1, p1]),
            metrics=binary_classification_metrics(y[va], (ps >= 5.5).astype(int), p1),
            best_iteration=int(model.get_best_iteration() or params["iterations"]),
            params=params,
            features=features,
            model_name="m2_reg",
            prob_cols=PROB_COLS_BINARY,
            y_true=y[va],
            pred_score=ps,
        )


    thr_rows = []
    best_thr, best_acc = 5.5, -1.0
    for thr in np.round(np.arange(4.0, 7.0 + 1e-12, 0.05), 10):
        pred = (pred_score >= thr).astype(int)
        acc = float((pred == y).mean())
        thr_rows.append({"threshold_score": float(thr), "accuracy": acc})
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)


    p_sig = _sigmoid(pred_score, best_thr, 0.8)
    try:
        lr = LogisticRegression(max_iter=200)
        lr.fit(pred_score.reshape(-1, 1), y)
        p_lr = lr.predict_proba(pred_score.reshape(-1, 1))[:, 1]
    except Exception:
        p_lr = p_sig
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pred_score, y)
        p_iso = iso.predict(pred_score)
    except Exception:
        p_iso = p_sig


    cands = {
        "sigmoid": p_sig,
        "logistic": p_lr,
        "isotonic": p_iso,
    }
    best_name, best_p = "sigmoid", p_sig
    best_map_acc = -1.0
    for name, p in cands.items():
        acc = float(((p >= 0.5).astype(int) == y).mean())
        if acc > best_map_acc:
            best_map_acc, best_name, best_p = acc, name, p

    oof = pd.DataFrame(
        {
            "sample_index": sample_index,
            "true_label": y,
            "true_score": score,
            "pred_score": pred_score,
            "prob_0": 1.0 - best_p,
            "prob_1": best_p,
            "fold": fold_assign,
            "predicted_label": (best_p >= 0.5).astype(int),
        }
    )
    info = {
        "best_score_threshold": best_thr,
        "score_threshold_accuracy": best_acc,
        "score_to_prob_method": best_name,
        "score_to_prob_accuracy": best_map_acc,
        "rmse": float(np.sqrt(np.mean((pred_score - score) ** 2))),
        "mae": float(np.mean(np.abs(pred_score - score))),
    }
    return oof, info


def persist_m2_outputs(oof: pd.DataFrame, info: Dict[str, Any], thr_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "tuning").mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "models" / "m2_regression_oof.csv", oof)
    atomic_write_csv(output_dir / "tuning" / "m2_regression_threshold.csv", thr_df)
    atomic_write_json(output_dir / "tuning" / "m2_score_to_probability.json", info)

    rows = []
    for s in range(1, 11):
        m = oof["true_score"].round().astype(int) == s
        if not m.any():
            continue
        sub = oof.loc[m]
        rows.append(
            {
                "score": s,
                "n": int(m.sum()),
                "accuracy": float((sub["predicted_label"] == sub["true_label"]).mean()),
                "mean_prob1": float(sub["prob_1"].mean()),
                "std_prob1": float(sub["prob_1"].std()),
            }
        )
    atomic_write_csv(output_dir / "models" / "m2_score_level_metrics.csv", pd.DataFrame(rows))
