"""任务二快速模型：Logistic + CatBoost OOF。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from src.common.fold_checkpoints import (
    PROB_COLS_BINARY,
    assemble_oof_from_checkpoints,
    fold_is_complete,
    save_fold_checkpoint,
)
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.config import TASK2_CATBOOST_COMMON, TASK2_EARLY_STOPPING_ROUNDS
from src.tasks.task2.metrics_ext import rich_binary_metrics
from src.tasks.task2.preprocessing import (
    fit_tabular,
    fit_task2_preprocessor,
    get_cat_indices,
    transform_tabular,
    transform_task2_data,
)

logger = logging.getLogger(__name__)


def _prob1_cb(model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = [int(c) for c in model.classes_]
    return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]


def _oof_logistic(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    features: Sequence[str],
    fold_assign: np.ndarray,
    n_folds: int,
    ckpt_dir: Path,
    class_weight: str | None = None,
    model_name: str = "logistic",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_index = X_dev.index.to_numpy().astype(int)
    y_arr = y_dev.to_numpy().astype(int)
    n = len(X_dev)
    cw = None if class_weight in (None, "none") else class_weight

    for fold_id in range(n_folds):
        va_mask = fold_assign == fold_id
        tr_mask = ~va_mask
        expected_n = int(va_mask.sum())
        if fold_is_complete(ckpt_dir, fold_id, expected_n, PROB_COLS_BINARY):
            continue
        state = fit_tabular(X_dev.loc[tr_mask], features, scale=True)
        X_tr = transform_tabular(X_dev.loc[tr_mask], state)
        X_va = transform_tabular(X_dev.loc[va_mask], state)
        y_tr, y_va = y_arr[tr_mask], y_arr[va_mask]
        lr = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=500,
            class_weight=cw,
            random_state=42,
        )
        lr.fit(X_tr, y_tr)
        p1 = lr.predict_proba(X_va)[:, 1]
        pred = (p1 >= 0.5).astype(int)
        m = rich_binary_metrics(y_va, pred, p1)
        save_fold_checkpoint(
            ckpt_dir,
            fold_id,
            sample_index=sample_index[va_mask],
            predicted_label=pred,
            proba=p1,
            metrics=m,
            best_iteration=None,
            params={"class_weight": cw, "penalty": "l2"},
            features=list(features),
            model_name=model_name,
            prob_cols=PROB_COLS_BINARY,
            y_true=y_va,
        )

    oof_p, oof_pred, fold_rows, _ = assemble_oof_from_checkpoints(
        ckpt_dir, n, n_folds, sample_index, PROB_COLS_BINARY
    )
    oof_df = pd.DataFrame(
        {
            "sample_index": sample_index,
            "true_label": y_arr,
            "predicted_label": oof_pred.astype(int),
            "prob_0": oof_p[:, 0],
            "prob_1": oof_p[:, 1],
            "fold": fold_assign,
        }
    )
    return oof_df, pd.DataFrame(fold_rows)


def _oof_catboost(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    features: Sequence[str],
    fold_assign: np.ndarray,
    n_folds: int,
    ckpt_dir: Path,
    balanced: bool = False,
    model_name: str = "catboost",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_index = X_dev.index.to_numpy().astype(int)
    y_arr = y_dev.to_numpy().astype(int)
    n = len(X_dev)
    params = dict(TASK2_CATBOOST_COMMON)
    if balanced:
        params["auto_class_weights"] = "Balanced"

    for fold_id in range(n_folds):
        va_mask = fold_assign == fold_id
        tr_mask = ~va_mask
        expected_n = int(va_mask.sum())
        if fold_is_complete(ckpt_dir, fold_id, expected_n, PROB_COLS_BINARY):
            continue
        state = fit_task2_preprocessor(X_dev.loc[tr_mask], features)
        X_tr = transform_task2_data(X_dev.loc[tr_mask], state)
        X_va = transform_task2_data(X_dev.loc[va_mask], state)
        y_tr, y_va = y_arr[tr_mask], y_arr[va_mask]
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            cat_features=get_cat_indices(state),
            eval_set=(X_va, y_va),
            early_stopping_rounds=TASK2_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        best_iter = int(model.get_best_iteration() or params["iterations"])
        p1 = _prob1_cb(model, X_va)
        pred = (p1 >= 0.5).astype(int)
        m = rich_binary_metrics(y_va, pred, p1)
        save_fold_checkpoint(
            ckpt_dir,
            fold_id,
            sample_index=sample_index[va_mask],
            predicted_label=pred,
            proba=p1,
            metrics=m,
            best_iteration=best_iter,
            params=params,
            features=list(features),
            model_name=model_name,
            prob_cols=PROB_COLS_BINARY,
            y_true=y_va,
        )

    oof_p, oof_pred, fold_rows, _ = assemble_oof_from_checkpoints(
        ckpt_dir, n, n_folds, sample_index, PROB_COLS_BINARY
    )
    oof_df = pd.DataFrame(
        {
            "sample_index": sample_index,
            "true_label": y_arr,
            "predicted_label": oof_pred.astype(int),
            "prob_0": oof_p[:, 0],
            "prob_1": oof_p[:, 1],
            "fold": fold_assign,
        }
    )
    return oof_df, pd.DataFrame(fold_rows)


def run_quick_models(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    fold_assign: np.ndarray,
    output_dir: Path,
    feat_a: Sequence[str],
    feat_b: Sequence[str],
    n_folds: int = 5,
    resume_folds: bool = True,
) -> Dict[str, Any]:
    """Logistic (none/balanced) + CatBoost (A/B) OOF。"""
    output_dir = Path(output_dir)
    oof_dir = output_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / "checkpoints" / "models"

    results: Dict[str, Any] = {}
    metric_rows: List[Dict[str, Any]] = []

    specs = [
        ("logistic_none", feat_b, {"class_weight": "none"}, "logistic"),
        ("logistic_balanced", feat_b, {"class_weight": "balanced"}, "logistic_bal"),
        ("catboost_a", feat_a, {"balanced": False}, "catboost_a"),
        ("catboost_b", feat_b, {"balanced": False}, "catboost_b"),
    ]

    for key, feats, kw, mname in specs:
        sub = ckpt / key
        if key.startswith("logistic"):
            oof, folds = _oof_logistic(
                X_dev,
                y_dev,
                feats,
                fold_assign,
                n_folds,
                sub,
                class_weight=kw.get("class_weight"),
                model_name=mname,
            )
        else:
            oof, folds = _oof_catboost(
                X_dev,
                y_dev,
                feats,
                fold_assign,
                n_folds,
                sub,
                balanced=kw.get("balanced", False),
                model_name=mname,
            )
        fname = f"{key}_oof_predictions.csv"
        atomic_write_csv(oof_dir / fname, oof)
        atomic_write_csv(oof_dir / f"{key}_fold_metrics.csv", folds)
        m = rich_binary_metrics(oof["true_label"], oof["predicted_label"], oof["prob_1"])
        m["model"] = key
        m["n_features"] = len(feats)
        metric_rows.append(m)
        results[key] = oof

    comp = pd.DataFrame(metric_rows)
    atomic_write_csv(oof_dir / "quick_models_comparison.csv", comp)
    atomic_write_json(oof_dir / "quick_models_metrics.json", {"models": metric_rows})
    # 兼容 audit 路径
    if "logistic_none" in results:
        atomic_write_csv(oof_dir / "logistic_oof_predictions.csv", results["logistic_none"])
    if "catboost_b" in results:
        atomic_write_csv(oof_dir / "catboost_b_oof_predictions.csv", results["catboost_b"])
    logger.info("quick models done n=%d", len(metric_rows))
    return {"models": results, "comparison": comp, "metric_rows": metric_rows}
