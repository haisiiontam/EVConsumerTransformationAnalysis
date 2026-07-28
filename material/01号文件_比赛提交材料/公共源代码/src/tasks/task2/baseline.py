"""任务二基线：B0 折内多数类 / B1 常数先验 / B2 官方五特征 CatBoost。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.common.fold_checkpoints import (
    PROB_COLS_BINARY,
    assemble_oof_from_checkpoints,
    fold_is_complete,
    save_fold_checkpoint,
)
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.metrics import majority_baseline
from src.config import (
    RANDOM_SEED,
    TASK2_BASE_FEATURES,
    TASK2_CATBOOST_COMMON,
    TASK2_EARLY_STOPPING_ROUNDS,
)
from src.tasks.task2.metrics_ext import rich_binary_metrics
from src.tasks.task2.preprocessing import (
    fit_task2_preprocessor,
    get_cat_indices,
    transform_task2_data,
)

logger = logging.getLogger(__name__)


def _prob1(model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = [int(c) for c in model.classes_]
    return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]


def _oof_majority_per_fold(
    y_dev: pd.Series,
    fold_assign: np.ndarray,
    n_folds: int,
    ckpt_dir: Path,
    model_name: str = "B0_majority",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """B0：每折仅用训练折多数类预测验证折。"""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_index = y_dev.index.to_numpy().astype(int)
    y_arr = y_dev.to_numpy().astype(int)
    n = len(y_dev)

    for fold_id in range(n_folds):
        va_mask = fold_assign == fold_id
        tr_mask = ~va_mask
        expected_n = int(va_mask.sum())
        if fold_is_complete(ckpt_dir, fold_id, expected_n, PROB_COLS_BINARY):
            continue
        y_tr = y_arr[tr_mask]
        maj = int(pd.Series(y_tr).value_counts().idxmax())
        p1 = np.full(int(va_mask.sum()), float(maj), dtype=float)
        pred = np.full(int(va_mask.sum()), maj, dtype=int)
        m = rich_binary_metrics(y_arr[va_mask], pred, p1)
        save_fold_checkpoint(
            ckpt_dir,
            fold_id,
            sample_index=sample_index[va_mask],
            predicted_label=pred,
            proba=p1,
            metrics=m,
            best_iteration=None,
            params={"rule": "train_fold_majority", "majority_class": maj},
            features=[],
            model_name=model_name,
            prob_cols=PROB_COLS_BINARY,
            y_true=y_arr[va_mask],
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


def _oof_constant_prior(
    y_dev: pd.Series,
    fold_assign: np.ndarray,
    n_folds: int,
    ckpt_dir: Path,
    model_name: str = "B1_constant_prior",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """B1：每折用训练折 class-1 比例作为常数概率。"""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_index = y_dev.index.to_numpy().astype(int)
    y_arr = y_dev.to_numpy().astype(int)
    n = len(y_dev)

    for fold_id in range(n_folds):
        va_mask = fold_assign == fold_id
        tr_mask = ~va_mask
        expected_n = int(va_mask.sum())
        if fold_is_complete(ckpt_dir, fold_id, expected_n, PROB_COLS_BINARY):
            continue
        y_tr = y_arr[tr_mask]
        prior = float(y_tr.mean())
        p1 = np.full(int(va_mask.sum()), prior, dtype=float)
        pred = (p1 >= 0.5).astype(int)
        m = rich_binary_metrics(y_arr[va_mask], pred, p1)
        save_fold_checkpoint(
            ckpt_dir,
            fold_id,
            sample_index=sample_index[va_mask],
            predicted_label=pred,
            proba=p1,
            metrics=m,
            best_iteration=None,
            params={"rule": "train_fold_prior", "prior_1": prior},
            features=[],
            model_name=model_name,
            prob_cols=PROB_COLS_BINARY,
            y_true=y_arr[va_mask],
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


def _oof_catboost_official(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    fold_assign: np.ndarray,
    n_folds: int,
    ckpt_dir: Path,
    features: Sequence[str] | None = None,
    model_name: str = "B2_catboost_official",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """B2：官方五特征 CatBoost OOF。"""
    features = list(features or TASK2_BASE_FEATURES)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_index = X_dev.index.to_numpy().astype(int)
    y_arr = y_dev.to_numpy().astype(int)
    n = len(X_dev)
    params = dict(TASK2_CATBOOST_COMMON)

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
        p1 = _prob1(model, X_va)
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
            features=features,
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


def run_baselines(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    fold_assign: np.ndarray,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """运行 B0/B1/B2 并写入 baseline/。"""
    output_dir = Path(output_dir)
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = output_dir / "checkpoints" / "baseline"

    maj = majority_baseline(y_dev)
    atomic_write_json(baseline_dir / "majority_baseline.json", maj)

    oof_b0, folds_b0 = _oof_majority_per_fold(
        y_dev, fold_assign, n_folds, ckpt_root / "B0", "B0_majority"
    )
    oof_b1, folds_b1 = _oof_constant_prior(
        y_dev, fold_assign, n_folds, ckpt_root / "B1", "B1_constant_prior"
    )
    oof_b2, folds_b2 = _oof_catboost_official(
        X_dev, y_dev, fold_assign, n_folds, ckpt_root / "B2", TASK2_BASE_FEATURES
    )

    # 汇总 metrics
    rows = []
    for name, oof, folds in [
        ("B0_majority", oof_b0, folds_b0),
        ("B1_constant_prior", oof_b1, folds_b1),
        ("B2_catboost_official", oof_b2, folds_b2),
    ]:
        m = rich_binary_metrics(oof["true_label"], oof["predicted_label"], oof["prob_1"])
        m["baseline"] = name
        m["fold_mean_acc"] = float(folds["accuracy"].mean()) if "accuracy" in folds else None
        rows.append(m)
        atomic_write_csv(baseline_dir / f"{name}_oof_predictions.csv", oof)
        atomic_write_csv(baseline_dir / f"{name}_fold_metrics.csv", folds)

    comp = pd.DataFrame(rows)
    atomic_write_csv(baseline_dir / "baseline_comparison.csv", comp)
    atomic_write_json(
        baseline_dir / "baseline_metrics.json",
        {
            "majority_baseline": maj,
            "baselines": rows,
            "n_folds": n_folds,
            "seed": seed,
        },
    )
    # 兼容 task3 命名
    atomic_write_csv(baseline_dir / "baseline_oof_predictions.csv", oof_b0)
    logger.info("task2 baselines done: B0 acc=%.4f B2 acc=%.4f", rows[0]["accuracy"], rows[2]["accuracy"])
    return {
        "B0": oof_b0,
        "B1": oof_b1,
        "B2": oof_b2,
        "majority": maj,
        "comparison": comp,
    }
