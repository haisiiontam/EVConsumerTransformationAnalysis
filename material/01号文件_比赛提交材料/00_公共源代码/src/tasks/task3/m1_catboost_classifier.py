"""M1：CatBoost 二分类 + 可选 Optuna。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from src.common.fold_checkpoints import (
    PROB_COLS_BINARY,
    assemble_oof_from_checkpoints,
    fold_is_complete,
    persist_assembled_oof,
    save_fold_checkpoint,
)
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.metrics import binary_classification_metrics
from src.config import (
    RANDOM_SEED,
    TASK3_CATBOOST_COMMON,
    TASK3_EARLY_STOPPING_ROUNDS,
    TASK3_N_FOLDS,
    TASK3_OPTUNA_COARSE_FOLDS,
    TASK3_OPTUNA_TRIALS_CB,
)
from src.tasks.task3.preprocessing import cat_indices, fit_cat_preprocessor, transform_cat

logger = logging.getLogger(__name__)


def _prob1(model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = [int(c) for c in model.classes_]
    return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]


def oof_catboost_binary(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    features: Sequence[str],
    params: Dict[str, Any],
    fold_assign: np.ndarray,
    ckpt_dir: Path,
    model_name: str = "m1",
    resume_folds: bool = True,
    n_folds: int = TASK3_N_FOLDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_index = X_dev.index.to_numpy().astype(int)
    y_arr = y_dev.to_numpy().astype(int)
    n = len(X_dev)

    for fold_id in range(n_folds):
        va_mask = fold_assign == fold_id
        tr_mask = ~va_mask
        expected_n = int(va_mask.sum())
        if resume_folds and fold_is_complete(ckpt_dir, fold_id, expected_n, PROB_COLS_BINARY):
            logger.info("resume fold %s fold=%d", model_name, fold_id)
            continue
        state = fit_cat_preprocessor(X_dev.loc[tr_mask], features)
        X_tr = transform_cat(X_dev.loc[tr_mask], state)
        X_va = transform_cat(X_dev.loc[va_mask], state)
        y_tr, y_va = y_arr[tr_mask], y_arr[va_mask]
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_indices(state),
            eval_set=(X_va, y_va),
            early_stopping_rounds=TASK3_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        best_iter = int(model.get_best_iteration() or params.get("iterations", 500))
        p1 = _prob1(model, X_va)
        pred = (p1 >= 0.5).astype(int)
        m = binary_classification_metrics(y_va, pred, p1)
        proba = np.column_stack([1.0 - p1, p1])
        save_fold_checkpoint(
            ckpt_dir,
            fold_id,
            sample_index=sample_index[va_mask],
            predicted_label=pred,
            proba=proba,
            metrics=m,
            best_iteration=best_iter,
            params=params,
            features=features,
            model_name=model_name,
            prob_cols=PROB_COLS_BINARY,
            y_true=y_va,
        )

    oof_p, oof_pred, fold_rows, best_iters = assemble_oof_from_checkpoints(
        ckpt_dir, n, n_folds, sample_index, PROB_COLS_BINARY
    )
    oof_df = pd.DataFrame(
        {
            "sample_index": sample_index,
            "true_label": y_arr,
            "predicted_label": oof_pred,
            "prob_0": oof_p[:, 0],
            "prob_1": oof_p[:, 1],
            "fold": fold_assign,
        }
    )
    return oof_df, pd.DataFrame(fold_rows), best_iters


def run_optuna_catboost(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    features: Sequence[str],
    fold_assign: np.ndarray,
    output_dir: Path,
    n_trials: int = TASK3_OPTUNA_TRIALS_CB,
    quick: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        best = dict(TASK3_CATBOOST_COMMON)
        atomic_write_json(output_dir / "m1_best_params.json", best)
        atomic_write_csv(
            output_dir / "catboost_trials.csv",
            pd.DataFrame([{"trial": 0, "accuracy": np.nan, "note": "quick_skip"}]),
        )
        return best

    try:
        import optuna
    except ImportError:
        logger.warning("optuna missing; use TASK3_CATBOOST_COMMON")
        best = dict(TASK3_CATBOOST_COMMON)
        atomic_write_json(output_dir / "m1_best_params.json", best)
        return best

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y = y_dev.to_numpy().astype(int)
    # 粗搜：用前 coarse folds
    coarse_folds = list(range(min(TASK3_OPTUNA_COARSE_FOLDS, int(fold_assign.max()) + 1)))

    def objective(trial: "optuna.Trial") -> float:
        params = {
            **TASK3_CATBOOST_COMMON,
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
            "iterations": trial.suggest_int("iterations", 400, 1500),
            "verbose": False,
        }
        accs = []
        for fold_id in coarse_folds:
            va = fold_assign == fold_id
            tr = ~va
            state = fit_cat_preprocessor(X_dev.loc[tr], features)
            model = CatBoostClassifier(**params)
            model.fit(
                transform_cat(X_dev.loc[tr], state),
                y[tr],
                cat_features=cat_indices(state),
                eval_set=(transform_cat(X_dev.loc[va], state), y[va]),
                early_stopping_rounds=TASK3_EARLY_STOPPING_ROUNDS,
                use_best_model=True,
            )
            p1 = _prob1(model, transform_cat(X_dev.loc[va], state))
            accs.append(float(((p1 >= 0.5).astype(int) == y[va]).mean()))
        return float(np.mean(accs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials_rows = []
    for t in study.trials:
        trials_rows.append({"trial": t.number, "accuracy": t.value, **(t.params or {})})
    atomic_write_csv(output_dir / "catboost_trials.csv", pd.DataFrame(trials_rows))

    # Top5 完整 5 折确认
    top = sorted(study.trials, key=lambda t: (t.value is not None, t.value or -1), reverse=True)[:5]
    conf_rows = []
    best_params = dict(TASK3_CATBOOST_COMMON)
    best_acc = -1.0
    for rank, t in enumerate(top):
        params = {**TASK3_CATBOOST_COMMON, **t.params, "verbose": False, "allow_writing_files": False}
        accs = []
        for fold_id in range(int(fold_assign.max()) + 1):
            va = fold_assign == fold_id
            tr = ~va
            state = fit_cat_preprocessor(X_dev.loc[tr], features)
            model = CatBoostClassifier(**params)
            model.fit(
                transform_cat(X_dev.loc[tr], state),
                y[tr],
                cat_features=cat_indices(state),
                eval_set=(transform_cat(X_dev.loc[va], state), y[va]),
                early_stopping_rounds=TASK3_EARLY_STOPPING_ROUNDS,
                use_best_model=True,
            )
            p1 = _prob1(model, transform_cat(X_dev.loc[va], state))
            accs.append(float(((p1 >= 0.5).astype(int) == y[va]).mean()))
        mean_acc = float(np.mean(accs))
        conf_rows.append({"rank": rank, "mean_acc": mean_acc, "std_acc": float(np.std(accs)), **t.params})
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = params
    atomic_write_csv(output_dir / "catboost_top5_cv.csv", pd.DataFrame(conf_rows))
    atomic_write_json(output_dir / "m1_best_params.json", best_params)
    return best_params
