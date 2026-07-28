"""M4 LightGBM / M5 XGBoost / M6 序数 Logistic。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import (
    RANDOM_SEED,
    TASK1_LABEL_ORDER,
    TASK1_OPTUNA_COARSE_FOLDS,
    TASK1_OPTUNA_TRIALS_LGB,
)
from src.tasks.task1.checkpoints import (
    assemble_oof_from_checkpoints,
    fold_is_complete,
    load_fold_predictions,
    save_fold_checkpoint,
)
from src.tasks.task1.metrics_ext import evaluate_multiclass, normalize_label_array, pred_from_proba
from src.tasks.task1.ordinal import ordinal_probs_from_boundaries
from src.tasks.task1.preprocess import fit_tabular_preprocessor, transform_tabular

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    lgb = None  # type: ignore

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None  # type: ignore

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None  # type: ignore


def _y_codes(y: pd.Series) -> np.ndarray:
    y_n = normalize_label_array(y)
    return np.array([TASK1_LABEL_ORDER.index(lab) for lab in y_n], dtype=int)


def oof_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    params: Optional[Dict[str, Any]] = None,
    tag: str = "M4_lgbm",
    checkpoint_dir: Optional[Path] = None,
    sample_index: Optional[Sequence[int]] = None,
    resume_folds: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    if not HAS_LGBM:
        raise RuntimeError("lightgbm 未安装")
    X_pos = X.reset_index(drop=True)
    y_pos = pd.Series(normalize_label_array(y.reset_index(drop=True)))
    y_code = _y_codes(y_pos)
    n = len(X_pos)
    oof_p = np.zeros((n, 3), dtype=float)
    fold_rows: List[Dict[str, Any]] = []
    if sample_index is None:
        sample_index = list(range(n))
    sample_index = np.asarray(sample_index).astype(int)
    base = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "objective": "multiclass",
        "num_class": 3,
    }
    if params:
        base.update(params)
    # 强制整型，避免 CSV/JSON 恢复后的 710.0
    for ik in ("n_estimators", "num_leaves", "min_child_samples", "random_state", "num_class"):
        if ik in base and base[ik] is not None:
            base[ik] = int(base[ik])

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

        state = fit_tabular_preprocessor(X_pos.iloc[tr], features, scale=False)
        X_tr = transform_tabular(X_pos.iloc[tr], state)
        X_va = transform_tabular(X_pos.iloc[va], state)
        model = lgb.LGBMClassifier(**base)
        model.fit(
            X_tr,
            y_code[tr],
            eval_set=[(X_va, y_code[va])],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        proba = model.predict_proba(X_va)
        out = np.zeros((len(va), 3), dtype=float)
        for i, c in enumerate(model.classes_):
            out[:, int(c)] = proba[:, i]
        oof_p[va] = out
        pred = pred_from_proba(out)
        m = evaluate_multiclass(y_pos.iloc[va], pred, out)
        bi = int(getattr(model, "best_iteration_", base.get("n_estimators", 500)) or 0)
        fold_rows.append({"fold": fold_id, "model": tag, **m, "best_iteration": bi})
        logger.info("%s fold=%d acc=%.4f", tag, fold_id, m["accuracy"])
        if checkpoint_dir is not None:
            save_fold_checkpoint(
                Path(checkpoint_dir),
                fold_id,
                sample_index=sample_index[va],
                predicted_label=pred,
                proba=out,
                metrics=m,
                best_iteration=bi,
                params=base,
                features=features,
                model_name=tag,
            )
    return oof_p, pred_from_proba(oof_p), fold_rows


def optuna_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    seed: int = RANDOM_SEED,
    n_trials: Optional[int] = None,
    storage_path: Optional[Path] = None,
    study_name: str = "task1_lightgbm",
    existing_trials_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    default = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "verbose": -1,
        "objective": "multiclass",
        "num_class": 3,
    }
    if not HAS_OPTUNA or not HAS_LGBM:
        return pd.DataFrame(), default
    n_trials = n_trials or TASK1_OPTUNA_TRIALS_LGB

    if existing_trials_csv is not None and Path(existing_trials_csv).exists():
        df_exist = pd.read_csv(existing_trials_csv)
        if "trial" in df_exist.columns and df_exist["trial"].nunique() >= n_trials:
            ids = sorted(df_exist["trial"].astype(int).tolist())
            if ids[:n_trials] == list(range(n_trials)):
                logger.info("LightGBM Optuna: reuse existing CSV (%d trials)", n_trials)
                best = default.copy()
                df_s = df_exist.sort_values("oof_accuracy", ascending=False)
                row0 = df_s.iloc[0]
                for k in row0.index:
                    if k in {"trial", "oof_accuracy"}:
                        continue
                    val = row0[k]
                    if k in {"n_estimators", "num_leaves", "min_child_samples", "random_state", "num_class"}:
                        best[k] = int(val)
                    elif k in {"learning_rate", "subsample", "colsample_bytree"}:
                        best[k] = float(val)
                    else:
                        best[k] = val
                best["random_state"] = int(seed)
                best["verbose"] = -1
                best["objective"] = "multiclass"
                best["num_class"] = 3
                # pandas 读 CSV 常把整数变成 float（如 710.0），LightGBM 会报错
                best["n_estimators"] = int(best["n_estimators"])
                return df_s, best

    y_pos = pd.Series(normalize_label_array(y.reset_index(drop=True)))
    X_pos = X.reset_index(drop=True)
    folds = list(
        StratifiedKFold(
            n_splits=TASK1_OPTUNA_COARSE_FOLDS, shuffle=True, random_state=seed
        ).split(np.zeros(len(y_pos)), y_pos)
    )

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "random_state": seed,
            "verbose": -1,
            "objective": "multiclass",
            "num_class": 3,
        }
        oof_p, oof_pred, _ = oof_lgbm(X_pos, y_pos, features, folds, params, tag=f"lgb_{trial.number}")
        acc = float((oof_pred == y_pos.to_numpy()).mean())
        logger.info("LightGBM Optuna trial=%d acc=%.6f", trial.number, acc)
        return acc

    storage = None
    if storage_path is not None:
        storage_path = Path(storage_path)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_path.resolve().as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=True,
    )
    done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    remain = max(0, n_trials - done)
    if remain:
        logger.info("LightGBM Optuna: %d done, running %d more", done, remain)
        study.optimize(objective, n_trials=remain, show_progress_bar=False)
    else:
        logger.info("LightGBM Optuna study already complete (%d)", done)
    rows = [{"trial": t.number, "oof_accuracy": t.value, **t.params} for t in study.trials if t.value is not None]
    df = pd.DataFrame(rows).sort_values("oof_accuracy", ascending=False)
    best = default.copy()
    if len(df):
        best.update({k: df.iloc[0][k] for k in df.iloc[0].index if k not in {"trial", "oof_accuracy"}})
        best["random_state"] = seed
        best["verbose"] = -1
        best["objective"] = "multiclass"
        best["num_class"] = 3
    return df, best


def oof_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    tag: str = "M5_xgb",
    checkpoint_dir: Optional[Path] = None,
    sample_index: Optional[Sequence[int]] = None,
    resume_folds: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    if not HAS_XGB:
        raise RuntimeError("xgboost 未安装")
    X_pos = X.reset_index(drop=True)
    y_pos = pd.Series(normalize_label_array(y.reset_index(drop=True)))
    y_code = _y_codes(y_pos)
    n = len(X_pos)
    oof_p = np.zeros((n, 3), dtype=float)
    fold_rows: List[Dict[str, Any]] = []
    if sample_index is None:
        sample_index = list(range(n))
    sample_index = np.asarray(sample_index).astype(int)

    if checkpoint_dir is not None and resume_folds:
        ckpt = Path(checkpoint_dir)
        if all(fold_is_complete(ckpt, fid, expected_n=len(va)) for fid, (_, va) in enumerate(folds)):
            oof_p, oof_pred, fold_rows, _ = assemble_oof_from_checkpoints(
                ckpt, n, len(folds), sample_index
            )
            return oof_p, oof_pred, fold_rows

    xgb_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_SEED,
        "n_jobs": 0,
    }
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

        state = fit_tabular_preprocessor(X_pos.iloc[tr], features, scale=False)
        X_tr = transform_tabular(X_pos.iloc[tr], state)
        X_va = transform_tabular(X_pos.iloc[va], state)
        model = xgb.XGBClassifier(**xgb_params)
        try:
            model.fit(
                X_tr,
                y_code[tr],
                eval_set=[(X_va, y_code[va])],
                verbose=False,
                early_stopping_rounds=80,
            )
        except TypeError:
            model.fit(X_tr, y_code[tr], eval_set=[(X_va, y_code[va])], verbose=False)
        proba = model.predict_proba(X_va)
        out = np.zeros((len(va), 3), dtype=float)
        for i, c in enumerate(model.classes_):
            out[:, int(c)] = proba[:, i]
        oof_p[va] = out
        pred = pred_from_proba(out)
        m = evaluate_multiclass(y_pos.iloc[va], pred, out)
        fold_rows.append({"fold": fold_id, "model": tag, **m})
        logger.info("%s fold=%d acc=%.4f", tag, fold_id, m["accuracy"])
        if checkpoint_dir is not None:
            save_fold_checkpoint(
                Path(checkpoint_dir),
                fold_id,
                sample_index=sample_index[va],
                predicted_label=pred,
                proba=out,
                metrics=m,
                best_iteration=int(getattr(model, "best_iteration", 0) or 0),
                params=xgb_params,
                features=features,
                model_name=tag,
            )
    return oof_p, pred_from_proba(oof_p), fold_rows


def oof_ordinal_logistic(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    tag: str = "M6_ord_logit",
    checkpoint_dir: Optional[Path] = None,
    sample_index: Optional[Sequence[int]] = None,
    resume_folds: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], pd.DataFrame]:
    """双累积边界 Logistic，可解释线性基线。"""
    X_pos = X.reset_index(drop=True)
    y_pos = pd.Series(normalize_label_array(y.reset_index(drop=True)))
    ge_med = np.array([0 if lab == "Low" else 1 for lab in y_pos], dtype=int)
    ge_high = np.array([1 if lab == "High" else 0 for lab in y_pos], dtype=int)
    n = len(X_pos)
    oof_p = np.zeros((n, 3), dtype=float)
    fold_rows: List[Dict[str, Any]] = []
    coef_rows: List[Dict[str, Any]] = []
    if sample_index is None:
        sample_index = list(range(n))
    sample_index = np.asarray(sample_index).astype(int)

    if checkpoint_dir is not None and resume_folds:
        ckpt = Path(checkpoint_dir)
        if all(fold_is_complete(ckpt, fid, expected_n=len(va)) for fid, (_, va) in enumerate(folds)):
            oof_p, oof_pred, fold_rows, _ = assemble_oof_from_checkpoints(
                ckpt, n, len(folds), sample_index
            )
            coef_path = Path(checkpoint_dir) / "coefficients.csv"
            coef_df = pd.read_csv(coef_path) if coef_path.exists() else pd.DataFrame()
            return oof_p, oof_pred, fold_rows, coef_df

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

        state = fit_tabular_preprocessor(X_pos.iloc[tr], features, scale=True)
        X_tr = transform_tabular(X_pos.iloc[tr], state)
        X_va = transform_tabular(X_pos.iloc[va], state)
        m1 = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        m2 = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        m1.fit(X_tr, ge_med[tr])
        m2.fit(X_tr, ge_high[tr])
        p_ge_m = m1.predict_proba(X_va)[:, 1]
        p_ge_h = m2.predict_proba(X_va)[:, 1]
        proba = ordinal_probs_from_boundaries(p_ge_m, p_ge_h)
        oof_p[va] = proba
        pred = pred_from_proba(proba)
        m = evaluate_multiclass(y_pos.iloc[va], pred, proba)
        fold_rows.append({"fold": fold_id, "model": tag, **m})
        for name, coef in zip(state.feature_names, m1.coef_.ravel()):
            coef_rows.append({"fold": fold_id, "boundary": "ge_medium", "feature": name, "coef": float(coef)})
        for name, coef in zip(state.feature_names, m2.coef_.ravel()):
            coef_rows.append({"fold": fold_id, "boundary": "ge_high", "feature": name, "coef": float(coef)})
        logger.info("%s fold=%d acc=%.4f", tag, fold_id, m["accuracy"])
        if checkpoint_dir is not None:
            save_fold_checkpoint(
                Path(checkpoint_dir),
                fold_id,
                sample_index=sample_index[va],
                predicted_label=pred,
                proba=proba,
                metrics=m,
                best_iteration=None,
                params={"model": "LogisticRegression"},
                features=features,
                model_name=tag,
            )

    coef_df = pd.DataFrame(coef_rows)
    if len(coef_df):
        coef_df = coef_df.groupby(["boundary", "feature"], as_index=False)["coef"].mean()
        coef_df["abs_coef"] = coef_df["coef"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
    return oof_p, pred_from_proba(oof_p), fold_rows, coef_df
