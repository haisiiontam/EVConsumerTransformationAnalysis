"""M1：CatBoost 多分类 + Optuna。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from src.config import (
    RANDOM_SEED,
    TASK1_CATBOOST_COMMON,
    TASK1_EARLY_STOPPING_ROUNDS,
    TASK1_LABEL_ORDER,
    TASK1_OPTUNA_COARSE_FOLDS,
    TASK1_OPTUNA_TRIALS_CB,
)
from src.tasks.task1.checkpoints import (
    assemble_oof_from_checkpoints,
    fold_is_complete,
    load_fold_predictions,
    save_fold_checkpoint,
)
from src.tasks.task1.metrics_ext import evaluate_multiclass, normalize_label_array, pred_from_proba
from src.tasks.task1.preprocess import cat_indices, fit_cat_preprocessor, transform_cat

logger = logging.getLogger(__name__)

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:  # pragma: no cover
    HAS_OPTUNA = False
    optuna = None  # type: ignore


def _align_proba(model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    class_names = [str(c) for c in model.classes_]
    out = np.zeros((len(X), 3), dtype=float)
    for i, name in enumerate(class_names):
        key = {"low": "Low", "medium": "Medium", "high": "High"}.get(name.lower(), name)
        if key in TASK1_LABEL_ORDER:
            out[:, TASK1_LABEL_ORDER.index(key)] = proba[:, i]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def oof_catboost_multiclass(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    params: Dict[str, Any],
    tag: str = "M1",
    fill_median: bool = True,
    checkpoint_dir: Optional[Path] = None,
    sample_index: Optional[Sequence[int]] = None,
    resume_folds: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[int]]:
    """返回 oof_proba (n,3), oof_pred, fold_rows, best_iters。"""
    X_pos = X.reset_index(drop=True)
    y_pos = normalize_label_array(y.reset_index(drop=True))
    y_pos = pd.Series(y_pos)
    n = len(X_pos)
    oof_p = np.zeros((n, 3), dtype=float)
    oof_pred = np.empty(n, dtype=object)
    fold_rows: List[Dict[str, Any]] = []
    best_iters: List[int] = []
    if sample_index is None:
        sample_index = list(range(n))
    sample_index = np.asarray(sample_index).astype(int)

    # 若全部 fold 已完整，直接组装
    if checkpoint_dir is not None and resume_folds:
        ckpt = Path(checkpoint_dir)
        if all(fold_is_complete(ckpt, fid, expected_n=len(va)) for fid, (_, va) in enumerate(folds)):
            logger.info("%s: all %d folds loaded from checkpoints", tag, len(folds))
            return assemble_oof_from_checkpoints(ckpt, n, len(folds), sample_index)

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
                oof_pred[pos] = row["predicted_label"]
            fold_rows.append(met)
            if met.get("best_iteration") is not None:
                best_iters.append(int(met["best_iteration"]))
            logger.info("%s fold=%d resumed from checkpoint", tag, fold_id)
            continue

        state = fit_cat_preprocessor(X_pos.iloc[tr], features, fill_numeric_median=fill_median)
        X_tr, X_va = transform_cat(X_pos.iloc[tr], state), transform_cat(X_pos.iloc[va], state)
        p = dict(params)
        p.setdefault("loss_function", "MultiClass")
        p.setdefault("eval_metric", "MultiClass")
        p.setdefault("verbose", False)
        p.setdefault("allow_writing_files", False)
        model = CatBoostClassifier(**p)
        model.fit(
            X_tr,
            y_pos.iloc[tr],
            cat_features=cat_indices(state),
            eval_set=(X_va, y_pos.iloc[va]),
            early_stopping_rounds=TASK1_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        it = int(model.get_best_iteration() or p.get("iterations", 500))
        best_iters.append(it)
        proba = _align_proba(model, X_va)
        pred = pred_from_proba(proba)
        oof_p[va] = proba
        oof_pred[va] = pred
        m = evaluate_multiclass(y_pos.iloc[va], pred, proba)
        fold_rows.append({"fold": fold_id, "model": tag, **m, "best_iteration": it})
        logger.info("%s fold=%d acc=%.4f qwk=%.4f", tag, fold_id, m["accuracy"], m["qwk"])
        if checkpoint_dir is not None:
            save_fold_checkpoint(
                Path(checkpoint_dir),
                fold_id,
                sample_index=sample_index[va],
                predicted_label=pred,
                proba=proba,
                metrics=m,
                best_iteration=it,
                params=p,
                features=features,
                model_name=tag,
            )
    return oof_p, oof_pred, fold_rows, best_iters


def default_catboost_params(seed: int = RANDOM_SEED) -> Dict[str, Any]:
    p = dict(TASK1_CATBOOST_COMMON)
    p["random_seed"] = seed
    p["eval_metric"] = "MultiClass"
    p["verbose"] = False
    return p


def _trials_df_from_study(study: "optuna.Study") -> pd.DataFrame:
    rows = []
    for t in study.trials:
        if t.state.name != "COMPLETE" or t.value is None:
            continue
        rows.append(
            {
                "trial": t.number,
                "oof_accuracy": t.value,
                "fold_std": t.user_attrs.get("fold_std"),
                "macro_f1": t.user_attrs.get("macro_f1"),
                **t.params,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["oof_accuracy", "fold_std", "macro_f1"], ascending=[False, True, False]
    )


def _best_from_trials_df(trials_df: pd.DataFrame, seed: int) -> Dict[str, Any]:
    best = default_catboost_params(seed)
    if trials_df is None or len(trials_df) == 0:
        return best
    r = trials_df.iloc[0]
    best.update(
        {
            "depth": int(r["depth"]),
            "learning_rate": float(r["learning_rate"]),
            "l2_leaf_reg": float(r["l2_leaf_reg"]),
            "random_strength": float(r["random_strength"]),
            "bagging_temperature": float(r["bagging_temperature"]),
            "border_count": int(r["border_count"]),
            "iterations": int(r["iterations"]),
        }
    )
    acw = r.get("auto_class_weights")
    if acw is not None and str(acw) not in {"None", "nan"}:
        best["auto_class_weights"] = acw
    return best


def optuna_search_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    seed: int = RANDOM_SEED,
    n_trials: Optional[int] = None,
    storage_path: Optional[Path] = None,
    study_name: str = "task1_catboost",
    existing_trials_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not HAS_OPTUNA:
        logger.warning("未安装 optuna，跳过 M1 搜索，使用默认参数")
        return pd.DataFrame(), default_catboost_params(seed)

    n_trials = n_trials or TASK1_OPTUNA_TRIALS_CB

    # 若 CSV 已完整，直接复用，绝不重跑
    if existing_trials_csv is not None and Path(existing_trials_csv).exists():
        trials_df = pd.read_csv(existing_trials_csv)
        if "trial" in trials_df.columns and trials_df["trial"].nunique() >= n_trials:
            ids = sorted(trials_df["trial"].astype(int).tolist())
            if ids[:n_trials] == list(range(n_trials)):
                logger.info(
                    "CatBoost Optuna: reuse existing CSV (%d trials), skip optimize",
                    n_trials,
                )
                return trials_df.sort_values(
                    ["oof_accuracy", "fold_std", "macro_f1"],
                    ascending=[False, True, False],
                ), _best_from_trials_df(trials_df, seed)

    y_pos = pd.Series(normalize_label_array(y.reset_index(drop=True)))
    X_pos = X.reset_index(drop=True)
    skf = StratifiedKFold(
        n_splits=TASK1_OPTUNA_COARSE_FOLDS, shuffle=True, random_state=seed
    )
    folds = list(skf.split(np.zeros(len(y_pos)), y_pos))

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "verbose": False,
            "allow_writing_files": False,
            "random_seed": seed,
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
            "iterations": trial.suggest_int("iterations", 400, 1500),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights", [None, "Balanced"]
            ),
        }
        if params["auto_class_weights"] is None:
            params.pop("auto_class_weights")
        oof_p, oof_pred, fold_rows, _ = oof_catboost_multiclass(
            X_pos, y_pos, features, folds, params, tag=f"optuna_{trial.number}"
        )
        acc = float((oof_pred == y_pos.to_numpy()).mean())
        std = float(np.std([r["accuracy"] for r in fold_rows]))
        trial.set_user_attr("fold_std", std)
        trial.set_user_attr(
            "macro_f1",
            float(evaluate_multiclass(y_pos, oof_pred, oof_p)["macro_f1"]),
        )
        logger.info("CatBoost Optuna trial=%d acc=%.6f", trial.number, acc)
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
    if remain == 0:
        logger.info("CatBoost Optuna study already has %d COMPLETE trials", done)
    else:
        logger.info("CatBoost Optuna: %d done, running %d more (target=%d)", done, remain, n_trials)
        study.optimize(objective, n_trials=remain, show_progress_bar=False)
    trials_df = _trials_df_from_study(study)
    return trials_df, _best_from_trials_df(trials_df, seed)


def confirm_top_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    trials_df: pd.DataFrame,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    top_k: int = 5,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rows = []
    if trials_df is None or len(trials_df) == 0:
        params = default_catboost_params(seed)
        oof_p, oof_pred, fold_rows, iters = oof_catboost_multiclass(
            X, y, features, folds, params, tag="M1_default_confirm"
        )
        m = evaluate_multiclass(y.reset_index(drop=True), oof_pred, oof_p)
        rows.append({**params, **m, "fold_std": float(np.std([r["accuracy"] for r in fold_rows]))})
        return pd.DataFrame(rows)

    seen = set()
    for _, r in trials_df.head(top_k).iterrows():
        key = (
            int(r["depth"]),
            round(float(r["learning_rate"]), 6),
            round(float(r["l2_leaf_reg"]), 6),
            int(r["iterations"]),
            int(r["border_count"]),
        )
        if key in seen:
            continue
        seen.add(key)
        params = default_catboost_params(seed)
        params.update(
            {
                "depth": key[0],
                "learning_rate": key[1],
                "l2_leaf_reg": key[2],
                "iterations": key[3],
                "border_count": key[4],
                "random_strength": float(r.get("random_strength", 1.0)),
                "bagging_temperature": float(r.get("bagging_temperature", 1.0)),
            }
        )
        acw = r.get("auto_class_weights")
        if acw is not None and str(acw) not in {"None", "nan"}:
            params["auto_class_weights"] = acw
        oof_p, oof_pred, fold_rows, _ = oof_catboost_multiclass(
            X, y, features, folds, params, tag=f"M1_confirm_{len(rows)}"
        )
        m = evaluate_multiclass(y.reset_index(drop=True), oof_pred, oof_p)
        rows.append(
            {
                **{k: params[k] for k in ["depth", "learning_rate", "l2_leaf_reg", "iterations", "border_count", "random_strength", "bagging_temperature"]},
                **m,
                "fold_std": float(np.std([fr["accuracy"] for fr in fold_rows])),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["accuracy", "fold_std", "macro_f1", "qwk"], ascending=[False, True, False, False]
    )
