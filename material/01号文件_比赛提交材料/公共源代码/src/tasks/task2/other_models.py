from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from src.common.fold_checkpoints import PROB_COLS_BINARY, fold_is_complete, save_fold_checkpoint
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.tasks.task2.metrics_ext import rich_binary_metrics
from src.tasks.task2.preprocessing import fit_tabular, transform_tabular

logger = logging.getLogger(__name__)


def _oof_model(
    name: str,
    builder: Callable[[], Any],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    features: Sequence[str],
    fold_assign: np.ndarray,
    ckpt_dir: Path,
    n_folds: int = 5,
    resume_folds: bool = True,
    scale: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    y = y_dev.to_numpy().astype(int)
    sample_index = X_dev.index.to_numpy().astype(int)
    p1_all = np.zeros(len(X_dev), dtype=float)
    fold_rows: List[Dict[str, Any]] = []

    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        if resume_folds and fold_is_complete(ckpt_dir, fold_id, int(va.sum()), PROB_COLS_BINARY):
            pred = pd.read_csv(ckpt_dir / f"fold_{fold_id}_predictions.csv")
            idx_map = {int(s): i for i, s in enumerate(sample_index)}
            for _, row in pred.iterrows():
                p1_all[idx_map[int(row["sample_index"])]] = float(row["prob_1"])
            continue
        state = fit_tabular(X_dev.loc[tr], features, scale=scale)
        X_tr = transform_tabular(X_dev.loc[tr], state)
        X_va = transform_tabular(X_dev.loc[va], state)
        model = builder()
        model.fit(X_tr, y[tr])
        p1 = model.predict_proba(X_va)[:, 1]
        p1_all[va] = p1
        pred_lab = (p1 >= 0.5).astype(int)
        m = rich_binary_metrics(y[va], pred_lab, p1)
        fold_rows.append({"fold": fold_id, "model": name, **m})
        save_fold_checkpoint(
            ckpt_dir,
            fold_id,
            sample_index=sample_index[va],
            predicted_label=pred_lab,
            proba=np.column_stack([1.0 - p1, p1]),
            metrics=m,
            best_iteration=None,
            params={"model": name},
            features=list(features),
            model_name=name,
            prob_cols=PROB_COLS_BINARY,
            y_true=y[va],
        )

    oof = pd.DataFrame(
        {
            "sample_index": sample_index,
            "true_label": y,
            "prob_0": 1.0 - p1_all,
            "prob_1": p1_all,
            "predicted_label": (p1_all >= 0.5).astype(int),
            "fold": fold_assign,
        }
    )
    return oof, pd.DataFrame(fold_rows)


def oof_lightgbm(X_dev, y_dev, features, fold_assign, ckpt_dir, **kw):
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError(f"lightgbm unavailable: {e}") from e

    def builder():
        return lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )

    return _oof_model("lightgbm", builder, X_dev, y_dev, features, fold_assign, ckpt_dir, **kw)


def oof_xgboost(X_dev, y_dev, features, fold_assign, ckpt_dir, **kw):
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise RuntimeError(f"xgboost unavailable: {e}") from e

    def builder():
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )

    return _oof_model("xgboost", builder, X_dev, y_dev, features, fold_assign, ckpt_dir, **kw)


def oof_extra_trees(X_dev, y_dev, features, fold_assign, ckpt_dir, **kw):
    def builder():
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=4,
        )

    return _oof_model("extra_trees", builder, X_dev, y_dev, features, fold_assign, ckpt_dir, scale=True, **kw)


def persist_model_oof(output_dir: Path, name: str, oof: pd.DataFrame, fold_metrics: pd.DataFrame) -> None:
    out = Path(output_dir) / "oof"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(out / f"{name}_oof_predictions.csv", oof)
    atomic_write_csv(out / f"{name}_fold_metrics.csv", fold_metrics)
    m = rich_binary_metrics(oof["true_label"], oof["predicted_label"], oof["prob_1"])
    atomic_write_json(out / f"{name}_oof_metrics.json", m)


def run_heterogeneous_models(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    fold_assign: np.ndarray,
    features: Sequence[str],
    output_dir: Path,
    n_folds: int = 5,
    resume_folds: bool = True,
    enable: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """返回 name -> oof DataFrame。失败的模型跳过并记录。"""
    wanted = set(enable or ("lightgbm", "xgboost", "extra_trees"))
    results: Dict[str, pd.DataFrame] = {}
    runners = {
        "lightgbm": oof_lightgbm,
        "xgboost": oof_xgboost,
        "extra_trees": oof_extra_trees,
    }
    for name in ("lightgbm", "xgboost", "extra_trees"):
        if name not in wanted:
            continue
        try:
            oof, folds = runners[name](
                df_dev,
                y_dev,
                features,
                fold_assign,
                Path(output_dir) / "checkpoints" / name,
                n_folds=n_folds,
                resume_folds=resume_folds,
            )
            persist_model_oof(output_dir, name, oof, folds)
            results[name] = oof
            logger.info("%s oof done acc=%.6f", name, rich_binary_metrics(oof["true_label"], oof["predicted_label"], oof["prob_1"])["accuracy"])
        except Exception as e:
            logger.warning("%s skipped: %s", name, e)
            atomic_write_json(
                Path(output_dir) / "oof" / f"{name}_skip.json",
                {"available": False, "error": str(e)},
            )
    return results
