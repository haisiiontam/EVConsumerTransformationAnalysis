from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.common.fold_checkpoints import PROB_COLS_BINARY, fold_is_complete, save_fold_checkpoint
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.metrics import binary_classification_metrics
from src.config import RANDOM_SEED, TASK3_N_FOLDS
from src.tasks.task3.preprocessing import fit_tabular_preprocessor, transform_tabular

logger = logging.getLogger(__name__)


def _oof_sklearnish(
    name: str,
    builder,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    features: Sequence[str],
    fold_assign: np.ndarray,
    ckpt_dir: Path,
    n_folds: int = TASK3_N_FOLDS,
    resume_folds: bool = True,
    scale: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    y = y_dev.to_numpy().astype(int)
    sample_index = X_dev.index.to_numpy().astype(int)
    p1_all = np.zeros(len(X_dev))
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
        state = fit_tabular_preprocessor(X_dev.loc[tr], features, scale=scale)
        X_tr = transform_tabular(X_dev.loc[tr], state)
        X_va = transform_tabular(X_dev.loc[va], state)
        model = builder()
        model.fit(X_tr, y[tr])
        if hasattr(model, "predict_proba"):
            p1 = model.predict_proba(X_va)[:, 1]
        else:
            p1 = model.predict(X_va).astype(float)
        p1_all[va] = p1
        pred = (p1 >= 0.5).astype(int)
        m = binary_classification_metrics(y[va], pred, p1)
        fold_rows.append({"fold": fold_id, "model": name, **m})
        save_fold_checkpoint(
            ckpt_dir,
            fold_id,
            sample_index=sample_index[va],
            predicted_label=pred,
            proba=np.column_stack([1 - p1, p1]),
            metrics=m,
            best_iteration=None,
            params={"model": name},
            features=features,
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
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            verbosity=-1,
        )

    return _oof_sklearnish("m4_lgbm", builder, X_dev, y_dev, features, fold_assign, ckpt_dir, **kw)


def oof_xgboost(X_dev, y_dev, features, fold_assign, ckpt_dir, **kw):
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise RuntimeError(f"xgboost unavailable: {e}") from e

    def builder():
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            verbosity=0,
        )

    return _oof_sklearnish("m5_xgb", builder, X_dev, y_dev, features, fold_assign, ckpt_dir, **kw)


def oof_extra_trees(X_dev, y_dev, features, fold_assign, ckpt_dir, **kw):
    from sklearn.ensemble import ExtraTreesClassifier

    def builder():
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

    return _oof_sklearnish("m6_et", builder, X_dev, y_dev, features, fold_assign, ckpt_dir, scale=False, **kw)


def try_gam_skip(output_dir: Path, reason: str = "disabled") -> None:
    atomic_write_json(
        Path(output_dir) / "models" / "m7_gam_skip.json",
        {"skipped": True, "reason": reason},
    )


def try_tabnet_skip(output_dir: Path, reason: str = "disabled") -> None:
    atomic_write_json(
        Path(output_dir) / "models" / "m8_tabnet_skip.json",
        {"skipped": True, "reason": reason},
    )


def persist_oof(oof: pd.DataFrame, folds: pd.DataFrame, path: Path, fold_path: Path) -> None:
    atomic_write_csv(path, oof)
    atomic_write_csv(fold_path, folds)
