"""鲁棒性与学习曲线（轻量实现）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from src.common.io_utils import atomic_write_csv
from src.config import RANDOM_SEED, TASK3_CATBOOST_COMMON
from src.tasks.task3.m1_catboost_classifier import _prob1
from src.tasks.task3.preprocessing import cat_indices, fit_cat_preprocessor, transform_cat


def learning_curve(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
) -> pd.DataFrame:
    yv = y.to_numpy().astype(int)
    rows = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    for frac in fractions:
        accs = []
        for tr, va in skf.split(X, yv):
            n = max(50, int(len(tr) * frac))
            tr_use = tr[:n]
            state = fit_cat_preprocessor(X.iloc[tr_use], features)
            model = CatBoostClassifier(**{**TASK3_CATBOOST_COMMON, "iterations": 200})
            model.fit(
                transform_cat(X.iloc[tr_use], state),
                yv[tr_use],
                cat_features=cat_indices(state),
            )
            p1 = _prob1(model, transform_cat(X.iloc[va], state))
            accs.append(float(((p1 >= 0.5).astype(int) == yv[va]).mean()))
        rows.append({"fraction": frac, "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs))})
    return pd.DataFrame(rows)


def noise_sensitivity(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    noise_levels: Sequence[float] = (0.01, 0.05),
) -> pd.DataFrame:
    yv = y.to_numpy().astype(int)
    state = fit_cat_preprocessor(X, features)
    Xt = transform_cat(X, state)
    model = CatBoostClassifier(**{**TASK3_CATBOOST_COMMON, "iterations": 200})
    model.fit(Xt, yv, cat_features=cat_indices(state))
    base_p = _prob1(model, Xt)
    base_acc = float(((base_p >= 0.5).astype(int) == yv).mean())
    rows = [{"noise": 0.0, "accuracy": base_acc}]
    rng = np.random.RandomState(RANDOM_SEED)
    for nl in noise_levels:
        Xn = Xt.copy()
        for c in state.numeric_columns:
            if c in Xn.columns:
                vals = pd.to_numeric(Xn[c], errors="coerce").to_numpy()
                Xn[c] = vals * (1.0 + rng.normal(0, nl, size=len(vals)))
        p = _prob1(model, Xn)
        rows.append({"noise": nl, "accuracy": float(((p >= 0.5).astype(int) == yv).mean())})
    return pd.DataFrame(rows)


def persist_robustness(output_dir: Path, lc: pd.DataFrame, noise: pd.DataFrame) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "learning_curve.csv", lc)
    # also under diagnostics per prompt
    diag = output_dir.parent / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(diag / "learning_curve.csv", lc)
    atomic_write_csv(output_dir / "noise_sensitivity.csv", noise)
    # placeholders
    atomic_write_csv(
        output_dir / "feature_dropout.csv",
        pd.DataFrame([{"note": "see ablation for group dropout"}]),
    )
    atomic_write_csv(
        output_dir / "uncertainty_analysis.csv",
        pd.DataFrame([{"note": "margin/entropy computed in diversity stage when available"}]),
    )
