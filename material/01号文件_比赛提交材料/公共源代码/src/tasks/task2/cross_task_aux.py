"""任务二跨任务安全 OOF 表征：绝不用其他任务真标签作输入。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.config import (
    TASK1_TARGET,
    TASK2_SAFE_CANDIDATE_FEATURES,
    TASK2_TARGET,
    TASK3_SOURCE_SCORE,
    TASK3_THRESHOLD,
)

logger = logging.getLogger(__name__)

AUX_FEATURE_NAMES = [
    "aux_t1_p_Low",
    "aux_t1_p_Medium",
    "aux_t1_p_High",
    "aux_t1_expected_level",
    "aux_t1_entropy",
    "aux_t3_p_high_anxiety",
    "aux_t3_pred_score",
]


def _aux_base_features(df: pd.DataFrame) -> List[str]:
    """辅助模型输入：安全候选 ∪ 常见问卷字段，排除任务二标签。"""
    ban = {TASK2_TARGET, "home_charging_available"}
    cols: List[str] = []
    for c in TASK2_SAFE_CANDIDATE_FEATURES:
        if c in df.columns and c not in ban and c not in cols:
            cols.append(c)
    # 任务一辅助可用焦虑分（问卷），但不把任务一真标签当输入
    for c in ("range_anxiety_score", "monthly_energy_consumption_kwh", "monthly_charging_cost"):
        if c in df.columns and c not in ban and c not in cols:
            # 能耗/费用可能后验：辅助模型仍排除，仅允许 range_anxiety_score
            if c == "range_anxiety_score":
                cols.append(c)
    return cols


def _prepare_xy(
    df: pd.DataFrame,
    features: Sequence[str],
) -> Tuple[pd.DataFrame, List[int]]:
    X = df.loc[:, [c for c in features if c in df.columns]].copy()
    cat_idx: List[int] = []
    for i, c in enumerate(X.columns):
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).fillna("Unknown")
            cat_idx.append(i)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, cat_idx


def _entropy(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -np.sum(p * np.log(p), axis=1)


def generate_cross_task_oof_features(
    df_dev: pd.DataFrame,
    fold_assign: np.ndarray,
    output_dir: Optional[Path] = None,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    在 develop 上生成折内 OOF 辅助特征（与任务二标签协议隔离）。

    T1-aux: 预测 ev_adoption_likelihood（真实标签仅作训练目标）
    T3-aux: 回归 range_anxiety_score + 映射 P(score>5)
    """
    feats = _aux_base_features(df_dev)
    # T1 目标：不得把 home_charging 当特征（已排除）
    if TASK1_TARGET not in df_dev.columns:
        raise ValueError(f"missing {TASK1_TARGET} for T1-aux target")
    if TASK3_SOURCE_SCORE not in df_dev.columns:
        raise ValueError(f"missing {TASK3_SOURCE_SCORE} for T3-aux target")

    y1 = df_dev[TASK1_TARGET].astype(str).str.strip()
    y1 = y1.replace({"low": "Low", "medium": "Medium", "high": "High"})
    score = pd.to_numeric(df_dev[TASK3_SOURCE_SCORE], errors="coerce")
    y3_bin = (score > TASK3_THRESHOLD).astype(int)

    n = len(df_dev)
    p_low = np.zeros(n)
    p_med = np.zeros(n)
    p_high = np.zeros(n)
    p_anx = np.zeros(n)
    pred_score = np.zeros(n)

    X_all, cat_idx = _prepare_xy(df_dev, feats)
    sample_index = df_dev.index.to_numpy().astype(int)

    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        X_tr, X_va = X_all.loc[tr], X_all.loc[va]
        # --- T1 multiclass ---
        clf = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=250,
            depth=6,
            learning_rate=0.06,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        clf.fit(X_tr, y1.loc[tr], cat_features=cat_idx or None)
        proba = clf.predict_proba(X_va)
        classes = [str(c) for c in clf.classes_]
        colmap = {c: i for i, c in enumerate(classes)}
        p_low[va] = proba[:, colmap["Low"]] if "Low" in colmap else proba[:, 0]
        p_med[va] = proba[:, colmap["Medium"]] if "Medium" in colmap else (
            proba[:, 1] if proba.shape[1] > 1 else 0.0
        )
        p_high[va] = proba[:, colmap["High"]] if "High" in colmap else (
            proba[:, -1] if proba.shape[1] > 2 else 0.0
        )

        # --- T3: regress score + binary clf for P(high anxiety) ---
        reg = CatBoostRegressor(
            loss_function="RMSE",
            iterations=250,
            depth=6,
            learning_rate=0.06,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        # T3-aux 输入需排除 range_anxiety_score 本身（目标）
        feat_t3 = [c for c in feats if c != TASK3_SOURCE_SCORE]
        X3, cat3 = _prepare_xy(df_dev, feat_t3)
        reg.fit(X3.loc[tr], score.loc[tr].fillna(score.loc[tr].median()), cat_features=cat3 or None)
        pred_score[va] = reg.predict(X3.loc[va])

        clf3 = CatBoostClassifier(
            loss_function="Logloss",
            iterations=250,
            depth=6,
            learning_rate=0.06,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        clf3.fit(X3.loc[tr], y3_bin.loc[tr], cat_features=cat3 or None)
        pr = clf3.predict_proba(X3.loc[va])
        classes3 = [int(c) for c in clf3.classes_]
        p_anx[va] = pr[:, classes3.index(1)] if 1 in classes3 else pr[:, -1]

        logger.info("cross_task_aux fold=%d done", fold_id)

    p_stack = np.column_stack([p_low, p_med, p_high])
    expected = p_low * 0.0 + p_med * 1.0 + p_high * 2.0
    ent = _entropy(p_stack)

    out = pd.DataFrame(
        {
            "sample_index": sample_index,
            "aux_t1_p_Low": p_low,
            "aux_t1_p_Medium": p_med,
            "aux_t1_p_High": p_high,
            "aux_t1_expected_level": expected,
            "aux_t1_entropy": ent,
            "aux_t3_p_high_anxiety": p_anx,
            "aux_t3_pred_score": pred_score,
        },
        index=df_dev.index,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(output_dir / "cross_task_oof_features.csv", out.reset_index(drop=True))
        atomic_write_json(
            output_dir / "cross_task_aux_meta.json",
            {
                "aux_input_features": feats,
                "t3_input_features": [c for c in feats if c != TASK3_SOURCE_SCORE],
                "note": "OOF predictions only; never use true ev_adoption_likelihood as task2 input",
                "banned_as_input": [TASK2_TARGET, TASK1_TARGET],
            },
        )
    return out


def attach_aux_to_frame(df: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    """按 index 合并辅助特征列。"""
    cols = [c for c in AUX_FEATURE_NAMES if c in aux.columns]
    merged = df.copy()
    for c in cols:
        merged[c] = aux[c].reindex(merged.index).to_numpy()
    return merged
