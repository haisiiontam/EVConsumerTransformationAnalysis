"""任务二折内预处理：CatBoost 原生类别 / 表模型 One-Hot+StandardScaler。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TASK2_KNOWN_CATEGORICAL
from src.tasks.task2.features import (
    _add_train_relative_features,
    add_mechanism_features,
    to_num,
)


@dataclass
class Task2PreprocessorState:
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_medians: Dict[str, float] = field(default_factory=dict)
    final_feature_order: List[str] = field(default_factory=list)
    income_median: float | None = None


@dataclass
class TabularPrepState:
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_medians: Dict[str, float] = field(default_factory=dict)
    ohe: Optional[OneHotEncoder] = None
    scaler: Optional[StandardScaler] = None
    feature_names: List[str] = field(default_factory=list)


def identify_task2_types(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """识别任务二数值/类别特征。"""
    categorical: List[str] = []
    numerical: List[str] = []
    known = set(TASK2_KNOWN_CATEGORICAL)

    for col in feature_cols:
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        is_known = col in known
        is_object_like = pd.api.types.is_object_dtype(dtype) or isinstance(
            dtype, pd.CategoricalDtype
        )
        is_bool = pd.api.types.is_bool_dtype(dtype)
        if is_known or is_object_like or is_bool:
            categorical.append(col)
        else:
            numerical.append(col)
    return numerical, categorical


def fit_task2_preprocessor(
    X_train: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Task2PreprocessorState:
    """仅根据训练集拟合预处理状态。"""
    X_work = add_mechanism_features(X_train)
    income_median: float | None = None
    if "annual_income" in X_work.columns:
        income_s, _ = to_num(X_work["annual_income"])
        if income_s.notna().any():
            income_median = float(income_s.median())
    X_work = _add_train_relative_features(X_work, income_median)

    cols = [c for c in feature_cols if c in X_work.columns]
    numerical, categorical = identify_task2_types(X_work, cols)

    medians: Dict[str, float] = {}
    for col in numerical:
        series, _ = to_num(X_work[col])
        medians[col] = float(series.median()) if series.notna().any() else 0.0

    return Task2PreprocessorState(
        numeric_columns=numerical,
        categorical_columns=categorical,
        numeric_medians=medians,
        final_feature_order=cols,
        income_median=income_median,
    )


def transform_task2_data(
    X: pd.DataFrame,
    state: Task2PreprocessorState,
) -> pd.DataFrame:
    """使用已拟合状态变换数据。"""
    X_work = add_mechanism_features(X)
    X_work = _add_train_relative_features(X_work, state.income_median)
    out = pd.DataFrame(index=X_work.index)

    for col in state.final_feature_order:
        if col not in X_work.columns:
            if col in state.numeric_columns:
                out[col] = state.numeric_medians.get(col, 0.0)
            else:
                out[col] = "Unknown"
            continue

        if col in state.numeric_columns:
            series, _ = to_num(X_work[col])
            out[col] = series.fillna(state.numeric_medians.get(col, 0.0))
        else:
            mask_missing = X_work[col].isna()
            as_str = X_work[col].astype(str).str.strip()
            as_str = as_str.replace(
                {"nan": "Unknown", "None": "Unknown", "<NA>": "Unknown", "": "Unknown"}
            )
            as_str = as_str.where(~mask_missing, "Unknown")
            out[col] = as_str

    return out.loc[:, state.final_feature_order]


def get_cat_indices(state: Task2PreprocessorState) -> List[int]:
    """CatBoost 类别特征索引。"""
    cat_set = set(state.categorical_columns)
    return [i for i, c in enumerate(state.final_feature_order) if c in cat_set]


def fit_tabular(
    X_train: pd.DataFrame,
    feature_cols: Sequence[str],
    scale: bool = True,
) -> TabularPrepState:
    """Logistic 等表模型：OneHot + StandardScaler（折内拟合）。"""
    Xw = add_mechanism_features(X_train)
    income_median: float | None = None
    if "annual_income" in Xw.columns:
        income_s, _ = to_num(Xw["annual_income"])
        if income_s.notna().any():
            income_median = float(income_s.median())
    Xw = _add_train_relative_features(Xw, income_median)

    cols = [c for c in feature_cols if c in Xw.columns]
    num, cat = identify_task2_types(Xw, cols)
    med = {
        c: float(s.median()) if (s := to_num(Xw[c])[0]).notna().any() else 0.0 for c in num
    }
    ohe = None
    if cat:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(Xw[cat].astype(str).fillna("Unknown"))
    scaler = None
    Xn = (
        np.column_stack([to_num(Xw[c])[0].fillna(med[c]).to_numpy() for c in num])
        if num
        else np.zeros((len(Xw), 0))
    )
    if scale and Xn.shape[1] > 0:
        scaler = StandardScaler()
        scaler.fit(Xn)
    names = list(num)
    if ohe is not None:
        names.extend(list(ohe.get_feature_names_out(cat)))
    return TabularPrepState(num, cat, med, ohe, scaler, names)


def transform_tabular(X: pd.DataFrame, state: TabularPrepState) -> np.ndarray:
    """将原始特征矩阵转为表模型输入。"""
    Xw = add_mechanism_features(X)
    parts = []
    if state.numeric_columns:
        Xn = np.column_stack(
            [
                to_num(Xw[c])[0].fillna(state.numeric_medians.get(c, 0.0)).to_numpy()
                if c in Xw.columns
                else np.full(len(Xw), state.numeric_medians.get(c, 0.0))
                for c in state.numeric_columns
            ]
        )
        if state.scaler is not None:
            Xn = state.scaler.transform(Xn)
        parts.append(Xn)
    if state.ohe is not None and state.categorical_columns:
        Xc = Xw.reindex(columns=state.categorical_columns).astype(str).fillna("Unknown")
        parts.append(state.ohe.transform(Xc))
    if not parts:
        return np.zeros((len(Xw), 0))
    return np.hstack(parts)
