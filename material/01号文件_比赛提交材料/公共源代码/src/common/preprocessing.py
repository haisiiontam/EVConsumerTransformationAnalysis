from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.config import KNOWN_CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


def identify_feature_types(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    known_categorical: Sequence[str] | None = None,
) -> Tuple[List[str], List[str]]:
    """识别数值特征与类别特征。"""
    known = set(known_categorical) if known_categorical is not None else set(
        KNOWN_CATEGORICAL_FEATURES
    )
    categorical: List[str] = []
    numerical: List[str] = []

    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"特征列不存在：{col}")

        dtype = df[col].dtype
        is_known_cat = col in known
        is_object_like = pd.api.types.is_object_dtype(dtype) or isinstance(
            dtype, pd.CategoricalDtype
        )
        is_bool = pd.api.types.is_bool_dtype(dtype)

        if is_known_cat or is_object_like or is_bool:
            categorical.append(col)
        else:
            numerical.append(col)

    logger.info(
        "特征类型识别：数值=%d，类别=%d；类别列=%s",
        len(numerical),
        len(categorical),
        categorical,
    )
    return numerical, categorical


def fit_preprocess_stats(
    X_train: pd.DataFrame,
    numerical_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> Dict[str, float]:
    """仅根据训练集计算数值特征中位数，避免数据泄露。"""
    medians: Dict[str, float] = {}
    for col in numerical_cols:
        series = pd.to_numeric(X_train[col], errors="coerce")
        median_value = float(series.median()) if series.notna().any() else 0.0
        medians[col] = median_value
    _ = categorical_cols
    return medians


def transform_features(
    X: pd.DataFrame,
    numerical_cols: Sequence[str],
    categorical_cols: Sequence[str],
    medians: Dict[str, float],
) -> pd.DataFrame:
    """使用训练集统计量对特征做变换。"""
    X_out = X.copy()

    for col in numerical_cols:
        X_out[col] = pd.to_numeric(X_out[col], errors="coerce")
        fill_value = medians.get(col, 0.0)
        X_out[col] = X_out[col].fillna(fill_value)

    for col in categorical_cols:
        mask_missing = X_out[col].isna()
        as_str = X_out[col].astype(str).str.strip()
        as_str = as_str.replace({"nan": "Unknown", "None": "Unknown", "<NA>": "Unknown"})
        as_str = as_str.where(~mask_missing, "Unknown")
        as_str = as_str.replace({"": "Unknown"})
        X_out[col] = as_str

    return X_out


def get_cat_feature_indices(
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> List[int]:
    """生成 CatBoost 所需的类别特征列索引。"""
    cat_set = set(categorical_cols)
    return [i for i, col in enumerate(feature_cols) if col in cat_set]


def prepare_xy(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[int]]:
    """完整预处理流水线：识别类型 -> 拟合训练集统计量 -> 变换 train/valid。"""
    numerical_cols, categorical_cols = identify_feature_types(X_train, feature_cols)
    medians = fit_preprocess_stats(X_train, numerical_cols, categorical_cols)

    X_train_p = transform_features(X_train, numerical_cols, categorical_cols, medians)
    X_valid_p = transform_features(X_valid, numerical_cols, categorical_cols, medians)

    X_train_p = X_train_p.loc[:, list(feature_cols)]
    X_valid_p = X_valid_p.loc[:, list(feature_cols)]

    cat_indices = get_cat_feature_indices(feature_cols, categorical_cols)
    return X_train_p, X_valid_p, numerical_cols, categorical_cols, cat_indices
