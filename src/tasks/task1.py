"""任务一：电动汽车采用意愿分类（朴素基线，逻辑保持不变）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.common.baseline import run_baseline_task
from src.config import (
    TASK1_FEATURES,
    TASK1_LABEL_ORDER,
    TASK1_MODEL_PARAMS,
    TASK1_NAME,
    TASK1_TARGET,
)

logger = logging.getLogger(__name__)


def _normalize_task1_labels(series: pd.Series) -> pd.Series:
    """清理任务一标签的空格与大小写，统一为 Low / Medium / High。"""
    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {"low": "Low", "medium": "Medium", "high": "High"}
    result = cleaned.map(mapping)
    invalid = result.isna() & series.notna()
    if invalid.any():
        bad_values = series[invalid].unique().tolist()
        raise ValueError(
            f"任务一目标列存在无法识别的标签：{bad_values}。"
            f"期望为 Low / Medium / High。"
        )
    if result.isna().any():
        raise ValueError("任务一目标列存在缺失值，无法进行分层划分。")
    return result


def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """构造任务一的 X 与 y。"""
    X = df.loc[:, TASK1_FEATURES].copy()
    y = _normalize_task1_labels(df[TASK1_TARGET])
    return X, y


def run(df: pd.DataFrame, output_dir: Path, test_size: float) -> Dict[str, Any]:
    """任务一训练入口。"""
    X, y = build_xy(df)
    return run_baseline_task(
        task_name=TASK1_NAME,
        X=X,
        y=y,
        feature_cols=TASK1_FEATURES,
        label_order=TASK1_LABEL_ORDER,
        model_params=TASK1_MODEL_PARAMS,
        output_dir=output_dir,
        test_size=test_size,
        is_multiclass=True,
        prob_class_names=TASK1_LABEL_ORDER,
    )
