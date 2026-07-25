"""任务三：里程焦虑敏感度识别（朴素基线，逻辑保持不变）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.common.baseline import run_baseline_task
from src.config import (
    TASK3_FEATURES,
    TASK3_LABEL_ORDER,
    TASK3_MODEL_PARAMS,
    TASK3_NAME,
    TASK3_SOURCE_SCORE,
    TASK3_THRESHOLD,
)

logger = logging.getLogger(__name__)


def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    构造任务三的 X 与 y。

    标签：range_anxiety_score > 5 -> 1，否则 0。
    注意：range_anxiety_score 不得进入特征。
    """
    X = df.loc[:, TASK3_FEATURES].copy()
    score = pd.to_numeric(df[TASK3_SOURCE_SCORE], errors="coerce")
    if score.isna().any():
        raise ValueError("任务三原始评分列存在缺失或无法解析的值。")
    y = (score > TASK3_THRESHOLD).astype(int)
    return X, y


def run(df: pd.DataFrame, output_dir: Path, test_size: float) -> Dict[str, Any]:
    """任务三训练入口。"""
    X, y = build_xy(df)
    return run_baseline_task(
        task_name=TASK3_NAME,
        X=X,
        y=y,
        feature_cols=TASK3_FEATURES,
        label_order=TASK3_LABEL_ORDER,
        model_params=TASK3_MODEL_PARAMS,
        output_dir=output_dir,
        test_size=test_size,
        is_multiclass=False,
        prob_class_names=TASK3_LABEL_ORDER,
    )
