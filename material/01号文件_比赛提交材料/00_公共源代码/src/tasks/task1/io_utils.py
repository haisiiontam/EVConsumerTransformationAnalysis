"""任务一 io_utils：兼容 re-export + 任务一 STAGES / 日志名。"""

from __future__ import annotations

from pathlib import Path

from src.common.io_utils import (  # noqa: F401
    RunState,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    config_hash,
    now_iso,
)
from src.common.io_utils import setup_file_logging as _setup_file_logging

STAGES = [
    "split",
    "baseline",
    "ablation",
    "feature_selection",
    "catboost_optuna",
    "catboost_confirmation",
    "m1_catboost",
    "m2_ordinal",
    "m3_regression",
    "lightgbm_optuna",
    "lightgbm_model",
    "xgboost",
    "logistic",
    "diversity",
    "soft_blend",
    "stacking",
    "class_bias",
    "multiseed",
    "final_validation",
    "shap",
    "report",
]


def setup_file_logging(log_path: Path, run_id: str):
    return _setup_file_logging(
        log_path, run_id, logger_name="task1", handler_attr="_task1_file"
    )
