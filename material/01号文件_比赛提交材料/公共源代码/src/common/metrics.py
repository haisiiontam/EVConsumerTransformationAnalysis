from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """兼容 numpy / pandas 标量，避免 json.dump 失败。"""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_json(data: Dict[str, Any], path: Path) -> None:
    """保存 JSON（UTF-8，ensure_ascii=False）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("已保存 JSON：%s", path)


def binary_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_prob_1: Sequence[float] | None = None,
) -> Dict[str, Any]:
    """计算二分类常用指标。"""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision_0": float(
            precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)
        ),
        "recall_0": float(
            recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)
        ),
        "f1_0": float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)),
        "precision_1": float(
            precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
        ),
        "recall_1": float(
            recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
        ),
        "f1_1": float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "predicted_0_count": int((y_pred_arr == 0).sum()),
        "predicted_1_count": int((y_pred_arr == 1).sum()),
    }

    if y_prob_1 is not None:
        prob = np.asarray(y_prob_1, dtype=float)

        prob = np.clip(prob, 1e-15, 1.0 - 1e-15)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["log_loss"] = float(log_loss(y_true_arr, prob))
        except ValueError:
            metrics["log_loss"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["log_loss"] = float("nan")

    return metrics


def majority_baseline(y: Sequence[Any]) -> Dict[str, Any]:
    """多数类基准：数量最多的类别样本数 / 总样本数。"""
    series = pd.Series(y)
    counts = series.value_counts()
    majority_class = counts.index[0]
    majority_count = int(counts.iloc[0])
    total = int(len(series))
    return {
        "majority_class": int(majority_class) if pd.api.types.is_number(majority_class) else majority_class,
        "majority_class_count": majority_count,
        "majority_baseline_accuracy": float(majority_count / total) if total else 0.0,
    }


def make_confusion_matrix_df(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    """生成混淆矩阵数组与 DataFrame（行=真实，列=预测）。"""
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{lab}" for lab in labels],
        columns=[f"pred_{lab}" for lab in labels],
    )
    return cm, cm_df


def classification_report_dict(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any],
) -> Dict[str, Any]:
    """classification_report 字典形式。"""
    return classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
