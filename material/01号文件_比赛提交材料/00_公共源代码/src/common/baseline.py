"""任务一 / 任务三共用的朴素基线训练流程（逻辑保持不变）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.common.metrics import (
    classification_report_dict,
    make_confusion_matrix_df,
    save_json,
)
from src.common.preprocessing import prepare_xy
from src.common.visualization import plot_confusion_matrix, plot_feature_importance
from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """分层划分训练集与验证集，保留原始索引。"""
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "分层划分完成：train=%d，valid=%d，test_size=%.2f",
        len(X_train),
        len(X_valid),
        test_size,
    )
    return X_train, X_valid, y_train, y_valid


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: Sequence[int],
    params: Dict[str, Any],
) -> CatBoostClassifier:
    """训练 CatBoostClassifier。"""
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=list(cat_features))
    return model


def extract_feature_importance(
    model: CatBoostClassifier,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """提取特征重要性并按从高到低排序。"""
    importances = model.get_feature_importance()
    return (
        pd.DataFrame({"feature": list(feature_names), "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _value_counts_dict(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def run_baseline_task(
    *,
    task_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: Sequence[str],
    label_order: Sequence[str],
    model_params: Dict[str, Any],
    output_dir: Path,
    test_size: float,
    is_multiclass: bool,
    prob_class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """执行朴素基线：单次分层划分 + 单一 CatBoost + 默认阈值。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("开始任务：%s", task_name)
    logger.info("特征数=%d，特征=%s", len(feature_cols), list(feature_cols))

    X_train, X_valid, y_train, y_valid = stratified_split(X, y, test_size=test_size)
    X_train_p, X_valid_p, _, _, cat_indices = prepare_xy(X_train, X_valid, feature_cols)
    model = train_catboost(X_train_p, y_train, cat_indices, model_params)

    y_pred = model.predict(X_valid_p)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    y_pred = pd.Series(y_pred, index=y_valid.index)

    if is_multiclass:
        y_pred = y_pred.astype(str).str.strip()
        y_pred = y_pred.replace({"low": "Low", "medium": "Medium", "high": "High"})
        labels_for_cm: List[Any] = list(label_order)
        y_true_cm = y_valid.astype(str)
        y_pred_cm = y_pred.astype(str)
    else:
        y_pred = pd.to_numeric(y_pred, errors="coerce").astype(int)
        labels_for_cm = [0, 1]
        y_true_cm = y_valid.astype(int)
        y_pred_cm = y_pred.astype(int)

    y_proba = model.predict_proba(X_valid_p)
    class_names_str = [str(c) for c in model.classes_]

    acc = float(accuracy_score(y_valid, y_pred))
    report = classification_report_dict(y_valid, y_pred, labels_for_cm)
    logger.info("任务 %s 验证集 Accuracy = %.6f", task_name, acc)
    logger.info(
        "classification_report:\n%s",
        classification_report(y_valid, y_pred, labels=labels_for_cm, zero_division=0),
    )

    cm, cm_df = make_confusion_matrix_df(y_true_cm, y_pred_cm, labels_for_cm)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    plot_confusion_matrix(
        cm,
        labels=[str(lab) for lab in labels_for_cm],
        title=f"{task_name} Confusion Matrix",
        save_path=output_dir / "confusion_matrix.png",
    )

    pred_df = pd.DataFrame(
        {
            "sample_index": y_valid.index.astype(int),
            "true_label": y_valid.values,
            "predicted_label": y_pred.values,
        }
    )
    desired_prob_names = list(prob_class_names) if prob_class_names is not None else (
        list(label_order) if is_multiclass else ["0", "1"]
    )
    proba_map = {name: y_proba[:, idx] for idx, name in enumerate(class_names_str)}
    for name in desired_prob_names:
        col_name = f"prob_{name}"
        if name in proba_map:
            pred_df[col_name] = proba_map[name]
        else:
            matched = False
            for key, vals in proba_map.items():
                try:
                    if float(key) == float(name):
                        pred_df[col_name] = vals
                        matched = True
                        break
                except ValueError:
                    continue
            if not matched:
                pred_df[col_name] = np.nan

    pred_df.to_csv(
        output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )

    fi = extract_feature_importance(model, feature_cols)
    fi.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_feature_importance(
        fi,
        title=f"{task_name} Feature Importance",
        save_path=output_dir / "feature_importance.png",
    )

    model.save_model(str(output_dir / "model.cbm"))

    metrics: Dict[str, Any] = {
        "task_name": task_name,
        "accuracy": acc,
        "train_samples": int(len(X_train_p)),
        "validation_samples": int(len(X_valid_p)),
        "feature_count": int(len(feature_cols)),
        "target_distribution_train": _value_counts_dict(y_train),
        "target_distribution_validation": _value_counts_dict(y_valid),
        "classification_report": report,
    }
    save_json(metrics, output_dir / "metrics.json")
    logger.info("任务 %s 完成。", task_name)
    return metrics
