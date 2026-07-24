"""建模：标签构造、训练划分、CatBoost 训练、指标与模型保存。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_SEED,
    TASK1_FEATURES,
    TASK1_LABEL_ORDER,
    TASK1_MODEL_PARAMS,
    TASK1_NAME,
    TASK1_TARGET,
    TASK2_FEATURES,
    TASK2_LABEL_ORDER,
    TASK2_MODEL_PARAMS,
    TASK2_NAME,
    TASK2_TARGET,
    TASK3_FEATURES,
    TASK3_LABEL_ORDER,
    TASK3_MODEL_PARAMS,
    TASK3_NAME,
    TASK3_SOURCE_SCORE,
    TASK3_THRESHOLD,
)
from src.preprocessing import prepare_xy
from src.visualization import plot_confusion_matrix, plot_feature_importance

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


def build_task1_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """构造任务一的 X 与 y。"""
    X = df.loc[:, TASK1_FEATURES].copy()
    y = _normalize_task1_labels(df[TASK1_TARGET])
    return X, y


def build_task2_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """构造任务二的 X 与 y（home_charging_available，0/1）。"""
    X = df.loc[:, TASK2_FEATURES].copy()
    y_raw = pd.to_numeric(df[TASK2_TARGET], errors="coerce")
    if y_raw.isna().any():
        raise ValueError("任务二目标列存在缺失或无法解析为数值的值。")
    # 规范为整数 0/1
    unique_vals = set(y_raw.unique().tolist())
    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"任务二目标值异常，期望为 0/1，实际出现：{sorted(unique_vals)}")
    y = y_raw.astype(int)
    return X, y


def build_task3_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
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


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
    model.fit(
        X_train,
        y_train,
        cat_features=list(cat_features),
    )
    return model


def _value_counts_dict(series: pd.Series) -> Dict[str, int]:
    """将类别分布转为 JSON 可序列化字典。"""
    counts = series.value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def extract_feature_importance(
    model: CatBoostClassifier,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """提取特征重要性并按从高到低排序。"""
    importances = model.get_feature_importance()
    fi = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": importances,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """保存 metrics.json（UTF-8，ensure_ascii=False）。"""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("已保存指标文件：%s", path)


def run_task(
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
    """
    执行单个任务的完整训练与验证流程。

    返回包含 accuracy 等关键信息的字典，供总成绩汇总。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("开始任务：%s", task_name)
    logger.info("特征数=%d，特征=%s", len(feature_cols), list(feature_cols))

    # 1. 分层划分
    X_train, X_valid, y_train, y_valid = stratified_split(X, y, test_size=test_size)

    # 2. 预处理（仅用训练集统计量）
    X_train_p, X_valid_p, _, _, cat_indices = prepare_xy(
        X_train, X_valid, feature_cols
    )

    # 3. 训练
    model = train_catboost(X_train_p, y_train, cat_indices, model_params)

    # 4. 预测类别与概率
    y_pred = model.predict(X_valid_p)
    # CatBoost 多分类可能返回二维数组
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    y_pred = pd.Series(y_pred, index=y_valid.index)

    # 统一任务一标签字符串形式
    if is_multiclass:
        y_pred = y_pred.astype(str).str.strip()
        # 某些版本可能返回带空格的标签
        y_pred = y_pred.replace({"low": "Low", "medium": "Medium", "high": "High"})
    else:
        y_pred = pd.to_numeric(y_pred, errors="coerce").astype(int)

    y_proba = model.predict_proba(X_valid_p)
    class_names = list(model.classes_)
    # 统一类别名为字符串，便于对齐概率列
    class_names_str = [str(c) for c in class_names]

    # 5. 指标
    acc = float(accuracy_score(y_valid, y_pred))
    report = classification_report(
        y_valid,
        y_pred,
        labels=list(label_order) if is_multiclass else [0, 1],
        output_dict=True,
        zero_division=0,
    )
    logger.info("任务 %s 验证集 Accuracy = %.6f", task_name, acc)
    logger.info(
        "classification_report:\n%s",
        classification_report(
            y_valid,
            y_pred,
            labels=list(label_order) if is_multiclass else [0, 1],
            zero_division=0,
        ),
    )

    # 6. 混淆矩阵
    if is_multiclass:
        labels_for_cm: List[Any] = list(label_order)
        y_true_cm = y_valid.astype(str)
        y_pred_cm = y_pred.astype(str)
    else:
        labels_for_cm = [0, 1]
        y_true_cm = y_valid.astype(int)
        y_pred_cm = y_pred.astype(int)

    cm = confusion_matrix(y_true_cm, y_pred_cm, labels=labels_for_cm)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{lab}" for lab in labels_for_cm],
        columns=[f"pred_{lab}" for lab in labels_for_cm],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    plot_confusion_matrix(
        cm,
        labels=[str(lab) for lab in labels_for_cm],
        title=f"{task_name} Confusion Matrix",
        save_path=output_dir / "confusion_matrix.png",
    )

    # 7. 验证预测结果（保留原始索引）
    pred_df = pd.DataFrame(
        {
            "sample_index": y_valid.index.astype(int),
            "true_label": y_valid.values,
            "predicted_label": y_pred.values,
        }
    )

    # 概率列顺序：任务一 Low/Medium/High；任务二/三 0/1
    if prob_class_names is None:
        if is_multiclass:
            desired_prob_names = list(TASK1_LABEL_ORDER)
        else:
            desired_prob_names = ["0", "1"]
    else:
        desired_prob_names = list(prob_class_names)

    proba_map = {name: y_proba[:, idx] for idx, name in enumerate(class_names_str)}
    for name in desired_prob_names:
        col_name = f"prob_{name}"
        if name in proba_map:
            pred_df[col_name] = proba_map[name]
        elif name in {"0", "1"} and str(int(name)) in proba_map:
            pred_df[col_name] = proba_map[str(int(name))]
        else:
            # 尝试数值匹配（例如类名为 0 而期望 "0"）
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
                logger.warning("任务 %s 缺少概率列对应类别：%s", task_name, name)

    pred_df.to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 8. 特征重要性
    fi = extract_feature_importance(model, feature_cols)
    fi.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_feature_importance(
        fi,
        title=f"{task_name} Feature Importance",
        save_path=output_dir / "feature_importance.png",
    )

    # 9. 保存模型
    model_path = output_dir / "model.cbm"
    model.save_model(str(model_path))
    logger.info("模型已保存：%s", model_path)

    # 10. metrics.json
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
    save_metrics(metrics, output_dir / "metrics.json")

    logger.info("任务 %s 完成。", task_name)
    return metrics


def run_all_tasks(
    df: pd.DataFrame,
    output_dirs: Dict[str, Path],
    test_size: float,
) -> Dict[str, float]:
    """依次运行三个独立任务，返回各任务 Accuracy。"""
    accuracies: Dict[str, float] = {}

    # 任务一
    X1, y1 = build_task1_xy(df)
    m1 = run_task(
        task_name=TASK1_NAME,
        X=X1,
        y=y1,
        feature_cols=TASK1_FEATURES,
        label_order=TASK1_LABEL_ORDER,
        model_params=TASK1_MODEL_PARAMS,
        output_dir=output_dirs["task1"],
        test_size=test_size,
        is_multiclass=True,
        prob_class_names=TASK1_LABEL_ORDER,
    )
    accuracies["ACC1"] = float(m1["accuracy"])

    # 任务二
    X2, y2 = build_task2_xy(df)
    m2 = run_task(
        task_name=TASK2_NAME,
        X=X2,
        y=y2,
        feature_cols=TASK2_FEATURES,
        label_order=TASK2_LABEL_ORDER,
        model_params=TASK2_MODEL_PARAMS,
        output_dir=output_dirs["task2"],
        test_size=test_size,
        is_multiclass=False,
        prob_class_names=TASK2_LABEL_ORDER,
    )
    accuracies["ACC2"] = float(m2["accuracy"])

    # 任务三
    X3, y3 = build_task3_xy(df)
    m3 = run_task(
        task_name=TASK3_NAME,
        X=X3,
        y=y3,
        feature_cols=TASK3_FEATURES,
        label_order=TASK3_LABEL_ORDER,
        model_params=TASK3_MODEL_PARAMS,
        output_dir=output_dirs["task3"],
        test_size=test_size,
        is_multiclass=False,
        prob_class_names=TASK3_LABEL_ORDER,
    )
    accuracies["ACC3"] = float(m3["accuracy"])

    return accuracies
