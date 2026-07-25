"""任务二优化：诊断、机制特征、OOF、融合权重与阈值搜索、最终验证。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.common.baseline import extract_feature_importance
from src.common.metrics import (
    binary_classification_metrics,
    majority_baseline,
    make_confusion_matrix_df,
    save_json,
)
from src.common.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_probability_distribution,
    plot_roc_curve,
)
from src.common.separability import (
    assess_weak_signal,
    leak_risk_probe,
    univariate_separability,
    write_weak_signal_markdown,
)
from src.config import (
    RANDOM_SEED,
    TASK2_BASE_FEATURES,
    TASK2_CATBOOST_COMMON,
    TASK2_DIAG_CATEGORICAL,
    TASK2_EARLY_STOPPING_ROUNDS,
    TASK2_EXCLUDED_FEATURES,
    TASK2_KNOWN_CATEGORICAL,
    TASK2_LEAK_RISK_PROBE_FEATURES,
    TASK2_MECHANISM_FEATURES,
    TASK2_NAME,
    TASK2_SAFE_CANDIDATE_FEATURES,
    TASK2_TARGET,
    TASK2_TIE_EPS,
    TASK2_WEAK_SIGNAL_AUC_GAP,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 特征解析与机制特征
# ---------------------------------------------------------------------------

def resolve_safe_features(df: pd.DataFrame) -> List[str]:
    """从安全候选中筛选实际存在的列；缺失则警告并跳过。"""
    available: List[str] = []
    for col in TASK2_SAFE_CANDIDATE_FEATURES:
        if col in TASK2_EXCLUDED_FEATURES:
            continue
        if col not in df.columns:
            logger.warning("任务二安全候选特征缺失，已跳过：%s", col)
            continue
        if col not in available:
            available.append(col)
    return available


def check_task2_base_columns(df: pd.DataFrame) -> None:
    """官方基础字段必须存在。"""
    required = TASK2_BASE_FEATURES + [TASK2_TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("任务二缺少官方基础字段：" + ", ".join(missing))


def build_target(df: pd.DataFrame) -> pd.Series:
    """构造任务二标签 0/1。"""
    y_raw = pd.to_numeric(df[TASK2_TARGET], errors="coerce")
    if y_raw.isna().any():
        raise ValueError("任务二目标列存在缺失或无法解析为数值的值。")
    unique_vals = set(y_raw.unique().tolist())
    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"任务二目标值异常，期望为 0/1，实际出现：{sorted(unique_vals)}")
    return y_raw.astype(int)


def _to_numeric_series(series: pd.Series) -> Tuple[pd.Series, int]:
    """清洗数值：去逗号/空格后 to_numeric，返回转换失败数。"""
    if pd.api.types.is_numeric_dtype(series):
        converted = pd.to_numeric(series, errors="coerce")
        fail = int(series.notna().sum() - converted.notna().sum())
        return converted, fail

    as_str = series.astype(str).str.strip().str.replace(",", "", regex=False)
    as_str = as_str.replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan, "": np.nan})
    # 去掉常见货币符号
    as_str = as_str.str.replace(r"[$¥€]", "", regex=True)
    converted = pd.to_numeric(as_str, errors="coerce")
    # 原始非缺失但转换失败
    fail = int((series.notna() & converted.isna()).sum())
    return converted, fail


def add_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造机制特征与住房/充电代理特征。

    仅使用行内信息（不加目标、不用全表统计），避免泄露。
    依赖训练集分位数的相对特征在 preprocessor 中另行生成。
    """
    out = df.copy()
    eps = 1.0

    nearest = access = income = daily = weekly = fuel = None
    has_nearest = "nearest_charging_station_km" in out.columns
    has_access = "charging_station_accessibility" in out.columns

    if has_nearest:
        nearest, _ = _to_numeric_series(out["nearest_charging_station_km"])
    if has_access:
        access, _ = _to_numeric_series(out["charging_station_accessibility"])
    if has_nearest and has_access:
        out["charging_gap_index"] = nearest / (access + eps)
        out["charging_convenience_index"] = access / (nearest + eps)

    if "daily_commute_km" in out.columns and "weekly_travel_distance_km" in out.columns:
        daily, _ = _to_numeric_series(out["daily_commute_km"])
        weekly, _ = _to_numeric_series(out["weekly_travel_distance_km"])
        out["commute_ratio"] = 7.0 * daily / (weekly + eps)
        out["noncommute_km"] = (weekly - 7.0 * daily).clip(lower=0)

    if "annual_income" in out.columns:
        income, _ = _to_numeric_series(out["annual_income"])
        out["log_annual_income"] = np.log1p(income.clip(lower=0))

    if "fuel_expense_per_month" in out.columns and income is not None:
        fuel, _ = _to_numeric_series(out["fuel_expense_per_month"])
        out["fuel_income_pressure"] = 12.0 * fuel / (income + eps)

    if "vehicle_age_years" in out.columns and "fuel_expense_per_month" in out.columns:
        vage, _ = _to_numeric_series(out["vehicle_age_years"])
        if fuel is None:
            fuel, _ = _to_numeric_series(out["fuel_expense_per_month"])
        out["replacement_pressure"] = vage * fuel

    # 城市类型 one-hot 风格代理（字符串匹配，缺失为 0）
    if "city_type" in out.columns:
        city = out["city_type"].astype(str).str.strip().str.lower()
        out["city_is_urban"] = (city == "urban").astype(float)
        out["city_is_suburban"] = (city == "suburban").astype(float)
        out["city_is_rural"] = (city == "rural").astype(float)
        if income is not None:
            out["income_x_urban"] = income * out["city_is_urban"]
            out["income_x_suburban"] = income * out["city_is_suburban"]
            out["income_x_rural"] = income * out["city_is_rural"]
        if nearest is not None:
            out["near_x_rural"] = nearest * out["city_is_rural"]

    if has_access and "previous_ev_experience" in out.columns:
        prev, _ = _to_numeric_series(out["previous_ev_experience"])
        out["access_x_prev_ev"] = access * prev

    if income is not None and nearest is not None:
        out["income_over_near"] = income / (nearest + eps)

    if "electricity_cost_per_kwh" in out.columns and income is not None:
        elec, _ = _to_numeric_series(out["electricity_cost_per_kwh"])
        out["electricity_x_income"] = elec * income

    return out


def _add_train_relative_features(
    df: pd.DataFrame,
    income_median: float | None,
) -> pd.DataFrame:
    """基于训练集收入中位数的相对特征（防泄露：中位数只来自当前训练折）。"""
    out = df.copy()
    if income_median is None or "annual_income" not in out.columns:
        return out
    income, _ = _to_numeric_series(out["annual_income"])
    high = (income >= income_median).astype(float)
    out["income_ge_train_median"] = high
    if "city_is_urban" in out.columns:
        out["urban_high_income"] = out["city_is_urban"] * high
    elif "city_type" in out.columns:
        city = out["city_type"].astype(str).str.strip().str.lower()
        out["urban_high_income"] = ((city == "urban").astype(float) * high)
    if "city_is_suburban" in out.columns:
        out["suburban_high_income"] = out["city_is_suburban"] * high
    elif "city_type" in out.columns:
        city = out["city_type"].astype(str).str.strip().str.lower()
        out["suburban_high_income"] = ((city == "suburban").astype(float) * high)
    return out


# ---------------------------------------------------------------------------
# 预处理器（仅用当前训练折 / 开发集统计量）
# ---------------------------------------------------------------------------

@dataclass
class Task2PreprocessorState:
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_medians: Dict[str, float] = field(default_factory=dict)
    final_feature_order: List[str] = field(default_factory=list)
    income_median: float | None = None


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
        income_s, _ = _to_numeric_series(X_work["annual_income"])
        if income_s.notna().any():
            income_median = float(income_s.median())
    X_work = _add_train_relative_features(X_work, income_median)

    # 机制特征可能新增，按 feature_cols 过滤实际存在的
    cols = [c for c in feature_cols if c in X_work.columns]
    numerical, categorical = identify_task2_types(X_work, cols)

    medians: Dict[str, float] = {}
    for col in numerical:
        series, _ = _to_numeric_series(X_work[col])
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
            series, _ = _to_numeric_series(X_work[col])
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


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------

def _label_dist_row(name: str, y: pd.Series) -> Dict[str, Any]:
    n = len(y)
    c0 = int((y == 0).sum())
    c1 = int((y == 1).sum())
    return {
        "dataset": name,
        "class_0_count": c0,
        "class_1_count": c1,
        "class_0_rate": float(c0 / n) if n else 0.0,
        "class_1_rate": float(c1 / n) if n else 0.0,
    }


def run_diagnostics(
    df: pd.DataFrame,
    y_all: pd.Series,
    y_dev: pd.Series,
    y_val: pd.Series,
    candidate_features: Sequence[str],
    output_dir: Path,
) -> Dict[str, Any]:
    """生成任务二诊断报告。"""
    diag_dir = Path(output_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # 多数类基准
    maj = majority_baseline(y_all)
    save_json(maj, diag_dir / "majority_baseline.json")
    logger.info(
        "多数类基准：class=%s, count=%d, accuracy=%.6f",
        maj["majority_class"],
        maj["majority_class_count"],
        maj["majority_baseline_accuracy"],
    )

    # 标签分布
    dist_df = pd.DataFrame(
        [
            _label_dist_row("full", y_all),
            _label_dist_row("develop", y_dev),
            _label_dist_row("final_validation", y_val),
        ]
    )
    dist_df.to_csv(diag_dir / "target_distribution.csv", index=False, encoding="utf-8-sig")

    # 字段质量
    quality_rows: List[Dict[str, Any]] = []
    X_mech = add_mechanism_features(df)
    for col in candidate_features:
        if col not in X_mech.columns and col not in df.columns:
            continue
        src = X_mech[col] if col in X_mech.columns else df[col]
        dtype_before = str(src.dtype)
        missing_before = int(src.isna().sum())
        missing_rate_before = float(missing_before / len(src)) if len(src) else 0.0

        # 尝试数值转换诊断
        converted, fail_count = _to_numeric_series(src)
        # 若转换失败过多，仍按类别看待
        if fail_count > 0.2 * src.notna().sum() and not pd.api.types.is_numeric_dtype(src):
            dtype_after = "categorical_string"
            cleaned = src.astype(str).str.strip()
            missing_after = int(src.isna().sum())
            unique_count = int(cleaned.nunique(dropna=True))
            numeric_fail = fail_count
        else:
            dtype_after = "numeric"
            missing_after = int(converted.isna().sum())
            unique_count = int(converted.nunique(dropna=True))
            numeric_fail = fail_count

        quality_rows.append(
            {
                "feature": col,
                "dtype_before": dtype_before,
                "dtype_after": dtype_after,
                "missing_count_before": missing_before,
                "missing_rate_before": missing_rate_before,
                "missing_count_after": missing_after,
                "unique_count": unique_count,
                "numeric_conversion_failure_count": numeric_fail,
            }
        )
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(diag_dir / "feature_quality.csv", index=False, encoding="utf-8-sig")

    # 类别特征分组目标率（仅诊断，不作为模型输入）
    cat_rows: List[Dict[str, Any]] = []
    for col in TASK2_DIAG_CATEGORICAL:
        if col not in df.columns:
            logger.warning("诊断类别特征缺失，跳过：%s", col)
            continue
        tmp = pd.DataFrame({"category": df[col].astype(str).str.strip(), "y": y_all.values})
        tmp.loc[df[col].isna(), "category"] = "Unknown"
        grouped = tmp.groupby("category", dropna=False)["y"].agg(["count", "sum"])
        for cat, row in grouped.iterrows():
            cat_rows.append(
                {
                    "feature": col,
                    "category": cat,
                    "sample_count": int(row["count"]),
                    "target_1_count": int(row["sum"]),
                    "target_1_rate": float(row["sum"] / row["count"]) if row["count"] else 0.0,
                }
            )
    pd.DataFrame(cat_rows).to_csv(
        diag_dir / "categorical_target_rates.csv", index=False, encoding="utf-8-sig"
    )

    # 数值特征按目标分组统计
    num_rows: List[Dict[str, Any]] = []
    for col in candidate_features:
        if col not in df.columns and col not in X_mech.columns:
            continue
        src = X_mech[col] if col in X_mech.columns else df[col]
        series, fail = _to_numeric_series(src)
        # 跳过明显类别列
        if fail > 0.2 * src.notna().sum() and not pd.api.types.is_numeric_dtype(src):
            continue
        for label in (0, 1):
            part = series[y_all.values == label]
            num_rows.append(
                {
                    "feature": col,
                    "target": label,
                    "count": int(part.notna().sum()),
                    "mean": float(part.mean()) if part.notna().any() else np.nan,
                    "median": float(part.median()) if part.notna().any() else np.nan,
                    "std": float(part.std()) if part.notna().sum() > 1 else np.nan,
                    "min": float(part.min()) if part.notna().any() else np.nan,
                    "max": float(part.max()) if part.notna().any() else np.nan,
                }
            )
    pd.DataFrame(num_rows).to_csv(
        diag_dir / "numeric_group_statistics.csv", index=False, encoding="utf-8-sig"
    )

    return maj


# ---------------------------------------------------------------------------
# OOF 训练
# ---------------------------------------------------------------------------

def _catboost_params(balanced: bool) -> Dict[str, Any]:
    params = dict(TASK2_CATBOOST_COMMON)
    if balanced:
        params["auto_class_weights"] = "Balanced"
    return params


def _prob_class1(model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = [int(c) for c in model.classes_]
    if 1 in classes:
        return proba[:, classes.index(1)]
    # 退化情况
    return proba[:, -1]


def generate_oof_predictions(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    feature_cols: Sequence[str],
    n_folds: int,
    balanced: bool,
    model_tag: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """
    分层 K 折生成 OOF 概率。

    返回：oof_predictions_df, fold_metrics_df, best_iterations
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_prob0 = np.zeros(len(X_dev), dtype=float)
    oof_prob1 = np.zeros(len(X_dev), dtype=float)
    oof_fold = np.full(len(X_dev), -1, dtype=int)
    fold_rows: List[Dict[str, Any]] = []
    best_iterations: List[int] = []

    # 保留原始样本索引；特征矩阵按位置与 y 对齐
    orig_index = X_dev.index.to_numpy()
    X_feat = X_dev.reset_index(drop=True)
    y_arr = y_dev.reset_index(drop=True)

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_feat, y_arr)):
        X_tr_raw = X_feat.iloc[tr_idx]
        X_va_raw = X_feat.iloc[va_idx]
        y_tr = y_arr.iloc[tr_idx]
        y_va = y_arr.iloc[va_idx]

        state = fit_task2_preprocessor(X_tr_raw, feature_cols)
        X_tr = transform_task2_data(X_tr_raw, state)
        X_va = transform_task2_data(X_va_raw, state)
        cat_idx = get_cat_indices(state)

        params = _catboost_params(balanced)
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_idx,
            eval_set=(X_va, y_va),
            early_stopping_rounds=TASK2_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        best_iter = int(model.get_best_iteration() or params["iterations"])
        best_iterations.append(best_iter)

        prob1 = _prob_class1(model, X_va)
        prob0 = 1.0 - prob1
        oof_prob1[va_idx] = prob1
        oof_prob0[va_idx] = prob0
        oof_fold[va_idx] = fold_id

        y_pred = (prob1 >= 0.5).astype(int)
        m = binary_classification_metrics(y_va, y_pred, prob1)
        fold_rows.append(
            {
                "fold": fold_id,
                "model": model_tag,
                "accuracy": m["accuracy"],
                "roc_auc": m["roc_auc"],
                "log_loss": m["log_loss"],
                "balanced_accuracy": m["balanced_accuracy"],
                "recall_0": m["recall_0"],
                "recall_1": m["recall_1"],
                "predicted_0_count": m["predicted_0_count"],
                "predicted_1_count": m["predicted_1_count"],
                "best_iteration": best_iter,
            }
        )
        logger.info(
            "OOF %s fold=%d acc=%.4f bal_acc=%.4f recall_0=%.4f best_iter=%d",
            model_tag,
            fold_id,
            m["accuracy"],
            m["balanced_accuracy"],
            m["recall_0"],
            best_iter,
        )

    if (oof_fold < 0).any():
        raise RuntimeError("OOF 未覆盖全部开发集样本，请检查交叉验证划分。")

    oof_df = pd.DataFrame(
        {
            "sample_index": orig_index.astype(int),
            "true_label": y_arr.values.astype(int),
            "prob_0": oof_prob0,
            "prob_1": oof_prob1,
            "fold": oof_fold,
        }
    )
    fold_df = pd.DataFrame(fold_rows)
    return oof_df, fold_df, best_iterations


def evaluate_oof_at_threshold(
    oof_df: pd.DataFrame,
    threshold: float,
    prob_col: str = "prob_1",
) -> Dict[str, Any]:
    """在给定阈值下评估 OOF。"""
    y_true = oof_df["true_label"].values
    prob = oof_df[prob_col].values
    y_pred = (prob >= threshold).astype(int)
    return binary_classification_metrics(y_true, y_pred, prob)


# ---------------------------------------------------------------------------
# 融合搜索
# ---------------------------------------------------------------------------

def search_blend_threshold(
    plain_oof: pd.DataFrame,
    balanced_oof: pd.DataFrame,
    alpha_step: float,
    thr_min: float,
    thr_max: float,
    thr_step: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """在开发集 OOF 上搜索 alpha 与 threshold。"""
    # 按 sample_index 对齐
    plain = plain_oof.set_index("sample_index").sort_index()
    bal = balanced_oof.set_index("sample_index").sort_index()
    if not plain.index.equals(bal.index):
        common = plain.index.intersection(bal.index)
        plain = plain.loc[common]
        bal = bal.loc[common]

    y_true = plain["true_label"].values.astype(int)
    p_plain = plain["prob_1"].values.astype(float)
    p_bal = bal["prob_1"].values.astype(float)

    alphas = np.round(np.arange(0.0, 1.0 + 1e-12, alpha_step), 10)
    thresholds = np.round(np.arange(thr_min, thr_max + 1e-12, thr_step), 10)

    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        p_final = alpha * p_plain + (1.0 - alpha) * p_bal
        for thr in thresholds:
            y_pred = (p_final >= thr).astype(int)
            m = binary_classification_metrics(y_true, y_pred, p_final)
            rows.append(
                {
                    "alpha": float(alpha),
                    "threshold": float(thr),
                    "accuracy": m["accuracy"],
                    "roc_auc": m["roc_auc"],
                    "log_loss": m["log_loss"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "precision_0": m["precision_0"],
                    "recall_0": m["recall_0"],
                    "f1_0": m["f1_0"],
                    "precision_1": m["precision_1"],
                    "recall_1": m["recall_1"],
                    "f1_1": m["f1_1"],
                    "predicted_0_count": m["predicted_0_count"],
                    "predicted_1_count": m["predicted_1_count"],
                }
            )

    search_df = pd.DataFrame(rows)
    best = select_best_blend_row(search_df)
    best_config = {
        "alpha": float(best["alpha"]),
        "threshold": float(best["threshold"]),
        "selection_metric": "oof_accuracy",
        "oof_accuracy": float(best["accuracy"]),
        "oof_balanced_accuracy": float(best["balanced_accuracy"]),
        "oof_roc_auc": float(best["roc_auc"]),
        "oof_recall_0": float(best["recall_0"]),
        "oof_recall_1": float(best["recall_1"]),
    }
    return search_df, best_config


def select_best_blend_row(search_df: pd.DataFrame) -> pd.Series:
    """
    主目标 OOF Accuracy；平局规则：
    1) Balanced Accuracy 更高
    2) 0 类 Recall 更高
    3) 阈值更接近 0.5
    4) alpha 更大
    """
    max_acc = search_df["accuracy"].max()
    cand = search_df[search_df["accuracy"] >= max_acc - TASK2_TIE_EPS].copy()
    cand["thr_dist"] = (cand["threshold"] - 0.5).abs()
    cand = cand.sort_values(
        by=["accuracy", "balanced_accuracy", "recall_0", "thr_dist", "alpha"],
        ascending=[False, False, False, True, False],
    )
    return cand.iloc[0]


# ---------------------------------------------------------------------------
# 最终训练与验证
# ---------------------------------------------------------------------------

def train_final_model(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    feature_cols: Sequence[str],
    balanced: bool,
    iterations: int,
) -> Tuple[CatBoostClassifier, Task2PreprocessorState]:
    """在完整开发集上训练最终模型。"""
    state = fit_task2_preprocessor(X_dev, feature_cols)
    X_tr = transform_task2_data(X_dev, state)
    cat_idx = get_cat_indices(state)
    params = _catboost_params(balanced)
    params["iterations"] = int(iterations)
    # 最终重训不再用外层验证集早停
    model = CatBoostClassifier(**params)
    model.fit(X_tr, y_dev, cat_features=cat_idx)
    return model, state


def predict_with_state(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    state: Task2PreprocessorState,
) -> np.ndarray:
    """变换后预测 P(y=1)。"""
    X_t = transform_task2_data(X, state)
    return _prob_class1(model, X_t)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

@dataclass
class Task2RunConfig:
    test_size: float = 0.2
    n_folds: int = 5
    alpha_step: float = 0.05
    threshold_min: float = 0.30
    threshold_max: float = 0.80
    threshold_step: float = 0.005
    test_df: Optional[pd.DataFrame] = None


def run(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.2,
    n_folds: int = 5,
    alpha_step: float = 0.05,
    threshold_min: float = 0.30,
    threshold_max: float = 0.80,
    threshold_step: float = 0.005,
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """任务二优化完整流程。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = output_dir / "oof"
    tuning_dir = output_dir / "tuning"
    final_dir = output_dir / "final_validation"
    for d in (oof_dir, tuning_dir, final_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("开始任务二优化流程")

    check_task2_base_columns(df)
    y_all = build_target(df)
    safe_features = resolve_safe_features(df)

    # 为了让机制特征出现在 feature list，需要源列存在
    feature_set_official = list(TASK2_BASE_FEATURES)
    feature_set_safe = list(safe_features)
    feature_set_mech = list(safe_features)
    # 仅当源字段存在时追加机制特征名
    X_probe = add_mechanism_features(df.head(2))
    for mcol in TASK2_MECHANISM_FEATURES:
        if mcol in X_probe.columns and mcol not in feature_set_mech:
            feature_set_mech.append(mcol)

    logger.info("官方特征：%s", feature_set_official)
    logger.info("安全扩展特征数：%d", len(feature_set_safe))
    logger.info("安全+机制特征数：%d", len(feature_set_mech))

    # 外层 80/20 分层划分（最终验证集不参与调参）
    indices = df.index.to_series()
    idx_dev, idx_val = train_test_split(
        indices,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y_all,
    )
    df_dev = df.loc[idx_dev].copy()
    df_val = df.loc[idx_val].copy()
    y_dev = y_all.loc[idx_dev]
    y_val = y_all.loc[idx_val]

    logger.info(
        "外层划分：develop=%d，final_validation=%d",
        len(df_dev),
        len(df_val),
    )

    # 诊断（含完整/开发/验证标签分布）
    diag_features = sorted(
        set(feature_set_mech) | set(feature_set_official) | set(TASK2_DIAG_CATEGORICAL)
    )
    maj = run_diagnostics(df, y_all, y_dev, y_val, diag_features, output_dir)
    majority_acc = float(maj["majority_baseline_accuracy"])
    majority_class = int(maj["majority_class"])

    # ---- 可分性诊断（仅开发集）+ 泄露风险探查 ----
    diag_dir = Path(output_dir) / "diagnostics"
    df_dev_fe = add_mechanism_features(df_dev)
    # 用开发集收入中位数生成相对特征，仅用于诊断展示
    dev_income_med = None
    if "annual_income" in df_dev.columns:
        _inc, _ = _to_numeric_series(df_dev["annual_income"])
        if _inc.notna().any():
            dev_income_med = float(_inc.median())
    df_dev_fe = _add_train_relative_features(df_dev_fe, dev_income_med)
    sep_cols = [c for c in feature_set_mech if c in df_dev_fe.columns]
    sep_df = univariate_separability(df_dev_fe, y_dev, sep_cols)
    sep_df.to_csv(diag_dir / "separability.csv", index=False, encoding="utf-8-sig")
    leak_df = leak_risk_probe(df_dev, y_dev, TASK2_LEAK_RISK_PROBE_FEATURES)
    leak_df.to_csv(diag_dir / "leak_risk_probe.csv", index=False, encoding="utf-8-sig")
    weak_report = assess_weak_signal(
        sep_df, leak_df, auc_gap_threshold=TASK2_WEAK_SIGNAL_AUC_GAP
    )
    save_json(weak_report, diag_dir / "weak_signal_report.json")
    write_weak_signal_markdown(weak_report, diag_dir / "WEAK_SIGNAL_REPORT.md")

    # ---- OOF：官方五特征 plain (A) ----
    logger.info("生成配置 A OOF：官方五特征 + plain")
    oof_a, folds_a, best_a = generate_oof_predictions(
        df_dev, y_dev, feature_set_official, n_folds, balanced=False, model_tag="A_plain_official"
    )

    # ---- OOF：安全扩展 plain (B) ----
    logger.info("生成配置 B OOF：安全扩展 + plain")
    oof_b, folds_b, best_b = generate_oof_predictions(
        df_dev, y_dev, feature_set_safe, n_folds, balanced=False, model_tag="B_plain_safe"
    )

    # ---- OOF：安全+机制 plain (C) 与 balanced (D/E) ----
    logger.info("生成配置 C OOF：安全+机制代理 + plain")
    oof_plain, folds_plain, best_plain = generate_oof_predictions(
        df_dev, y_dev, feature_set_mech, n_folds, balanced=False, model_tag="C_plain_mech"
    )
    logger.info("生成配置 D/E OOF：安全+机制代理 + balanced")
    oof_bal, folds_bal, best_bal = generate_oof_predictions(
        df_dev, y_dev, feature_set_mech, n_folds, balanced=True, model_tag="D_balanced_mech"
    )

    oof_plain.to_csv(oof_dir / "plain_oof_predictions.csv", index=False, encoding="utf-8-sig")
    oof_bal.to_csv(oof_dir / "balanced_oof_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics = pd.concat([folds_a, folds_b, folds_plain, folds_bal], ignore_index=True)
    fold_metrics.to_csv(oof_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    # 配置 A-D：阈值 0.5
    m_a = evaluate_oof_at_threshold(oof_a, 0.5)
    m_b = evaluate_oof_at_threshold(oof_b, 0.5)
    m_c = evaluate_oof_at_threshold(oof_plain, 0.5)
    m_d = evaluate_oof_at_threshold(oof_bal, 0.5)

    # 配置 E：融合搜索
    search_df, best_blend = search_blend_threshold(
        oof_plain,
        oof_bal,
        alpha_step=alpha_step,
        thr_min=threshold_min,
        thr_max=threshold_max,
        thr_step=threshold_step,
    )
    search_df.to_csv(tuning_dir / "blend_threshold_search.csv", index=False, encoding="utf-8-sig")
    save_json(best_blend, tuning_dir / "best_blend_config.json")

    # 融合 OOF 指标
    plain_aligned = oof_plain.set_index("sample_index").sort_index()
    bal_aligned = oof_bal.set_index("sample_index").sort_index()
    p_final_oof = (
        best_blend["alpha"] * plain_aligned["prob_1"].values
        + (1.0 - best_blend["alpha"]) * bal_aligned["prob_1"].values
    )
    y_pred_e = (p_final_oof >= best_blend["threshold"]).astype(int)
    m_e = binary_classification_metrics(
        plain_aligned["true_label"].values, y_pred_e, p_final_oof
    )

    comparison = pd.DataFrame(
        [
            {
                "configuration": "A",
                "feature_set": "official_5",
                "model_type": "plain",
                "threshold": 0.5,
                "alpha": np.nan,
                **{k: m_a[k] for k in [
                    "accuracy", "roc_auc", "log_loss", "balanced_accuracy",
                    "recall_0", "recall_1", "predicted_0_count", "predicted_1_count",
                ]},
            },
            {
                "configuration": "B",
                "feature_set": "safe_raw",
                "model_type": "plain",
                "threshold": 0.5,
                "alpha": np.nan,
                **{k: m_b[k] for k in [
                    "accuracy", "roc_auc", "log_loss", "balanced_accuracy",
                    "recall_0", "recall_1", "predicted_0_count", "predicted_1_count",
                ]},
            },
            {
                "configuration": "C",
                "feature_set": "safe_plus_mechanism",
                "model_type": "plain",
                "threshold": 0.5,
                "alpha": np.nan,
                **{k: m_c[k] for k in [
                    "accuracy", "roc_auc", "log_loss", "balanced_accuracy",
                    "recall_0", "recall_1", "predicted_0_count", "predicted_1_count",
                ]},
            },
            {
                "configuration": "D",
                "feature_set": "safe_plus_mechanism",
                "model_type": "balanced",
                "threshold": 0.5,
                "alpha": np.nan,
                **{k: m_d[k] for k in [
                    "accuracy", "roc_auc", "log_loss", "balanced_accuracy",
                    "recall_0", "recall_1", "predicted_0_count", "predicted_1_count",
                ]},
            },
            {
                "configuration": "E",
                "feature_set": "safe_plus_mechanism",
                "model_type": "plain_balanced_blend",
                "threshold": best_blend["threshold"],
                "alpha": best_blend["alpha"],
                **{k: m_e[k] for k in [
                    "accuracy", "roc_auc", "log_loss", "balanced_accuracy",
                    "recall_0", "recall_1", "predicted_0_count", "predicted_1_count",
                ]},
            },
        ]
    )

    # 配置 M：多数类基线（弱信号下 Accuracy 最优策略）
    y_dev_arr = y_dev.to_numpy().astype(int)
    maj_pred = np.full(len(y_dev_arr), majority_class, dtype=int)
    maj_prob = np.full(len(y_dev_arr), float(majority_class), dtype=float)
    m_m = binary_classification_metrics(y_dev_arr, maj_pred, maj_prob)
    comparison = pd.concat(
        [
            comparison,
            pd.DataFrame(
                [
                    {
                        "configuration": "M",
                        "feature_set": "none",
                        "model_type": "majority_baseline",
                        "threshold": 0.5,
                        "alpha": np.nan,
                        **{k: m_m[k] for k in [
                            "accuracy", "roc_auc", "log_loss", "balanced_accuracy",
                            "recall_0", "recall_1", "predicted_0_count", "predicted_1_count",
                        ]},
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    comparison.to_csv(oof_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    logger.info("OOF 模型对比：\n%s", comparison.to_string(index=False))

    # 选择最佳配置（仅基于 OOF）；弱信号且无法显著超过多数类时强制选 M
    best_row = select_best_config_row(comparison)
    selected_config = str(best_row["configuration"])
    model_best_acc = float(
        comparison.loc[comparison["configuration"] != "M", "accuracy"].max()
    )
    if weak_report["is_weak_signal"] and model_best_acc <= float(m_m["accuracy"]) + TASK2_TIE_EPS:
        selected_config = "M"
        best_row = comparison.loc[comparison["configuration"] == "M"].iloc[0]
        logger.warning(
            "弱信号协议生效：模型 OOF Accuracy (%.6f) 未能显著超过多数类 (%.6f)，最终选用配置 M。",
            model_best_acc,
            float(m_m["accuracy"]),
        )

    logger.info(
        "OOF 选定配置=%s，OOF Accuracy=%.6f，多数类基准=%.6f，weak_signal=%s",
        selected_config,
        best_row["accuracy"],
        majority_acc,
        weak_report["is_weak_signal"],
    )
    if float(best_row["accuracy"]) <= majority_acc + 1e-12:
        logger.warning(
            "当前最佳 OOF Accuracy 未高于多数类基准，特征仍可能未形成有效分类提升。"
        )

    # 最终迭代次数：各折最佳迭代中位数
    median_iter_plain = int(np.median(best_plain)) if best_plain else 500
    median_iter_bal = int(np.median(best_bal)) if best_bal else 500
    median_iter_a = int(np.median(best_a)) if best_a else 500
    median_iter_b = int(np.median(best_b)) if best_b else 500

    # ---- 最终验证：仅评估一次 ----
    final_metrics = _run_final_validation(
        selected_config=selected_config,
        df_dev=df_dev,
        y_dev=y_dev,
        df_val=df_val,
        y_val=y_val,
        feature_set_official=feature_set_official,
        feature_set_safe=feature_set_safe,
        feature_set_mech=feature_set_mech,
        best_blend=best_blend,
        median_iters={
            "A": median_iter_a,
            "B": median_iter_b,
            "C": median_iter_plain,
            "D": median_iter_bal,
            "E_plain": median_iter_plain,
            "E_bal": median_iter_bal,
        },
        majority_acc=majority_acc,
        majority_class=majority_class,
        final_dir=final_dir,
        weak_signal=bool(weak_report["is_weak_signal"]),
    )

    # 可选测试集预测
    if test_df is not None:
        _predict_test_set(
            test_df=test_df,
            df_all=df,
            y_all=y_all,
            selected_config=selected_config,
            feature_set_official=feature_set_official,
            feature_set_safe=feature_set_safe,
            feature_set_mech=feature_set_mech,
            best_blend=best_blend,
            median_iters={
                "A": median_iter_a,
                "B": median_iter_b,
                "C": median_iter_plain,
                "D": median_iter_bal,
                "E_plain": median_iter_plain,
                "E_bal": median_iter_bal,
            },
            majority_class=majority_class,
            output_path=output_dir / "test_predictions.csv",
        )

    result = {
        "task_name": TASK2_NAME,
        "accuracy": float(final_metrics["accuracy"]),
        "selected_configuration": selected_config,
        "oof_best_accuracy": float(best_row["accuracy"]),
        "majority_baseline_accuracy": majority_acc,
        "best_alpha": float(best_blend["alpha"]),
        "best_threshold": float(best_blend["threshold"]),
        "is_weak_signal": bool(weak_report["is_weak_signal"]),
        "weak_signal_recommendation": weak_report["recommendation"],
        "final_validation": final_metrics,
    }
    save_json(result, output_dir / "metrics.json")
    # 兼容 overall_summary 读取
    result["accuracy"] = float(final_metrics["accuracy"])
    logger.info("任务二完成。最终验证 Accuracy=%.6f", result["accuracy"])
    return result


def select_best_config_row(comparison: pd.DataFrame) -> pd.Series:
    """从 A-E 中按 OOF Accuracy 及平局规则选最优。"""
    df = comparison.copy()
    max_acc = df["accuracy"].max()
    cand = df[df["accuracy"] >= max_acc - TASK2_TIE_EPS].copy()
    cand["thr_dist"] = (cand["threshold"].fillna(0.5) - 0.5).abs()
    cand["alpha_fill"] = cand["alpha"].fillna(-1.0)
    cand = cand.sort_values(
        by=["accuracy", "balanced_accuracy", "recall_0", "thr_dist", "alpha_fill"],
        ascending=[False, False, False, True, False],
    )
    return cand.iloc[0]


def _run_final_validation(
    *,
    selected_config: str,
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    df_val: pd.DataFrame,
    y_val: pd.Series,
    feature_set_official: List[str],
    feature_set_safe: List[str],
    feature_set_mech: List[str],
    best_blend: Dict[str, Any],
    median_iters: Dict[str, int],
    majority_acc: float,
    majority_class: int,
    final_dir: Path,
    weak_signal: bool = False,
) -> Dict[str, Any]:
    """在最终 20% 验证集上只评估一次。"""
    alpha = float(best_blend["alpha"])
    threshold = float(best_blend["threshold"])

    train_vs_val_rows: List[Dict[str, Any]] = []
    models_for_fi: List[Tuple[str, CatBoostClassifier, Task2PreprocessorState]] = []

    if selected_config == "M":
        p_final = np.full(len(y_val), float(majority_class), dtype=float)
        p_plain_val = p_final.copy()
        p_bal_val = np.full(len(y_val), np.nan)
        threshold = 0.5
        alpha = np.nan
        p_dev = np.full(len(y_dev), float(majority_class), dtype=float)
        _append_train_val(train_vs_val_rows, "develop", "majority", y_dev, p_dev, 0.5)
        _append_train_val(train_vs_val_rows, "final_validation", "majority", y_val, p_final, 0.5)

    elif selected_config == "A":
        features = feature_set_official
        model, state = train_final_model(df_dev, y_dev, features, False, median_iters["A"])
        p_val = predict_with_state(model, df_val, state)
        p_dev = predict_with_state(model, df_dev, state)
        p_plain_val = p_val
        p_bal_val = np.full_like(p_val, np.nan)
        p_final = p_val
        threshold = 0.5
        alpha = np.nan
        models_for_fi = [("plain", model, state)]
        _append_train_val(train_vs_val_rows, "develop", "plain", y_dev, p_dev, 0.5)
        _append_train_val(train_vs_val_rows, "final_validation", "plain", y_val, p_val, 0.5)
        model.save_model(str(final_dir / "model_plain.cbm"))

    elif selected_config == "B":
        features = feature_set_safe
        model, state = train_final_model(df_dev, y_dev, features, False, median_iters["B"])
        p_val = predict_with_state(model, df_val, state)
        p_dev = predict_with_state(model, df_dev, state)
        p_plain_val = p_val
        p_bal_val = np.full_like(p_val, np.nan)
        p_final = p_val
        threshold = 0.5
        alpha = np.nan
        models_for_fi = [("plain", model, state)]
        _append_train_val(train_vs_val_rows, "develop", "plain", y_dev, p_dev, 0.5)
        _append_train_val(train_vs_val_rows, "final_validation", "plain", y_val, p_val, 0.5)
        model.save_model(str(final_dir / "model_plain.cbm"))

    elif selected_config == "C":
        features = feature_set_mech
        model, state = train_final_model(df_dev, y_dev, features, False, median_iters["C"])
        p_val = predict_with_state(model, df_val, state)
        p_dev = predict_with_state(model, df_dev, state)
        p_plain_val = p_val
        p_bal_val = np.full_like(p_val, np.nan)
        p_final = p_val
        threshold = 0.5
        alpha = np.nan
        models_for_fi = [("plain", model, state)]
        _append_train_val(train_vs_val_rows, "develop", "plain", y_dev, p_dev, 0.5)
        _append_train_val(train_vs_val_rows, "final_validation", "plain", y_val, p_val, 0.5)
        model.save_model(str(final_dir / "model_plain.cbm"))

    elif selected_config == "D":
        features = feature_set_mech
        model, state = train_final_model(df_dev, y_dev, features, True, median_iters["D"])
        p_val = predict_with_state(model, df_val, state)
        p_dev = predict_with_state(model, df_dev, state)
        p_plain_val = np.full_like(p_val, np.nan)
        p_bal_val = p_val
        p_final = p_val
        threshold = 0.5
        alpha = np.nan
        models_for_fi = [("balanced", model, state)]
        _append_train_val(train_vs_val_rows, "develop", "balanced", y_dev, p_dev, 0.5)
        _append_train_val(train_vs_val_rows, "final_validation", "balanced", y_val, p_val, 0.5)
        model.save_model(str(final_dir / "model_balanced.cbm"))

    else:  # E
        features = feature_set_mech
        model_p, state_p = train_final_model(
            df_dev, y_dev, features, False, median_iters["E_plain"]
        )
        model_b, state_b = train_final_model(
            df_dev, y_dev, features, True, median_iters["E_bal"]
        )
        p_plain_val = predict_with_state(model_p, df_val, state_p)
        p_bal_val = predict_with_state(model_b, df_val, state_b)
        p_plain_dev = predict_with_state(model_p, df_dev, state_p)
        p_bal_dev = predict_with_state(model_b, df_dev, state_b)
        p_final = alpha * p_plain_val + (1.0 - alpha) * p_bal_val
        p_final_dev = alpha * p_plain_dev + (1.0 - alpha) * p_bal_dev
        models_for_fi = [("plain", model_p, state_p), ("balanced", model_b, state_b)]
        _append_train_val(train_vs_val_rows, "develop", "plain", y_dev, p_plain_dev, 0.5)
        _append_train_val(train_vs_val_rows, "develop", "balanced", y_dev, p_bal_dev, 0.5)
        _append_train_val(
            train_vs_val_rows, "develop", "blend", y_dev, p_final_dev, threshold
        )
        _append_train_val(
            train_vs_val_rows, "final_validation", "plain", y_val, p_plain_val, 0.5
        )
        _append_train_val(
            train_vs_val_rows, "final_validation", "balanced", y_val, p_bal_val, 0.5
        )
        _append_train_val(
            train_vs_val_rows, "final_validation", "blend", y_val, p_final, threshold
        )
        model_p.save_model(str(final_dir / "model_plain.cbm"))
        model_b.save_model(str(final_dir / "model_balanced.cbm"))

    # 配置 M：常数多数类；其余按阈值切分
    if selected_config == "M":
        y_pred = np.full(len(y_val), majority_class, dtype=int)
    else:
        y_pred = (p_final >= threshold).astype(int)
    metrics = binary_classification_metrics(y_val, y_pred, p_final)
    lift = float(metrics["accuracy"] - majority_acc)

    # 过拟合提示
    train_blend = [
        r for r in train_vs_val_rows
        if r["dataset"] == "develop" and r["model"] in {"blend", "plain", "balanced"}
    ]
    if train_blend:
        best_train_acc = max(r["accuracy"] for r in train_blend)
        if best_train_acc > 0.85 and metrics["accuracy"] <= majority_acc + 0.01:
            logger.warning(
                "可能存在过拟合：开发集 Accuracy 较高 (%.4f)，"
                "最终验证 Accuracy (%.4f) 接近多数类基准 (%.4f)。",
                best_train_acc,
                metrics["accuracy"],
                majority_acc,
            )

    metrics_out: Dict[str, Any] = {
        "selected_configuration": selected_config,
        "alpha": None if (isinstance(alpha, float) and np.isnan(alpha)) or pd.isna(alpha) else float(alpha),
        "threshold": float(threshold),
        "majority_class": int(majority_class),
        "majority_baseline_accuracy": majority_acc,
        "accuracy_lift_vs_majority": lift,
        "is_weak_signal": bool(weak_signal),
        **metrics,
        "confusion_matrix": make_confusion_matrix_df(y_val, y_pred, [0, 1])[0].tolist(),
    }
    save_json(metrics_out, final_dir / "metrics.json")

    pred_df = pd.DataFrame(
        {
            "sample_index": y_val.index.astype(int),
            "true_label": y_val.values.astype(int),
            "predicted_label": y_pred,
            "prob_plain_1": p_plain_val,
            "prob_balanced_1": p_bal_val,
            "prob_final_1": p_final,
            "threshold": float(threshold),
        }
    )
    pred_df.to_csv(final_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    cm, cm_df = make_confusion_matrix_df(y_val, y_pred, [0, 1])
    cm_df.to_csv(final_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    plot_confusion_matrix(
        cm,
        labels=["0", "1"],
        title="Task2 Final Validation Confusion Matrix",
        save_path=final_dir / "confusion_matrix.png",
    )
    plot_probability_distribution(
        y_val,
        p_final,
        threshold=float(threshold),
        title="Task2 Final Validation Probability Distribution",
        save_path=final_dir / "probability_distribution.png",
    )
    # 常数概率（多数类基线）时 ROC 无意义，跳过以免报错
    if selected_config != "M" and float(np.nanstd(p_final)) > 1e-12:
        plot_roc_curve(
            y_val,
            p_final,
            title="Task2 Final Validation ROC Curve",
            save_path=final_dir / "roc_curve.png",
        )
    else:
        logger.info("跳过 ROC 曲线：配置为多数类基线或概率无方差。")

    for tag, model, state in models_for_fi:
        fi = extract_feature_importance(model, state.final_feature_order)
        fi.to_csv(
            final_dir / f"feature_importance_{tag}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_feature_importance(
            fi,
            title=f"Task2 Feature Importance ({tag})",
            save_path=final_dir / f"feature_importance_{tag}.png",
        )

    pd.DataFrame(train_vs_val_rows).to_csv(
        final_dir / "train_vs_validation.csv", index=False, encoding="utf-8-sig"
    )

    logger.info(
        "最终验证：Accuracy=%.6f，多数类基准=%.6f，提升=%.6f，recall_0=%.4f，recall_1=%.4f",
        metrics["accuracy"],
        majority_acc,
        lift,
        metrics["recall_0"],
        metrics["recall_1"],
    )
    return metrics_out


def _append_train_val(
    rows: List[Dict[str, Any]],
    dataset: str,
    model: str,
    y_true: pd.Series,
    prob_1: np.ndarray,
    threshold: float,
) -> None:
    y_pred = (np.asarray(prob_1) >= threshold).astype(int)
    m = binary_classification_metrics(y_true, y_pred, prob_1)
    rows.append(
        {
            "dataset": dataset,
            "model": model,
            "accuracy": m["accuracy"],
            "roc_auc": m["roc_auc"],
            "log_loss": m["log_loss"],
            "balanced_accuracy": m["balanced_accuracy"],
            "recall_0": m["recall_0"],
            "recall_1": m["recall_1"],
        }
    )


def _predict_test_set(
    *,
    test_df: pd.DataFrame,
    df_all: pd.DataFrame,
    y_all: pd.Series,
    selected_config: str,
    feature_set_official: List[str],
    feature_set_safe: List[str],
    feature_set_mech: List[str],
    best_blend: Dict[str, Any],
    median_iters: Dict[str, int],
    majority_class: int,
    output_path: Path,
) -> None:
    """使用全部有标签数据重训后预测独立测试集。"""
    logger.info("对独立测试集进行预测（不使用测试标签）。")
    alpha = float(best_blend["alpha"])
    threshold = float(best_blend["threshold"])

    if selected_config == "M":
        p_plain = np.full(len(test_df), float(majority_class))
        p_bal = np.full(len(test_df), np.nan)
        p_final = p_plain
        threshold = 0.5
        pred = np.full(len(test_df), majority_class, dtype=int)
        out = pd.DataFrame(
            {
                "sample_index": test_df.index.astype(int),
                "predicted_label": pred,
                "prob_plain_1": p_plain,
                "prob_balanced_1": p_bal,
                "prob_final_1": p_final,
                "threshold": float(threshold),
            }
        )
        out.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("测试集预测已保存（多数类基线）：%s", output_path)
        return

    if selected_config == "A":
        features, balanced, iters, thr = feature_set_official, False, median_iters["A"], 0.5
        model, state = train_final_model(df_all, y_all, features, balanced, iters)
        p_plain = predict_with_state(model, test_df, state)
        p_bal = np.full(len(test_df), np.nan)
        p_final = p_plain
        threshold = thr
    elif selected_config == "B":
        model, state = train_final_model(
            df_all, y_all, feature_set_safe, False, median_iters["B"]
        )
        p_plain = predict_with_state(model, test_df, state)
        p_bal = np.full(len(test_df), np.nan)
        p_final = p_plain
        threshold = 0.5
    elif selected_config == "C":
        model, state = train_final_model(
            df_all, y_all, feature_set_mech, False, median_iters["C"]
        )
        p_plain = predict_with_state(model, test_df, state)
        p_bal = np.full(len(test_df), np.nan)
        p_final = p_plain
        threshold = 0.5
    elif selected_config == "D":
        model, state = train_final_model(
            df_all, y_all, feature_set_mech, True, median_iters["D"]
        )
        p_bal = predict_with_state(model, test_df, state)
        p_plain = np.full(len(test_df), np.nan)
        p_final = p_bal
        threshold = 0.5
    else:
        model_p, state_p = train_final_model(
            df_all, y_all, feature_set_mech, False, median_iters["E_plain"]
        )
        model_b, state_b = train_final_model(
            df_all, y_all, feature_set_mech, True, median_iters["E_bal"]
        )
        p_plain = predict_with_state(model_p, test_df, state_p)
        p_bal = predict_with_state(model_b, test_df, state_b)
        p_final = alpha * p_plain + (1.0 - alpha) * p_bal

    pred = (p_final >= threshold).astype(int)
    out = pd.DataFrame(
        {
            "sample_index": test_df.index.astype(int),
            "predicted_label": pred,
            "prob_plain_1": p_plain,
            "prob_balanced_1": p_bal,
            "prob_final_1": p_final,
            "threshold": float(threshold),
        }
    )
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("测试集预测已保存：%s", output_path)
