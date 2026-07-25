"""任务三优化：里程焦虑敏感度识别（可分性诊断 + 扩展特征 + OOF 阈值）。"""

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
from src.common.separability import univariate_separability
from src.common.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_probability_distribution,
    plot_roc_curve,
)
from src.config import (
    RANDOM_SEED,
    TASK3_CATBOOST_COMMON,
    TASK3_EARLY_STOPPING_ROUNDS,
    TASK3_EXCLUDED_FEATURES,
    TASK3_KNOWN_CATEGORICAL,
    TASK3_MECHANISM_FEATURES,
    TASK3_NAME,
    TASK3_N_FOLDS,
    TASK3_OFFICIAL_FEATURES,
    TASK3_SAFE_CANDIDATE_FEATURES,
    TASK3_SOURCE_SCORE,
    TASK3_STRONG_FEATURES,
    TASK3_THRESHOLD,
    TASK3_THRESHOLD_MAX,
    TASK3_THRESHOLD_MIN,
    TASK3_THRESHOLD_STEP,
    TASK3_TIE_EPS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 标签与特征
# ---------------------------------------------------------------------------

def build_target(df: pd.DataFrame) -> pd.Series:
    """构造任务三标签：range_anxiety_score > 5 -> 1。"""
    score = pd.to_numeric(df[TASK3_SOURCE_SCORE], errors="coerce")
    if score.isna().any():
        raise ValueError("任务三原始评分列存在缺失或无法解析的值。")
    return (score > TASK3_THRESHOLD).astype(int)


def check_task3_base_columns(df: pd.DataFrame) -> None:
    """官方基础字段与标签源必须存在。"""
    required = TASK3_OFFICIAL_FEATURES + [TASK3_SOURCE_SCORE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("任务三缺少必需字段：" + ", ".join(missing))


def resolve_safe_features(df: pd.DataFrame) -> List[str]:
    """筛选实际存在的安全扩展特征。"""
    available: List[str] = []
    for col in TASK3_SAFE_CANDIDATE_FEATURES:
        if col in TASK3_EXCLUDED_FEATURES:
            continue
        if col not in df.columns:
            logger.warning("任务三安全候选特征缺失，已跳过：%s", col)
            continue
        if col not in available:
            available.append(col)
    return available


def _to_numeric_series(series: pd.Series) -> Tuple[pd.Series, int]:
    if pd.api.types.is_numeric_dtype(series):
        converted = pd.to_numeric(series, errors="coerce")
        fail = int(series.notna().sum() - converted.notna().sum())
        return converted, fail
    as_str = series.astype(str).str.strip().str.replace(",", "", regex=False)
    as_str = as_str.replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan, "": np.nan})
    converted = pd.to_numeric(as_str, errors="coerce")
    fail = int((series.notna() & converted.isna()).sum())
    return converted, fail


def add_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    """行内机制特征（不含标签，不用全表统计）。"""
    out = df.copy()
    eps = 1.0

    if "annual_income" in out.columns:
        income, _ = _to_numeric_series(out["annual_income"])
        out["log_annual_income"] = np.log1p(income.clip(lower=0))
    else:
        income = None

    if "daily_commute_km" in out.columns and "weekly_travel_distance_km" in out.columns:
        daily, _ = _to_numeric_series(out["daily_commute_km"])
        weekly, _ = _to_numeric_series(out["weekly_travel_distance_km"])
        out["commute_ratio"] = 7.0 * daily / (weekly + eps)

    if "ev_knowledge_score" in out.columns:
        know, _ = _to_numeric_series(out["ev_knowledge_score"])
        out["knowledge_deficit"] = 10.0 - know

    if (
        "environmental_awareness_score" in out.columns
        and "technology_affinity_score" in out.columns
    ):
        eco, _ = _to_numeric_series(out["environmental_awareness_score"])
        tech, _ = _to_numeric_series(out["technology_affinity_score"])
        out["eco_tech_mean"] = (eco + tech) / 2.0
        out["eco_minus_tech"] = eco - tech

    if income is not None and "fuel_expense_per_month" in out.columns:
        fuel, _ = _to_numeric_series(out["fuel_expense_per_month"])
        out["fuel_income_pressure"] = 12.0 * fuel / (income + eps)

    return out


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------

@dataclass
class Task3PreprocessorState:
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_medians: Dict[str, float] = field(default_factory=dict)
    final_feature_order: List[str] = field(default_factory=list)


def identify_types(
    df: pd.DataFrame, feature_cols: Sequence[str]
) -> Tuple[List[str], List[str]]:
    categorical, numerical = [], []
    known = set(TASK3_KNOWN_CATEGORICAL)
    for col in feature_cols:
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        is_cat = (
            col in known
            or pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
        )
        (categorical if is_cat else numerical).append(col)
    return numerical, categorical


def fit_preprocessor(
    X_train: pd.DataFrame, feature_cols: Sequence[str]
) -> Task3PreprocessorState:
    X_work = add_mechanism_features(X_train)
    cols = [c for c in feature_cols if c in X_work.columns]
    numerical, categorical = identify_types(X_work, cols)
    medians: Dict[str, float] = {}
    for col in numerical:
        series, _ = _to_numeric_series(X_work[col])
        medians[col] = float(series.median()) if series.notna().any() else 0.0
    return Task3PreprocessorState(
        numeric_columns=numerical,
        categorical_columns=categorical,
        numeric_medians=medians,
        final_feature_order=cols,
    )


def transform_data(X: pd.DataFrame, state: Task3PreprocessorState) -> pd.DataFrame:
    X_work = add_mechanism_features(X)
    out = pd.DataFrame(index=X_work.index)
    for col in state.final_feature_order:
        if col not in X_work.columns:
            out[col] = (
                state.numeric_medians.get(col, 0.0)
                if col in state.numeric_columns
                else "Unknown"
            )
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
            out[col] = as_str.where(~mask_missing, "Unknown")
    return out.loc[:, state.final_feature_order]


def cat_indices(state: Task3PreprocessorState) -> List[int]:
    cat_set = set(state.categorical_columns)
    return [i for i, c in enumerate(state.final_feature_order) if c in cat_set]


# ---------------------------------------------------------------------------
# OOF / 阈值
# ---------------------------------------------------------------------------

def _prob1(model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = [int(c) for c in model.classes_]
    return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]


def generate_oof(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    feature_cols: Sequence[str],
    n_folds: int,
    model_tag: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """分层 K 折 OOF 概率。"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    orig_index = X_dev.index.to_numpy()
    X_feat = X_dev.reset_index(drop=True)
    y_arr = y_dev.reset_index(drop=True)

    oof_p1 = np.zeros(len(X_feat))
    oof_fold = np.full(len(X_feat), -1, dtype=int)
    fold_rows: List[Dict[str, Any]] = []
    best_iters: List[int] = []

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_feat, y_arr)):
        state = fit_preprocessor(X_feat.iloc[tr_idx], feature_cols)
        X_tr = transform_data(X_feat.iloc[tr_idx], state)
        X_va = transform_data(X_feat.iloc[va_idx], state)
        y_tr, y_va = y_arr.iloc[tr_idx], y_arr.iloc[va_idx]

        params = dict(TASK3_CATBOOST_COMMON)
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_indices(state),
            eval_set=(X_va, y_va),
            early_stopping_rounds=TASK3_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        best_iter = int(model.get_best_iteration() or params["iterations"])
        best_iters.append(best_iter)

        p1 = _prob1(model, X_va)
        oof_p1[va_idx] = p1
        oof_fold[va_idx] = fold_id
        m = binary_classification_metrics(y_va, (p1 >= 0.5).astype(int), p1)
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
                "best_iteration": best_iter,
            }
        )
        logger.info(
            "OOF %s fold=%d acc=%.4f auc=%.4f best_iter=%d",
            model_tag,
            fold_id,
            m["accuracy"],
            m["roc_auc"],
            best_iter,
        )

    if (oof_fold < 0).any():
        raise RuntimeError("任务三 OOF 未覆盖全部开发集样本。")

    oof_df = pd.DataFrame(
        {
            "sample_index": orig_index.astype(int),
            "true_label": y_arr.values.astype(int),
            "prob_0": 1.0 - oof_p1,
            "prob_1": oof_p1,
            "fold": oof_fold,
        }
    )
    return oof_df, pd.DataFrame(fold_rows), best_iters


def search_threshold(
    oof_df: pd.DataFrame,
    thr_min: float,
    thr_max: float,
    thr_step: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """仅在开发集 OOF 上搜索分类阈值。"""
    y_true = oof_df["true_label"].to_numpy()
    p1 = oof_df["prob_1"].to_numpy()
    rows = []
    for thr in np.round(np.arange(thr_min, thr_max + 1e-12, thr_step), 10):
        pred = (p1 >= thr).astype(int)
        m = binary_classification_metrics(y_true, pred, p1)
        rows.append({"threshold": float(thr), **m})
    search = pd.DataFrame(rows)
    max_acc = search["accuracy"].max()
    cand = search[search["accuracy"] >= max_acc - TASK3_TIE_EPS].copy()
    cand["thr_dist"] = (cand["threshold"] - 0.5).abs()
    cand = cand.sort_values(
        by=["accuracy", "balanced_accuracy", "recall_0", "thr_dist"],
        ascending=[False, False, False, True],
    )
    best = cand.iloc[0]
    best_cfg = {
        "threshold": float(best["threshold"]),
        "selection_metric": "oof_accuracy",
        "oof_accuracy": float(best["accuracy"]),
        "oof_roc_auc": float(best["roc_auc"]),
        "oof_balanced_accuracy": float(best["balanced_accuracy"]),
    }
    return search, best_cfg


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.2,
    n_folds: int = TASK3_N_FOLDS,
    threshold_min: float = TASK3_THRESHOLD_MIN,
    threshold_max: float = TASK3_THRESHOLD_MAX,
    threshold_step: float = TASK3_THRESHOLD_STEP,
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """任务三优化完整流程。"""
    output_dir = Path(output_dir)
    for sub in ("diagnostics", "oof", "tuning", "final_validation"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("开始任务三优化流程")
    check_task3_base_columns(df)
    y_all = build_target(df)
    maj = majority_baseline(y_all)
    save_json(maj, output_dir / "diagnostics" / "majority_baseline.json")

    feature_official = list(TASK3_OFFICIAL_FEATURES)
    feature_safe = resolve_safe_features(df)
    # 确保强信号特征在扩展集中
    for c in TASK3_STRONG_FEATURES:
        if c in df.columns and c not in feature_safe and c not in TASK3_EXCLUDED_FEATURES:
            feature_safe.append(c)

    feature_mech = list(feature_safe)
    probe = add_mechanism_features(df.head(2))
    for mcol in TASK3_MECHANISM_FEATURES:
        if mcol in probe.columns and mcol not in feature_mech:
            feature_mech.append(mcol)

    logger.info("官方特征数=%d，安全扩展=%d，安全+机制=%d", len(feature_official), len(feature_safe), len(feature_mech))
    logger.info("强信号特征：%s", TASK3_STRONG_FEATURES)

    # 外层划分：最终验证不参与调参
    idx_dev, idx_val = train_test_split(
        df.index.to_series(),
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y_all,
    )
    df_dev, df_val = df.loc[idx_dev].copy(), df.loc[idx_val].copy()
    y_dev, y_val = y_all.loc[idx_dev], y_all.loc[idx_val]

    # 可分性（开发集）
    df_dev_fe = add_mechanism_features(df_dev)
    sep_cols = [c for c in feature_mech if c in df_dev_fe.columns]
    sep = univariate_separability(df_dev_fe, y_dev, sep_cols)
    sep.to_csv(output_dir / "diagnostics" / "separability.csv", index=False, encoding="utf-8-sig")
    logger.info(
        "可分性：最强特征=%s，|AUC-0.5|=%.4f",
        sep.iloc[0]["feature"] if len(sep) else "NA",
        float(sep.iloc[0]["auc_gap_from_0_5"]) if len(sep) else float("nan"),
    )

    # OOF：A 官方 / B 安全扩展 / C 安全+机制
    oof_a, folds_a, best_a = generate_oof(
        df_dev, y_dev, feature_official, n_folds, "A_official"
    )
    oof_b, folds_b, best_b = generate_oof(
        df_dev, y_dev, feature_safe, n_folds, "B_safe"
    )
    oof_c, folds_c, best_c = generate_oof(
        df_dev, y_dev, feature_mech, n_folds, "C_safe_mech"
    )
    oof_c.to_csv(output_dir / "oof" / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat([folds_a, folds_b, folds_c], ignore_index=True).to_csv(
        output_dir / "oof" / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )

    # 各配置：默认 0.5 + 搜阈值
    comparison_rows = []
    best_cfgs: Dict[str, Dict[str, Any]] = {}
    for name, oof_df, fset in [
        ("A", oof_a, "official_5"),
        ("B", oof_b, "safe_expanded"),
        ("C", oof_c, "safe_plus_mechanism"),
    ]:
        m05 = binary_classification_metrics(
            oof_df["true_label"], (oof_df["prob_1"] >= 0.5).astype(int), oof_df["prob_1"]
        )
        search, best = search_threshold(oof_df, threshold_min, threshold_max, threshold_step)
        search.to_csv(
            output_dir / "tuning" / f"threshold_search_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        best_cfgs[name] = best
        comparison_rows.append(
            {
                "configuration": f"{name}@0.5",
                "feature_set": fset,
                "threshold": 0.5,
                **{k: m05[k] for k in ["accuracy", "roc_auc", "log_loss", "balanced_accuracy", "recall_0", "recall_1"]},
            }
        )
        m_best = binary_classification_metrics(
            oof_df["true_label"],
            (oof_df["prob_1"] >= best["threshold"]).astype(int),
            oof_df["prob_1"],
        )
        comparison_rows.append(
            {
                "configuration": f"{name}@tuned",
                "feature_set": fset,
                "threshold": best["threshold"],
                **{
                    k: m_best[k]
                    for k in [
                        "accuracy",
                        "roc_auc",
                        "log_loss",
                        "balanced_accuracy",
                        "recall_0",
                        "recall_1",
                    ]
                },
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "oof" / "model_comparison.csv", index=False, encoding="utf-8-sig")
    logger.info("OOF 对比：\n%s", comparison.to_string(index=False))

    # 选最优（OOF Accuracy）
    max_acc = comparison["accuracy"].max()
    cand = comparison[comparison["accuracy"] >= max_acc - TASK3_TIE_EPS].copy()
    cand["thr_dist"] = (cand["threshold"] - 0.5).abs()
    cand = cand.sort_values(
        by=["accuracy", "balanced_accuracy", "recall_0", "thr_dist"],
        ascending=[False, False, False, True],
    )
    best_row = cand.iloc[0]
    selected_name = str(best_row["configuration"]).split("@")[0]
    selected_threshold = float(best_row["threshold"])
    feature_map = {"A": feature_official, "B": feature_safe, "C": feature_mech}
    selected_features = feature_map[selected_name]
    # 折内早停迭代可能偏小；最终重训使用中位数与下限取 max，避免欠拟合
    raw_median = {
        "A": int(np.median(best_a)),
        "B": int(np.median(best_b)),
        "C": int(np.median(best_c)),
    }[selected_name]
    median_iter = max(raw_median, 200)
    logger.info("最终重训 iterations=%d（折内中位数=%d）", median_iter, raw_median)

    save_json(
        {
            "selected_configuration": str(best_row["configuration"]),
            "feature_set": selected_name,
            "threshold": selected_threshold,
            "oof_accuracy": float(best_row["accuracy"]),
            "oof_roc_auc": float(best_row["roc_auc"]),
            "iterations": median_iter,
            "n_features": len(selected_features),
            "features": selected_features,
        },
        output_dir / "tuning" / "best_config.json",
    )
    logger.info(
        "选定 %s，threshold=%.3f，OOF Acc=%.4f，features=%d",
        best_row["configuration"],
        selected_threshold,
        best_row["accuracy"],
        len(selected_features),
    )

    # 最终验证：完整开发集重训，验证集只评一次
    state = fit_preprocessor(df_dev, selected_features)
    X_tr = transform_data(df_dev, state)
    params = dict(TASK3_CATBOOST_COMMON)
    params["iterations"] = median_iter
    model = CatBoostClassifier(**params)
    model.fit(X_tr, y_dev, cat_features=cat_indices(state))

    p_val = _prob1(model, transform_data(df_val, state))
    p_dev = _prob1(model, X_tr)
    y_pred = (p_val >= selected_threshold).astype(int)
    metrics = binary_classification_metrics(y_val, y_pred, p_val)
    metrics_dev = binary_classification_metrics(
        y_dev, (p_dev >= selected_threshold).astype(int), p_dev
    )

    final_dir = output_dir / "final_validation"
    metrics_out = {
        "selected_configuration": str(best_row["configuration"]),
        "threshold": selected_threshold,
        "majority_baseline_accuracy": float(maj["majority_baseline_accuracy"]),
        "accuracy_lift_vs_majority": float(
            metrics["accuracy"] - maj["majority_baseline_accuracy"]
        ),
        "develop_accuracy": float(metrics_dev["accuracy"]),
        **metrics,
        "confusion_matrix": make_confusion_matrix_df(y_val, y_pred, [0, 1])[0].tolist(),
        "n_features": len(selected_features),
        "features": selected_features,
    }
    save_json(metrics_out, final_dir / "metrics.json")

    pd.DataFrame(
        {
            "sample_index": y_val.index.astype(int),
            "true_label": y_val.values.astype(int),
            "predicted_label": y_pred,
            "prob_0": 1.0 - p_val,
            "prob_1": p_val,
            "threshold": selected_threshold,
        }
    ).to_csv(final_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    cm, cm_df = make_confusion_matrix_df(y_val, y_pred, [0, 1])
    cm_df.to_csv(final_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    plot_confusion_matrix(
        cm, ["0", "1"], "Task3 Final Validation Confusion Matrix", final_dir / "confusion_matrix.png"
    )
    plot_probability_distribution(
        y_val,
        p_val,
        selected_threshold,
        "Task3 Final Validation Probability Distribution",
        final_dir / "probability_distribution.png",
    )
    plot_roc_curve(
        y_val, p_val, "Task3 Final Validation ROC Curve", final_dir / "roc_curve.png"
    )

    fi = extract_feature_importance(model, state.final_feature_order)
    fi.to_csv(final_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_feature_importance(
        fi, "Task3 Feature Importance", final_dir / "feature_importance.png"
    )
    model.save_model(str(final_dir / "model.cbm"))

    pd.DataFrame(
        [
            {"dataset": "develop", **{k: metrics_dev[k] for k in ["accuracy", "roc_auc", "log_loss", "balanced_accuracy", "recall_0", "recall_1"]}},
            {"dataset": "final_validation", **{k: metrics[k] for k in ["accuracy", "roc_auc", "log_loss", "balanced_accuracy", "recall_0", "recall_1"]}},
        ]
    ).to_csv(final_dir / "train_vs_validation.csv", index=False, encoding="utf-8-sig")

    # 兼容旧输出位置：metrics.json + validation_predictions 风格
    result = {
        "task_name": TASK3_NAME,
        "accuracy": float(metrics["accuracy"]),
        "selected_configuration": str(best_row["configuration"]),
        "threshold": selected_threshold,
        "oof_accuracy": float(best_row["accuracy"]),
        "majority_baseline_accuracy": float(maj["majority_baseline_accuracy"]),
        "final_validation": metrics_out,
    }
    save_json(result, output_dir / "metrics.json")

    # 可选测试集
    if test_df is not None:
        state_all = fit_preprocessor(df, selected_features)
        X_all = transform_data(df, state_all)
        model_all = CatBoostClassifier(**{**TASK3_CATBOOST_COMMON, "iterations": median_iter})
        model_all.fit(X_all, y_all, cat_features=cat_indices(state_all))
        p_test = _prob1(model_all, transform_data(test_df, state_all))
        pd.DataFrame(
            {
                "sample_index": test_df.index.astype(int),
                "predicted_label": (p_test >= selected_threshold).astype(int),
                "prob_1": p_test,
                "threshold": selected_threshold,
            }
        ).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    logger.info(
        "任务三完成。最终验证 Accuracy=%.4f（相对多数类 +%.4f），ROC-AUC=%.4f",
        metrics["accuracy"],
        metrics["accuracy"] - maj["majority_baseline_accuracy"],
        metrics["roc_auc"],
    )
    return result


# 兼容旧接口名称
def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """供外部快速取官方五特征 X/y。"""
    y = build_target(df)
    X = df.loc[:, TASK3_OFFICIAL_FEATURES].copy()
    return X, y
