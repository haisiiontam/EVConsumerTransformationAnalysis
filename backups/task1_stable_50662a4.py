"""任务一优化：电动汽车采用意愿三分类（可分性诊断 + 扩展特征 + OOF）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from src.common.baseline import extract_feature_importance
from src.common.metrics import make_confusion_matrix_df, save_json
from src.common.visualization import plot_confusion_matrix, plot_feature_importance
from src.config import (
    RANDOM_SEED,
    TASK1_CATBOOST_COMMON,
    TASK1_EARLY_STOPPING_ROUNDS,
    TASK1_EXCLUDED_FEATURES,
    TASK1_KNOWN_CATEGORICAL,
    TASK1_LABEL_ORDER,
    TASK1_MECHANISM_FEATURES,
    TASK1_NAME,
    TASK1_N_FOLDS,
    TASK1_OFFICIAL_FEATURES,
    TASK1_SAFE_CANDIDATE_FEATURES,
    TASK1_STRONG_FEATURES,
    TASK1_TARGET,
    TASK1_TIE_EPS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 标签与特征
# ---------------------------------------------------------------------------

def normalize_labels(series: pd.Series) -> pd.Series:
    """清理标签空格与大小写，统一为 Low / Medium / High。"""
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


def build_target(df: pd.DataFrame) -> pd.Series:
    if TASK1_TARGET not in df.columns:
        raise ValueError(f"缺少任务一目标列：{TASK1_TARGET}")
    return normalize_labels(df[TASK1_TARGET])


def check_task1_base_columns(df: pd.DataFrame) -> None:
    required = TASK1_OFFICIAL_FEATURES + [TASK1_TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("任务一缺少必需字段：" + ", ".join(missing))


def resolve_safe_features(df: pd.DataFrame) -> List[str]:
    available: List[str] = []
    for col in TASK1_SAFE_CANDIDATE_FEATURES:
        if col in TASK1_EXCLUDED_FEATURES:
            continue
        if col not in df.columns:
            logger.warning("任务一安全候选特征缺失，已跳过：%s", col)
            continue
        if col not in available:
            available.append(col)
    for col in TASK1_STRONG_FEATURES:
        if col in df.columns and col not in available and col not in TASK1_EXCLUDED_FEATURES:
            available.append(col)
    return available


def _to_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    as_str = series.astype(str).str.strip().str.replace(",", "", regex=False)
    as_str = as_str.replace({"nan": np.nan, "None": np.nan, "<NA>": np.nan, "": np.nan})
    return pd.to_numeric(as_str, errors="coerce")


def add_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    """行内机制特征（不含目标，不用全表统计）。"""
    out = df.copy()
    eps = 1.0
    income = None

    if "annual_income" in out.columns:
        income = _to_numeric_series(out["annual_income"])
        out["log_annual_income"] = np.log1p(income.clip(lower=0))

    if "ev_knowledge_score" in out.columns:
        know = _to_numeric_series(out["ev_knowledge_score"])
        out["knowledge_deficit"] = 10.0 - know
    else:
        know = None

    if (
        "environmental_awareness_score" in out.columns
        and "technology_affinity_score" in out.columns
    ):
        eco = _to_numeric_series(out["environmental_awareness_score"])
        tech = _to_numeric_series(out["technology_affinity_score"])
        out["eco_tech_mean"] = (eco + tech) / 2.0
        out["eco_minus_tech"] = eco - tech

    if "range_anxiety_score" in out.columns and know is not None:
        anxiety = _to_numeric_series(out["range_anxiety_score"])
        out["anxiety_knowledge_gap"] = anxiety * (10.0 - know)

    if income is not None and "fuel_expense_per_month" in out.columns:
        fuel = _to_numeric_series(out["fuel_expense_per_month"])
        out["fuel_income_pressure"] = 12.0 * fuel / (income + eps)

    if "daily_commute_km" in out.columns and "weekly_travel_distance_km" in out.columns:
        daily = _to_numeric_series(out["daily_commute_km"])
        weekly = _to_numeric_series(out["weekly_travel_distance_km"])
        out["commute_ratio"] = 7.0 * daily / (weekly + eps)

    return out


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------

@dataclass
class Task1PreprocessorState:
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_medians: Dict[str, float] = field(default_factory=dict)
    final_feature_order: List[str] = field(default_factory=list)


def identify_types(
    df: pd.DataFrame, feature_cols: Sequence[str]
) -> Tuple[List[str], List[str]]:
    categorical, numerical = [], []
    known = set(TASK1_KNOWN_CATEGORICAL)
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
) -> Task1PreprocessorState:
    X_work = add_mechanism_features(X_train)
    cols = [c for c in feature_cols if c in X_work.columns]
    numerical, categorical = identify_types(X_work, cols)
    medians: Dict[str, float] = {}
    for col in numerical:
        series = _to_numeric_series(X_work[col])
        medians[col] = float(series.median()) if series.notna().any() else 0.0
    return Task1PreprocessorState(
        numeric_columns=numerical,
        categorical_columns=categorical,
        numeric_medians=medians,
        final_feature_order=cols,
    )


def transform_data(X: pd.DataFrame, state: Task1PreprocessorState) -> pd.DataFrame:
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
            series = _to_numeric_series(X_work[col])
            out[col] = series.fillna(state.numeric_medians.get(col, 0.0))
        else:
            mask_missing = X_work[col].isna()
            as_str = X_work[col].astype(str).str.strip()
            as_str = as_str.replace(
                {"nan": "Unknown", "None": "Unknown", "<NA>": "Unknown", "": "Unknown"}
            )
            out[col] = as_str.where(~mask_missing, "Unknown")
    return out.loc[:, state.final_feature_order]


def cat_indices(state: Task1PreprocessorState) -> List[int]:
    cat_set = set(state.categorical_columns)
    return [i for i, c in enumerate(state.final_feature_order) if c in cat_set]


# ---------------------------------------------------------------------------
# 诊断与 OOF
# ---------------------------------------------------------------------------

def run_mi_separability(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """三分类互信息可分性（开发集）。"""
    y_enc = LabelEncoder().fit_transform(y)
    rows = []
    for col in feature_cols:
        if col not in X.columns:
            continue
        s = X[col]
        if pd.api.types.is_numeric_dtype(s) or col not in TASK1_KNOWN_CATEGORICAL:
            # 数值或机制特征
            if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                le = LabelEncoder()
                x = le.fit_transform(s.astype(str).fillna("Unknown")).reshape(-1, 1).astype(float)
                discrete = True
            else:
                sc = _to_numeric_series(s)
                x = sc.fillna(sc.median()).to_numpy().reshape(-1, 1)
                discrete = False
        else:
            le = LabelEncoder()
            x = le.fit_transform(s.astype(str).fillna("Unknown")).reshape(-1, 1).astype(float)
            discrete = True
        mi = float(mutual_info_classif(x, y_enc, discrete_features=discrete, random_state=42)[0])
        rows.append(
            {
                "feature": col,
                "mutual_info": mi,
                "official": col in TASK1_OFFICIAL_FEATURES,
                "strong": col in TASK1_STRONG_FEATURES,
            }
        )
    out = pd.DataFrame(rows).sort_values("mutual_info", ascending=False).reset_index(drop=True)
    return out


def generate_oof(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    feature_cols: Sequence[str],
    n_folds: int,
    model_tag: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """分层 K 折 OOF 多分类预测。"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    orig_index = X_dev.index.to_numpy()
    X_feat = X_dev.reset_index(drop=True)
    y_arr = y_dev.reset_index(drop=True)

    oof_pred = np.empty(len(X_feat), dtype=object)
    oof_fold = np.full(len(X_feat), -1, dtype=int)
    # 概率矩阵按固定标签顺序
    proba_store = {lab: np.full(len(X_feat), np.nan) for lab in TASK1_LABEL_ORDER}
    fold_rows: List[Dict[str, Any]] = []
    best_iters: List[int] = []

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_feat, y_arr)):
        state = fit_preprocessor(X_feat.iloc[tr_idx], feature_cols)
        X_tr = transform_data(X_feat.iloc[tr_idx], state)
        X_va = transform_data(X_feat.iloc[va_idx], state)
        y_tr, y_va = y_arr.iloc[tr_idx], y_arr.iloc[va_idx]

        params = dict(TASK1_CATBOOST_COMMON)
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_indices(state),
            eval_set=(X_va, y_va),
            early_stopping_rounds=TASK1_EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
        best_iter = int(model.get_best_iteration() or params["iterations"])
        best_iters.append(best_iter)

        pred = model.predict(X_va)
        if isinstance(pred, np.ndarray) and pred.ndim > 1:
            pred = pred.ravel()
        pred = pd.Series(pred).astype(str).str.strip().replace(
            {"low": "Low", "medium": "Medium", "high": "High"}
        ).to_numpy()
        oof_pred[va_idx] = pred
        oof_fold[va_idx] = fold_id

        proba = model.predict_proba(X_va)
        class_names = [str(c) for c in model.classes_]
        for i, name in enumerate(class_names):
            # 规范化
            key = {"low": "Low", "medium": "Medium", "high": "High"}.get(name.lower(), name)
            if key in proba_store:
                proba_store[key][va_idx] = proba[:, i]

        acc = float(accuracy_score(y_va, pred))
        f1 = float(f1_score(y_va, pred, average="macro", labels=TASK1_LABEL_ORDER, zero_division=0))
        fold_rows.append(
            {
                "fold": fold_id,
                "model": model_tag,
                "accuracy": acc,
                "macro_f1": f1,
                "best_iteration": best_iter,
            }
        )
        logger.info(
            "OOF %s fold=%d acc=%.4f macroF1=%.4f best_iter=%d",
            model_tag,
            fold_id,
            acc,
            f1,
            best_iter,
        )

    if (oof_fold < 0).any():
        raise RuntimeError("任务一 OOF 未覆盖全部开发集样本。")

    oof_df = pd.DataFrame(
        {
            "sample_index": orig_index.astype(int),
            "true_label": y_arr.values,
            "predicted_label": oof_pred,
            "fold": oof_fold,
            "prob_Low": proba_store["Low"],
            "prob_Medium": proba_store["Medium"],
            "prob_High": proba_store["High"],
        }
    )
    return oof_df, pd.DataFrame(fold_rows), best_iters


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.2,
    n_folds: int = TASK1_N_FOLDS,
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """任务一优化完整流程。"""
    output_dir = Path(output_dir)
    for sub in ("diagnostics", "oof", "tuning", "final_validation"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("开始任务一优化流程")
    check_task1_base_columns(df)
    y_all = build_target(df)

    maj_rate = float(y_all.value_counts(normalize=True).iloc[0])
    maj_class = str(y_all.value_counts().index[0])
    save_json(
        {
            "majority_class": maj_class,
            "majority_baseline_accuracy": maj_rate,
            "label_distribution": y_all.value_counts().to_dict(),
        },
        output_dir / "diagnostics" / "majority_baseline.json",
    )

    feature_official = list(TASK1_OFFICIAL_FEATURES)
    feature_safe = resolve_safe_features(df)
    feature_mech = list(feature_safe)
    probe = add_mechanism_features(df.head(2))
    for mcol in TASK1_MECHANISM_FEATURES:
        if mcol in probe.columns and mcol not in feature_mech:
            feature_mech.append(mcol)

    logger.info(
        "官方=%d，安全扩展=%d，安全+机制=%d；强信号=%s",
        len(feature_official),
        len(feature_safe),
        len(feature_mech),
        TASK1_STRONG_FEATURES,
    )

    idx_dev, idx_val = train_test_split(
        df.index.to_series(),
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y_all,
    )
    df_dev, df_val = df.loc[idx_dev].copy(), df.loc[idx_val].copy()
    y_dev, y_val = y_all.loc[idx_dev], y_all.loc[idx_val]

    # 可分性（开发集 MI）
    df_dev_fe = add_mechanism_features(df_dev)
    sep_cols = [c for c in feature_mech if c in df_dev_fe.columns]
    sep = run_mi_separability(df_dev_fe, y_dev, sep_cols)
    sep.to_csv(output_dir / "diagnostics" / "separability.csv", index=False, encoding="utf-8-sig")
    logger.info(
        "可分性 Top3：%s",
        ", ".join(
            f"{r.feature}(MI={r.mutual_info:.3f})" for r in sep.head(3).itertuples()
        ),
    )

    # OOF A/B/C
    oof_a, folds_a, best_a = generate_oof(df_dev, y_dev, feature_official, n_folds, "A_official")
    oof_b, folds_b, best_b = generate_oof(df_dev, y_dev, feature_safe, n_folds, "B_safe")
    oof_c, folds_c, best_c = generate_oof(df_dev, y_dev, feature_mech, n_folds, "C_safe_mech")

    oof_c.to_csv(output_dir / "oof" / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat([folds_a, folds_b, folds_c], ignore_index=True).to_csv(
        output_dir / "oof" / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )

    comparison_rows = []
    for name, oof_df, fset in [
        ("A", oof_a, "official_6"),
        ("B", oof_b, "safe_expanded"),
        ("C", oof_c, "safe_plus_mechanism"),
    ]:
        acc = float(accuracy_score(oof_df["true_label"], oof_df["predicted_label"]))
        f1 = float(
            f1_score(
                oof_df["true_label"],
                oof_df["predicted_label"],
                average="macro",
                labels=TASK1_LABEL_ORDER,
                zero_division=0,
            )
        )
        comparison_rows.append(
            {
                "configuration": name,
                "feature_set": fset,
                "accuracy": acc,
                "macro_f1": f1,
                "n_features": {
                    "A": len(feature_official),
                    "B": len(feature_safe),
                    "C": len(feature_mech),
                }[name],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "oof" / "model_comparison.csv", index=False, encoding="utf-8-sig")
    logger.info("OOF 对比：\n%s", comparison.to_string(index=False))

    max_acc = comparison["accuracy"].max()
    cand = comparison[comparison["accuracy"] >= max_acc - TASK1_TIE_EPS].copy()
    cand = cand.sort_values(
        by=["accuracy", "macro_f1"],
        ascending=[False, False],
    )
    best_row = cand.iloc[0]
    selected_name = str(best_row["configuration"])
    feature_map = {"A": feature_official, "B": feature_safe, "C": feature_mech}
    selected_features = feature_map[selected_name]
    raw_median = {
        "A": int(np.median(best_a)),
        "B": int(np.median(best_b)),
        "C": int(np.median(best_c)),
    }[selected_name]
    iterations = max(raw_median, 200)

    save_json(
        {
            "selected_configuration": selected_name,
            "oof_accuracy": float(best_row["accuracy"]),
            "oof_macro_f1": float(best_row["macro_f1"]),
            "iterations": iterations,
            "fold_median_iterations": raw_median,
            "n_features": len(selected_features),
            "features": selected_features,
        },
        output_dir / "tuning" / "best_config.json",
    )
    logger.info(
        "选定配置=%s，OOF Acc=%.4f，features=%d，iterations=%d",
        selected_name,
        best_row["accuracy"],
        len(selected_features),
        iterations,
    )

    # 最终验证：完整开发集重训
    state = fit_preprocessor(df_dev, selected_features)
    X_tr = transform_data(df_dev, state)
    X_va = transform_data(df_val, state)
    params = dict(TASK1_CATBOOST_COMMON)
    params["iterations"] = iterations
    model = CatBoostClassifier(**params)
    model.fit(X_tr, y_dev, cat_features=cat_indices(state))

    y_pred = model.predict(X_va)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    y_pred = (
        pd.Series(y_pred, index=y_val.index)
        .astype(str)
        .str.strip()
        .replace({"low": "Low", "medium": "Medium", "high": "High"})
    )
    y_proba = model.predict_proba(X_va)
    class_names = [str(c) for c in model.classes_]

    acc = float(accuracy_score(y_val, y_pred))
    macro_f1 = float(
        f1_score(y_val, y_pred, average="macro", labels=TASK1_LABEL_ORDER, zero_division=0)
    )
    report = classification_report(
        y_val, y_pred, labels=TASK1_LABEL_ORDER, output_dict=True, zero_division=0
    )
    logger.info("最终验证 Accuracy=%.4f macroF1=%.4f（多数类基准=%.4f）", acc, macro_f1, maj_rate)

    final_dir = output_dir / "final_validation"
    cm, cm_df = make_confusion_matrix_df(y_val.astype(str), y_pred.astype(str), TASK1_LABEL_ORDER)
    cm_df.to_csv(final_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    plot_confusion_matrix(
        cm,
        list(TASK1_LABEL_ORDER),
        "Task1 Final Validation Confusion Matrix",
        final_dir / "confusion_matrix.png",
    )

    pred_df = pd.DataFrame(
        {
            "sample_index": y_val.index.astype(int),
            "true_label": y_val.values,
            "predicted_label": y_pred.values,
        }
    )
    proba_map = {str(n): y_proba[:, i] for i, n in enumerate(class_names)}
    for lab in TASK1_LABEL_ORDER:
        col = f"prob_{lab}"
        if lab in proba_map:
            pred_df[col] = proba_map[lab]
        else:
            matched = False
            for key, vals in proba_map.items():
                if key.lower() == lab.lower():
                    pred_df[col] = vals
                    matched = True
                    break
            if not matched:
                pred_df[col] = np.nan
    pred_df.to_csv(final_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    # 兼容旧文件名
    pred_df.to_csv(output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")

    fi = extract_feature_importance(model, state.final_feature_order)
    fi.to_csv(final_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    fi.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_feature_importance(
        fi, "Task1 Feature Importance", final_dir / "feature_importance.png"
    )
    model.save_model(str(final_dir / "model.cbm"))
    model.save_model(str(output_dir / "model.cbm"))

    y_pred_dev = model.predict(X_tr)
    if isinstance(y_pred_dev, np.ndarray) and y_pred_dev.ndim > 1:
        y_pred_dev = y_pred_dev.ravel()
    y_pred_dev = pd.Series(y_pred_dev).astype(str).str.strip()
    dev_acc = float(accuracy_score(y_dev, y_pred_dev))

    metrics_out = {
        "selected_configuration": selected_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "majority_baseline_accuracy": maj_rate,
        "accuracy_lift_vs_majority": acc - maj_rate,
        "develop_accuracy": dev_acc,
        "train_samples": int(len(X_tr)),
        "validation_samples": int(len(X_va)),
        "feature_count": len(selected_features),
        "features": selected_features,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    save_json(metrics_out, final_dir / "metrics.json")

    pd.DataFrame(
        [
            {"dataset": "develop", "accuracy": dev_acc},
            {"dataset": "final_validation", "accuracy": acc, "macro_f1": macro_f1},
        ]
    ).to_csv(final_dir / "train_vs_validation.csv", index=False, encoding="utf-8-sig")

    # 兼容旧 metrics.json 结构
    result = {
        "task_name": TASK1_NAME,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "selected_configuration": selected_name,
        "oof_accuracy": float(best_row["accuracy"]),
        "majority_baseline_accuracy": maj_rate,
        "train_samples": int(len(X_tr)),
        "validation_samples": int(len(X_va)),
        "feature_count": len(selected_features),
        "classification_report": report,
        "final_validation": metrics_out,
    }
    save_json(result, output_dir / "metrics.json")
    cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    plot_confusion_matrix(
        cm,
        list(TASK1_LABEL_ORDER),
        "Task1 Confusion Matrix",
        output_dir / "confusion_matrix.png",
    )

    if test_df is not None:
        state_all = fit_preprocessor(df, selected_features)
        X_all = transform_data(df, state_all)
        model_all = CatBoostClassifier(**{**TASK1_CATBOOST_COMMON, "iterations": iterations})
        model_all.fit(X_all, y_all, cat_features=cat_indices(state_all))
        pred_test = model_all.predict(transform_data(test_df, state_all))
        if isinstance(pred_test, np.ndarray) and pred_test.ndim > 1:
            pred_test = pred_test.ravel()
        proba_test = model_all.predict_proba(transform_data(test_df, state_all))
        out = pd.DataFrame(
            {
                "sample_index": test_df.index.astype(int),
                "predicted_label": pd.Series(pred_test).astype(str).str.strip().values,
            }
        )
        for i, name in enumerate(model_all.classes_):
            key = {"low": "Low", "medium": "Medium", "high": "High"}.get(str(name).lower(), str(name))
            out[f"prob_{key}"] = proba_test[:, i]
        out.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    logger.info("任务一完成。最终验证 Accuracy=%.4f", acc)
    return result


def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """兼容旧接口：官方六特征 X/y。"""
    return df.loc[:, TASK1_OFFICIAL_FEATURES].copy(), build_target(df)
