from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def _safe_auc(y: pd.Series, score: pd.Series) -> float:
    """计算 ROC-AUC；无效时返回 nan。"""
    mask = score.notna() & y.notna()
    if mask.sum() < 50 or y.loc[mask].nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y.loc[mask].astype(int), score.loc[mask].astype(float)))
    except ValueError:
        return float("nan")


def univariate_separability(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    在给定样本上评估每个特征的单变量可分性。

    注意：应使用开发集，不要用最终验证集做模型选择。
    """
    y = y.loc[X.index]
    rows: List[Dict[str, Any]] = []

    for col in feature_cols:
        if col not in X.columns:
            continue
        series = X[col]
        dtype = str(series.dtype)
        missing_rate = float(series.isna().mean())

        if pd.api.types.is_numeric_dtype(series):
            score = pd.to_numeric(series, errors="coerce")
            auc = _safe_auc(y, score)

            m0 = score[y == 0].mean()
            m1 = score[y == 1].mean()
            std = score.std()
            mean_diff_z = float((m1 - m0) / std) if std and np.isfinite(std) and std > 0 else 0.0
            rate_gap = float("nan")
            nunique = int(score.nunique(dropna=True))
        else:

            tmp = pd.DataFrame({"cat": series.astype(str).fillna("Unknown"), "y": y.values})
            rates = tmp.groupby("cat")["y"].mean()
            score = tmp["cat"].map(rates)
            auc = _safe_auc(y, score)
            rate_gap = float(rates.max() - rates.min()) if len(rates) else float("nan")
            mean_diff_z = float("nan")
            nunique = int(series.astype(str).nunique(dropna=False))

        auc_gap = float(abs(auc - 0.5)) if np.isfinite(auc) else float("nan")
        rows.append(
            {
                "feature": col,
                "dtype": dtype,
                "missing_rate": missing_rate,
                "unique_count": nunique,
                "roc_auc": auc,
                "auc_gap_from_0_5": auc_gap,
                "mean_diff_z": mean_diff_z,
                "category_target_rate_gap": rate_gap,
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("auc_gap_from_0_5", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def leak_risk_probe(
    df: pd.DataFrame,
    y: pd.Series,
    candidate_cols: Sequence[str],
) -> pd.DataFrame:
    """
    对被排除/高风险字段做可分性探查（仅诊断，绝不进入正式模型）。

    用于回答：即便放宽泄露约束，是否仍无信号？
    """
    rows: List[Dict[str, Any]] = []
    for col in candidate_cols:
        if col not in df.columns:
            rows.append(
                {
                    "feature": col,
                    "available": False,
                    "roc_auc": float("nan"),
                    "auc_gap_from_0_5": float("nan"),
                    "note": "column_missing",
                }
            )
            continue

        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            score = pd.to_numeric(series, errors="coerce")
        else:
            enc = LabelEncoder()
            filled = series.astype(str).fillna("Unknown")
            score = pd.Series(enc.fit_transform(filled), index=series.index)

        auc = _safe_auc(y, score)
        rows.append(
            {
                "feature": col,
                "available": True,
                "roc_auc": auc,
                "auc_gap_from_0_5": float(abs(auc - 0.5)) if np.isfinite(auc) else float("nan"),
                "note": "diagnostic_only_not_used_in_model",
            }
        )
    return pd.DataFrame(rows).sort_values(
        "auc_gap_from_0_5", ascending=False, na_position="last"
    ).reset_index(drop=True)


def assess_weak_signal(
    separability_df: pd.DataFrame,
    leak_df: pd.DataFrame,
    auc_gap_threshold: float = 0.02,
) -> Dict[str, Any]:
    """
    判定是否属于弱信号场景。

    规则：安全特征最大 |AUC-0.5| < 阈值，且泄露探查同样无强信号。

    说明：其他任务的类别标签（如 ev_adoption_likelihood）经 LabelEncoder 后的 AUC
    不能当作可用信号；判定时只看数值型探查字段，并把任务标签单独标注。
    """
    safe_max_gap = (
        float(separability_df["auc_gap_from_0_5"].max())
        if len(separability_df) and separability_df["auc_gap_from_0_5"].notna().any()
        else 0.0
    )

    other_task_labels = {"ev_adoption_likelihood", "range_anxiety_score"}
    leak_usable = leak_df[
        ~leak_df["feature"].isin(other_task_labels) & leak_df["available"].fillna(False)
    ] if len(leak_df) else leak_df
    leak_max_gap = (
        float(leak_usable["auc_gap_from_0_5"].max())
        if len(leak_usable) and leak_usable["auc_gap_from_0_5"].notna().any()
        else 0.0
    )
    leak_max_gap_including_other_tasks = (
        float(leak_df["auc_gap_from_0_5"].max())
        if len(leak_df) and leak_df["auc_gap_from_0_5"].notna().any()
        else 0.0
    )
    safe_max_auc = (
        float(separability_df["roc_auc"].max())
        if len(separability_df) and separability_df["roc_auc"].notna().any()
        else 0.5
    )

    is_weak = safe_max_gap < auc_gap_threshold
    leak_also_weak = leak_max_gap < auc_gap_threshold

    if is_weak and leak_also_weak:
        conclusion = (
            "weak_signal_confirmed: 安全特征与可探查数值字段均接近随机；"
            "即便查看被排除的能耗/充电费用字段也几乎无信息。"
            "标签在当前数据下几乎不可预测；Accuracy 最优策略是多数类基线。"
        )
        recommendation = (
            "正式提交使用配置 M（多数类）稳住 ACC2；竞赛提升应转向任务一/三；"
            "不要再靠阈值/融合硬抬任务二 Accuracy。"
            "其他任务标签（ev_adoption_likelihood 等）即使编码后略有 AUC 偏离，也绝不能用作特征。"
        )
    elif is_weak and not leak_also_weak:
        conclusion = (
            "safe_features_weak_but_leak_probe_has_signal: "
            "安全特征无信号，但被排除的数值字段可能有信息（需警惕后验泄露）。"
        )
        recommendation = (
            "先核查 monthly_charging_cost / monthly_energy_consumption_kwh "
            "是否属于后验信息；若确认安全再考虑纳入，否则保持排除。"
        )
    else:
        conclusion = "signal_detected: 至少部分安全特征具备可分性，可继续特征与模型优化。"
        recommendation = "保留可分性靠前的特征，继续 OOF 对比与阈值搜索。"

    report = {
        "is_weak_signal": bool(is_weak),
        "leak_probe_also_weak": bool(leak_also_weak),
        "auc_gap_threshold": auc_gap_threshold,
        "safe_max_auc_gap": safe_max_gap,
        "safe_max_roc_auc": safe_max_auc,
        "leak_max_auc_gap": leak_max_gap,
        "leak_max_auc_gap_including_other_task_labels": leak_max_gap_including_other_tasks,
        "conclusion": conclusion,
        "recommendation": recommendation,
    }
    logger.info("可分性判定：%s", conclusion)
    return report


def write_weak_signal_markdown(report: Dict[str, Any], path: Path) -> None:
    """写一份给人看的弱信号说明。"""
    path = Path(path)
    lines = [
        "# Task2 Weak Signal Report",
        "",
        f"- is_weak_signal: **{report['is_weak_signal']}**",
        f"- leak_probe_also_weak: **{report['leak_probe_also_weak']}**",
        f"- safe_max_|AUC-0.5|: {report['safe_max_auc_gap']:.6f}",
        f"- leak_max_|AUC-0.5|: {report['leak_max_auc_gap']:.6f}",
        f"- threshold: {report['auc_gap_threshold']}",
        "",
        "## Conclusion",
        "",
        report["conclusion"],
        "",
        "## Recommendation",
        "",
        report["recommendation"],
        "",
        "## If AUC is truly ~0.5",
        "",
        "1. Accuracy 的理论上界近似等于多数类比例；强行拉开 0 类 Recall 会牺牲官方 Accuracy。",
        "2. 阈值搜索 / 模型融合无法创造不存在的信息。",
        "3. 竞赛策略：任务二用多数类（或等价配置）稳住分数，把提升空间放在任务一和任务三。",
        "4. 只有找到住房/车位等真正相关的新字段，或确认某排除字段可合法使用时，才值得继续挖任务二。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
