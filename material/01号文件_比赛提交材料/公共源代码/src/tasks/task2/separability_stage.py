from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.separability import univariate_separability
from src.config import TASK2_DIAG_CATEGORICAL
from src.tasks.task2.features import add_mechanism_features, resolve_safe_features

logger = logging.getLogger(__name__)


def _beta_binomial_rate(count: int, total: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    return float((count + alpha) / (total + alpha + beta))


def categorical_group_rates(
    df: pd.DataFrame,
    y: pd.Series,
    categorical_cols: Sequence[str] | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    """类别特征分组目标率（Beta-Binomial 平滑）。"""
    categorical_cols = list(categorical_cols or TASK2_DIAG_CATEGORICAL)
    rows: List[Dict[str, Any]] = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        tmp = pd.DataFrame(
            {
                "category": df[col].astype(str).str.strip().fillna("Unknown"),
                "y": y.values,
            }
        )
        grp = tmp.groupby("category")["y"].agg(["count", "sum"])
        global_rate = float(y.mean())
        for cat, row in grp.iterrows():
            sm_rate = _beta_binomial_rate(int(row["sum"]), int(row["count"]), alpha, beta)
            rows.append(
                {
                    "feature": col,
                    "category": cat,
                    "sample_count": int(row["count"]),
                    "target_1_count": int(row["sum"]),
                    "raw_rate": float(row["sum"] / row["count"]) if row["count"] else 0.0,
                    "smoothed_rate": sm_rate,
                    "rate_gap_from_global": sm_rate - global_rate,
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("rate_gap_from_global", key=abs, ascending=False)
    return out


def _optional_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    try:
        from sklearn.feature_selection import mutual_info_classif
    except ImportError:
        return pd.DataFrame()
    rows = []
    for col in feature_cols:
        if col not in X.columns:
            continue
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            arr = pd.to_numeric(s, errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
        else:
            codes, _ = pd.factorize(s.astype(str).fillna("Unknown"))
            arr = codes.reshape(-1, 1)
        try:
            mi = float(mutual_info_classif(arr, y, discrete_features=False, random_state=42)[0])
        except Exception:
            mi = float("nan")
        rows.append({"feature": col, "mutual_info": mi})
    return pd.DataFrame(rows).sort_values("mutual_info", ascending=False)


def run_separability_stage(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    output_dir: Path,
    feature_cols: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """单变量可分性 + 可选互信息 + 类别平滑目标率。"""
    output_dir = Path(output_dir) / "separability"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe = resolve_safe_features(df_dev)
    feature_cols = list(feature_cols or safe)
    df_fe = add_mechanism_features(df_dev)
    sep_cols = [c for c in feature_cols if c in df_fe.columns]
    sep = univariate_separability(df_fe, y_dev, sep_cols)
    atomic_write_csv(output_dir / "separability.csv", sep)

    cat_rates = categorical_group_rates(df_dev, y_dev)
    atomic_write_csv(output_dir / "categorical_smoothed_rates.csv", cat_rates)

    mi = _optional_mutual_info(df_fe, y_dev, sep_cols)
    if len(mi):
        atomic_write_csv(output_dir / "mutual_info.csv", mi)

    max_gap = (
        float(sep["auc_gap_from_0_5"].max())
        if len(sep) and sep["auc_gap_from_0_5"].notna().any()
        else float("nan")
    )
    summary = {
        "n_features": len(sep),
        "max_auc_gap_from_0_5": max_gap,
        "mutual_info_computed": len(mi) > 0,
    }
    atomic_write_json(output_dir / "separability_summary.json", summary)
    logger.info("separability max |AUC-0.5|=%.4f", max_gap if max_gap == max_gap else -1)
    return {"separability": sep, "summary": summary, "cat_rates": cat_rates}
