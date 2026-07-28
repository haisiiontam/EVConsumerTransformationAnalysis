from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.tasks.task2.features import resolve_safe_features, to_num

logger = logging.getLogger(__name__)


def run_missingness_analysis(
    df: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    features: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """若安全特征存在缺失，分析 __is_missing 指示器；否则如实报告无缺失。"""
    output_dir = Path(output_dir) / "missingness"
    output_dir.mkdir(parents=True, exist_ok=True)
    features = list(features or resolve_safe_features(df))

    rows: List[Dict[str, Any]] = []
    any_missing = False
    for col in features:
        if col not in df.columns:
            continue
        miss = df[col].isna()
        n_miss = int(miss.sum())
        rate = float(n_miss / len(df)) if len(df) else 0.0
        if n_miss > 0:
            any_missing = True
        rows.append(
            {
                "feature": col,
                "missing_count": n_miss,
                "missing_rate": rate,
            }
        )
        if n_miss > 0:
            ind = miss.astype(int)
            rate1 = float(y[miss].mean()) if miss.any() else float("nan")
            rate0 = float(y[~miss].mean()) if (~miss).any() else float("nan")
            rows.append(
                {
                    "feature": f"{col}__is_missing",
                    "missing_count": n_miss,
                    "missing_rate": rate,
                    "target_1_rate_when_missing": rate1,
                    "target_1_rate_when_present": rate0,
                    "rate_gap": rate1 - rate0 if rate1 == rate1 and rate0 == rate0 else float("nan"),
                }
            )

    report_df = pd.DataFrame(rows)
    atomic_write_csv(output_dir / "missingness_by_feature.csv", report_df)

    if not any_missing:
        summary = {
            "has_missing_in_safe_features": False,
            "message": "安全特征列无缺失；未创建 __is_missing 建模列。",
            "n_features_checked": len(features),
        }
    else:
        summary = {
            "has_missing_in_safe_features": True,
            "message": "存在缺失；已记录 __is_missing 指示器与目标率差异（诊断用）。",
            "n_features_with_missing": int(sum(1 for r in rows if r.get("feature", "").endswith("__is_missing") is False and r.get("missing_count", 0) > 0)),
        }

    atomic_write_json(output_dir / "missingness_report.json", summary)
    atomic_write_csv(
        output_dir / "missingness_summary.csv",
        pd.DataFrame([summary]),
    )
    logger.info("missingness: any_missing=%s", any_missing)
    return summary
