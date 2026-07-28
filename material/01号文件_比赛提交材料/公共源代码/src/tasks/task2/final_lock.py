from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.config import TASK2_TIE_EPS, TASK2_WEAK_SIGNAL_AUC_GAP
from src.tasks.task2.metrics_ext import rich_binary_metrics

logger = logging.getLogger(__name__)


def _row_from_oof(name: str, oof: pd.DataFrame, extra: Optional[Dict] = None) -> Dict[str, Any]:
    m = rich_binary_metrics(oof["true_label"], oof["predicted_label"], oof["prob_1"])
    row = {
        "candidate": name,
        "accuracy": m["accuracy"],
        "roc_auc": m["roc_auc"],
        "balanced_accuracy": m["balanced_accuracy"],
        "pr_auc": m.get("pr_auc"),
        **(extra or {}),
    }
    return row


def compare_candidates(
    majority_class: int,
    majority_acc: float,
    y_dev: pd.Series,
    candidates: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """汇总各候选 OOF 指标，含配置 M。"""
    rows: List[Dict[str, Any]] = []
    for name, oof in candidates.items():
        rows.append(_row_from_oof(name, oof))
    y_arr = y_dev.to_numpy().astype(int)
    maj_pred = np.full(len(y_arr), majority_class, dtype=int)
    maj_prob = np.full(len(y_arr), float(majority_class), dtype=float)
    m_m = rich_binary_metrics(y_arr, maj_pred, maj_prob)
    rows.append(
        {
            "candidate": "M_majority",
            "accuracy": m_m["accuracy"],
            "roc_auc": m_m["roc_auc"],
            "balanced_accuracy": m_m["balanced_accuracy"],
            "pr_auc": m_m.get("pr_auc"),
            "majority_class": majority_class,
        }
    )
    df = pd.DataFrame(rows)
    df["lift_vs_majority"] = df["accuracy"] - majority_acc
    df["auc_gap_from_0_5"] = (df["roc_auc"] - 0.5).abs()
    return df.sort_values("accuracy", ascending=False)


def lock_final_scheme(
    comparison: pd.DataFrame,
    majority_acc: float,
    max_auc_gap: float = TASK2_WEAK_SIGNAL_AUC_GAP,
    tie_eps: float = TASK2_TIE_EPS,
) -> Dict[str, Any]:
    """
    锁定规则：
    - 若无候选 Acc  lift > tie_eps 超过多数类，或
    - 所有非 M 候选 max |AUC-0.5| < max_auc_gap
    → 锁定 M。
    """
    non_m = comparison[comparison["candidate"] != "M_majority"]
    best_model_acc = float(non_m["accuracy"].max()) if len(non_m) else 0.0
    max_auc_dev = (
        float(non_m["auc_gap_from_0_5"].max())
        if "auc_gap_from_0_5" in non_m.columns and len(non_m)
        else float("nan")
    )

    weak_no_lift = best_model_acc <= majority_acc + tie_eps
    weak_auc = max_auc_dev == max_auc_dev and max_auc_dev < max_auc_gap

    if weak_no_lift or weak_auc:
        locked = "M"
        reason = []
        if weak_no_lift:
            reason.append(f"no Acc lift > {tie_eps} over majority")
        if weak_auc:
            reason.append(f"max |AUC-0.5|={max_auc_dev:.4f} < {max_auc_gap}")
        locked_row = comparison[comparison["candidate"] == "M_majority"].iloc[0]
    else:
        locked = str(non_m.sort_values("accuracy", ascending=False).iloc[0]["candidate"])
        reason = ["best OOF accuracy among non-majority candidates"]
        locked_row = non_m[non_m["candidate"] == locked].iloc[0]

    return {
        "locked_scheme": locked,
        "lock_reason": "; ".join(reason),
        "majority_baseline_accuracy": majority_acc,
        "best_model_accuracy": best_model_acc,
        "max_auc_gap_from_0_5": max_auc_dev,
        "is_weak_signal_lock": locked == "M",
        "oof_accuracy": float(locked_row["accuracy"]),
        "oof_roc_auc": float(locked_row.get("roc_auc", float("nan"))),
    }


def persist_final_lock(
    output_dir: Path,
    lock_cfg: Dict[str, Any],
    comparison: pd.DataFrame,
    features: Sequence[str] | None = None,
) -> None:
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(final_dir / "candidate_comparison.csv", comparison)
    atomic_write_json(final_dir / "task2_final_config.json", lock_cfg)
    manifest = {
        "scheme": lock_cfg["locked_scheme"],
        "features": list(features or []),
        "threshold": 0.5 if lock_cfg["locked_scheme"] != "M" else None,
        "weak_signal_lock": lock_cfg.get("is_weak_signal_lock", False),
    }
    atomic_write_json(final_dir / "model_manifest.json", manifest)
    logger.info("final lock -> %s (%s)", lock_cfg["locked_scheme"], lock_cfg.get("lock_reason"))
