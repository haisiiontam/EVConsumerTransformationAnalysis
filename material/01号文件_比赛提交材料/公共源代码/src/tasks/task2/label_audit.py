from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.metrics import majority_baseline
from src.tasks.task2.features import add_mechanism_features, resolve_safe_features, to_num
from src.tasks.task2.preprocessing import fit_tabular, transform_tabular

logger = logging.getLogger(__name__)


def _prepare_matrix(
    df: pd.DataFrame,
    features: Sequence[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """用全 develop 拟合 tabular 仅用于标签审计（非正式建模）。"""
    state = fit_tabular(df, features, scale=True)
    X = transform_tabular(df, state)
    return X, pd.DataFrame(X, index=df.index)


def duplicate_conflicts(
    df: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
) -> pd.DataFrame:
    """完全相同安全特征行但标签冲突。"""
    state = fit_tabular(df, features, scale=False)
    X = transform_tabular(df, state)

    key_df = pd.DataFrame(X).round(6)
    key_df["y"] = y.values
    grp = key_df.groupby(list(range(X.shape[1])), dropna=False)["y"].agg(["count", "nunique", "mean"])
    grp = grp.reset_index()
    conflicts = grp[(grp["count"] > 1) & (grp["nunique"] > 1)].copy()
    conflicts["conflict_type"] = "exact_duplicate_features"
    return conflicts


def profile_bin_conflicts(
    df: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    n_bins: int = 5,
) -> pd.DataFrame:
    """数值特征分位分箱后的标签冲突率。"""
    X_mech = add_mechanism_features(df)
    rows: List[Dict[str, Any]] = []
    for col in features:
        if col not in X_mech.columns:
            continue
        series, fail = to_num(X_mech[col])
        if fail > 0.2 * len(series) and not pd.api.types.is_numeric_dtype(X_mech[col]):
            continue
        if series.notna().sum() < 20:
            continue
        try:
            bins = pd.qcut(series, q=n_bins, duplicates="drop")
        except ValueError:
            continue
        tmp = pd.DataFrame({"bin": bins.astype(str), "y": y.values})
        for b, g in tmp.groupby("bin"):
            rows.append(
                {
                    "feature": col,
                    "bin": b,
                    "count": int(len(g)),
                    "label_1_rate": float(g["y"].mean()),
                    "label_entropy": float(-sum(p * np.log(p + 1e-12) for p in [g["y"].mean(), 1 - g["y"].mean()])),
                    "conflict_score": float(min(g["y"].mean(), 1 - g["y"].mean())),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("conflict_score", ascending=True)
    return out


def knn_label_consistency(
    df: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    ks: Sequence[int] = (5, 10, 20, 50),
) -> pd.DataFrame:
    """kNN 邻居标签一致率 vs 全局先验与随机基线。"""
    X, _ = _prepare_matrix(df, features)
    y_arr = y.to_numpy().astype(int)
    global_prior = float(y_arr.mean())
    rng = np.random.default_rng(42)
    random_acc = max(global_prior, 1 - global_prior)

    rows: List[Dict[str, Any]] = []
    nn = NearestNeighbors(n_neighbors=max(ks) + 1, metric="euclidean")
    nn.fit(X)
    dist, idx = nn.kneighbors(X)

    idx = idx[:, 1:]

    for k in ks:
        neighbor_labels = y_arr[idx[:, :k]]

        mode_pred = (neighbor_labels.mean(axis=1) >= 0.5).astype(int)
        knn_acc = float((mode_pred == y_arr).mean())
        rows.append(
            {
                "k": k,
                "knn_accuracy": knn_acc,
                "global_prior_accuracy": random_acc,
                "random_baseline": random_acc,
                "lift_vs_prior": knn_acc - random_acc,
            }
        )
    return pd.DataFrame(rows)


def _try_cleanlab(
    df: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    output_dir: Path,
) -> Optional[pd.DataFrame]:
    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        logger.info("cleanlab 未安装，跳过 label issue 检测")
        return None
    X, _ = _prepare_matrix(df, features)

    p = np.full(len(y), float(y.mean()))
    issues = find_label_issues(y.to_numpy(), p.reshape(-1, 1))
    out = pd.DataFrame({"sample_index": df.index.astype(int), "is_label_issue": issues})
    atomic_write_csv(output_dir / "cleanlab_issues.csv", out)
    return out


def run_label_audit(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    output_dir: Path,
) -> Dict[str, Any]:
    """在 develop 集运行标签审计，不修改标签。"""
    output_dir = Path(output_dir) / "label_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    features = resolve_safe_features(df_dev)

    dup = duplicate_conflicts(df_dev, y_dev, features)
    atomic_write_csv(output_dir / "duplicate_conflicts.csv", dup)

    prof = profile_bin_conflicts(df_dev, y_dev, features)
    atomic_write_csv(output_dir / "profile_bin_conflicts.csv", prof)

    knn = knn_label_consistency(df_dev, y_dev, features)
    atomic_write_csv(output_dir / "knn_label_consistency.csv", knn)

    cleanlab_df = _try_cleanlab(df_dev, y_dev, features, output_dir)

    maj = majority_baseline(y_dev)
    summary = {
        "n_duplicate_conflict_groups": int(len(dup)),
        "n_profile_bins": int(len(prof)),
        "knn_best_lift": float(knn["lift_vs_prior"].max()) if len(knn) else None,
        "majority_baseline": maj,
        "cleanlab_ran": cleanlab_df is not None,
        "note": "标签未修改；审计仅用于弱信号说明",
    }
    atomic_write_json(output_dir / "label_audit_summary.json", summary)
    logger.info("label audit done conflicts=%d", len(dup))
    return summary
