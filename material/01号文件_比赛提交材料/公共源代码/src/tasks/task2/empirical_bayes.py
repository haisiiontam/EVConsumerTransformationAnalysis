from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.common.fold_checkpoints import PROB_COLS_BINARY, save_fold_checkpoint
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.config import RANDOM_SEED
from src.tasks.task2.metrics_ext import rich_binary_metrics

logger = logging.getLogger(__name__)

ALPHA_CANDIDATES = [1, 5, 10, 20, 50, 100]

GROUP_COLS_1WAY = [
    "city_type",
    "education_level",
    "current_vehicle_type",
]
NUM_BIN_SOURCES = {
    "income_bin": "annual_income",
    "vehicle_age_bin": "vehicle_age_years",
    "station_distance_bin": "nearest_charging_station_km",
}


def _qcut_train(
    train_series: pd.Series,
    val_series: pd.Series,
    n_bins: int = 4,
) -> Tuple[pd.Series, pd.Series]:
    """分位分箱：边界来自训练折。"""
    tr = pd.to_numeric(train_series, errors="coerce")
    va = pd.to_numeric(val_series, errors="coerce")
    if tr.notna().sum() < n_bins:
        return tr.astype(str).fillna("Unknown"), va.astype(str).fillna("Unknown")
    try:
        _, edges = pd.qcut(tr.dropna(), q=n_bins, retbins=True, duplicates="drop")
        edges[0] = -np.inf
        edges[-1] = np.inf
        tr_b = pd.cut(tr, bins=edges, duplicates="drop").astype(str)
        va_b = pd.cut(va, bins=edges, duplicates="drop").astype(str)
        return tr_b.fillna("Unknown"), va_b.fillna("Unknown")
    except Exception:
        return tr.astype(str).fillna("Unknown"), va.astype(str).fillna("Unknown")


def _build_group_columns(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """为训练/验证折构造分组列（边界仅来自训练折）。"""
    tr = df_tr.copy()
    va = df_va.copy()
    for gcol in GROUP_COLS_1WAY:
        if gcol in tr.columns:
            tr[f"g_{gcol}"] = tr[gcol].astype(str).str.strip().fillna("Unknown")
            va[f"g_{gcol}"] = va[gcol].astype(str).str.strip().fillna("Unknown")
    for bname, src in NUM_BIN_SOURCES.items():
        if src in tr.columns:
            tr[bname], va[bname] = _qcut_train(tr[src], va[src])
    return tr, va


def _group_keys(tr: pd.DataFrame, va: pd.DataFrame) -> Tuple[List[str], List[str]]:
    keys_1: List[str] = []
    keys_2: List[str] = []
    one_way = [f"g_{c}" for c in GROUP_COLS_1WAY if f"g_{c}" in tr.columns]
    one_way += [b for b in NUM_BIN_SOURCES if b in tr.columns]
    keys_1.extend(one_way)
    keys_2.extend([f"{a}|{b}" for a, b in combinations(one_way, 2)])
    return keys_1, keys_2


def _eb_rate(
    count: int,
    total: int,
    global_rate: float,
    alpha: float,
) -> float:
    """向全局率收缩：alpha 越大收缩越强。"""
    if total <= 0:
        return global_rate
    local = count / total
    w = total / (total + alpha)
    return w * local + (1 - w) * global_rate


def _fit_rates(
    tr: pd.DataFrame,
    y_tr: np.ndarray,
    keys_1: Sequence[str],
    keys_2: Sequence[str],
    alpha: float,
) -> Dict[str, Dict[str, float]]:

    y_s = pd.Series(np.asarray(y_tr).astype(int), index=tr.index)
    global_rate = float(y_s.mean())
    rates: Dict[str, Dict[str, float]] = {"global": {"__global__": global_rate}}
    for key in keys_1:
        rates[key] = {}
        for lvl, idx in tr.groupby(key).groups.items():
            yy = y_s.loc[list(idx)].to_numpy()
            rates[key][str(lvl)] = _eb_rate(int(yy.sum()), len(yy), global_rate, alpha)
    for key in keys_2:
        a, b = key.split("|")
        rates[key] = {}
        tr["_k2"] = tr[a].astype(str) + "||" + tr[b].astype(str)
        for lvl, idx in tr.groupby("_k2").groups.items():
            yy = y_s.loc[list(idx)].to_numpy()
            rates[key][str(lvl)] = _eb_rate(int(yy.sum()), len(yy), global_rate, alpha)
    return rates


def _predict_row(
    row: pd.Series,
    rates: Dict[str, Dict[str, float]],
    keys_1: Sequence[str],
    keys_2: Sequence[str],
) -> float:
    global_rate = rates["global"]["__global__"]

    for key in keys_2:
        a, b = key.split("|")
        if a in row.index and b in row.index:
            k = f"{row[a]}||{row[b]}"
            if k in rates.get(key, {}):
                return rates[key][k]
    for key in keys_1:
        if key in row.index:
            k = str(row[key])
            if k in rates.get(key, {}):
                return rates[key][k]
    return global_rate


def _oof_eb_for_alpha(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    fold_assign: np.ndarray,
    alpha: float,
    n_folds: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    y_arr = y_dev.to_numpy().astype(int)
    n = len(df_dev)
    oof_p = np.zeros(n, dtype=float)
    fold_rows: List[Dict[str, Any]] = []

    for fold_id in range(n_folds):
        va_mask = fold_assign == fold_id
        tr_mask = ~va_mask
        tr_df, va_df = _build_group_columns(df_dev.loc[tr_mask], df_dev.loc[va_mask])
        keys_1, keys_2 = _group_keys(tr_df, va_df)
        rates = _fit_rates(tr_df, y_arr[tr_mask], keys_1, keys_2, alpha)
        probs = []
        for _, row in va_df.iterrows():
            probs.append(_predict_row(row, rates, keys_1, keys_2))
        probs = np.array(probs, dtype=float)
        oof_p[va_mask] = probs
        pred = (probs >= 0.5).astype(int)
        m = rich_binary_metrics(y_arr[va_mask], pred, probs)
        fold_rows.append({"fold": fold_id, "alpha": alpha, **m})
    return oof_p, fold_rows


def run_empirical_bayes(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    fold_assign: np.ndarray,
    output_dir: Path,
    n_folds: int = 5,
    alphas: Sequence[float] | None = None,
) -> Dict[str, Any]:
    """搜索 alpha，输出 OOF 最优 EB 配置。"""
    output_dir = Path(output_dir) / "empirical_bayes"
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = list(alphas or ALPHA_CANDIDATES)

    search_rows: List[Dict[str, Any]] = []
    best_alpha = alphas[0]
    best_acc = -1.0
    best_oof: Optional[np.ndarray] = None

    for alpha in alphas:
        oof_p, fold_rows = _oof_eb_for_alpha(df_dev, y_dev, fold_assign, float(alpha), n_folds)
        y_arr = y_dev.to_numpy().astype(int)
        pred = (oof_p >= 0.5).astype(int)
        m = rich_binary_metrics(y_arr, pred, oof_p)
        row = {"alpha": float(alpha), **m}
        search_rows.append(row)
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_alpha = float(alpha)
            best_oof = oof_p.copy()

    search_df = pd.DataFrame(search_rows)
    atomic_write_csv(output_dir / "eb_search.csv", search_df)

    assert best_oof is not None
    oof_df = pd.DataFrame(
        {
            "sample_index": df_dev.index.astype(int),
            "true_label": y_dev.to_numpy().astype(int),
            "prob_0": 1.0 - best_oof,
            "prob_1": best_oof,
            "predicted_label": (best_oof >= 0.5).astype(int),
            "fold": fold_assign,
        }
    )
    atomic_write_csv(output_dir / "eb_best_oof_predictions.csv", oof_df)

    best_cfg = {
        "alpha": best_alpha,
        "oof_accuracy": best_acc,
        "groups_1way": GROUP_COLS_1WAY + list(NUM_BIN_SOURCES.keys()),
        "backoff": "2way->1way->global",
        "seed": RANDOM_SEED,
    }
    atomic_write_json(output_dir / "eb_best_config.json", best_cfg)
    logger.info("EB best alpha=%s acc=%.4f", best_alpha, best_acc)
    return {"search": search_df, "best_config": best_cfg, "oof": oof_df, "oof_prob": best_oof}
