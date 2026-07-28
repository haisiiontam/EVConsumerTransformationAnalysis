from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.tasks.task2.metrics_ext import rich_binary_metrics
from src.tasks.task2.preprocessing import fit_tabular, transform_tabular

logger = logging.getLogger(__name__)

K_CANDIDATES = [2, 3, 4, 5, 6, 8, 10]


def _eb_rate(n_pos: int, n: int, global_rate: float, alpha: float = 20.0) -> float:
    if n <= 0:
        return global_rate
    w = n / (n + alpha)
    return w * (n_pos / n) + (1.0 - w) * global_rate


def _oof_cluster_prior(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    features: Sequence[str],
    fold_assign: np.ndarray,
    k: int,
    n_folds: int,
    alpha: float = 20.0,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    y = y_dev.to_numpy().astype(int)
    n = len(X_dev)
    oof_p = np.zeros(n, dtype=float)
    fold_rows: List[Dict[str, Any]] = []
    sil_scores: List[float] = []

    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        state = fit_tabular(X_dev.loc[tr], features, scale=True)
        X_tr = transform_tabular(X_dev.loc[tr], state)
        X_va = transform_tabular(X_dev.loc[va], state)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab_tr = km.fit_predict(X_tr)
        lab_va = km.predict(X_va)
        try:
            if len(np.unique(lab_tr)) > 1 and X_tr.shape[0] > k:
                sil_scores.append(float(silhouette_score(X_tr, lab_tr, sample_size=min(2000, len(X_tr)))))
        except Exception:
            pass

        global_rate = float(y[tr].mean())
        rates = {}
        for c in range(k):
            mask = lab_tr == c
            rates[c] = _eb_rate(int(y[tr][mask].sum()), int(mask.sum()), global_rate, alpha)

        p = np.array([rates.get(int(c), global_rate) for c in lab_va], dtype=float)
        oof_p[va] = p
        pred = (p >= 0.5).astype(int)
        m = rich_binary_metrics(y[va], pred, p)
        fold_rows.append({"fold": fold_id, "k": k, **m, "cluster_rate_span": float(max(rates.values()) - min(rates.values()))})

    meta = {
        "k": k,
        "mean_silhouette": float(np.mean(sil_scores)) if sil_scores else float("nan"),
        "n_sil_folds": len(sil_scores),
    }
    return oof_p, fold_rows, meta


def run_clustering_prior(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    fold_assign: np.ndarray,
    features: Sequence[str],
    output_dir: Path,
    n_folds: int = 5,
    k_candidates: Sequence[int] | None = None,
) -> Dict[str, Any]:
    out = Path(output_dir) / "clustering"
    out.mkdir(parents=True, exist_ok=True)
    ks = list(k_candidates or K_CANDIDATES)
    rows: List[Dict[str, Any]] = []
    best_acc = -1.0
    best_k = ks[0]
    best_oof: np.ndarray | None = None
    best_meta: Dict[str, Any] = {}

    for k in ks:
        oof_p, fold_rows, meta = _oof_cluster_prior(
            df_dev, y_dev, features, fold_assign, int(k), n_folds
        )
        y = y_dev.to_numpy().astype(int)
        pred = (oof_p >= 0.5).astype(int)
        m = rich_binary_metrics(y, pred, oof_p)
        row = {"k": int(k), **m, **meta}
        rows.append(row)
        pd.DataFrame(fold_rows).to_csv(out / f"fold_metrics_k{k}.csv", index=False, encoding="utf-8-sig")
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_k = int(k)
            best_oof = oof_p.copy()
            best_meta = meta

    comparison = pd.DataFrame(rows).sort_values(["accuracy", "roc_auc"], ascending=False)
    atomic_write_csv(out / "cluster_comparison.csv", comparison)

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
    atomic_write_csv(out / "cluster_prior_oof.csv", oof_df)
    cfg = {
        "best_k": best_k,
        "oof_accuracy": float(best_acc),
        "oof_roc_auc": float(comparison.loc[comparison["k"] == best_k, "roc_auc"].iloc[0]),
        **best_meta,
        "note": "cluster fit on train fold only; rates beta-binomial smoothed",
    }
    atomic_write_json(out / "cluster_best_config.json", cfg)


    try:
        state = fit_tabular(df_dev, features, scale=True)
        X = transform_tabular(df_dev, state)
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labs = km.fit_predict(X)
        prof = []
        for c in range(best_k):
            mask = labs == c
            yy = y_dev.to_numpy()[mask]
            prof.append(
                {
                    "cluster": c,
                    "n": int(mask.sum()),
                    "rate_1": float(yy.mean()) if mask.sum() else float("nan"),
                    "center_l2_norm": float(np.linalg.norm(km.cluster_centers_[c])),
                }
            )
        atomic_write_csv(out / "cluster_profiles.csv", pd.DataFrame(prof))
    except Exception as e:
        logger.warning("cluster profile failed: %s", e)

    logger.info("clustering best k=%d acc=%.6f", best_k, best_acc)
    return {"oof": oof_df, "best_config": cfg, "comparison": comparison}
