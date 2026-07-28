from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.config import TASK2_BOOTSTRAP_N, TASK2_PERMUTATION_N
from src.tasks.task2.metrics_ext import rich_binary_metrics
from src.tasks.task2.preprocessing import fit_tabular, transform_tabular

logger = logging.getLogger(__name__)


def _fast_oof_auc_acc(
    X_dev: pd.DataFrame,
    y: np.ndarray,
    features: Sequence[str],
    fold_assign: np.ndarray,
    n_folds: int,
) -> Tuple[float, float]:
    """固定快模型（L2 Logistic）OOF，用于置换检验。"""
    n = len(X_dev)
    p1 = np.zeros(n, dtype=float)
    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        state = fit_tabular(X_dev.loc[tr], features, scale=True)
        X_tr = transform_tabular(X_dev.loc[tr], state)
        X_va = transform_tabular(X_dev.loc[va], state)
        lr = LogisticRegression(max_iter=400, C=1.0, random_state=42)

        if len(np.unique(y[tr])) < 2:
            p1[va] = float(y[tr].mean()) if len(y[tr]) else 0.5
            continue
        lr.fit(X_tr, y[tr])
        p1[va] = lr.predict_proba(X_va)[:, 1]
    pred = (p1 >= 0.5).astype(int)
    acc = float((pred == y).mean())
    try:
        auc = float(roc_auc_score(y, p1)) if len(np.unique(y)) > 1 else 0.5
    except ValueError:
        auc = 0.5
    return acc, auc


def run_permutation_test(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    fold_assign: np.ndarray,
    features: Sequence[str],
    output_dir: Path,
    n_perm: int = TASK2_PERMUTATION_N,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    out = Path(output_dir) / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    y = y_dev.to_numpy().astype(int)
    majority_acc = float((y == (1 if y.mean() >= 0.5 else 0)).mean())
    obs_acc, obs_auc = _fast_oof_auc_acc(df_dev, y, features, fold_assign, n_folds)
    obs_lift = obs_acc - majority_acc

    rng = np.random.RandomState(seed)
    rows = []
    for i in range(int(n_perm)):
        y_perm = rng.permutation(y)
        acc, auc = _fast_oof_auc_acc(df_dev, y_perm, features, fold_assign, n_folds)
        rows.append(
            {
                "perm_id": i,
                "oof_accuracy": acc,
                "oof_auc": auc,
                "lift_vs_majority": acc - majority_acc,
                "auc_minus_0_5": auc - 0.5,
            }
        )
        if (i + 1) % 20 == 0:
            logger.info("permutation %d/%d", i + 1, n_perm)

    df = pd.DataFrame(rows)
    atomic_write_csv(out / "permutation_results.csv", df)

    p_auc = float((df["oof_auc"] >= obs_auc).mean())
    p_lift = float((df["lift_vs_majority"] >= obs_lift).mean())
    summary = {
        "n_perm": int(n_perm),
        "model": "logistic_l2_oof",
        "observed_accuracy": obs_acc,
        "observed_auc": obs_auc,
        "observed_lift_vs_majority": obs_lift,
        "majority_accuracy": majority_acc,
        "perm_auc_mean": float(df["oof_auc"].mean()),
        "perm_auc_std": float(df["oof_auc"].std()),
        "p_value_auc_ge_observed": p_auc,
        "p_value_lift_ge_observed": p_lift,
        "supports_signal": bool(p_auc < 0.05 and obs_auc > 0.52),
        "note": "Association test under label permutation; not causation.",
    }
    atomic_write_json(out / "permutation_summary.json", summary)
    logger.info(
        "permutation done obs_auc=%.4f p_auc=%.4f supports=%s",
        obs_auc,
        p_auc,
        summary["supports_signal"],
    )
    return summary


def run_bootstrap_delta(
    oof: pd.DataFrame,
    majority_class: int,
    majority_acc: float,
    output_dir: Path,
    n_boot: int = TASK2_BOOTSTRAP_N,
    seed: int = 42,
) -> Dict[str, Any]:
    """对已有 OOF 做配对 Bootstrap：Acc-majority 与 AUC-0.5。"""
    out = Path(output_dir) / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    y = oof["true_label"].to_numpy().astype(int)
    p1 = oof["prob_1"].to_numpy().astype(float)
    pred = oof["predicted_label"].to_numpy().astype(int)
    n = len(y)
    rng = np.random.RandomState(seed)
    lifts = []
    auc_gaps = []
    for _ in range(int(n_boot)):
        idx = rng.randint(0, n, size=n)
        yy, pp, pr = y[idx], p1[idx], pred[idx]
        acc = float((pr == yy).mean())
        lifts.append(acc - majority_acc)
        try:
            auc = float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else 0.5
        except ValueError:
            auc = 0.5
        auc_gaps.append(auc - 0.5)

    lifts_a = np.asarray(lifts)
    gaps_a = np.asarray(auc_gaps)
    summary = {
        "n_boot": int(n_boot),
        "majority_class": int(majority_class),
        "majority_accuracy": float(majority_acc),
        "delta_acc_mean": float(lifts_a.mean()),
        "delta_acc_ci95": [float(np.percentile(lifts_a, 2.5)), float(np.percentile(lifts_a, 97.5))],
        "p_delta_acc_gt_0": float((lifts_a > 0).mean()),
        "auc_minus_0_5_mean": float(gaps_a.mean()),
        "auc_minus_0_5_ci95": [float(np.percentile(gaps_a, 2.5)), float(np.percentile(gaps_a, 97.5))],
        "p_auc_gap_gt_0": float((gaps_a > 0).mean()),
    }
    atomic_write_json(out / "bootstrap_summary.json", summary)
    atomic_write_csv(
        out / "bootstrap_samples.csv",
        pd.DataFrame({"delta_acc": lifts_a, "auc_minus_0_5": gaps_a}),
    )
    return summary


def run_mcnemar(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_majority: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    """McNemar：模型 vs 多数类。"""
    out = Path(output_dir) / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true).astype(int)
    a = np.asarray(pred_model).astype(int)
    b = np.asarray(pred_majority).astype(int)

    model_wrong = a != y
    maj_wrong = b != y
    n01 = int((model_wrong & ~maj_wrong).sum())
    n10 = int((~model_wrong & maj_wrong).sum())

    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)

    try:
        from scipy.stats import chi2

        p = float(chi2.sf(stat, 1))
    except Exception:
        p = float("nan")
    summary = {
        "n01_model_wrong_majority_right": n01,
        "n10_model_right_majority_wrong": n10,
        "statistic": float(stat),
        "p_value": p,
        "note": "If both always predict majority, n01=n10=0.",
    }
    atomic_write_json(out / "mcnemar_test.json", summary)
    return summary
