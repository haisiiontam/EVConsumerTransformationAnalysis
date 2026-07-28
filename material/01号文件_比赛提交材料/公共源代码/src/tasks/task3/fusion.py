from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text
from src.common.metrics import binary_classification_metrics
from src.config import TASK3_TIE_EPS

logger = logging.getLogger(__name__)


def _align_p1(dfs: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    base = next(iter(dfs.values()))
    y = base["true_label"].to_numpy().astype(int)
    out = {}
    for k, df in dfs.items():
        m = df.set_index("sample_index").loc[base["sample_index"]]
        out[k] = m["prob_1"].to_numpy().astype(float)
    return y, out


def diversity_report(
    model_oofs: Dict[str, pd.DataFrame],
    score: Optional[np.ndarray],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y, ps = _align_p1(model_oofs)
    names = list(ps.keys())

    corr = pd.DataFrame({n: ps[n] for n in names}).corr(method="pearson")
    atomic_write_csv(output_dir / "probability_correlation.csv", corr.reset_index())

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pa, pb = (ps[a] >= 0.5).astype(int), (ps[b] >= 0.5).astype(int)
            a_err, b_err = pa != y, pb != y
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "agree_rate": float((pa == pb).mean()),
                    "both_wrong": int((a_err & b_err).sum()),
                    "a_wrong_b_right": int((a_err & ~b_err).sum()),
                    "b_wrong_a_right": int((~a_err & b_err).sum()),
                }
            )
    atomic_write_csv(output_dir / "error_overlap.csv", pd.DataFrame(rows))

    if score is not None:
        base = next(iter(model_oofs.values()))
        sc = pd.Series(score, index=base["sample_index"]).loc[base["sample_index"]].to_numpy()
        crow = []
        for s in (5, 6):
            m = np.round(sc).astype(int) == s
            if not m.any():
                continue
            for n in names:
                pred = (ps[n][m] >= 0.5).astype(int)
                crow.append({"score": s, "model": n, "accuracy": float((pred == y[m]).mean()), "n": int(m.sum())})
        atomic_write_csv(output_dir / "complementarity_by_score.csv", pd.DataFrame(crow))

    md = ["# Diversity Summary", "", f"Models: {', '.join(names)}", ""]
    if rows:
        best = max(rows, key=lambda r: r["a_wrong_b_right"] + r["b_wrong_a_right"])
        md.append(
            f"Most complementary pair: `{best['model_a']}` vs `{best['model_b']}` "
            f"(cross-correct={best['a_wrong_b_right'] + best['b_wrong_a_right']})"
        )
    atomic_write_text(output_dir / "diversity_summary.md", "\n".join(md) + "\n")


def classifier_regressor_fusion(
    p_cls: np.ndarray,
    p_reg: np.ndarray,
    y: np.ndarray,
    p_ord: Optional[np.ndarray] = None,
    step: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    rows = []
    best = {"accuracy": -1.0}
    best_p = p_cls.copy()
    gammas = [0.0] if p_ord is None else np.round(np.arange(0.0, 1.0 + 1e-12, step), 10)
    for alpha in np.round(np.arange(0.0, 1.0 + 1e-12, step), 10):
        for beta in np.round(np.arange(0.0, 1.0 - alpha + 1e-12, step), 10):
            for gamma in gammas:
                if abs(alpha + beta + gamma - 1.0) > 1e-6:
                    if p_ord is None and abs(alpha + beta - 1.0) > 1e-6:
                        continue
                    if p_ord is not None:
                        continue
                if p_ord is None:
                    if abs(alpha + beta - 1.0) > 1e-6:
                        continue
                    p = alpha * p_cls + beta * p_reg
                    g = 0.0
                else:
                    if abs(alpha + beta + gamma - 1.0) > 1e-6:
                        continue
                    p = alpha * p_cls + beta * p_reg + gamma * p_ord
                    g = float(gamma)

                for thr in (0.45, 0.5, 0.55):
                    pred = (p >= thr).astype(int)
                    acc = float((pred == y).mean())
                    row = {
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "gamma": g,
                        "threshold": thr,
                        "accuracy": acc,
                    }
                    rows.append(row)
                    if acc > best["accuracy"] + TASK3_TIE_EPS or (
                        abs(acc - best["accuracy"]) <= TASK3_TIE_EPS and thr == 0.5
                    ):
                        if acc >= best["accuracy"]:
                            best = row
                            best_p = p.copy()


    for w in np.round(np.arange(0.0, 1.0 + 1e-12, 0.1), 10):
        unc = 1.0 - 2.0 * np.abs(p_cls - 0.5)
        p = (1 - w * unc) * p_cls + (w * unc) * p_reg
        for thr in (0.45, 0.5, 0.55):
            acc = float(((p >= thr).astype(int) == y).mean())
            row = {
                "alpha": "boundary",
                "beta": float(w),
                "gamma": 0.0,
                "threshold": thr,
                "accuracy": acc,
            }
            rows.append(row)
            if acc > best["accuracy"]:
                best = row
                best_p = p.copy()

    grid = pd.DataFrame(rows)
    return grid, best, best_p


def soft_blend_search(
    model_p1: Dict[str, np.ndarray],
    y: np.ndarray,
    step: float = 0.1,
    max_models: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    names = list(model_p1.keys())[:max_models]

    rows = []
    best = {"accuracy": -1.0, "weights": {}}
    best_p = model_p1[names[0]].copy()


    stack = np.column_stack([model_p1[n] for n in names])
    p_eq = stack.mean(axis=1)
    acc_eq = float(((p_eq >= 0.5).astype(int) == y).mean())
    rows.append({"weights": "equal", "accuracy": acc_eq})
    if acc_eq > best["accuracy"]:
        best = {"accuracy": acc_eq, "weights": {n: 1.0 / len(names) for n in names}, "threshold": 0.5}
        best_p = p_eq


    for i, a in enumerate(names[:3]):
        for b in names[i + 1 : 4]:
            for wa in np.round(np.arange(0.0, 1.0 + 1e-12, step), 10):
                p = wa * model_p1[a] + (1 - wa) * model_p1[b]
                acc = float(((p >= 0.5).astype(int) == y).mean())
                rows.append({"weights": f"{a}:{wa:.2f},{b}:{1-wa:.2f}", "accuracy": acc})
                if acc > best["accuracy"]:
                    best = {
                        "accuracy": acc,
                        "weights": {a: float(wa), b: float(1 - wa)},
                        "threshold": 0.5,
                    }
                    best_p = p
    return pd.DataFrame(rows), best, best_p


def persist_fusion(
    output_dir: Path,
    grid: pd.DataFrame,
    best: Dict[str, Any],
    oof_p1: np.ndarray,
    sample_index: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "classifier_regressor_grid.csv", grid)
    atomic_write_json(output_dir / "classifier_regressor_best.json", best)
    thr = float(best.get("threshold", 0.5))
    oof = pd.DataFrame(
        {
            "sample_index": sample_index,
            "true_label": y,
            "prob_1": oof_p1,
            "prob_0": 1.0 - oof_p1,
            "predicted_label": (oof_p1 >= thr).astype(int),
        }
    )
    atomic_write_csv(output_dir / "classifier_regressor_oof.csv", oof)
    return oof
