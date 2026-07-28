from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text
from src.tasks.task2.metrics_ext import rich_binary_metrics

logger = logging.getLogger(__name__)


def diversity_report(
    candidates: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    out = Path(output_dir) / "ensemble"
    out.mkdir(parents=True, exist_ok=True)
    names = list(candidates.keys())
    rows: List[Dict[str, Any]] = []
    for a, b in combinations(names, 2):
        oa, ob = candidates[a], candidates[b]

        m = oa.merge(ob, on="sample_index", suffixes=("_a", "_b"))
        if m.empty:
            continue
        pa = m["prob_1_a"].to_numpy()
        pb = m["prob_1_b"].to_numpy()
        ya = m["true_label_a"].to_numpy().astype(int)
        prea = m["predicted_label_a"].to_numpy().astype(int)
        preb = m["predicted_label_b"].to_numpy().astype(int)
        corr = float(np.corrcoef(pa, pb)[0, 1]) if len(pa) > 1 else float("nan")
        agree = float((prea == preb).mean())
        a_wrong = prea != ya
        b_wrong = preb != ya
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "prob_corr": corr,
                "pred_agree": agree,
                "both_wrong": float((a_wrong & b_wrong).mean()),
                "a_wrong_b_right": float((a_wrong & ~b_wrong).mean()),
                "b_wrong_a_right": float((b_wrong & ~a_wrong).mean()),
            }
        )
    df = pd.DataFrame(rows)
    atomic_write_csv(out / "diversity_pairs.csv", df)
    md = [
        "# Diversity",
        "",
        "If all models collapse to majority predictions, diversity is near-zero and fusion is pointless.",
        "",
        f"n_pairs={len(df)}",
        "",
    ]
    if len(df):
        md.append("See `diversity_pairs.csv`.")
        md.append(
            f"mean_prob_corr={float(df['prob_corr'].mean()):.4f}, mean_agree={float(df['pred_agree'].mean()):.4f}"
        )
    atomic_write_text(out / "diversity_summary.md", "\n".join(md))
    return df


def soft_blend_search(
    candidates: Dict[str, pd.DataFrame],
    y_dev: pd.Series,
    output_dir: Path,
    max_models: int = 4,
) -> Dict[str, Any]:
    """最多 4 个候选的网格软融合；若无提升则返回 empty。"""
    out = Path(output_dir) / "ensemble"
    out.mkdir(parents=True, exist_ok=True)

    scored = []
    for name, oof in candidates.items():
        if name.startswith("B0") or name.startswith("M"):
            continue
        m = rich_binary_metrics(oof["true_label"], oof["predicted_label"], oof["prob_1"])
        scored.append((name, m["roc_auc"], m["accuracy"], oof))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    picked = scored[:max_models]
    if len(picked) < 2:
        cfg = {"available": False, "reason": "fewer than 2 non-majority candidates"}
        atomic_write_json(out / "soft_blend_best.json", cfg)
        return cfg


    base = picked[0][3][["sample_index", "true_label"]].copy()
    probs = []
    names = []
    for name, _, _, oof in picked:
        names.append(name)
        tmp = base.merge(oof[["sample_index", "prob_1"]], on="sample_index", how="left")
        probs.append(tmp["prob_1"].to_numpy(dtype=float))
    P = np.vstack(probs)
    y = base["true_label"].to_numpy().astype(int)

    best = {"accuracy": -1.0}
    rows = []

    m = len(names)
    if m == 2:
        grid = [(w, 1 - w) for w in np.round(np.arange(0.0, 1.01, 0.1), 10)]
    elif m == 3:
        grid = []
        for a in np.round(np.arange(0.0, 1.01, 0.2), 10):
            for b in np.round(np.arange(0.0, 1.01 - a, 0.2), 10):
                c = round(1.0 - a - b, 10)
                if c < -1e-9:
                    continue
                grid.append((a, b, max(c, 0.0)))
    else:

        grid = [tuple([1.0 / m] * m)]
        for i in range(m):
            w = [0.0] * m
            w[i] = 1.0
            grid.append(tuple(w))

    for w in grid:
        w = np.asarray(w, dtype=float)
        w = w / max(w.sum(), 1e-12)
        p = (w.reshape(-1, 1) * P).sum(axis=0)
        pred = (p >= 0.5).astype(int)
        met = rich_binary_metrics(y, pred, p)
        row = {f"w_{names[i]}": float(w[i]) for i in range(m)}
        row.update({"accuracy": met["accuracy"], "roc_auc": met["roc_auc"]})
        rows.append(row)
        if met["accuracy"] > best.get("accuracy", -1):
            best = {
                "available": True,
                "weights": {names[i]: float(w[i]) for i in range(m)},
                "accuracy": met["accuracy"],
                "roc_auc": met["roc_auc"],
                "models": names,
            }

    atomic_write_csv(out / "soft_blend_search.csv", pd.DataFrame(rows))
    atomic_write_json(out / "soft_blend_best.json", best)

    if best.get("available"):
        w = np.array([best["weights"][n] for n in names], dtype=float)
        p = (w.reshape(-1, 1) * P).sum(axis=0)
        oof = pd.DataFrame(
            {
                "sample_index": base["sample_index"],
                "true_label": y,
                "prob_0": 1.0 - p,
                "prob_1": p,
                "predicted_label": (p >= 0.5).astype(int),
            }
        )
        atomic_write_csv(out / "soft_blend_oof.csv", oof)
        best["oof"] = oof
    logger.info("soft blend best acc=%.6f", best.get("accuracy", float("nan")))
    return best
