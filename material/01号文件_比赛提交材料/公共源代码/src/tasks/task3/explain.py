from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.common.io_utils import atomic_write_csv

logger = logging.getLogger(__name__)

try:
    import shap

    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False


C_POS = "#C44E52"
C_NEG = "#4C72B0"
C_MAIN = "#2F6F8F"
C_ACCENT = "#E8A838"


def _safe_name(feat: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in feat)


def _prepare_pool(
    Xs: pd.DataFrame,
    cat_feature_indices: Optional[Sequence[int]],
) -> Tuple[pd.DataFrame, List[int]]:
    if cat_feature_indices is None:
        cat_feature_indices = [
            i
            for i, c in enumerate(Xs.columns)
            if not pd.api.types.is_numeric_dtype(Xs[c])
        ]
    cat_idx = list(cat_feature_indices)
    Xs_pool = Xs.copy()
    for i in cat_idx:
        col = Xs_pool.columns[i]
        Xs_pool[col] = Xs_pool[col].astype(str).fillna("Unknown")
    return Xs_pool, cat_idx


def _catboost_shap_values(
    model: CatBoostClassifier,
    Xs: pd.DataFrame,
    cat_feature_indices: Optional[Sequence[int]],
) -> Tuple[np.ndarray, float, str]:
    """返回 (n, f) SHAP（正类）与平均 base value。"""
    Xs_pool, cat_idx = _prepare_pool(Xs, cat_feature_indices)
    pool = Pool(Xs_pool, cat_features=cat_idx or None)
    raw = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"))
    if raw.ndim == 2:

        shap_vals = raw[:, :-1]
        base = float(np.mean(raw[:, -1]))
    elif raw.ndim == 3:

        cidx = 1 if raw.shape[1] > 1 else 0
        shap_vals = raw[:, cidx, :-1]
        base = float(np.mean(raw[:, cidx, -1]))
    else:
        raise ValueError(f"unexpected ShapValues shape: {raw.shape}")
    return shap_vals, base, "catboost_native"


def _plot_global_bar(fi: pd.DataFrame, output_dir: Path, top_k: int = 20) -> None:
    top = fi.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, max(4.5, 0.30 * len(top))))
    ax.barh(top["feature"], top["mean_abs_shap"], color=C_MAIN, height=0.72)
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title("Task3 Global Feature Importance (mean |SHAP|)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "shap_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_direction_bar(fi: pd.DataFrame, output_dir: Path, top_k: int = 18) -> None:
    """平均符号 SHAP：正=推高高焦虑概率，负=压低。"""
    top = fi.reindex(fi["mean_abs_shap"].abs().sort_values(ascending=False).index).head(top_k)
    top = top.iloc[::-1]
    colors = [C_POS if v >= 0 else C_NEG for v in top["mean_shap"]]
    fig, ax = plt.subplots(figsize=(8.2, max(4.5, 0.30 * len(top))))
    ax.barh(top["feature"], top["mean_shap"], color=colors, height=0.72)
    ax.axvline(0.0, color="#333333", lw=0.9)
    ax.set_xlabel("mean(SHAP)  (+ → higher range anxiety)")
    ax.set_title("Task3 Feature Direction (signed mean SHAP)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "shap_direction.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_beeswarm(Xs: pd.DataFrame, shap_vals: np.ndarray, output_dir: Path) -> bool:
    if not HAS_SHAP:
        return False
    try:
        plt.figure(figsize=(9.0, 6.5))
        shap.summary_plot(
            shap_vals,
            Xs,
            show=False,
            max_display=18,
            plot_size=None,
            color_bar_label="Feature value",
        )
        plt.title("Task3 SHAP Beeswarm (class=high anxiety)", pad=12)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=220, bbox_inches="tight")
        plt.close("all")
        return True
    except Exception as e:
        logger.warning("SHAP beeswarm skipped: %s", e)
        plt.close("all")
        return False


def _plot_dependence(
    Xs: pd.DataFrame,
    shap_vals: np.ndarray,
    feat: str,
    output_dir: Path,
    color_feat: Optional[str] = None,
) -> None:
    j = list(Xs.columns).index(feat)
    x = Xs[feat]
    if x.dtype == object or str(x.dtype).startswith("category"):
        x_plot = pd.Categorical(x).codes.astype(float)
        xlab = f"{feat} (codes)"
    else:
        x_plot = pd.to_numeric(x, errors="coerce")
        xlab = feat
    yv = shap_vals[:, j]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    if color_feat is not None and color_feat in Xs.columns:
        craw = Xs[color_feat]
        if craw.dtype == object or str(craw.dtype).startswith("category"):
            c = pd.Categorical(craw).codes.astype(float)
        else:
            c = pd.to_numeric(craw, errors="coerce")
        sc = ax.scatter(x_plot, yv, c=c, s=22, alpha=0.72, cmap="coolwarm", edgecolors="none")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(color_feat, fontsize=9)
    else:
        ax.scatter(x_plot, yv, s=22, alpha=0.72, c=C_MAIN, edgecolors="none")
    ax.axhline(0.0, color="#888888", lw=0.8, ls="--")
    ax.set_xlabel(xlab)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"Dependence: {feat}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / f"shap_dependence_{_safe_name(feat)}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_waterfall_bar(
    feature_names: Sequence[str],
    values: np.ndarray,
    base: float,
    label: str,
    output_dir: Path,
    top_k: int = 12,
    subtitle: str = "",
) -> None:
    order = np.argsort(np.abs(values))[::-1][:top_k]
    names = [feature_names[i] for i in order][::-1]
    vals = values[order][::-1]
    colors = [C_POS if v >= 0 else C_NEG for v in vals]
    fig, ax = plt.subplots(figsize=(8.2, max(4.2, 0.34 * len(names))))
    ax.barh(names, vals, color=colors, height=0.72)
    ax.axvline(0.0, color="#333333", lw=0.9)
    ax.set_xlabel("SHAP contribution")
    title = f"Typical {label}"
    if subtitle:
        title = f"{title}  ({subtitle})"
    ax.set_title(f"{title}  |  base={base:.3f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / f"waterfall_typical_{label}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_class_compare(
    Xs: pd.DataFrame,
    shap_vals: np.ndarray,
    y: np.ndarray,
    fi: pd.DataFrame,
    output_dir: Path,
    top_k: int = 10,
) -> None:
    """按真实标签分组的 mean|SHAP| 对比（高焦虑 vs 低焦虑样本）。"""
    feats = list(fi.head(top_k)["feature"])
    cols = list(Xs.columns)
    rows = []
    for f in feats:
        j = cols.index(f)
        for lab, name in [(0, "low"), (1, "high")]:
            mask = y == lab
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "feature": f,
                    "group": name,
                    "mean_abs_shap": float(np.abs(shap_vals[mask, j]).mean()),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return
    atomic_write_csv(output_dir / "shap_by_class.csv", df)

    pivot = df.pivot(index="feature", columns="group", values="mean_abs_shap").reindex(feats)
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.38 * len(feats))))
    y_pos = np.arange(len(feats))
    h = 0.36
    low = pivot["low"].fillna(0).values if "low" in pivot.columns else np.zeros(len(feats))
    high = pivot["high"].fillna(0).values if "high" in pivot.columns else np.zeros(len(feats))
    ax.barh(y_pos - h / 2, low, height=h, color=C_NEG, label="true low (y=0)", alpha=0.9)
    ax.barh(y_pos + h / 2, high, height=h, color=C_POS, label="true high (y=1)", alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP|) within group")
    ax.set_title("Task3 SHAP Magnitude by True Label")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "shap_by_class.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _pick_typical_indices(shap_vals: np.ndarray, y: np.ndarray) -> Dict[str, int]:
    """选代表性样本：高/低焦虑（强贡献）+ 边界样本（总贡献接近 0）。"""
    total = shap_vals.sum(axis=1)
    out: Dict[str, int] = {}
    high = np.where(y == 1)[0]
    low = np.where(y == 0)[0]
    if len(high):
        out["high"] = int(high[np.argmax(total[high])])
    if len(low):
        out["low"] = int(low[np.argmin(total[low])])

    out["borderline"] = int(np.argmin(np.abs(total)))
    return out


def run_shap_analysis(
    model: CatBoostClassifier,
    X_trans: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    cat_feature_indices: Optional[Sequence[int]] = None,
    max_samples: int = 300,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {"available": True, "backend": "catboost_native", "figures": []}
    try:
        y_arr = pd.Series(y).astype(int).to_numpy()
        rng = np.random.RandomState(42)
        idx = np.arange(len(X_trans))
        if len(idx) > max_samples:

            pos = idx[y_arr == 1]
            neg = idx[y_arr == 0]
            n_pos = min(len(pos), max_samples // 2)
            n_neg = min(len(neg), max_samples - n_pos)
            chosen = np.concatenate(
                [
                    rng.choice(pos, size=n_pos, replace=False) if n_pos else np.array([], dtype=int),
                    rng.choice(neg, size=n_neg, replace=False) if n_neg else np.array([], dtype=int),
                ]
            )
            if len(chosen) < max_samples and len(idx) > len(chosen):
                rest = np.setdiff1d(idx, chosen)
                extra = min(max_samples - len(chosen), len(rest))
                if extra > 0:
                    chosen = np.concatenate([chosen, rng.choice(rest, size=extra, replace=False)])
            idx = chosen
        Xs = X_trans.iloc[idx].copy()
        ys = y_arr[idx]

        shap_vals, base, backend = _catboost_shap_values(model, Xs, cat_feature_indices)
        info["backend"] = backend

        abs_mean = np.abs(shap_vals).mean(axis=0)
        mean_signed = shap_vals.mean(axis=0)
        fi = pd.DataFrame(
            {
                "feature": list(Xs.columns),
                "mean_abs_shap": abs_mean,
                "mean_shap": mean_signed,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        atomic_write_csv(output_dir / "global_mean_abs_shap.csv", fi)

        _plot_global_bar(fi, output_dir)
        info["figures"].append("shap_summary.png")

        _plot_direction_bar(fi, output_dir)
        info["figures"].append("shap_direction.png")

        if _plot_beeswarm(Xs, shap_vals, output_dir):
            info["figures"].append("shap_summary_beeswarm.png")

        top_feats = [str(f) for f in fi.head(4)["feature"]]
        color_feat = top_feats[1] if len(top_feats) > 1 else None
        for feat in top_feats:
            try:
                _plot_dependence(
                    Xs,
                    shap_vals,
                    feat,
                    output_dir,
                    color_feat=color_feat if feat != color_feat else (top_feats[0] if top_feats else None),
                )
                info["figures"].append(f"shap_dependence_{_safe_name(feat)}.png")
            except Exception as e:
                logger.warning("dependence %s failed: %s", feat, e)

        try:
            _plot_class_compare(Xs, shap_vals, ys, fi, output_dir, top_k=10)
            info["figures"].append("shap_by_class.png")
        except Exception as e:
            logger.warning("class compare failed: %s", e)

        try:
            picks = _pick_typical_indices(shap_vals, ys)
            names = list(Xs.columns)
            for label, i in picks.items():
                sub = f"y={int(ys[i])}, sumSHAP={float(shap_vals[i].sum()):.3f}"
                _plot_waterfall_bar(names, shap_vals[i], base, label, output_dir, subtitle=sub)
                info["figures"].append(f"waterfall_typical_{label}.png")
            info["typical_indices"] = {k: int(idx[v]) for k, v in picks.items()}
        except Exception as e:
            logger.warning("waterfall failed: %s", e)

        info["top_feature"] = str(fi.iloc[0]["feature"]) if len(fi) else None
        info["n_shap_samples"] = int(len(Xs))
        info["base_value"] = float(base)
        info["has_shap_pkg"] = bool(HAS_SHAP)
    except Exception as e:
        logger.warning("SHAP failed: %s", e)
        info["available"] = False
        info["error"] = str(e)
    return info
