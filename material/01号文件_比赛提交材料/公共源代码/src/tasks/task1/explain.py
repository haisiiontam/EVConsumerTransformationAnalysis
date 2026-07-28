\
\
\

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.config import TASK1_LABEL_ORDER
from src.tasks.task1.metrics_ext import normalize_label_array
from src.tasks.task1.preprocess import CatPrepState, cat_indices, fit_cat_preprocessor, transform_cat

logger = logging.getLogger(__name__)

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None

C_POS = "#C44E52"
C_NEG = "#4C72B0"
C_MAIN = "#2F6F8F"
CLASS_COLORS = {"Low": "#4C72B0", "Medium": "#E8A838", "High": "#C44E52"}


def train_final_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    params: Dict[str, Any],
) -> tuple[CatBoostClassifier, CatPrepState, pd.DataFrame]:
    state = fit_cat_preprocessor(X, features)
    Xt = transform_cat(X, state)
    y_n = pd.Series(normalize_label_array(y))
    model = CatBoostClassifier(**params)
    model.fit(Xt, y_n, cat_features=cat_indices(state))
    return model, state, Xt


def run_shap_analysis(
    model: CatBoostClassifier,
    X_trans: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    max_samples: int = 300,
    cat_feature_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {"available": True, "shap_package": HAS_SHAP, "figures": []}
    try:
        return _run_shap_analysis_impl(
            model, X_trans, y, output_dir, max_samples, cat_feature_indices, info
        )
    except Exception as e:
        logger.warning("SHAP 分析失败（已跳过，不影响主结果）：%s", e)
        info["error"] = str(e)
        info["available"] = False
        return info


def _safe_name(feat: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in feat)


def _catboost_shap_values(
    model: CatBoostClassifier,
    Xs: pd.DataFrame,
    cat_feature_indices: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """返回 shap_3d (n, f, c), base_values (c,), backend。"""
    from catboost import Pool

    if cat_feature_indices is None:
        cat_feature_indices = [
            i
            for i, c in enumerate(Xs.columns)
            if Xs[c].dtype == object
            or str(Xs[c].dtype).startswith("category")
            or not pd.api.types.is_numeric_dtype(Xs[c])
        ]
    Xs_pool = Xs.copy()
    for i in cat_feature_indices:
        col = Xs_pool.columns[i]
        Xs_pool[col] = Xs_pool[col].astype(str).fillna("Unknown")
    pool = Pool(Xs_pool, cat_features=list(cat_feature_indices) or None)
    raw = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"))
    if raw.ndim == 3:
        base = raw[:, :, -1].mean(axis=0)
        shap_3d = np.transpose(raw[:, :, :-1], (0, 2, 1))
    elif raw.ndim == 2:
        base = np.array([raw[:, -1].mean()])
        shap_3d = raw[:, :-1][:, :, None]
    else:
        raise ValueError(f"unexpected ShapValues shape: {raw.shape}")
    return shap_3d, base, "catboost_native"


def _plot_global_bar(fi: pd.DataFrame, output_dir: Path, top_k: int = 20) -> None:
    top = fi.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, max(4.5, 0.30 * len(top))))
    ax.barh(top["feature"], top["mean_abs_shap"], color=C_MAIN, height=0.72)
    ax.set_xlabel("mean(|SHAP|) across classes")
    ax.set_title("Task1 Global Feature Importance (mean |SHAP|)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "shap_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_class_compare(fi: pd.DataFrame, output_dir: Path, top_k: int = 12) -> None:
    cols = [f"mean_abs_shap_{lab}" for lab in TASK1_LABEL_ORDER if f"mean_abs_shap_{lab}" in fi.columns]
    if not cols:
        return
    top = fi.head(top_k)
    feats = list(top["feature"])
    y_pos = np.arange(len(feats))
    h = 0.25
    fig, ax = plt.subplots(figsize=(9.0, max(4.8, 0.40 * len(feats))))
    for i, lab in enumerate(TASK1_LABEL_ORDER):
        col = f"mean_abs_shap_{lab}"
        if col not in top.columns:
            continue
        ax.barh(
            y_pos + (i - 1) * h,
            top[col].values,
            height=h,
            label=lab,
            color=CLASS_COLORS.get(lab, C_MAIN),
            alpha=0.9,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title("Task1 SHAP Magnitude by Class")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "shap_by_class.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_beeswarm(Xs: pd.DataFrame, shap_3d: np.ndarray, output_dir: Path) -> bool:
    if not HAS_SHAP:
        return False
    try:
        shap_list = [shap_3d[:, :, i] for i in range(shap_3d.shape[-1])]
        plt.figure(figsize=(9.5, 7.0))
        shap.summary_plot(
            shap_list,
            Xs,
            class_names=list(TASK1_LABEL_ORDER),
            show=False,
            max_display=18,
            plot_size=None,
        )
        plt.title("Task1 SHAP Beeswarm (multiclass)", pad=12)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=220, bbox_inches="tight")
        plt.close("all")
        return True
    except Exception as e:
        logger.warning("SHAP beeswarm 跳过：%s", e)
        plt.close("all")
        return False


def _plot_dependence(
    Xs: pd.DataFrame,
    shap_3d: np.ndarray,
    feat: str,
    class_idx: int,
    output_dir: Path,
    color_feat: Optional[str] = None,
    suffix: str = "",
) -> None:
    j = list(Xs.columns).index(feat)
    x = Xs[feat]
    if x.dtype == object or str(x.dtype).startswith("category"):
        x_plot = pd.Categorical(x).codes.astype(float)
        xlab = f"{feat} (codes)"
    else:
        x_plot = pd.to_numeric(x, errors="coerce")
        xlab = feat
    yv = shap_3d[:, j, class_idx]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    if color_feat is not None and color_feat in Xs.columns and color_feat != feat:
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
    lab = TASK1_LABEL_ORDER[class_idx]
    ax.set_xlabel(xlab)
    ax.set_ylabel(f"SHAP ({lab})")
    ax.set_title(f"Dependence: {feat} → {lab}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    tag = suffix or f"{_safe_name(feat)}_{lab}"
    fig.savefig(output_dir / f"shap_dependence_{tag}.png", dpi=220, bbox_inches="tight")
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


def _pick_typical_indices(
    shap_3d: np.ndarray,
    ys: np.ndarray,
    proba: Optional[np.ndarray],
) -> Dict[str, int]:
    """每类选贡献最强的典型样本；另选 Medium 边界/误判候选。"""
    out: Dict[str, int] = {}
    for cidx, lab in enumerate(TASK1_LABEL_ORDER):
        hits = np.where(ys == lab)[0]
        if len(hits) == 0:
            continue
        score = shap_3d[hits, :, cidx].sum(axis=1)
        out[lab] = int(hits[int(np.argmax(score))])


    med_idx = np.where(ys == "Medium")[0]
    if len(med_idx) and proba is not None and proba.shape[1] >= 3:
        p = proba[med_idx]

        rival = np.maximum(p[:, 0], p[:, 2])
        margin = p[:, 1] - rival
        out["Medium_borderline"] = int(med_idx[int(np.argmin(margin))])

        out["Medium_confident"] = int(med_idx[int(np.argmax(p[:, 1]))])
    return out


def _plot_medium_error_waterfalls(
    Xs: pd.DataFrame,
    ys: np.ndarray,
    shap_3d: np.ndarray,
    base_values: np.ndarray,
    proba: np.ndarray,
    output_dir: Path,
) -> List[str]:
    """Medium 被预测为 Low / High 的个案瀑布（对真实类 Medium 的 SHAP）。"""
    figs: List[str] = []
    if proba is None or proba.ndim != 2 or proba.shape[1] < 3:
        return figs
    pred = np.array([TASK1_LABEL_ORDER[i] for i in np.argmax(proba, axis=1)])
    med_c = TASK1_LABEL_ORDER.index("Medium")
    names = list(Xs.columns)
    for wrong in ("Low", "High"):
        hits = np.where((ys == "Medium") & (pred == wrong))[0]
        if len(hits) == 0:
            continue

        i = int(hits[int(np.argmax(proba[hits, med_c]))])
        label = f"Medium_as_{wrong}"
        _plot_waterfall_bar(
            names,
            shap_3d[i, :, med_c],
            float(base_values[med_c]),
            label,
            output_dir,
            subtitle=f"true=Medium pred={wrong} pM={proba[i, med_c]:.3f}",
        )
        figs.append(f"waterfall_typical_{label}.png")
    return figs


def _stratified_sample(y_n: np.ndarray, max_samples: int, rng: np.random.RandomState) -> np.ndarray:
    idx = np.arange(len(y_n))
    if len(idx) <= max_samples:
        return idx

    per = max(max_samples // max(len(TASK1_LABEL_ORDER), 1), 1)
    chosen: List[int] = []
    for lab in TASK1_LABEL_ORDER:
        hits = idx[y_n == lab]
        if len(hits) == 0:
            continue
        take = min(len(hits), per)
        chosen.extend(rng.choice(hits, size=take, replace=False).tolist())
    chosen_arr = np.unique(np.asarray(chosen, dtype=int))
    if len(chosen_arr) < max_samples:
        rest = np.setdiff1d(idx, chosen_arr)
        extra = min(max_samples - len(chosen_arr), len(rest))
        if extra > 0:
            chosen_arr = np.concatenate([chosen_arr, rng.choice(rest, size=extra, replace=False)])
    if len(chosen_arr) > max_samples:
        chosen_arr = rng.choice(chosen_arr, size=max_samples, replace=False)
    return chosen_arr


def _run_shap_analysis_impl(
    model: CatBoostClassifier,
    X_trans: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    max_samples: int,
    cat_feature_indices: Optional[Sequence[int]],
    info: Dict[str, Any],
) -> Dict[str, Any]:
    y_n = normalize_label_array(y)
    rng = np.random.RandomState(42)
    idx = _stratified_sample(y_n, max_samples, rng)
    Xs = X_trans.iloc[idx].copy()
    ys = y_n[idx]

    logger.info("SHAP (CatBoost native) on n=%d features=%d", len(Xs), Xs.shape[1])
    shap_3d, base_values, backend = _catboost_shap_values(model, Xs, cat_feature_indices)
    info["backend"] = backend


    try:
        proba = np.asarray(model.predict_proba(Xs), dtype=float)

        classes = [str(c) for c in model.classes_]
        order = [classes.index(lab) if lab in classes else i for i, lab in enumerate(TASK1_LABEL_ORDER)]
        if list(order) != list(range(len(order))):
            proba = proba[:, order]
    except Exception:
        proba = None

    abs_mean = np.abs(shap_3d).mean(axis=0).mean(axis=-1)
    fi = pd.DataFrame({"feature": list(Xs.columns), "mean_abs_shap": abs_mean})
    for i, lab in enumerate(TASK1_LABEL_ORDER):
        if i < shap_3d.shape[-1]:
            fi[f"mean_abs_shap_{lab}"] = np.abs(shap_3d[:, :, i]).mean(axis=0)
    fi = fi.sort_values("mean_abs_shap", ascending=False)
    fi.to_csv(output_dir / "global_mean_abs_shap.csv", index=False, encoding="utf-8-sig")

    try:
        _plot_global_bar(fi, output_dir)
        info["figures"].append("shap_summary.png")
    except Exception as e:
        logger.warning("SHAP summary 绘图失败：%s", e)

    try:
        _plot_class_compare(fi, output_dir)
        info["figures"].append("shap_by_class.png")
    except Exception as e:
        logger.warning("class compare 失败：%s", e)

    if _plot_beeswarm(Xs, shap_3d, output_dir):
        info["figures"].append("shap_summary_beeswarm.png")

    top_feats = [str(f) for f in fi.head(4)["feature"]]
    color_feat = top_feats[1] if len(top_feats) > 1 else None

    for feat in top_feats:
        try:
            _plot_dependence(Xs, shap_3d, feat, class_idx=1, output_dir=output_dir, color_feat=color_feat)
            info["figures"].append(f"shap_dependence_{_safe_name(feat)}_Medium.png")
        except Exception as e:
            logger.warning("dependence Medium %s: %s", feat, e)
    if top_feats:
        for cidx, lab in ((0, "Low"), (2, "High")):
            try:
                _plot_dependence(
                    Xs,
                    shap_3d,
                    top_feats[0],
                    class_idx=cidx,
                    output_dir=output_dir,
                    color_feat=color_feat,
                )
                info["figures"].append(f"shap_dependence_{_safe_name(top_feats[0])}_{lab}.png")
            except Exception as e:
                logger.warning("dependence %s: %s", lab, e)


    if top_feats:
        try:
            _plot_dependence(
                Xs,
                shap_3d,
                top_feats[0],
                class_idx=1,
                output_dir=output_dir,
                color_feat=color_feat,
                suffix=_safe_name(top_feats[0]),
            )
            info["figures"].append(f"shap_dependence_{_safe_name(top_feats[0])}.png")
        except Exception:
            pass

    picks = _pick_typical_indices(shap_3d, ys, proba)
    names = list(Xs.columns)
    for lab, i in picks.items():
        try:
            if lab in TASK1_LABEL_ORDER:
                cidx = TASK1_LABEL_ORDER.index(lab)
                sub = f"sumSHAP={float(shap_3d[i, :, cidx].sum()):.3f}"
                _plot_waterfall_bar(
                    names,
                    shap_3d[i, :, cidx],
                    float(base_values[cidx]) if cidx < len(base_values) else float(base_values[0]),
                    lab,
                    output_dir,
                    subtitle=sub,
                )
                info["figures"].append(f"waterfall_typical_{lab}.png")
            elif lab.startswith("Medium"):
                cidx = TASK1_LABEL_ORDER.index("Medium")
                _plot_waterfall_bar(
                    names,
                    shap_3d[i, :, cidx],
                    float(base_values[cidx]),
                    lab,
                    output_dir,
                    subtitle=f"y=Medium",
                )
                info["figures"].append(f"waterfall_typical_{lab}.png")
        except Exception as e:
            logger.warning("waterfall %s: %s", lab, e)

    if proba is not None:
        try:
            err_figs = _plot_medium_error_waterfalls(Xs, ys, shap_3d, base_values, proba, output_dir)
            info["figures"].extend(err_figs)
        except Exception as e:
            logger.warning("medium error waterfalls: %s", e)


    readme = "\n".join(
        [
            "Task1 SHAP artifacts (CatBoost native ShapValues; association ≠ causation).",
            f"Backend: {backend}. Explains M1 CatBoost (final submitter is soft_vote).",
            f"n_samples={len(Xs)} (stratified). top_feature={top_feats[0] if top_feats else None}.",
            "",
            "Core figures:",
            "- shap_summary.png / shap_summary_beeswarm.png / shap_by_class.png",
            "- shap_dependence_*_Medium/Low/High.png",
            "- waterfall_typical_{Low,Medium,High}.png",
            "- waterfall_typical_Medium_{borderline,confident}.png",
            "- waterfall_typical_Medium_as_{Low,High}.png  (misclassification cases)",
            "- global_mean_abs_shap.csv",
            "",
        ]
    )
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    info.update(
        {
            "n_shap_samples": int(len(Xs)),
            "top_feature": top_feats[0] if top_feats else None,
            "typical_labels": list(picks.keys()),
            "note": "SHAP on M1 CatBoost; locked predictor is soft_vote",
        }
    )
    return info


def plot_confusion_and_curves(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    proba: np.ndarray,
    output_dir: Path,
    title_prefix: str = "Task1",
) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
    from sklearn.preprocessing import label_binarize

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(TASK1_LABEL_ORDER)
    y_t = normalize_label_array(y_true)
    y_p = normalize_label_array(y_pred)
    cm = confusion_matrix(y_t, y_p, labels=labels)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{title_prefix} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    y_bin = label_binarize(y_t, classes=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, lab in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        ax.plot(fpr, tpr, label=f"{lab} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{title_prefix} ROC (OvR)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_ovr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
