"""SHAP 与论文图表（关联解释，非因果）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
    shap = None  # type: ignore


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
    max_samples: int = 200,
    cat_feature_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {"available": True, "shap_package": HAS_SHAP}
    try:
        return _run_shap_analysis_impl(
            model, X_trans, y, output_dir, max_samples, cat_feature_indices
        )
    except Exception as e:
        logger.warning("SHAP 分析失败（已跳过，不影响主结果）：%s", e)
        info["error"] = str(e)
        info["available"] = False
        return info


def _catboost_shap_values(
    model: CatBoostClassifier,
    Xs: pd.DataFrame,
    cat_feature_indices: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """用 CatBoost 原生 ShapValues，避免 shap.TreeExplainer 在 Windows 上原生崩溃。

    Returns
    -------
    shap_3d : (n, n_features, n_classes)
    base_values : (n_classes,)
    backend : backend name
    """
    from catboost import Pool

    if cat_feature_indices is None:
        cat_feature_indices = [
            i
            for i, c in enumerate(Xs.columns)
            if Xs[c].dtype == object
            or str(Xs[c].dtype).startswith("category")
            or not pd.api.types.is_numeric_dtype(Xs[c])
        ]
    # 非数值列强制为字符串，避免 Pool 误判为 float
    Xs_pool = Xs.copy()
    for i in cat_feature_indices:
        col = Xs_pool.columns[i]
        Xs_pool[col] = Xs_pool[col].astype(str).fillna("Unknown")
    pool = Pool(Xs_pool, cat_features=list(cat_feature_indices) or None)
    raw = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"))
    # multiclass: (n, n_classes, n_features+1) 或 (n, n_features+1)
    if raw.ndim == 3:
        base = raw[:, :, -1].mean(axis=0)
        shap_3d = np.transpose(raw[:, :, :-1], (0, 2, 1))  # (n, f, c)
    elif raw.ndim == 2:
        base = np.array([raw[:, -1].mean()])
        shap_3d = raw[:, :-1][:, :, None]
    else:
        raise ValueError(f"unexpected ShapValues shape: {raw.shape}")
    return shap_3d, base, "catboost_native"


def _plot_global_bar(fi: pd.DataFrame, output_dir: Path, top_k: int = 20) -> None:
    top = fi.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.28 * len(top))))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2F6F8F")
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title("Global Feature Importance (mean |SHAP|)")
    fig.tight_layout()
    fig.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_dependence(
    Xs: pd.DataFrame,
    shap_3d: np.ndarray,
    feat: str,
    class_idx: int,
    output_dir: Path,
) -> None:
    j = list(Xs.columns).index(feat)
    x = Xs[feat]
    if x.dtype == object or str(x.dtype).startswith("category"):
        x_plot = pd.Categorical(x).codes.astype(float)
    else:
        x_plot = pd.to_numeric(x, errors="coerce")
    yv = shap_3d[:, j, class_idx]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(x_plot, yv, s=18, alpha=0.65, c="#2F6F8F", edgecolors="none")
    ax.set_xlabel(feat)
    ax.set_ylabel(f"SHAP ({TASK1_LABEL_ORDER[class_idx]})")
    ax.set_title(f"Dependence: {feat}")
    fig.tight_layout()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in feat)
    fig.savefig(output_dir / f"shap_dependence_{safe}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_waterfall_bar(
    feature_names: Sequence[str],
    values: np.ndarray,
    base: float,
    label: str,
    output_dir: Path,
    top_k: int = 12,
) -> None:
    order = np.argsort(np.abs(values))[::-1][:top_k]
    names = [feature_names[i] for i in order][::-1]
    vals = values[order][::-1]
    colors = ["#C44E52" if v >= 0 else "#4C72B0" for v in vals]
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.32 * len(names))))
    ax.barh(names, vals, color=colors)
    ax.axvline(0.0, color="#333333", lw=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_title(f"Typical {label}  (base={base:.3f})")
    fig.tight_layout()
    fig.savefig(output_dir / f"waterfall_typical_{label}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _run_shap_analysis_impl(
    model: CatBoostClassifier,
    X_trans: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    max_samples: int,
    cat_feature_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": True}
    y_n = normalize_label_array(y)
    rng = np.random.RandomState(42)
    idx = np.arange(len(X_trans))
    if len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    Xs = X_trans.iloc[idx].copy()
    ys = y_n[idx]

    logger.info("SHAP (CatBoost native) on n=%d features=%d", len(Xs), Xs.shape[1])
    shap_3d, base_values, backend = _catboost_shap_values(model, Xs, cat_feature_indices)
    info["backend"] = backend

    # shap_3d: (n, f, c)
    abs_mean = np.abs(shap_3d).mean(axis=0).mean(axis=-1)
    per_class = {
        TASK1_LABEL_ORDER[i]: np.abs(shap_3d[:, :, i]).mean(axis=0)
        for i in range(min(3, shap_3d.shape[-1]))
    }

    fi = pd.DataFrame({"feature": list(Xs.columns), "mean_abs_shap": abs_mean})
    for lab, vals in per_class.items():
        if len(vals) == len(fi):
            fi[f"mean_abs_shap_{lab}"] = vals
    fi = fi.sort_values("mean_abs_shap", ascending=False)
    fi.to_csv(output_dir / "global_mean_abs_shap.csv", index=False, encoding="utf-8-sig")

    try:
        _plot_global_bar(fi, output_dir)
    except Exception as e:  # pragma: no cover
        logger.warning("SHAP summary 绘图失败：%s", e)

    # 可选：仍尝试 shap 包画 beeswarm（失败不影响）
    if HAS_SHAP:
        try:
            shap_list = [shap_3d[:, :, i] for i in range(shap_3d.shape[-1])]
            plt.figure()
            shap.summary_plot(shap_list, Xs, class_names=TASK1_LABEL_ORDER, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
            plt.close("all")
        except Exception as e:  # pragma: no cover
            logger.warning("SHAP beeswarm 跳过：%s", e)

    top_feat = str(fi.iloc[0]["feature"]) if len(fi) else None
    if top_feat is not None:
        try:
            # Medium 类 dependence（中类最难）
            _plot_dependence(Xs, shap_3d, top_feat, class_idx=1, output_dir=output_dir)
        except Exception as e:  # pragma: no cover
            logger.warning("SHAP dependence 失败：%s", e)

    try:
        for lab in TASK1_LABEL_ORDER:
            hits = np.where(ys == lab)[0]
            if len(hits) == 0:
                continue
            i = int(hits[0])
            cidx = TASK1_LABEL_ORDER.index(lab)
            _plot_waterfall_bar(
                list(Xs.columns),
                shap_3d[i, :, cidx],
                float(base_values[cidx]) if cidx < len(base_values) else float(base_values[0]),
                lab,
                output_dir,
            )
    except Exception as e:  # pragma: no cover
        logger.warning("瀑布图失败：%s", e)

    info.update({"n_shap_samples": int(len(Xs)), "top_feature": top_feat})
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
    yt = normalize_label_array(y_true)
    yp = normalize_label_array(y_pred)
    cm = confusion_matrix(yt, yp, labels=TASK1_LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=TASK1_LABEL_ORDER).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{title_prefix} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    y_bin = label_binarize(yt, classes=TASK1_LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, lab in enumerate(TASK1_LABEL_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        ax.plot(fpr, tpr, label=f"{lab} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.legend()
    ax.set_title(f"{title_prefix} OVR ROC")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_ovr.png", dpi=160)
    plt.close(fig)

    # Medium 概率分布
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab in TASK1_LABEL_ORDER:
        ax.hist(proba[yt == lab, 1], bins=30, alpha=0.45, label=f"true={lab}")
    ax.set_xlabel("P(Medium)")
    ax.legend()
    ax.set_title(f"{title_prefix} Medium Probability by True Class")
    fig.tight_layout()
    fig.savefig(output_dir / "medium_prob_hist.png", dpi=160)
    plt.close(fig)
