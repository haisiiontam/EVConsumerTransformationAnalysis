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
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {"available": HAS_SHAP}
    if not HAS_SHAP:
        logger.warning("未安装 shap，跳过 SHAP 分析")
        return info

    try:
        return _run_shap_analysis_impl(model, X_trans, y, output_dir, max_samples)
    except Exception as e:
        logger.warning("SHAP 分析失败（已跳过，不影响主结果）：%s", e)
        info["error"] = str(e)
        return info


def _run_shap_analysis_impl(
    model: CatBoostClassifier,
    X_trans: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    max_samples: int,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": True}
    y_n = normalize_label_array(y)
    rng = np.random.RandomState(42)
    idx = np.arange(len(X_trans))
    if len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    Xs = X_trans.iloc[idx].copy()
    # TreeExplainer + 类别特征偶发原生崩溃：强制数值矩阵
    for c in Xs.columns:
        if Xs[c].dtype == object or str(Xs[c].dtype).startswith("category"):
            Xs[c] = pd.Categorical(Xs[c]).codes
    ys = y_n[idx]

    logger.info("SHAP TreeExplainer on n=%d features=%d", len(Xs), Xs.shape[1])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)
    # CatBoost multiclass: list of arrays or 3D
    if isinstance(shap_values, list):
        abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        per_class = {
            TASK1_LABEL_ORDER[i]: np.abs(shap_values[i]).mean(axis=0)
            for i in range(min(3, len(shap_values)))
        }
    else:
        # shape (n, f, c) or (n, f)
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            abs_mean = np.abs(arr).mean(axis=0).mean(axis=-1)
            per_class = {
                TASK1_LABEL_ORDER[i]: np.abs(arr[:, :, i]).mean(axis=0)
                for i in range(min(3, arr.shape[-1]))
            }
        else:
            abs_mean = np.abs(arr).mean(axis=0)
            per_class = {"all": abs_mean}

    fi = pd.DataFrame({"feature": list(Xs.columns), "mean_abs_shap": abs_mean})
    for lab, vals in per_class.items():
        if len(vals) == len(fi):
            fi[f"mean_abs_shap_{lab}"] = vals
    fi = fi.sort_values("mean_abs_shap", ascending=False)
    fi.to_csv(output_dir / "global_mean_abs_shap.csv", index=False, encoding="utf-8-sig")

    try:
        plt.figure()
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values, Xs, class_names=TASK1_LABEL_ORDER, show=False)
        else:
            shap.summary_plot(shap_values, Xs, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close("all")
    except Exception as e:  # pragma: no cover
        logger.warning("SHAP summary 绘图失败：%s", e)

    # dependence：取 top1 特征
    top_feat = str(fi.iloc[0]["feature"]) if len(fi) else None
    if top_feat is not None:
        try:
            plt.figure()
            if isinstance(shap_values, list):
                shap.dependence_plot(top_feat, shap_values[1], Xs, show=False)
            else:
                shap.dependence_plot(top_feat, shap_values, Xs, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_dependence_{top_feat}.png", dpi=160, bbox_inches="tight")
            plt.close("all")
        except Exception as e:  # pragma: no cover
            logger.warning("SHAP dependence 失败：%s", e)

    # 典型样本瀑布图
    try:
        for lab in TASK1_LABEL_ORDER:
            hits = np.where(ys == lab)[0]
            if len(hits) == 0:
                continue
            i = int(hits[0])
            if isinstance(shap_values, list):
                sv = shap_values[TASK1_LABEL_ORDER.index(lab)][i]
                base = (
                    explainer.expected_value[TASK1_LABEL_ORDER.index(lab)]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                )
            else:
                arr = np.asarray(shap_values)
                if arr.ndim == 3:
                    sv = arr[i, :, TASK1_LABEL_ORDER.index(lab)]
                    base = explainer.expected_value[TASK1_LABEL_ORDER.index(lab)]
                else:
                    sv = arr[i]
                    base = explainer.expected_value
            exp_v = shap.Explanation(
                values=sv,
                base_values=base,
                data=Xs.iloc[i].to_numpy(),
                feature_names=list(Xs.columns),
            )
            shap.plots.waterfall(exp_v, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"waterfall_typical_{lab}.png", dpi=160, bbox_inches="tight")
            plt.close("all")
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
