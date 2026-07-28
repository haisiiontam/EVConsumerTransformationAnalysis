"""可视化：混淆矩阵、特征重要性、概率分布与 ROC。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    title: str,
    save_path: Path,
    dpi: int = 180,
) -> None:
    """绘制混淆矩阵并保存为 PNG。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=list(labels),
        yticklabels=list(labels),
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("混淆矩阵图已保存：%s", save_path)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str,
    save_path: Path,
    dpi: int = 180,
) -> None:
    """绘制横向特征重要性柱状图。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = importance_df.sort_values("importance", ascending=True).copy()

    fig_height = max(3.5, 0.45 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(df["feature"].astype(str), df["importance"], color="#2E6F95")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("特征重要性图已保存：%s", save_path)


def plot_probability_distribution(
    y_true: Sequence[int],
    prob_1: Sequence[float],
    threshold: float,
    title: str,
    save_path: Path,
    dpi: int = 180,
) -> None:
    """绘制真实 0/1 类的 P(y=1) 分布，并标出 threshold。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_1, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p[y == 0], bins=40, alpha=0.55, label="True 0", density=True, color="#4C78A8")
    ax.hist(p[y == 1], bins=40, alpha=0.55, label="True 1", density=True, color="#F58518")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.3f}")
    ax.set_xlabel("Predicted probability of class 1")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("概率分布图已保存：%s", save_path)


def plot_roc_curve(
    y_true: Sequence[int],
    prob_1: Sequence[float],
    title: str,
    save_path: Path,
    dpi: int = 180,
) -> None:
    """绘制 ROC 曲线。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_1, dtype=float)
    fpr, tpr, _ = roc_curve(y, p)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#2E6F95", linewidth=2, label="ROC")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC 曲线已保存：%s", save_path)
