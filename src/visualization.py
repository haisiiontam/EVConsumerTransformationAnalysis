"""可视化：混淆矩阵与特征重要性图（英文标题，避免中文字体乱码）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

# 非交互后端，运行时不弹出窗口
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

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

    # 每个格子显示数量
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
    """绘制横向特征重要性柱状图（按重要性从高到低）。"""
    save_path = Path(save_path)
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
