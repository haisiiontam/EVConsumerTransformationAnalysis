"""任务三可解释性：CatBoost 原生 SHAP。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.common.io_utils import atomic_write_csv

logger = logging.getLogger(__name__)


def run_shap_analysis(
    model: CatBoostClassifier,
    X_trans: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    cat_feature_indices: Optional[Sequence[int]] = None,
    max_samples: int = 200,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {"available": True, "backend": "catboost_native"}
    try:
        rng = np.random.RandomState(42)
        idx = np.arange(len(X_trans))
        if len(idx) > max_samples:
            idx = rng.choice(idx, size=max_samples, replace=False)
        Xs = X_trans.iloc[idx].copy()
        if cat_feature_indices is None:
            cat_feature_indices = [
                i
                for i, c in enumerate(Xs.columns)
                if not pd.api.types.is_numeric_dtype(Xs[c])
            ]
        Xs_pool = Xs.copy()
        for i in cat_feature_indices:
            col = Xs_pool.columns[i]
            Xs_pool[col] = Xs_pool[col].astype(str).fillna("Unknown")
        pool = Pool(Xs_pool, cat_features=list(cat_feature_indices) or None)
        raw = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"))
        # binary: (n, f+1)
        if raw.ndim == 2:
            shap_vals = raw[:, :-1]
        elif raw.ndim == 3:
            shap_vals = raw[:, 1, :-1]  # class 1
        else:
            raise ValueError(f"unexpected shap shape {raw.shape}")
        abs_mean = np.abs(shap_vals).mean(axis=0)
        fi = pd.DataFrame({"feature": list(Xs.columns), "mean_abs_shap": abs_mean})
        fi = fi.sort_values("mean_abs_shap", ascending=False)
        atomic_write_csv(output_dir / "global_mean_abs_shap.csv", fi)

        top = fi.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4.5, 0.28 * len(top))))
        ax.barh(top["feature"], top["mean_abs_shap"], color="#2F6F8F")
        ax.set_xlabel("mean(|SHAP|)")
        ax.set_title("Task3 Global Feature Importance (mean |SHAP|)")
        fig.tight_layout()
        fig.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        info["top_feature"] = str(fi.iloc[0]["feature"]) if len(fi) else None
        info["n_shap_samples"] = int(len(Xs))
    except Exception as e:
        logger.warning("SHAP failed: %s", e)
        info["available"] = False
        info["error"] = str(e)
    return info
