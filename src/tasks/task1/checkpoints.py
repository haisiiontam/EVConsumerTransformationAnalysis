"""每折原子 checkpoint：保存 / 校验 / 组装 OOF。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.tasks.task1.io_utils import atomic_write_csv, atomic_write_json
from src.tasks.task1.metrics_ext import pred_from_proba

logger = logging.getLogger(__name__)


def fold_paths(ckpt_dir: Path, fold_id: int) -> Dict[str, Path]:
    d = Path(ckpt_dir)
    return {
        "predictions": d / f"fold_{fold_id}_predictions.csv",
        "metrics": d / f"fold_{fold_id}_metrics.json",
        "complete": d / f"fold_{fold_id}_complete.json",
    }


def fold_is_complete(ckpt_dir: Path, fold_id: int, expected_n: Optional[int] = None) -> bool:
    paths = fold_paths(ckpt_dir, fold_id)
    if not all(p.exists() for p in paths.values()):
        return False
    try:
        complete = json.loads(paths["complete"].read_text(encoding="utf-8"))
        if complete.get("status") != "complete":
            return False
        pred = pd.read_csv(paths["predictions"])
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        if expected_n is not None and len(pred) != expected_n:
            return False
        need = {"sample_index", "predicted_label", "prob_Low", "prob_Medium", "prob_High"}
        if not need.issubset(set(pred.columns)):
            return False
        if pred[["prob_Low", "prob_Medium", "prob_High"]].isna().any().any():
            return False
        s = pred[["prob_Low", "prob_Medium", "prob_High"]].sum(axis=1)
        if ((s - 1.0).abs() > 1e-2).mean() > 0.01:
            return False
        if "fold" in metrics and int(metrics["fold"]) != int(fold_id):
            return False
        return True
    except Exception as e:
        logger.warning("fold %s checkpoint invalid: %s", fold_id, e)
        return False


def save_fold_checkpoint(
    ckpt_dir: Path,
    fold_id: int,
    *,
    sample_index: np.ndarray,
    predicted_label: np.ndarray,
    proba: np.ndarray,
    metrics: Dict[str, Any],
    best_iteration: Optional[int],
    params: Dict[str, Any],
    features: Sequence[str],
    model_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths = fold_paths(ckpt_dir, fold_id)
    pred_df = pd.DataFrame(
        {
            "sample_index": sample_index.astype(int),
            "fold": fold_id,
            "predicted_label": predicted_label,
            "prob_Low": proba[:, 0],
            "prob_Medium": proba[:, 1],
            "prob_High": proba[:, 2],
        }
    )
    atomic_write_csv(paths["predictions"], pred_df)
    met = {
        "fold": fold_id,
        "model": model_name,
        "best_iteration": best_iteration,
        "n_val": int(len(pred_df)),
        "params": params,
        "features": list(features),
        **metrics,
    }
    if extra:
        met.update(extra)
    atomic_write_json(paths["metrics"], met)
    atomic_write_json(
        paths["complete"],
        {
            "status": "complete",
            "fold": fold_id,
            "model": model_name,
            "n_val": int(len(pred_df)),
            "predictions": str(paths["predictions"].name),
            "metrics": str(paths["metrics"].name),
        },
    )
    logger.info("checkpoint saved %s fold=%d n=%d", model_name, fold_id, len(pred_df))


def load_fold_predictions(ckpt_dir: Path, fold_id: int) -> pd.DataFrame:
    return pd.read_csv(fold_paths(ckpt_dir, fold_id)["predictions"])


def assemble_oof_from_checkpoints(
    ckpt_dir: Path,
    n_samples: int,
    n_folds: int,
    sample_index_all: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[int]]:
    """按 sample 位置索引组装 OOF（假定 folds 覆盖全部 0..n-1）。"""
    oof_p = np.zeros((n_samples, 3), dtype=float)
    oof_pred = np.empty(n_samples, dtype=object)
    fold_rows: List[Dict[str, Any]] = []
    best_iters: List[int] = []
    filled = np.zeros(n_samples, dtype=bool)
    idx_to_pos = {int(s): i for i, s in enumerate(sample_index_all)}

    for fold_id in range(n_folds):
        if not fold_is_complete(ckpt_dir, fold_id):
            raise FileNotFoundError(f"missing/corrupt checkpoint fold={fold_id} under {ckpt_dir}")
        pred = load_fold_predictions(ckpt_dir, fold_id)
        met = json.loads(fold_paths(ckpt_dir, fold_id)["metrics"].read_text(encoding="utf-8"))
        fold_rows.append(met)
        bi = met.get("best_iteration")
        if bi is not None:
            best_iters.append(int(bi))
        for _, row in pred.iterrows():
            pos = idx_to_pos[int(row["sample_index"])]
            oof_p[pos] = [row["prob_Low"], row["prob_Medium"], row["prob_High"]]
            oof_pred[pos] = row["predicted_label"]
            filled[pos] = True
    if not filled.all():
        raise RuntimeError(f"OOF incomplete: filled={filled.sum()}/{n_samples}")
    # 若 pred 为空字符串等，重算
    if any(p is None or (isinstance(p, float) and np.isnan(p)) for p in oof_pred):
        oof_pred = pred_from_proba(oof_p)
    return oof_p, oof_pred, fold_rows, best_iters


def persist_assembled_oof(
    output_path: Path,
    sample_index: Sequence[int],
    y_true: Sequence[Any],
    oof_pred: np.ndarray,
    oof_p: np.ndarray,
    fold_assign: Optional[np.ndarray] = None,
) -> None:
    data = {
        "sample_index": np.asarray(sample_index).astype(int),
        "true_label": list(y_true),
        "predicted_label": oof_pred,
        "prob_Low": oof_p[:, 0],
        "prob_Medium": oof_p[:, 1],
        "prob_High": oof_p[:, 2],
    }
    if fold_assign is not None:
        data["fold"] = fold_assign
    atomic_write_csv(output_path, pd.DataFrame(data))
