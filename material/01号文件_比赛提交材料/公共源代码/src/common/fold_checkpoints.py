from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.common.io_utils import atomic_write_csv, atomic_write_json

logger = logging.getLogger(__name__)


PROB_COLS_MULTICLASS = ("prob_Low", "prob_Medium", "prob_High")

PROB_COLS_BINARY = ("prob_0", "prob_1")


def fold_paths(ckpt_dir: Path, fold_id: int) -> Dict[str, Path]:
    d = Path(ckpt_dir)
    return {
        "predictions": d / f"fold_{fold_id}_predictions.csv",
        "metrics": d / f"fold_{fold_id}_metrics.json",
        "config": d / f"fold_{fold_id}_config.json",
        "complete": d / f"fold_{fold_id}_complete.json",
    }


def fold_is_complete(
    ckpt_dir: Path,
    fold_id: int,
    expected_n: Optional[int] = None,
    prob_cols: Sequence[str] = PROB_COLS_MULTICLASS,
) -> bool:
    paths = fold_paths(ckpt_dir, fold_id)
    required = [paths["predictions"], paths["metrics"], paths["complete"]]
    if not all(p.exists() for p in required):
        return False
    try:
        complete = json.loads(paths["complete"].read_text(encoding="utf-8"))
        if complete.get("status") != "complete":
            return False
        pred = pd.read_csv(paths["predictions"])
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        if expected_n is not None and len(pred) != expected_n:
            return False
        need = {"sample_index", "predicted_label", *prob_cols}
        if not need.issubset(set(pred.columns)):
            return False
        if pred[list(prob_cols)].isna().any().any():
            return False
        s = pred[list(prob_cols)].sum(axis=1)
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
    prob_cols: Sequence[str] = PROB_COLS_MULTICLASS,
    y_true: Optional[np.ndarray] = None,
    pred_score: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths = fold_paths(ckpt_dir, fold_id)
    data: Dict[str, Any] = {
        "sample_index": np.asarray(sample_index).astype(int),
        "fold": fold_id,
        "predicted_label": predicted_label,
    }
    if y_true is not None:
        data["true_label"] = y_true
    proba = np.asarray(proba)
    if proba.ndim == 1:

        data["prob_0"] = 1.0 - proba
        data["prob_1"] = proba
        used_cols = ("prob_0", "prob_1")
    else:
        used_cols = tuple(prob_cols)
        for i, c in enumerate(used_cols):
            data[c] = proba[:, i]
    if pred_score is not None:
        data["pred_score"] = pred_score
    pred_df = pd.DataFrame(data)
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
        paths["config"],
        {
            "fold": fold_id,
            "model": model_name,
            "params": params,
            "features": list(features),
            "prob_cols": list(used_cols),
        },
    )
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
    prob_cols: Sequence[str] = PROB_COLS_MULTICLASS,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[int]]:
    n_prob = len(prob_cols)
    oof_p = np.zeros((n_samples, n_prob), dtype=float)
    oof_pred = np.empty(n_samples, dtype=object)
    fold_rows: List[Dict[str, Any]] = []
    best_iters: List[int] = []
    filled = np.zeros(n_samples, dtype=bool)
    idx_to_pos = {int(s): i for i, s in enumerate(sample_index_all)}

    for fold_id in range(n_folds):
        if not fold_is_complete(ckpt_dir, fold_id, prob_cols=prob_cols):
            raise FileNotFoundError(f"missing/corrupt checkpoint fold={fold_id} under {ckpt_dir}")
        pred = load_fold_predictions(ckpt_dir, fold_id)
        met = json.loads(fold_paths(ckpt_dir, fold_id)["metrics"].read_text(encoding="utf-8"))
        fold_rows.append(met)
        bi = met.get("best_iteration")
        if bi is not None:
            best_iters.append(int(bi))
        for _, row in pred.iterrows():
            pos = idx_to_pos[int(row["sample_index"])]
            oof_p[pos] = [row[c] for c in prob_cols]
            oof_pred[pos] = row["predicted_label"]
            filled[pos] = True
    if not filled.all():
        raise RuntimeError(f"OOF incomplete: filled={filled.sum()}/{n_samples}")
    return oof_p, oof_pred, fold_rows, best_iters


def persist_assembled_oof(
    output_path: Path,
    sample_index: Sequence[int],
    y_true: Sequence[Any],
    oof_pred: np.ndarray,
    oof_p: np.ndarray,
    prob_cols: Sequence[str] = PROB_COLS_MULTICLASS,
    fold_assign: Optional[np.ndarray] = None,
    pred_score: Optional[np.ndarray] = None,
) -> None:
    data: Dict[str, Any] = {
        "sample_index": np.asarray(sample_index).astype(int),
        "true_label": list(y_true),
        "predicted_label": oof_pred,
    }
    for i, c in enumerate(prob_cols):
        data[c] = oof_p[:, i]
    if fold_assign is not None:
        data["fold"] = fold_assign
    if pred_score is not None:
        data["pred_score"] = pred_score
    atomic_write_csv(output_path, pd.DataFrame(data))
