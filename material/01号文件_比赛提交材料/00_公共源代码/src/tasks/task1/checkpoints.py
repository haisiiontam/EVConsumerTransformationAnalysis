"""任务一每折 checkpoint：兼容包装 common.fold_checkpoints。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.common.fold_checkpoints import (
    PROB_COLS_MULTICLASS,
    assemble_oof_from_checkpoints as _assemble,
    fold_is_complete as _fold_is_complete,
    fold_paths,
    load_fold_predictions,
    persist_assembled_oof as _persist,
    save_fold_checkpoint as _save,
)
from src.tasks.task1.metrics_ext import pred_from_proba

__all__ = [
    "fold_paths",
    "fold_is_complete",
    "save_fold_checkpoint",
    "load_fold_predictions",
    "assemble_oof_from_checkpoints",
    "persist_assembled_oof",
]


def fold_is_complete(
    ckpt_dir: Path, fold_id: int, expected_n: Optional[int] = None
) -> bool:
    return _fold_is_complete(
        ckpt_dir, fold_id, expected_n=expected_n, prob_cols=PROB_COLS_MULTICLASS
    )


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
    _save(
        ckpt_dir,
        fold_id,
        sample_index=sample_index,
        predicted_label=predicted_label,
        proba=proba,
        metrics=metrics,
        best_iteration=best_iteration,
        params=params,
        features=features,
        model_name=model_name,
        prob_cols=PROB_COLS_MULTICLASS,
        extra=extra,
    )


def assemble_oof_from_checkpoints(
    ckpt_dir: Path,
    n_samples: int,
    n_folds: int,
    sample_index_all: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[int]]:
    oof_p, oof_pred, fold_rows, best_iters = _assemble(
        ckpt_dir,
        n_samples,
        n_folds,
        sample_index_all,
        prob_cols=PROB_COLS_MULTICLASS,
    )
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
    _persist(
        output_path,
        sample_index,
        y_true,
        oof_pred,
        oof_p,
        prob_cols=PROB_COLS_MULTICLASS,
        fold_assign=fold_assign,
    )
