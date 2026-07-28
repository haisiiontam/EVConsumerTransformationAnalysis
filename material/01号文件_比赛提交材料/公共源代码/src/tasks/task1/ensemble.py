"""互补性、软投票、Stacking、类别偏置后处理。"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_SEED, TASK1_LABEL_ORDER, TASK1_N_FOLDS
from src.tasks.task1.metrics_ext import evaluate_multiclass, normalize_label_array, pred_from_proba

logger = logging.getLogger(__name__)


def complementarity(
    name_a: str,
    pa: np.ndarray,
    name_b: str,
    pb: np.ndarray,
    y: Sequence[Any],
) -> Dict[str, Any]:
    y_n = normalize_label_array(y)
    pred_a, pred_b = pred_from_proba(pa), pred_from_proba(pb)
    agree = float((pred_a == pred_b).mean())
    wrong_a = pred_a != y_n
    wrong_b = pred_b != y_n
    both_wrong = wrong_a & wrong_b
    overlap = float(both_wrong.sum() / wrong_a.sum()) if wrong_a.any() else 1.0
    a_wrong_b_right = int((wrong_a & ~wrong_b).sum())
    b_wrong_a_right = int((wrong_b & ~wrong_a).sum())
    disagree = pred_a != pred_b
    disagree_acc = float((pred_a[disagree] == y_n[disagree]).mean()) if disagree.any() else float("nan")
    # 平均三类概率相关
    corrs = []
    for k in range(3):
        if pa[:, k].std() > 1e-12 and pb[:, k].std() > 1e-12:
            corrs.append(float(np.corrcoef(pa[:, k], pb[:, k])[0, 1]))
    return {
        "model_a": name_a,
        "model_b": name_b,
        "prob_corr_mean": float(np.mean(corrs)) if corrs else float("nan"),
        "pred_agree_rate": agree,
        "error_overlap_rate": overlap,
        "a_wrong_b_right": a_wrong_b_right,
        "b_wrong_a_right": b_wrong_a_right,
        "disagree_acc_if_pick_a": disagree_acc,
    }


def diversity_table(models: Dict[str, np.ndarray], y: Sequence[Any]) -> pd.DataFrame:
    rows = []
    names = list(models.keys())
    for i, j in itertools.combinations(range(len(names)), 2):
        rows.append(complementarity(names[i], models[names[i]], names[j], models[names[j]], y))
    return pd.DataFrame(rows)


def _iter_simplex(m: int, step: float):
    """生成 m 维单纯形权重（非负且和为 1）。"""
    grid = np.round(np.arange(0.0, 1.0 + 1e-12, step), 10)
    if m == 1:
        yield (1.0,)
        return
    for weights in itertools.product(grid, repeat=m - 1):
        s = float(sum(weights))
        if s > 1.0 + 1e-8:
            continue
        last = round(1.0 - s, 10)
        if last < -1e-8:
            continue
        yield tuple(float(w) for w in weights) + (float(last),)


def soft_vote_search(
    models: Dict[str, np.ndarray],
    y: Sequence[Any],
    max_models: int = 4,
    step: float = 0.05,
) -> Tuple[Dict[str, Any], np.ndarray, pd.DataFrame]:
    """在最多 max_models 个模型上网格搜索权重。"""
    y_n = normalize_label_array(y)
    y_idx = np.array([{"Low": 0, "Medium": 1, "High": 2}[lab] for lab in y_n], dtype=int)
    # 先按单模 Acc 排序取前 max_models
    ranked = sorted(
        models.items(),
        key=lambda kv: float((pred_from_proba(kv[1]) == y_n).mean()),
        reverse=True,
    )[:max_models]
    names = [n for n, _ in ranked]
    probs = [p for _, p in ranked]
    m = len(names)
    rows = []
    best = None
    best_p = None

    for weights in _iter_simplex(m, step):
        if all(w == 0 for w in weights):
            continue
        p = np.zeros_like(probs[0])
        for w, pr in zip(weights, probs):
            if w:
                p = p + w * pr
        pred_idx = np.argmax(p, axis=1)
        acc = float((pred_idx == y_idx).mean())
        row = {f"w_{names[i]}": float(weights[i]) for i in range(m)}
        row["accuracy"] = acc
        rows.append(row)
        n_active = int(sum(w > 1e-12 for w in weights))
        if best is None or acc > best["accuracy"] + 1e-12 or (
            abs(acc - best["accuracy"]) <= 1e-12 and n_active < best.get("_n_active", 99)
        ):
            best = {
                "accuracy": acc,
                "weights": {names[i]: float(weights[i]) for i in range(m)},
                "_n_active": n_active,
                "_p": p,
            }
    assert best is not None
    best_p = best.pop("_p")
    best.pop("_n_active", None)
    # 对最优权重补全完整指标
    met = evaluate_multiclass(y_n, pred_from_proba(best_p), best_p)
    best.update(met)
    return best, best_p, pd.DataFrame(rows)


def _better(m: Dict[str, Any], best: Dict[str, Any], n_active: int, best_active: int) -> bool:
    if m["accuracy"] > best["accuracy"] + 1e-12:
        return True
    if abs(m["accuracy"] - best["accuracy"]) > 1e-12:
        return False
    for key in ("macro_f1", "qwk", "medium_recall"):
        if m[key] > best[key] + 1e-12:
            return True
        if abs(m[key] - best[key]) > 1e-12:
            return False
    if m["cross_two_level_errors"] < best["cross_two_level_errors"]:
        return True
    if m["cross_two_level_errors"] > best["cross_two_level_errors"]:
        return False
    return n_active < best_active


def _meta_features(prob_list: Sequence[np.ndarray]) -> np.ndarray:
    parts = list(prob_list)
    stack = np.concatenate(parts, axis=1)
    # 平均概率上的 max / margin / entropy
    avg = np.mean(prob_list, axis=0)
    sorted_p = np.sort(avg, axis=1)
    max_p = sorted_p[:, -1:]
    margin = (sorted_p[:, -1] - sorted_p[:, -2]).reshape(-1, 1)
    ent = (-np.sum(avg * np.log(np.clip(avg, 1e-15, 1.0)), axis=1)).reshape(-1, 1)
    return np.concatenate([stack, max_p, margin, ent], axis=1)


def stacking_nested_oof(
    models: Dict[str, np.ndarray],
    y: Sequence[Any],
    n_folds: int = TASK1_N_FOLDS,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y_n = normalize_label_array(y)
    y_code = np.array([TASK1_LABEL_ORDER.index(lab) for lab in y_n], dtype=int)
    names = list(models.keys())
    prob_list = [models[n] for n in names]
    X_meta_full = _meta_features(prob_list)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_p = np.zeros((len(y_n), 3), dtype=float)
    for tr, va in skf.split(X_meta_full, y_code):
        meta = LogisticRegression(max_iter=1000, random_state=seed)
        meta.fit(X_meta_full[tr], y_code[tr])
        proba = meta.predict_proba(X_meta_full[va])
        out = np.zeros((len(va), 3), dtype=float)
        for i, c in enumerate(meta.classes_):
            out[:, int(c)] = proba[:, i]
        oof_p[va] = out
    pred = pred_from_proba(oof_p)
    met = evaluate_multiclass(y_n, pred, oof_p)
    met["base_models"] = names
    return oof_p, met


def search_class_bias(
    proba: np.ndarray,
    y: Sequence[Any],
    step: float = 0.025,
    lim: float = 0.15,
) -> Tuple[Dict[str, Any], np.ndarray]:
    y_n = normalize_label_array(y)
    grid = np.round(np.arange(-lim, lim + 1e-12, step), 10)
    best = None
    best_p = None
    for b0 in grid:
        for b1 in grid:
            b2 = -b0 - b1
            if abs(b2) > lim + 1e-9:
                continue
            bias = np.array([b0, b1, b2], dtype=float)
            logits = np.log(np.clip(proba, 1e-15, 1.0)) + bias.reshape(1, -1)
            logits = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p = p / p.sum(axis=1, keepdims=True)
            pred = pred_from_proba(p)
            m = evaluate_multiclass(y_n, pred, p)
            row = {"bias_Low": float(b0), "bias_Medium": float(b1), "bias_High": float(b2), **m}
            if best is None or _better(m, best, 3, 3):
                best, best_p = row, p
    assert best is not None and best_p is not None
    return best, best_p


def apply_bias(proba: np.ndarray, bias: Sequence[float]) -> np.ndarray:
    b = np.asarray(bias, dtype=float).reshape(1, -1)
    logits = np.log(np.clip(proba, 1e-15, 1.0)) + b
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)
