"""任务二 ACC 冲高实验：跨任务 OOF / 焦虑消融 / 细阈值 / 缺失指示 / stacking / 多 seed 门控。

不自动改正式 lock；仅当稳定 lift 满足门控时写出 relock 建议并可选更新 final。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_utils import read_csv_compatible
from src.common.io_utils import atomic_write_csv, atomic_write_json
from src.common.metrics import majority_baseline
from src.config import (
    RANDOM_SEED,
    TASK1_TARGET,
    TASK2_ACC_BOOST_MIN_LIFT,
    TASK2_ALLOW_ANXIETY_ABLATION,
    TASK2_BOOTSTRAP_N,
    TASK2_N_FOLDS,
    TASK2_STABILITY_SEEDS,
    TASK2_THRESHOLD_MAX,
    TASK2_THRESHOLD_MIN,
    TASK2_TIE_EPS,
    TASK2_UNCERTAIN_FEATURES,
    TASK3_SOURCE_SCORE,
    TASK3_THRESHOLD,
)
from src.tasks.task2.cross_task_aux import (
    AUX_FEATURE_NAMES,
    attach_aux_to_frame,
    generate_cross_task_oof_features,
)
from src.tasks.task2.features import build_target, resolve_safe_features
from src.tasks.task2.metrics_ext import rich_binary_metrics
from src.tasks.task2.preprocessing import fit_tabular, transform_tabular, fit_task2_preprocessor, transform_task2_data, get_cat_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("task2_acc_boost")


def _fold_assign_from_file(df_dev: pd.DataFrame, fold_path: Path) -> np.ndarray:
    fold_df = pd.read_csv(fold_path)
    m = dict(zip(fold_df["sample_index"].astype(int), fold_df["fold"].astype(int)))
    return np.array([m[int(i)] for i in df_dev.index], dtype=int)


def _make_missing_indicators(df: pd.DataFrame, features: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    added: List[str] = []
    for c in features:
        if c not in out.columns:
            continue
        name = f"{c}__is_missing"
        if out[c].isna().any():
            out[name] = out[c].isna().astype(float)
            added.append(name)
    return out, added


def _oof_logistic(
    df: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    fold_assign: np.ndarray,
    n_folds: int,
) -> np.ndarray:
    y_arr = y.to_numpy().astype(int)
    p1 = np.zeros(len(df), dtype=float)
    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        state = fit_tabular(df.loc[tr], features, scale=True)
        X_tr = transform_tabular(df.loc[tr], state)
        X_va = transform_tabular(df.loc[va], state)
        if X_tr.shape[1] == 0 or len(np.unique(y_arr[tr])) < 2:
            p1[va] = float(y_arr[tr].mean()) if tr.sum() else 0.5
            continue
        lr = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        lr.fit(X_tr, y_arr[tr])
        p1[va] = lr.predict_proba(X_va)[:, 1]
    return p1


def _oof_catboost(
    df: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    fold_assign: np.ndarray,
    n_folds: int,
    iterations: int = 400,
) -> np.ndarray:
    y_arr = y.to_numpy().astype(int)
    p1 = np.zeros(len(df), dtype=float)
    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        state = fit_task2_preprocessor(df.loc[tr], features)
        X_tr = transform_task2_data(df.loc[tr], state)
        X_va = transform_task2_data(df.loc[va], state)
        cat_idx = get_cat_indices(state)
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Accuracy",
            iterations=iterations,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        if len(np.unique(y_arr[tr])) < 2:
            p1[va] = float(y_arr[tr].mean())
            continue
        model.fit(X_tr, y_arr[tr], cat_features=cat_idx or None)
        pr = model.predict_proba(X_va)
        classes = [int(c) for c in model.classes_]
        p1[va] = pr[:, classes.index(1)] if 1 in classes else pr[:, -1]
    return p1


def _metrics_row(name: str, y: np.ndarray, p1: np.ndarray, majority_acc: float, thr: float = 0.5) -> Dict[str, Any]:
    pred = (p1 >= thr).astype(int)
    m = rich_binary_metrics(y, pred, p1)
    fold_beats = None
    return {
        "candidate": name,
        "threshold": thr,
        "accuracy": m["accuracy"],
        "roc_auc": m["roc_auc"],
        "balanced_accuracy": m["balanced_accuracy"],
        "pred_0_rate": float((pred == 0).mean()),
        "lift_vs_majority": float(m["accuracy"] - majority_acc),
        **{k: m.get(k) for k in ("brier", "log_loss", "pr_auc")},
    }


def _fine_threshold(y: np.ndarray, p1: np.ndarray, majority_acc: float) -> Tuple[float, Dict[str, Any]]:
    best_thr = 0.5
    best = None
    rows = []
    for thr in np.round(np.arange(TASK2_THRESHOLD_MIN, TASK2_THRESHOLD_MAX + 1e-12, 0.001), 10):
        row = _metrics_row("tmp", y, p1, majority_acc, thr=float(thr))
        rows.append(row)
        if best is None or row["accuracy"] > best["accuracy"] + 1e-15:
            best = row
            best_thr = float(thr)
        elif abs(row["accuracy"] - best["accuracy"]) < 1e-15 and abs(thr - 0.5) < abs(best_thr - 0.5):
            best = row
            best_thr = float(thr)
    assert best is not None
    best["threshold"] = best_thr
    return best_thr, best


def _fold_beats_majority(y: np.ndarray, p1: np.ndarray, fold_assign: np.ndarray, thr: float = 0.5) -> int:
    n_ok = 0
    for f in np.unique(fold_assign):
        m = fold_assign == f
        yy = y[m]
        maj = int(pd.Series(yy).value_counts().index[0])
        maj_acc = float((yy == maj).mean())
        pred = (p1[m] >= thr).astype(int)
        acc = float((pred == yy).mean())
        if acc > maj_acc + 1e-12:
            n_ok += 1
    return n_ok


def _nested_stacking_oof(
    base_probs: Dict[str, np.ndarray],
    y: np.ndarray,
    fold_assign: np.ndarray,
    n_folds: int,
) -> np.ndarray:
    """用各折外基模型概率训练元 Logistic（二层 OOF）。"""
    names = list(base_probs.keys())[:4]
    X = np.column_stack([base_probs[n] for n in names])
    p_meta = np.zeros(len(y), dtype=float)
    for fold_id in range(n_folds):
        va = fold_assign == fold_id
        tr = ~va
        if len(np.unique(y[tr])) < 2:
            p_meta[va] = float(y[tr].mean())
            continue
        lr = LogisticRegression(max_iter=400, C=1.0, random_state=42)
        lr.fit(X[tr], y[tr])
        p_meta[va] = lr.predict_proba(X[va])[:, 1]
    return p_meta


def _multiseed_eval(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    features: Sequence[str],
    builder: str,
    seeds: Sequence[int],
    n_folds: int,
) -> pd.DataFrame:
    rows = []
    y_arr = y_dev.to_numpy().astype(int)
    for sd in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(sd))
        fold_assign = np.full(len(df_dev), -1, dtype=int)
        for k, (_, va) in enumerate(skf.split(df_dev, y_dev)):
            fold_assign[va] = k
        if builder == "logistic":
            p1 = _oof_logistic(df_dev, y_dev, features, fold_assign, n_folds)
        else:
            p1 = _oof_catboost(df_dev, y_dev, features, fold_assign, n_folds, iterations=300)
        # per-seed majority from full develop prior for reporting
        maj = majority_baseline(y_arr)
        maj_acc = float(maj["majority_baseline_accuracy"])
        thr, best = _fine_threshold(y_arr, p1, maj_acc)
        # fold-wise majority
        fold_accs = []
        fold_lifts = []
        for f in range(n_folds):
            m = fold_assign == f
            yy = y_arr[m]
            local_maj = int(pd.Series(yy).value_counts().index[0])
            local_maj_acc = float((yy == local_maj).mean())
            pred = (p1[m] >= thr).astype(int)
            acc = float((pred == yy).mean())
            fold_accs.append(acc)
            fold_lifts.append(acc - local_maj_acc)
        rows.append(
            {
                "seed": int(sd),
                "builder": builder,
                "threshold": thr,
                "mean_acc": float(np.mean(fold_accs)),
                "std_acc": float(np.std(fold_accs)),
                "mean_lift": float(np.mean(fold_lifts)),
                "n_folds_beat_majority": int(sum(1 for x in fold_lifts if x > 0)),
                "oof_accuracy": best["accuracy"],
                "oof_auc": best["roc_auc"],
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_delta(y: np.ndarray, pred: np.ndarray, majority_class: int, n_boot: int = 1000, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n = len(y)
    maj_pred = np.full(n, majority_class, dtype=int)
    lifts = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        acc_m = float((pred[idx] == y[idx]).mean())
        acc_j = float((maj_pred[idx] == y[idx]).mean())
        lifts.append(acc_m - acc_j)
    a = np.asarray(lifts)
    return {
        "n_boot": n_boot,
        "delta_acc_mean": float(a.mean()),
        "delta_acc_ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
        "p_delta_acc_gt_0": float((a > 0).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataSetB.csv")
    ap.add_argument("--output-dir", default="outputs/task2")
    ap.add_argument("--n-folds", type=int, default=TASK2_N_FOLDS)
    ap.add_argument("--apply-relock", action="store_true", help="门控通过时写回 final lock")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    boost_dir = output_dir / "diagnostics" / "acc_boost"
    boost_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_compatible(Path(args.data))
    dev_idx = pd.read_csv(output_dir / "splits" / "develop_indices.csv")["sample_index"].astype(int)
    df_dev = df.loc[dev_idx].copy()
    y_dev = build_target(df_dev)
    fold_assign = _fold_assign_from_file(df_dev, output_dir / "splits" / "fold_assignment_seed42.csv")
    n_folds = args.n_folds

    maj = majority_baseline(y_dev)
    majority_class = int(maj["majority_class"])
    majority_acc = float(maj["majority_baseline_accuracy"])
    y_arr = y_dev.to_numpy().astype(int)

    feat_b = resolve_safe_features(df_dev)
    atomic_write_json(
        boost_dir / "uncertain_features.json",
        {
            "uncertain": TASK2_UNCERTAIN_FEATURES,
            "note": "range_anxiety_score is questionnaire field; ablation only until gate passes",
            "allow_anxiety_ablation": TASK2_ALLOW_ANXIETY_ABLATION,
        },
    )

    # ---- cross-task OOF ----
    aux_path = boost_dir / "cross_task_oof_features.csv"
    if aux_path.exists():
        logger.info("loading cached cross-task OOF aux features...")
        aux = pd.read_csv(aux_path)
        if "sample_index" in aux.columns:
            aux = aux.set_index("sample_index")
            aux.index = df_dev.index  # align to develop order if same length
            # prefer reindex by sample_index values
            aux = pd.read_csv(aux_path)
            aux = aux.set_index("sample_index").reindex(df_dev.index.astype(int))
            aux.index = df_dev.index
    else:
        logger.info("generating cross-task OOF aux features...")
        aux = generate_cross_task_oof_features(df_dev, fold_assign, boost_dir, n_folds=n_folds)
    df_aux = attach_aux_to_frame(df_dev, aux)
    feat_aux = feat_b + [c for c in AUX_FEATURE_NAMES if c in df_aux.columns]

    # missingness
    df_miss, miss_cols = _make_missing_indicators(df_dev, feat_b)
    feat_miss = feat_b + miss_cols

    # anxiety ablation features
    feat_anx = list(feat_b)
    if TASK2_ALLOW_ANXIETY_ABLATION and TASK3_SOURCE_SCORE in df_dev.columns:
        feat_anx = feat_b + [TASK3_SOURCE_SCORE]
    feat_anx_aux = list(dict.fromkeys(feat_anx + [c for c in AUX_FEATURE_NAMES if c in df_aux.columns]))

    configs: List[Tuple[str, pd.DataFrame, List[str], str]] = [
        ("B_safe_logistic", df_dev, feat_b, "logistic"),
        ("B_safe_catboost", df_dev, feat_b, "catboost"),
        ("B_safe_missing_logistic", df_miss, feat_miss, "logistic"),
        ("B_safe_aux_logistic", df_aux, feat_aux, "logistic"),
        ("B_safe_aux_catboost", df_aux, feat_aux, "catboost"),
    ]
    if TASK2_ALLOW_ANXIETY_ABLATION and TASK3_SOURCE_SCORE in df_dev.columns:
        configs.extend(
            [
                ("B_safe_anxiety_logistic", df_dev, feat_anx, "logistic"),
                ("B_safe_anxiety_catboost", df_dev, feat_anx, "catboost"),
                ("B_safe_anxiety_aux_logistic", df_aux, feat_anx_aux, "logistic"),
                ("B_safe_anxiety_aux_catboost", df_aux, feat_anx_aux, "catboost"),
            ]
        )

    comparison_rows: List[Dict[str, Any]] = []
    oof_store: Dict[str, np.ndarray] = {}
    thr_store: Dict[str, float] = {}

    # majority baseline row
    comparison_rows.append(
        {
            "candidate": "M_majority",
            "threshold": 0.5,
            "accuracy": majority_acc,
            "roc_auc": 0.5,
            "balanced_accuracy": 0.5,
            "pred_0_rate": 0.0 if majority_class == 1 else 1.0,
            "lift_vs_majority": 0.0,
            "folds_beat_majority": 0,
        }
    )

    for name, dframe, feats, builder in configs:
        logger.info("OOF %s builder=%s n_feat=%d", name, builder, len(feats))
        if builder == "logistic":
            p1 = _oof_logistic(dframe, y_dev, feats, fold_assign, n_folds)
        else:
            p1 = _oof_catboost(dframe, y_dev, feats, fold_assign, n_folds)
        oof_store[name] = p1
        # default 0.5
        row05 = _metrics_row(name + "@0.5", y_arr, p1, majority_acc, 0.5)
        row05["folds_beat_majority"] = _fold_beats_majority(y_arr, p1, fold_assign, 0.5)
        comparison_rows.append(row05)
        # fine threshold
        thr, best = _fine_threshold(y_arr, p1, majority_acc)
        thr_store[name] = thr
        best["candidate"] = name + f"@thr{thr:.3f}"
        best["folds_beat_majority"] = _fold_beats_majority(y_arr, p1, fold_assign, thr)
        comparison_rows.append(best)
        atomic_write_csv(
            boost_dir / f"oof_{name}.csv",
            pd.DataFrame(
                {
                    "sample_index": df_dev.index.astype(int),
                    "true_label": y_arr,
                    "prob_1": p1,
                    "predicted_label": (p1 >= thr).astype(int),
                    "fold": fold_assign,
                }
            ),
        )

    # anxiety ablation summary table
    abl_rows = [r for r in comparison_rows if "anxiety" in r["candidate"] or r["candidate"].startswith("B_safe_logistic") or r["candidate"].startswith("B_safe_aux")]
    atomic_write_csv(boost_dir / "anxiety_ablation.csv", pd.DataFrame(abl_rows))

    # stacking from strongest base probs at thr=0.5 versions
    base_for_stack = {}
    for key in ("B_safe_logistic", "B_safe_catboost", "B_safe_aux_logistic", "B_safe_aux_catboost"):
        if key in oof_store:
            base_for_stack[key] = oof_store[key]
    if len(base_for_stack) >= 2:
        logger.info("nested stacking on %s", list(base_for_stack))
        p_stack = _nested_stacking_oof(base_for_stack, y_arr, fold_assign, n_folds)
        oof_store["stack_meta"] = p_stack
        row05 = _metrics_row("stack_meta@0.5", y_arr, p_stack, majority_acc, 0.5)
        row05["folds_beat_majority"] = _fold_beats_majority(y_arr, p_stack, fold_assign, 0.5)
        comparison_rows.append(row05)
        thr, best = _fine_threshold(y_arr, p_stack, majority_acc)
        thr_store["stack_meta"] = thr
        best["candidate"] = f"stack_meta@thr{thr:.3f}"
        best["folds_beat_majority"] = _fold_beats_majority(y_arr, p_stack, fold_assign, thr)
        comparison_rows.append(best)

    comp = pd.DataFrame(comparison_rows).sort_values(
        ["lift_vs_majority", "accuracy", "roc_auc"], ascending=False
    )
    atomic_write_csv(boost_dir / "acc_boost_comparison.csv", comp)

    # pick best non-majority by lift then acc
    non_m = comp[~comp["candidate"].astype(str).str.startswith("M_")].copy()
    best_row = non_m.iloc[0].to_dict() if len(non_m) else None
    gate = {
        "majority_accuracy": majority_acc,
        "majority_class": majority_class,
        "best_candidate": None if best_row is None else best_row["candidate"],
        "best_accuracy": None if best_row is None else float(best_row["accuracy"]),
        "best_lift": None if best_row is None else float(best_row["lift_vs_majority"]),
        "min_lift_required": TASK2_ACC_BOOST_MIN_LIFT,
        "passed_oof_lift": False,
        "passed_multiseed": False,
        "passed_bootstrap": False,
        "relock_recommended": False,
    }

    if best_row is not None and float(best_row["lift_vs_majority"]) >= TASK2_ACC_BOOST_MIN_LIFT:
        gate["passed_oof_lift"] = True
        # recover base name and features for multiseed
        cand = str(best_row["candidate"])
        base_name = cand.split("@")[0]
        # map to feature set
        feat_map = {
            "B_safe_logistic": (df_dev, feat_b, "logistic"),
            "B_safe_catboost": (df_dev, feat_b, "catboost"),
            "B_safe_missing_logistic": (df_miss, feat_miss, "logistic"),
            "B_safe_aux_logistic": (df_aux, feat_aux, "logistic"),
            "B_safe_aux_catboost": (df_aux, feat_aux, "catboost"),
            "B_safe_anxiety_logistic": (df_dev, feat_anx, "logistic"),
            "B_safe_anxiety_catboost": (df_dev, feat_anx, "catboost"),
            "B_safe_anxiety_aux_logistic": (df_aux, feat_anx_aux, "logistic"),
            "B_safe_anxiety_aux_catboost": (df_aux, feat_anx_aux, "catboost"),
            "stack_meta": (df_aux, feat_aux, "logistic"),  # proxy
        }
        if base_name in feat_map and base_name != "stack_meta":
            dframe, feats, builder = feat_map[base_name]
            logger.info("multiseed gate for %s", base_name)
            ms = _multiseed_eval(dframe, y_dev, feats, builder, TASK2_STABILITY_SEEDS, n_folds)
            atomic_write_csv(boost_dir / "multiseed_gate.csv", ms)
            mean_lift = float(ms["mean_lift"].mean())
            seeds_ok = int((ms["mean_lift"] >= 0).sum())
            gate["multiseed_mean_lift"] = mean_lift
            gate["multiseed_seeds_nonneg"] = seeds_ok
            gate["passed_multiseed"] = bool(mean_lift > 0 and seeds_ok >= 2)

            # bootstrap on seed42 OOF of best
            p1 = oof_store.get(base_name)
            thr = thr_store.get(base_name, 0.5)
            if p1 is not None:
                pred = (p1 >= thr).astype(int)
                boot = _bootstrap_delta(y_arr, pred, majority_class, n_boot=TASK2_BOOTSTRAP_N)
                atomic_write_json(boost_dir / "bootstrap_gate.json", boot)
                gate["bootstrap"] = boot
                ci_lo = boot["delta_acc_ci95"][0]
                gate["passed_bootstrap"] = bool(ci_lo > -0.002)  # not clearly negative
        else:
            gate["multiseed_note"] = f"skip multiseed for {base_name}"

    gate["relock_recommended"] = bool(
        gate["passed_oof_lift"] and gate.get("passed_multiseed") and gate.get("passed_bootstrap")
    )
    atomic_write_json(boost_dir / "gate_summary.json", gate)

    # report
    report = [
        "# Task2 ACC Boost Report",
        "",
        f"- Majority Acc: `{majority_acc}`",
        f"- Best candidate: `{gate.get('best_candidate')}`",
        f"- Best Acc / lift: `{gate.get('best_accuracy')}` / `{gate.get('best_lift')}`",
        f"- OOF lift gate: `{gate.get('passed_oof_lift')}`",
        f"- Multiseed gate: `{gate.get('passed_multiseed')}`",
        f"- Bootstrap gate: `{gate.get('passed_bootstrap')}`",
        f"- Relock recommended: `{gate.get('relock_recommended')}`",
        "",
        "See `acc_boost_comparison.csv`, `anxiety_ablation.csv`, `cross_task_oof_features.csv`.",
        "",
    ]
    (boost_dir / "acc_boost_report.md").write_text("\n".join(report), encoding="utf-8")

    # optional relock
    if gate["relock_recommended"] and args.apply_relock and best_row is not None:
        base_name = str(best_row["candidate"]).split("@")[0]
        thr = float(thr_store.get(base_name, best_row.get("threshold", 0.5)))
        lock = {
            "locked_scheme": base_name,
            "lock_reason": "acc_boost gate passed (OOF lift + multiseed + bootstrap)",
            "majority_baseline_accuracy": majority_acc,
            "best_model_accuracy": float(best_row["accuracy"]),
            "is_weak_signal_lock": False,
            "oof_accuracy": float(best_row["accuracy"]),
            "oof_roc_auc": float(best_row["roc_auc"]),
            "threshold": thr,
            "from": "acc_boost",
        }
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(final_dir / "task2_final_config.json", lock)
        # final validation
        val_idx = pd.read_csv(output_dir / "splits" / "final_validation_indices.csv")["sample_index"].astype(int)
        df_val = df.loc[val_idx].copy()
        y_val = build_target(df_val).to_numpy().astype(int)
        feat_map = {
            "B_safe_logistic": (feat_b, "logistic", False, False),
            "B_safe_catboost": (feat_b, "catboost", False, False),
            "B_safe_missing_logistic": (feat_miss, "logistic", True, False),
            "B_safe_aux_logistic": (feat_aux, "logistic", False, True),
            "B_safe_aux_catboost": (feat_aux, "catboost", False, True),
            "B_safe_anxiety_logistic": (feat_anx, "logistic", False, False),
            "B_safe_anxiety_catboost": (feat_anx, "catboost", False, False),
            "B_safe_anxiety_aux_logistic": (feat_anx_aux, "logistic", False, True),
            "B_safe_anxiety_aux_catboost": (feat_anx_aux, "catboost", False, True),
        }
        if base_name in feat_map:
            feats, builder, use_miss, use_aux = feat_map[base_name]
            # rebuild aux on full develop for test-time: train aux on all develop then transform val — for fair protocol
            # Here: refit task2 model on develop with chosen features; for aux cols, use develop-trained full aux model approx:
            # simplest: concatenate develop+val indices for aux would leak; instead train aux on develop only and predict val.
            df_tr = df_dev.copy()
            df_te = df_val.copy()
            if use_aux:
                # train aux on full develop (no task2 label), predict val
                from src.tasks.task2.cross_task_aux import _aux_base_features, _prepare_xy
                from catboost import CatBoostClassifier as CBC, CatBoostRegressor as CBR

                feats_aux_in = _aux_base_features(df_tr)
                Xtr, cat_idx = _prepare_xy(df_tr, feats_aux_in)
                Xte, _ = _prepare_xy(df_te, feats_aux_in)
                y1 = df_tr[TASK1_TARGET].astype(str).str.strip().replace({"low": "Low", "medium": "Medium", "high": "High"})
                clf = CBC(loss_function="MultiClass", iterations=250, depth=6, learning_rate=0.06, random_seed=42, verbose=False, allow_writing_files=False)
                clf.fit(Xtr, y1, cat_features=cat_idx or None)
                pr = clf.predict_proba(Xte)
                classes = [str(c) for c in clf.classes_]
                cmap = {c: i for i, c in enumerate(classes)}
                pL = pr[:, cmap.get("Low", 0)]
                pM = pr[:, cmap.get("Medium", 1 if pr.shape[1] > 1 else 0)]
                pH = pr[:, cmap.get("High", -1)]
                score = pd.to_numeric(df_tr[TASK3_SOURCE_SCORE], errors="coerce")
                feat_t3 = [c for c in feats_aux_in if c != TASK3_SOURCE_SCORE]
                X3tr, cat3 = _prepare_xy(df_tr, feat_t3)
                X3te, _ = _prepare_xy(df_te, feat_t3)
                reg = CBR(loss_function="RMSE", iterations=250, depth=6, learning_rate=0.06, random_seed=42, verbose=False, allow_writing_files=False)
                reg.fit(X3tr, score.fillna(score.median()), cat_features=cat3 or None)
                pred_s = reg.predict(X3te)
                y3 = (score > TASK3_THRESHOLD).astype(int)
                clf3 = CBC(loss_function="Logloss", iterations=250, depth=6, learning_rate=0.06, random_seed=42, verbose=False, allow_writing_files=False)
                clf3.fit(X3tr, y3, cat_features=cat3 or None)
                pr3 = clf3.predict_proba(X3te)
                c3 = [int(c) for c in clf3.classes_]
                pA = pr3[:, c3.index(1)] if 1 in c3 else pr3[:, -1]
                for col, arr in [
                    ("aux_t1_p_Low", pL),
                    ("aux_t1_p_Medium", pM),
                    ("aux_t1_p_High", pH),
                    ("aux_t1_expected_level", pM * 1 + pH * 2),
                    ("aux_t1_entropy", -np.sum(np.clip(np.column_stack([pL, pM, pH]), 1e-15, 1) * np.log(np.clip(np.column_stack([pL, pM, pH]), 1e-15, 1)), axis=1)),
                    ("aux_t3_p_high_anxiety", pA),
                    ("aux_t3_pred_score", pred_s),
                ]:
                    df_te[col] = arr
                # also need aux on train for fitting task2 model — use OOF aux already in df_aux
                df_tr = df_aux.copy()
            if use_miss:
                df_tr, _ = _make_missing_indicators(df_dev, feat_b)
                df_te, _ = _make_missing_indicators(df_val, feat_b)

            if builder == "logistic":
                state = fit_tabular(df_tr, feats, scale=True)
                Xtr = transform_tabular(df_tr, state)
                Xte = transform_tabular(df_te, state)
                lr = LogisticRegression(max_iter=500, C=1.0, random_state=42)
                lr.fit(Xtr, y_arr)
                p_val = lr.predict_proba(Xte)[:, 1]
            else:
                state = fit_task2_preprocessor(df_tr, feats)
                Xtr = transform_task2_data(df_tr, state)
                Xte = transform_task2_data(df_te, state)
                model = CatBoostClassifier(
                    loss_function="Logloss",
                    iterations=400,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=5.0,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                )
                model.fit(Xtr, y_arr, cat_features=get_cat_indices(state) or None)
                pr = model.predict_proba(Xte)
                classes = [int(c) for c in model.classes_]
                p_val = pr[:, classes.index(1)] if 1 in classes else pr[:, -1]
            pred = (p_val >= thr).astype(int)
            fv = rich_binary_metrics(y_val, pred, p_val)
            atomic_write_json(output_dir / "final_validation" / "metrics.json", {**fv, "locked_scheme": base_name, "threshold": thr})
            atomic_write_json(output_dir / "metrics.json", {
                "task_name": "task2",
                "accuracy": fv["accuracy"],
                "selected_scheme": base_name,
                "oof_accuracy": float(best_row["accuracy"]),
                "majority_baseline_accuracy": majority_acc,
                "is_weak_signal_lock": False,
                "threshold": thr,
                "final_validation_accuracy": fv["accuracy"],
            })
            logger.info("RELOCKED %s final_acc=%.6f", base_name, fv["accuracy"])
    else:
        logger.info("KEEP majority M | gate=%s", gate)

    # sync slim copy to results
    results_boost = ROOT / "results" / "task2" / "diagnostics" / "acc_boost"
    results_boost.mkdir(parents=True, exist_ok=True)
    for fname in (
        "acc_boost_comparison.csv",
        "anxiety_ablation.csv",
        "gate_summary.json",
        "acc_boost_report.md",
        "uncertain_features.json",
        "cross_task_aux_meta.json",
        "bootstrap_gate.json",
        "multiseed_gate.csv",
    ):
        src = boost_dir / fname
        if src.exists():
            (results_boost / fname).write_bytes(src.read_bytes())

    print("ACC_BOOST_DONE", json.dumps(gate, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
