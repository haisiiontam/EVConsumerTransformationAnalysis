"""任务二弱信号诊断全流程（StageGate + 断点续跑）。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.common.io_utils import (
    RunState,
    atomic_write_csv,
    atomic_write_json,
    setup_file_logging,
)
from src.common.metrics import majority_baseline, save_json
from src.common.stage_control import RunOptions, StageGate, completed_from_audit
from src.config import (
    RANDOM_SEED,
    TASK2_BASE_FEATURES,
    TASK2_CATBOOST_COMMON,
    TASK2_EARLY_STOPPING_ROUNDS,
    TASK2_ENABLE_CLUSTERING,
    TASK2_ENABLE_CROSS_TASK_AUX,
    TASK2_ENABLE_TABNET,
    TASK2_NAME,
    TASK2_N_FOLDS,
    TASK2_PERMUTATION_N,
    TASK2_BOOTSTRAP_N,
    TASK2_STAGES,
    TASK2_STABILITY_SEEDS,
    TASK2_THRESHOLD_MAX,
    TASK2_THRESHOLD_MIN,
    TASK2_THRESHOLD_STEP,
    TASK2_TIE_EPS,
    TASK2_WEAK_SIGNAL_AUC_GAP,
)
from src.tasks.task2.audit import archive_legacy_if_needed, run_audit
from src.tasks.task2.baseline import run_baselines
from src.tasks.task2.clustering_prior import run_clustering_prior
from src.tasks.task2.diversity_blend import diversity_report, soft_blend_search
from src.tasks.task2.empirical_bayes import run_empirical_bayes
from src.tasks.task2.feature_ablation import run_feature_ablation
from src.tasks.task2.features import (
    build_target,
    check_task2_base_columns,
    feature_sets,
)
from src.tasks.task2.final_lock import (
    compare_candidates,
    lock_final_scheme,
    persist_final_lock,
)
from src.tasks.task2.label_audit import run_label_audit
from src.tasks.task2.metrics_ext import rich_binary_metrics
from src.tasks.task2.missingness import run_missingness_analysis
from src.tasks.task2.models_quick import run_quick_models
from src.tasks.task2.other_models import run_heterogeneous_models
from src.tasks.task2.preprocessing import (
    fit_task2_preprocessor,
    get_cat_indices,
    transform_task2_data,
)
from src.tasks.task2.report import write_paper_reports
from src.tasks.task2.separability_stage import run_separability_stage
from src.tasks.task2.statistics import run_bootstrap_delta, run_mcnemar, run_permutation_test

logger = logging.getLogger(__name__)


def _task2_skip(stage: str, opts: RunOptions) -> bool:
    if opts.skip_multiseed and stage == "multiseed":
        return True
    if opts.skip_shap and stage == "explain":
        return True
    if opts.audit_only and stage != "audit":
        return True
    # 默认关闭的重模块
    if stage == "m6_tabnet" and not TASK2_ENABLE_TABNET:
        return True
    if stage == "cross_task_aux" and not TASK2_ENABLE_CROSS_TASK_AUX:
        return True
    if stage == "clustering_prior" and not TASK2_ENABLE_CLUSTERING:
        return True
    quick_skip = {
        "clustering_prior",
        "m3_lightgbm",
        "m4_xgboost",
        "m5_extra_trees",
        "m6_tabnet",
        "cross_task_aux",
        "diversity",
        "soft_blend",
        "stacking",
        "calibration",
        "permutation_test",
        "multiseed",
        "bootstrap",
        "explain",
        "submission",
        "feature_ablation",
    }
    if opts.mode == "quick" and stage in quick_skip:
        logger.info("skip %s (quick mode)", stage)
        return True
    # full 论文路线：仍跳过堆叠/校准深挖/提交（可后续补）
    if opts.mode == "full" and stage in {"stacking", "calibration", "explain", "submission"}:
        logger.info("skip %s (full paper-core path)", stage)
        return True
    return False


_RESUME_SKIP = {
    "audit",
    "split",
    "baseline",
    "label_audit",
    "separability",
    "missingness",
    "empirical_bayes",
    "m1_logistic_gam",
    "m2_catboost",
    "threshold",
}


def _ensure_dirs(output_dir: Path) -> None:
    for sub in (
        "diagnostics",
        "baseline",
        "label_audit",
        "separability",
        "missingness",
        "empirical_bayes",
        "oof",
        "tuning",
        "checkpoints",
        "final",
        "final_validation",
        "paper",
        "splits",
        "logs",
        "resume_audit",
    ):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def _freeze_or_load_splits(
    df: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    test_size: float,
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray]:
    split_dir = output_dir / "splits"
    dev_p = split_dir / "develop_indices.csv"
    val_p = split_dir / "final_validation_indices.csv"
    fold_p = split_dir / "fold_assignment_seed42.csv"
    if dev_p.exists() and val_p.exists() and fold_p.exists():
        idx_dev = pd.read_csv(dev_p)["sample_index"].astype(int).tolist()
        idx_val = pd.read_csv(val_p)["sample_index"].astype(int).tolist()
        fold_df = pd.read_csv(fold_p)
        fold_map = dict(zip(fold_df["sample_index"].astype(int), fold_df["fold"].astype(int)))
        df_dev, df_val = df.loc[idx_dev].copy(), df.loc[idx_val].copy()
        y_dev, y_val = y.loc[idx_dev], y.loc[idx_val]
        fold_assign = np.array([fold_map[int(i)] for i in df_dev.index], dtype=int)
        logger.info("loaded frozen splits develop=%d final=%d", len(df_dev), len(df_val))
        return df_dev, df_val, y_dev, y_val, fold_assign

    idx_dev, idx_val = train_test_split(
        df.index.to_series(),
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    df_dev, df_val = df.loc[idx_dev].copy(), df.loc[idx_val].copy()
    y_dev, y_val = y.loc[idx_dev], y.loc[idx_val]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_assign = np.full(len(df_dev), -1, dtype=int)
    for k, (_, va) in enumerate(skf.split(df_dev, y_dev)):
        fold_assign[va] = k
    atomic_write_csv(dev_p, pd.DataFrame({"sample_index": df_dev.index.astype(int)}))
    atomic_write_csv(val_p, pd.DataFrame({"sample_index": df_val.index.astype(int)}))
    atomic_write_csv(
        fold_p,
        pd.DataFrame({"sample_index": df_dev.index.astype(int), "fold": fold_assign}),
    )
    atomic_write_json(
        split_dir / "split_config.json",
        {
            "test_size": test_size,
            "random_state": RANDOM_SEED,
            "n_folds": n_folds,
            "stratify": True,
            "n_develop": int(len(df_dev)),
            "n_final": int(len(df_val)),
        },
    )
    logger.info("froze splits develop=%d final=%d", len(df_dev), len(df_val))
    return df_dev, df_val, y_dev, y_val, fold_assign


def _search_threshold(p1: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for thr in np.round(
        np.arange(TASK2_THRESHOLD_MIN, TASK2_THRESHOLD_MAX + 1e-12, TASK2_THRESHOLD_STEP), 10
    ):
        pred = (p1 >= thr).astype(int)
        m = rich_binary_metrics(y, pred, p1)
        rows.append({"threshold": float(thr), **m})
    search = pd.DataFrame(rows)
    max_acc = search["accuracy"].max()
    cand = search[search["accuracy"] >= max_acc - TASK2_TIE_EPS].copy()
    cand["thr_dist"] = (cand["threshold"] - 0.5).abs()
    cand = cand.sort_values(
        by=["accuracy", "balanced_accuracy", "recall_0", "thr_dist"],
        ascending=[False, False, False, True],
    )
    best = cand.iloc[0]
    return search, {
        "threshold": float(best["threshold"]),
        "oof_accuracy": float(best["accuracy"]),
        "oof_roc_auc": float(best["roc_auc"]),
    }


def _run_final_validation(
    locked_scheme: str,
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    df_val: pd.DataFrame,
    y_val: pd.Series,
    feat_a: List[str],
    feat_b: List[str],
    majority_class: int,
    threshold: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """在最终 holdout 上评估一次。"""
    y_val_arr = y_val.to_numpy().astype(int)

    if locked_scheme == "M" or locked_scheme == "M_majority":
        p_val = np.full(len(y_val_arr), float(majority_class), dtype=float)
        pred = np.full(len(y_val_arr), majority_class, dtype=int)
    elif locked_scheme.startswith("catboost"):
        feats = feat_a if locked_scheme == "catboost_a" else feat_b
        state = fit_task2_preprocessor(df_dev, feats)
        X_tr = transform_task2_data(df_dev, state)
        X_va = transform_task2_data(df_val, state)
        params = dict(TASK2_CATBOOST_COMMON)
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_dev, cat_features=get_cat_indices(state))
        proba = model.predict_proba(X_va)
        classes = [int(c) for c in model.classes_]
        p_val = proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
        pred = (p_val >= threshold).astype(int)
    elif locked_scheme.startswith("logistic"):
        from src.tasks.task2.preprocessing import fit_tabular, transform_tabular
        from sklearn.linear_model import LogisticRegression

        cw = "balanced" if "balanced" in locked_scheme else None
        state = fit_tabular(df_dev, feat_b, scale=True)
        X_tr = transform_tabular(df_dev, state)
        X_va = transform_tabular(df_val, state)
        lr = LogisticRegression(penalty="l2", max_iter=500, class_weight=cw, random_state=42)
        lr.fit(X_tr, y_dev)
        p_val = lr.predict_proba(X_va)[:, 1]
        pred = (p_val >= threshold).astype(int)
    elif locked_scheme.startswith("eb") or locked_scheme.startswith("B0"):
        # EB / majority：用 develop 全局率或多数类
        if locked_scheme.startswith("B0") or locked_scheme == "M":
            p_val = np.full(len(y_val_arr), float(majority_class), dtype=float)
            pred = np.full(len(y_val_arr), majority_class, dtype=int)
        else:
            rate = float(y_dev.mean())
            p_val = np.full(len(y_val_arr), rate, dtype=float)
            pred = (p_val >= threshold).astype(int)
    else:
        # 默认多数类
        p_val = np.full(len(y_val_arr), float(majority_class), dtype=float)
        pred = np.full(len(y_val_arr), majority_class, dtype=int)

    m = rich_binary_metrics(y_val_arr, pred, p_val)
    out_dir = output_dir / "final_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "metrics.json", {**m, "locked_scheme": locked_scheme, "threshold": threshold})
    atomic_write_csv(
        out_dir / "predictions.csv",
        pd.DataFrame(
            {
                "sample_index": df_val.index.astype(int),
                "true_label": y_val_arr,
                "predicted_label": pred,
                "prob_1": p_val,
            }
        ),
    )
    return m


def run(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.2,
    mode: str = "quick",
    resume: bool = False,
    start_stage: Optional[str] = None,
    stop_after_stage: Optional[str] = None,
    run_id: str = "task2",
    skip_optuna: bool = False,
    skip_multiseed: bool = False,
    skip_shap: bool = False,
    audit_only: bool = False,
    test_df: Optional[pd.DataFrame] = None,
    n_folds: int = TASK2_N_FOLDS,
    alpha_step: float = 0.05,
    threshold_min: float = TASK2_THRESHOLD_MIN,
    threshold_max: float = TASK2_THRESHOLD_MAX,
    threshold_step: float = TASK2_THRESHOLD_STEP,
    **kwargs: Any,
) -> Dict[str, Any]:
    """任务二弱信号诊断完整流水线。"""
    del kwargs, skip_optuna, alpha_step, threshold_min, threshold_max, threshold_step  # 保留 API 兼容
    output_dir = Path(output_dir)
    _ensure_dirs(output_dir)
    run_id = run_id or f"task2_{mode}_{int(time.time())}"
    setup_file_logging(output_dir / "logs" / f"task2_{run_id}.log", run_id, logger_name="task2")
    state = RunState(output_dir / "run_state.json", run_id)

    archive_legacy_if_needed(output_dir)
    audit_summary = run_audit(output_dir)
    if audit_only:
        state.data["exit_status"] = "audit_only"
        state.save()
        return {"task_name": TASK2_NAME, "accuracy": float("nan"), "audit": audit_summary}

    opts = RunOptions(
        mode=mode,
        resume=resume,
        start_stage=start_stage,
        stop_after_stage=stop_after_stage,
        run_id=run_id,
        skip_multiseed=skip_multiseed,
        skip_shap=skip_shap,
        audit_only=False,
        completed_stages=completed_from_audit(audit_summary) if resume else set(),
    )
    if resume and start_stage is None:
        opts.start_stage = audit_summary.get("first_resume_stage") or "split"
        logger.info("resume auto start_stage=%s", opts.start_stage)

    gate = StageGate(
        opts,
        state,
        TASK2_STAGES,
        skip_checker=_task2_skip,
        resume_skip_stages=_RESUME_SKIP,
    )

    check_task2_base_columns(df)
    y_all = build_target(df)
    maj = majority_baseline(y_all)
    majority_class = int(maj["majority_class"])
    majority_acc = float(maj["majority_baseline_accuracy"])
    save_json(maj, output_dir / "diagnostics" / "majority_baseline.json")

    sets = feature_sets(df)
    feat_a = sets["A_official"]
    feat_b = sets["B_safe_raw"]
    feat_mech = sets.get("B+all_mechanism", feat_b)

    # ---- audit stage ----
    if gate.should_execute("audit"):
        gate.begin("audit")
        gate.complete("audit")

    # ---- split ----
    if gate.should_execute("split"):
        gate.begin("split")
    df_dev, df_val, y_dev, y_val, fold_assign = _freeze_or_load_splits(
        df, y_all, output_dir, test_size, n_folds
    )
    if gate.should_execute("split"):
        gate.complete("split")
    elif state.status_of("split") != "completed":
        state.complete("split")

    candidate_oofs: Dict[str, pd.DataFrame] = {}
    label_summary: Dict[str, Any] = {}
    sep_summary: Dict[str, Any] = {}
    eb_result: Dict[str, Any] = {}
    lock_cfg: Dict[str, Any] = {}
    perm_summary: Dict[str, Any] = {}
    boot_summary: Dict[str, Any] = {}
    best_thr = 0.5
    locked_scheme = "M"
    oof_acc = majority_acc
    final_acc = float("nan")

    # ---- baseline ----
    if gate.should_execute("baseline"):
        gate.begin("baseline")
        base = run_baselines(df_dev, y_dev, fold_assign, output_dir, n_folds=n_folds)
        candidate_oofs["B0_majority"] = base["B0"]
        candidate_oofs["B1_constant_prior"] = base["B1"]
        candidate_oofs["B2_catboost_official"] = base["B2"]
        gate.complete("baseline")
    else:
        for key, fname in [
            ("B0_majority", "B0_majority_oof_predictions.csv"),
            ("B2_catboost_official", "B2_catboost_official_oof_predictions.csv"),
        ]:
            p = output_dir / "baseline" / fname
            if p.exists():
                candidate_oofs[key] = pd.read_csv(p)

    # ---- label_audit ----
    if gate.should_execute("label_audit"):
        gate.begin("label_audit")
        label_summary = run_label_audit(df_dev, y_dev, output_dir)
        gate.complete("label_audit")

    # ---- separability ----
    if gate.should_execute("separability"):
        gate.begin("separability")
        sep_out = run_separability_stage(df_dev, y_dev, output_dir, feat_mech)
        sep_summary = sep_out.get("summary", {})
        gate.complete("separability")

    # ---- missingness ----
    if gate.should_execute("missingness"):
        gate.begin("missingness")
        run_missingness_analysis(df_dev, y_dev, output_dir, feat_b)
        gate.complete("missingness")

    # ---- segment_analysis (light) ----
    if gate.should_execute("segment_analysis"):
        gate.begin("segment_analysis")
        rows = []
        for col in ("city_type", "education_level"):
            if col not in df_dev.columns:
                continue
            for seg, g in df_dev.groupby(col):
                yy = y_dev.loc[g.index]
                rows.append({"segment_col": col, "segment": str(seg), "n": len(g), "rate_1": float(yy.mean())})
        atomic_write_csv(output_dir / "diagnostics" / "segment_rates.csv", pd.DataFrame(rows))
        gate.complete("segment_analysis")

    # ---- feature_ablation ----
    if gate.should_execute("feature_ablation"):
        gate.begin("feature_ablation")
        run_feature_ablation(df_dev, y_dev, fold_assign, output_dir, majority_acc, n_folds=n_folds)
        gate.complete("feature_ablation")

    # ---- empirical_bayes ----
    if gate.should_execute("empirical_bayes"):
        gate.begin("empirical_bayes")
        eb_result = run_empirical_bayes(df_dev, y_dev, fold_assign, output_dir, n_folds=n_folds)
        candidate_oofs["eb_best"] = eb_result["oof"]
        gate.complete("empirical_bayes")
    else:
        p = output_dir / "empirical_bayes" / "eb_best_oof_predictions.csv"
        if p.exists():
            candidate_oofs["eb_best"] = pd.read_csv(p)

    # ---- clustering_prior ----
    if gate.should_execute("clustering_prior"):
        gate.begin("clustering_prior")
        clus = run_clustering_prior(df_dev, y_dev, fold_assign, feat_b, output_dir, n_folds=n_folds)
        candidate_oofs["cluster_prior"] = clus["oof"]
        gate.complete("clustering_prior", best_k=clus["best_config"].get("best_k"))
    else:
        p = output_dir / "clustering" / "cluster_prior_oof.csv"
        if p.exists():
            candidate_oofs["cluster_prior"] = pd.read_csv(p)

    # ---- m1_logistic_gam ----
    if gate.should_execute("m1_logistic_gam"):
        gate.begin("m1_logistic_gam")
        quick = run_quick_models(
            df_dev, y_dev, fold_assign, output_dir, feat_a, feat_b, n_folds=n_folds, resume_folds=resume
        )
        for k, oof in quick["models"].items():
            candidate_oofs[k] = oof
        gate.complete("m1_logistic_gam")
    elif gate.should_execute("m2_catboost"):
        pass
    else:
        for name in ("logistic_none", "logistic_balanced", "catboost_a", "catboost_b"):
            oof_p = output_dir / "oof" / f"{name}_oof_predictions.csv"
            if oof_p.exists():
                candidate_oofs[name] = pd.read_csv(oof_p)

    # ---- m2_catboost ---- (included in quick models; mark complete)
    if gate.should_execute("m2_catboost"):
        gate.begin("m2_catboost")
        if "catboost_b" not in candidate_oofs:
            quick = run_quick_models(
                df_dev, y_dev, fold_assign, output_dir, feat_a, feat_b, n_folds=n_folds, resume_folds=resume
            )
            for k, oof in quick["models"].items():
                candidate_oofs[k] = oof
        gate.complete("m2_catboost")

    # ---- m3/m4/m5 heterogeneous ----
    for stage_name, model_key in [
        ("m3_lightgbm", "lightgbm"),
        ("m4_xgboost", "xgboost"),
        ("m5_extra_trees", "extra_trees"),
    ]:
        if gate.should_execute(stage_name):
            gate.begin(stage_name)
            got = run_heterogeneous_models(
                df_dev,
                y_dev,
                fold_assign,
                feat_b,
                output_dir,
                n_folds=n_folds,
                resume_folds=resume,
                enable=(model_key,),
            )
            candidate_oofs.update(got)
            gate.complete(stage_name)
        else:
            p = output_dir / "oof" / f"{model_key}_oof_predictions.csv"
            if p.exists():
                candidate_oofs[model_key] = pd.read_csv(p)

    # ---- cross_task_aux (full) ----
    if gate.should_execute("cross_task_aux"):
        gate.begin("cross_task_aux")
        try:
            from src.tasks.task2.cross_task_aux import generate_cross_task_oof_features

            generate_cross_task_oof_features(
                df_dev, fold_assign, output_dir / "diagnostics" / "cross_task_aux", n_folds=n_folds
            )
            gate.complete("cross_task_aux")
        except Exception as e:
            gate.partial("cross_task_aux", error=str(e))

    # ---- diversity ----
    if gate.should_execute("diversity"):
        gate.begin("diversity")
        diversity_report(candidate_oofs, output_dir)
        gate.complete("diversity")

    # ---- soft_blend ----
    if gate.should_execute("soft_blend"):
        gate.begin("soft_blend")
        blend = soft_blend_search(candidate_oofs, y_dev, output_dir)
        if blend.get("available") and "oof" in blend:
            candidate_oofs["soft_blend"] = blend["oof"]
        gate.complete("soft_blend")
    else:
        p = output_dir / "ensemble" / "soft_blend_oof.csv"
        if p.exists():
            candidate_oofs["soft_blend"] = pd.read_csv(p)

    # ---- threshold ----
    if gate.should_execute("threshold"):
        gate.begin("threshold")
        comp_pre = compare_candidates(majority_class, majority_acc, y_dev, candidate_oofs)
        # 对最佳非 M 候选做阈值搜索
        non_m = comp_pre[comp_pre["candidate"] != "M_majority"]
        if len(non_m):
            best_cand = str(non_m.iloc[0]["candidate"])
            oof = candidate_oofs.get(best_cand)
            if oof is not None and best_cand not in ("B0_majority", "M_majority"):
                search, thr_cfg = _search_threshold(
                    oof["prob_1"].to_numpy(), oof["true_label"].to_numpy()
                )
                atomic_write_csv(output_dir / "tuning" / "threshold_search.csv", search)
                atomic_write_json(output_dir / "tuning" / "threshold_best.json", thr_cfg)
                best_thr = float(thr_cfg["threshold"])
        else:
            best_thr = 0.5
        atomic_write_json(
            output_dir / "tuning" / "default_threshold.json",
            {"threshold": 0.5, "note": "majority lock uses 0.5"},
        )
        gate.complete("threshold", threshold=best_thr)

    # ---- permutation_test ----
    if gate.should_execute("permutation_test"):
        gate.begin("permutation_test")
        perm_summary = run_permutation_test(
            df_dev,
            y_dev,
            fold_assign,
            feat_b,
            output_dir,
            n_perm=TASK2_PERMUTATION_N,
            n_folds=n_folds,
        )
        gate.complete("permutation_test", supports_signal=perm_summary.get("supports_signal"))

    # ---- multiseed (light: majority Acc across seeds) ----
    if gate.should_execute("multiseed"):
        gate.begin("multiseed")
        stab = output_dir / "stability"
        stab.mkdir(parents=True, exist_ok=True)
        seed_rows = []
        for sd in TASK2_STABILITY_SEEDS:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(sd))
            fold_accs = []
            for tr_i, va_i in skf.split(df_dev, y_dev):
                y_tr = y_dev.iloc[tr_i]
                maj = int(y_tr.value_counts().index[0])
                y_va = y_dev.iloc[va_i].to_numpy().astype(int)
                fold_accs.append(float((y_va == maj).mean()))
            seed_rows.append(
                {
                    "seed": int(sd),
                    "scheme": "majority",
                    "mean_acc": float(np.mean(fold_accs)),
                    "std_acc": float(np.std(fold_accs)),
                    "n_folds": n_folds,
                }
            )
        atomic_write_csv(stab / "multiseed_majority.csv", pd.DataFrame(seed_rows))
        gate.complete("multiseed")

    # ---- bootstrap ----
    if gate.should_execute("bootstrap"):
        gate.begin("bootstrap")
        # 用当前最佳非多数候选；若无则用 logistic_none / catboost_b
        ref = None
        for key in ("soft_blend", "catboost_b", "logistic_none", "eb_best", "lightgbm"):
            if key in candidate_oofs:
                ref = candidate_oofs[key]
                break
        if ref is not None:
            boot_summary = run_bootstrap_delta(
                ref, majority_class, majority_acc, output_dir, n_boot=TASK2_BOOTSTRAP_N
            )
            maj_pred = np.full(len(ref), majority_class, dtype=int)
            run_mcnemar(
                ref["true_label"].to_numpy(),
                ref["predicted_label"].to_numpy(),
                maj_pred,
                output_dir,
            )
        gate.complete("bootstrap")

    # ---- final_lock ----
    if gate.should_execute("final_lock"):
        gate.begin("final_lock")
        comp = compare_candidates(majority_class, majority_acc, y_dev, candidate_oofs)
        max_gap = sep_summary.get("max_auc_gap_from_0_5")
        if max_gap is None or max_gap != max_gap:
            sep_path = output_dir / "separability" / "separability_summary.json"
            if sep_path.exists():
                max_gap = json.loads(sep_path.read_text(encoding="utf-8")).get("max_auc_gap_from_0_5")
        lock_cfg = lock_final_scheme(comp, majority_acc)
        locked_scheme = lock_cfg["locked_scheme"]
        oof_acc = float(lock_cfg["oof_accuracy"])
        if locked_scheme not in ("M", "M_majority") and locked_scheme in candidate_oofs:
            oof = candidate_oofs[locked_scheme]
            _, thr_cfg = _search_threshold(oof["prob_1"].to_numpy(), oof["true_label"].to_numpy())
            best_thr = float(thr_cfg["threshold"])
        else:
            locked_scheme = "M"
            best_thr = 0.5
        feats = feat_b if locked_scheme.startswith(("logistic", "catboost_b", "eb")) else feat_a
        persist_final_lock(output_dir, lock_cfg, comp, feats)
        gate.complete("final_lock", scheme=locked_scheme)
    else:
        cfg_p = output_dir / "final" / "task2_final_config.json"
        if cfg_p.exists():
            lock_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            locked_scheme = lock_cfg.get("locked_scheme", "M")
            oof_acc = float(lock_cfg.get("oof_accuracy", majority_acc))

    # ---- final_validation ----
    if gate.should_execute("final_validation"):
        gate.begin("final_validation")
        fv = _run_final_validation(
            locked_scheme,
            df_dev,
            y_dev,
            df_val,
            y_val,
            feat_a,
            feat_b,
            majority_class,
            best_thr,
            output_dir,
        )
        final_acc = float(fv["accuracy"])
        gate.complete("final_validation", accuracy=final_acc)

    # ---- report ----
    if gate.should_execute("report"):
        gate.begin("report")
        write_paper_reports(
            output_dir,
            {
                "scheme": locked_scheme,
                "oof_accuracy": oof_acc,
                "final_accuracy": final_acc,
                "majority_accuracy": majority_acc,
                "is_weak_signal_lock": lock_cfg.get("is_weak_signal_lock", locked_scheme == "M"),
                "max_auc_gap": sep_summary.get("max_auc_gap_from_0_5"),
                "n_duplicate_conflicts": label_summary.get("n_duplicate_conflict_groups"),
                "eb_best_alpha": (eb_result.get("best_config") or {}).get("alpha"),
                "permutation": perm_summary,
                "bootstrap": boot_summary,
            },
        )
        gate.complete("report")

    result = {
        "task_name": TASK2_NAME,
        "accuracy": final_acc if final_acc == final_acc else oof_acc,
        "selected_scheme": locked_scheme,
        "oof_accuracy": oof_acc,
        "majority_baseline_accuracy": majority_acc,
        "is_weak_signal_lock": lock_cfg.get("is_weak_signal_lock", locked_scheme == "M"),
        "threshold": best_thr,
        "final_validation_accuracy": final_acc,
    }
    save_json(result, output_dir / "metrics.json")
    state.data["exit_status"] = "completed"
    state.save()
    logger.info("任务二完成 accuracy=%.6f scheme=%s", result["accuracy"], locked_scheme)
    return result
