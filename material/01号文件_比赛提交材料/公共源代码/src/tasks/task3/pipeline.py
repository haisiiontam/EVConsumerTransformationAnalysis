"""任务三全量优化流水线：分类 / 回归 / 有序 / 异构集成 + 断点续跑。"""

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

from src.common.baseline import extract_feature_importance
from src.common.fold_checkpoints import persist_assembled_oof
from src.common.io_utils import (
    RunState,
    atomic_write_csv,
    atomic_write_json,
    config_hash,
    setup_file_logging,
)
from src.common.metrics import (
    binary_classification_metrics,
    majority_baseline,
    make_confusion_matrix_df,
    save_json,
)
from src.common.separability import univariate_separability
from src.common.stage_control import RunOptions, StageGate, completed_from_audit
from src.common.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_probability_distribution,
    plot_roc_curve,
)
from src.config import (
    RANDOM_SEED,
    TASK3_CATBOOST_COMMON,
    TASK3_ENABLE_CROSS_TASK_AUX,
    TASK3_ENABLE_GAM,
    TASK3_ENABLE_TABNET,
    TASK3_EXCLUDED_FEATURES,
    TASK3_N_FOLDS,
    TASK3_NAME,
    TASK3_SOURCE_SCORE,
    TASK3_STABILITY_SEEDS,
    TASK3_STAGES,
    TASK3_THRESHOLD_MAX,
    TASK3_THRESHOLD_MIN,
    TASK3_THRESHOLD_STEP,
)
from src.tasks.task3.audit import archive_legacy_if_needed, run_audit
from src.tasks.task3.calibration import calibrate_oof, persist_calibration, search_threshold
from src.tasks.task3.explain import run_shap_analysis
from src.tasks.task3.features import (
    add_mechanism_features,
    assert_no_leakage,
    build_target,
    check_task3_base_columns,
    feature_sets,
    group_feature_lists,
    raw_score,
)
from src.tasks.task3.fusion import (
    classifier_regressor_fusion,
    diversity_report,
    persist_fusion,
    soft_blend_search,
)
from src.tasks.task3.m1_catboost_classifier import oof_catboost_binary, run_optuna_catboost, _prob1
from src.tasks.task3.m2_catboost_regressor import oof_regression_score, persist_m2_outputs
from src.tasks.task3.m3_ordinal import oof_ordinal_score, persist_m3
from src.tasks.task3.other_models import (
    oof_extra_trees,
    oof_lightgbm,
    oof_xgboost,
    persist_oof,
    try_gam_skip,
    try_tabnet_skip,
)
from src.tasks.task3.preprocessing import cat_indices, fit_cat_preprocessor, transform_cat
from src.tasks.task3.report import write_paper_reports
from src.tasks.task3.robustness import learning_curve, noise_sensitivity, persist_robustness
from src.tasks.task3.stacking import persist_stacking, stacking_oof

logger = logging.getLogger(__name__)


def _task3_skip(stage: str, opts: RunOptions) -> bool:
    if opts.skip_optuna and stage == "m1_optuna":
        logger.info("skip %s (--skip-optuna)", stage)
        return True
    if opts.skip_multiseed and stage == "multiseed":
        logger.info("skip %s (--skip-multiseed)", stage)
        return True
    if opts.skip_shap and stage == "shap":
        logger.info("skip %s (--skip-shap)", stage)
        return True
    if opts.skip_robustness and stage == "robustness":
        logger.info("skip %s (--skip-robustness)", stage)
        return True
    deep = {
        "m1_optuna",
        "m3_ordinal_score",
        "m4_lightgbm",
        "m5_xgboost",
        "m6_extra_trees",
        "m7_gam",
        "m8_tabnet",
        "cross_task_aux",
        "stacking",
        "multiseed",
        "robustness",
    }
    if opts.skip_deep and stage in deep:
        logger.info("skip %s (--skip-deep)", stage)
        return True
    if opts.mode == "quick" and stage in {
        "m1_optuna",
        "m3_ordinal_score",
        "m4_lightgbm",
        "m5_xgboost",
        "m6_extra_trees",
        "m7_gam",
        "m8_tabnet",
        "cross_task_aux",
        "stacking",
        "calibration",
        "multiseed",
        "robustness",
        "shap",
        "submission",
    }:
        logger.info("skip %s (quick mode)", stage)
        return True
    if stage == "m7_gam" and not TASK3_ENABLE_GAM:
        return True
    if stage == "m8_tabnet" and not TASK3_ENABLE_TABNET:
        return True
    if stage == "cross_task_aux" and not TASK3_ENABLE_CROSS_TASK_AUX:
        return True
    return False


_RESUME_SKIP = {
    "audit",
    "split",
    "baseline",
    "separability",
    "feature_ablation",
    "feature_selection",
    "m1_catboost_classifier",
    "m1_optuna",
    "m2_catboost_regressor",
    "m3_ordinal_score",
    "m4_lightgbm",
    "m5_xgboost",
    "m6_extra_trees",
    "diversity",
    "classifier_regressor_fusion",
}


def _ensure_dirs(output_dir: Path) -> None:
    for sub in (
        "diagnostics",
        "baseline",
        "ablation",
        "features",
        "oof",
        "tuning",
        "models",
        "ensemble",
        "checkpoints",
        "final",
        "final_validation",
        "explain",
        "paper",
        "robustness",
        "splits",
        "logs",
        "resume_audit",
    ):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def _freeze_or_load_splits(
    df: pd.DataFrame, y: pd.Series, output_dir: Path, test_size: float
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
    skf = StratifiedKFold(n_splits=TASK3_N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_assign = np.full(len(df_dev), -1, dtype=int)
    for k, (_, va) in enumerate(skf.split(df_dev, y_dev)):
        fold_assign[va] = k
    atomic_write_csv(
        dev_p, pd.DataFrame({"sample_index": df_dev.index.astype(int)})
    )
    atomic_write_csv(
        val_p, pd.DataFrame({"sample_index": df_val.index.astype(int)})
    )
    atomic_write_csv(
        fold_p,
        pd.DataFrame({"sample_index": df_dev.index.astype(int), "fold": fold_assign}),
    )
    atomic_write_json(
        split_dir / "split_config.json",
        {
            "test_size": test_size,
            "random_state": RANDOM_SEED,
            "n_folds": TASK3_N_FOLDS,
            "stratify": True,
            "n_develop": int(len(df_dev)),
            "n_final": int(len(df_val)),
        },
    )
    logger.info("froze splits develop=%d final=%d", len(df_dev), len(df_val))
    return df_dev, df_val, y_dev, y_val, fold_assign


def run(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.2,
    mode: str = "full",
    resume: bool = False,
    start_stage: Optional[str] = None,
    stop_after_stage: Optional[str] = None,
    run_id: Optional[str] = None,
    skip_optuna: bool = False,
    skip_multiseed: bool = False,
    skip_shap: bool = False,
    skip_deep: bool = False,
    skip_robustness: bool = False,
    audit_only: bool = False,
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    _ensure_dirs(output_dir)
    run_id = run_id or f"task3_{mode}_{int(time.time())}"
    setup_file_logging(output_dir / "logs" / f"task3_{run_id}.log", run_id, logger_name="task3")
    state = RunState(output_dir / "run_state.json", run_id)

    archive_legacy_if_needed(output_dir)
    audit_summary = run_audit(output_dir)
    if audit_only:
        state.data["exit_status"] = "audit_only"
        state.save()
        return {"task_name": TASK3_NAME, "accuracy": float("nan"), "audit": audit_summary}

    opts = RunOptions(
        mode=mode,
        resume=resume,
        start_stage=start_stage,
        stop_after_stage=stop_after_stage,
        run_id=run_id,
        skip_optuna=skip_optuna,
        skip_multiseed=skip_multiseed,
        skip_shap=skip_shap,
        skip_deep=skip_deep,
        skip_robustness=skip_robustness,
        audit_only=False,
        completed_stages=completed_from_audit(audit_summary) if resume else set(),
    )
    if resume and start_stage is None:
        opts.start_stage = audit_summary.get("first_resume_stage") or "split"
        logger.info("resume auto start_stage=%s", opts.start_stage)

    gate = StageGate(
        opts,
        state,
        TASK3_STAGES,
        skip_checker=_task3_skip,
        resume_skip_stages=_RESUME_SKIP,
    )

    check_task3_base_columns(df)
    y_all = build_target(df)
    maj = majority_baseline(y_all)
    save_json(maj, output_dir / "diagnostics" / "majority_baseline.json")

    # leakage audit artifacts
    atomic_write_json(
        output_dir / "diagnostics" / "leakage_audit.json",
        {
            "excluded": TASK3_EXCLUDED_FEATURES,
            "source_score": TASK3_SOURCE_SCORE,
            "rule": "score only for label / regression target, never classification X",
        },
    )
    atomic_write_json(
        output_dir / "diagnostics" / "uncertain_features.json",
        {"uncertain": ["monthly_energy_consumption_kwh", "monthly_charging_cost"]},
    )
    atomic_write_csv(
        output_dir / "diagnostics" / "feature_audit.csv",
        pd.DataFrame(
            [{"feature": c, "status": "excluded"} for c in TASK3_EXCLUDED_FEATURES]
            + [{"feature": TASK3_SOURCE_SCORE, "status": "label_source_only"}]
        ),
    )

    sets = feature_sets(df)
    feat_a, feat_b, feat_c = sets["A_official"], sets["B_safe_raw"], sets["C_safe_plus_mechanism"]
    assert_no_leakage(feat_c)

    # splits 始终加载/冻结（后续阶段依赖）；resume 已完成则只读
    if gate.should_execute("split"):
        gate.begin("split")
    df_dev, df_val, y_dev, y_val, fold_assign = _freeze_or_load_splits(
        df, y_all, output_dir, test_size
    )
    if gate.should_execute("split"):
        gate.complete("split")
    elif state.status_of("split") != "completed":
        state.complete("split")

    model_oofs: Dict[str, pd.DataFrame] = {}
    selected_features = list(feat_b)
    m1_params = dict(TASK3_CATBOOST_COMMON)
    best_scheme = "m1_baseline"
    best_p1: Optional[np.ndarray] = None
    best_thr = 0.5
    oof_acc = float("nan")
    final_acc = float("nan")
    pred_score_oof: Optional[np.ndarray] = None

    # resume 时从磁盘回填，避免 --start-stage 跳过已完成阶段后丢失 OOF/特征
    def _load_oof(path: Path, key: str) -> None:
        if path.exists():
            model_oofs[key] = pd.read_csv(path)
            logger.info("hydrate OOF %s from %s n=%d", key, path.name, len(model_oofs[key]))

    for jp in (
        output_dir / "features" / "selected_features.json",
        output_dir / "ablation" / "feature_selection.json",
    ):
        if jp.exists():
            try:
                meta = json.loads(jp.read_text(encoding="utf-8"))
                feats = meta.get("features") or meta.get("final_features")
                if feats:
                    selected_features = list(feats)
                    assert_no_leakage(selected_features)
                    if meta.get("oof_accuracy") is not None:
                        oof_acc = float(meta["oof_accuracy"])
                        best_scheme = str(meta.get("selected", best_scheme))
                    logger.info("hydrate features n=%d from %s", len(selected_features), jp.name)
                    break
            except Exception as e:
                logger.warning("hydrate features failed %s: %s", jp, e)
    mp = output_dir / "tuning" / "m1_best_params.json"
    if mp.exists():
        try:
            m1_params = json.loads(mp.read_text(encoding="utf-8"))
            logger.info("hydrate m1_params from m1_best_params.json")
        except Exception as e:
            logger.warning("hydrate m1_params failed: %s", e)
    _load_oof(output_dir / "baseline" / "baseline_oof_predictions.csv", "baseline")
    _load_oof(output_dir / "oof" / "m1_oof_predictions.csv", "M1")
    _load_oof(output_dir / "oof" / "m1_tuned_oof_predictions.csv", "M1")  # tuned 覆盖
    _load_oof(output_dir / "models" / "m2_regression_oof.csv", "M2")
    _load_oof(output_dir / "models" / "m3_ordinal_oof.csv", "M3")
    _load_oof(output_dir / "oof" / "lgbm_oof_predictions.csv", "M4")
    _load_oof(output_dir / "oof" / "xgb_oof_predictions.csv", "M5")
    _load_oof(output_dir / "oof" / "et_oof_predictions.csv", "M6")
    if "M2" in model_oofs and "pred_score" in model_oofs["M2"].columns:
        pred_score_oof = model_oofs["M2"]["pred_score"].to_numpy()
    if "M1" in model_oofs:
        best_p1 = model_oofs["M1"]["prob_1"].to_numpy()
        y_tmp = model_oofs["M1"]["true_label"].to_numpy()
        search_tmp, thr_tmp = search_threshold(best_p1, y_tmp)
        best_thr = float(thr_tmp["threshold"])
        oof_acc = float(thr_tmp["oof_accuracy"]) if not (oof_acc == oof_acc) else max(
            oof_acc, float(thr_tmp["oof_accuracy"])
        )
        best_scheme = "M1_tuned" if (output_dir / "oof" / "m1_tuned_oof_predictions.csv").exists() else "M1"

    # ---- baseline ----
    if gate.should_execute("baseline"):
        gate.begin("baseline")
        oof_b, folds_b, _ = oof_catboost_binary(
            df_dev,
            y_dev,
            feat_b,
            m1_params,
            fold_assign,
            output_dir / "checkpoints" / "baseline_b",
            model_name="baseline_B",
            resume_folds=resume,
        )
        search, thr_cfg = search_threshold(
            oof_b["prob_1"].to_numpy(), oof_b["true_label"].to_numpy()
        )
        m = binary_classification_metrics(
            oof_b["true_label"],
            (oof_b["prob_1"] >= thr_cfg["threshold"]).astype(int),
            oof_b["prob_1"],
        )
        atomic_write_csv(output_dir / "baseline" / "baseline_oof_predictions.csv", oof_b)
        atomic_write_csv(output_dir / "baseline" / "baseline_fold_metrics.csv", folds_b)
        atomic_write_csv(output_dir / "baseline" / "baseline_threshold_search.csv", search)
        cm, cm_df = make_confusion_matrix_df(
            oof_b["true_label"],
            (oof_b["prob_1"] >= thr_cfg["threshold"]).astype(int),
            [0, 1],
        )
        cm_df.to_csv(output_dir / "baseline" / "baseline_confusion_matrix.csv", encoding="utf-8-sig")
        base_metrics = {
            **m,
            **thr_cfg,
            "majority_baseline": maj,
            "feature_set": "B_safe_raw",
            "n_features": len(feat_b),
            "fold_mean_acc": float(folds_b["accuracy"].mean()) if "accuracy" in folds_b else None,
            "fold_std_acc": float(folds_b["accuracy"].std()) if "accuracy" in folds_b else None,
        }
        atomic_write_json(output_dir / "baseline" / "baseline_metrics.json", base_metrics)
        atomic_write_json(
            output_dir / "baseline" / "baseline_config.json",
            {"params": m1_params, "features": feat_b, "threshold": thr_cfg["threshold"]},
        )
        model_oofs["baseline"] = oof_b
        best_p1 = oof_b["prob_1"].to_numpy()
        best_thr = float(thr_cfg["threshold"])
        oof_acc = float(thr_cfg["oof_accuracy"])
        selected_features = list(feat_b)
        best_scheme = "baseline_B"
        gate.complete("baseline", oof_accuracy=oof_acc)

    # ---- separability ----
    if gate.should_execute("separability"):
        gate.begin("separability")
        df_fe = add_mechanism_features(df_dev)
        sep_cols = [c for c in feat_c if c in df_fe.columns]
        sep = univariate_separability(df_fe, y_dev, sep_cols)
        atomic_write_csv(output_dir / "diagnostics" / "separability.csv", sep)
        # score boundary analysis
        sc = raw_score(df_dev).to_numpy()
        # need probs: use baseline if available
        p1 = best_p1 if best_p1 is not None else np.full(len(df_dev), 0.5)
        rows = []
        for s in range(1, 11):
            m = np.round(sc).astype(int) == s
            if not m.any():
                continue
            yt = y_dev.to_numpy()[m]
            pr = (p1[m] >= best_thr).astype(int)
            rows.append(
                {
                    "score": s,
                    "n": int(m.sum()),
                    "accuracy": float((pr == yt).mean()),
                    "mean_prob": float(p1[m].mean()),
                    "std_prob": float(p1[m].std()),
                    "errors": int((pr != yt).sum()),
                }
            )
        atomic_write_csv(output_dir / "diagnostics" / "score_level_error_analysis.csv", pd.DataFrame(rows))
        atomic_write_csv(
            output_dir / "diagnostics" / "score_5_6_boundary_analysis.csv",
            pd.DataFrame([r for r in rows if r["score"] in (5, 6)]),
        )
        td = pd.DataFrame(
            {
                "label": [0, 1],
                "count": [(y_dev == 0).sum(), (y_dev == 1).sum()],
                "rate": [(y_dev == 0).mean(), (y_dev == 1).mean()],
            }
        )
        atomic_write_csv(output_dir / "diagnostics" / "target_distribution.csv", td)
        gate.complete("separability")

    # ---- feature ablation ----
    if gate.should_execute("feature_ablation"):
        gate.begin("feature_ablation")
        candidates = group_feature_lists(feat_b, sets["mechanism_only_pool"])
        candidates["A_official"] = feat_a
        candidates["C_safe_plus_mechanism"] = feat_c
        abl_rows = []
        best_abl_name, best_abl_acc = "B_safe_raw", -1.0
        best_abl_feats = feat_b
        for name, feats in candidates.items():
            # quick: only A/B/C
            if mode == "quick" and name not in {"A_official", "B_safe_raw", "C_safe_plus_mechanism"}:
                continue
            assert_no_leakage(feats)
            ck = output_dir / "checkpoints" / f"abl_{name.replace('+', '_')}"
            oof, folds, _ = oof_catboost_binary(
                df_dev, y_dev, feats, m1_params, fold_assign, ck, model_name=name, resume_folds=resume
            )
            search, thr_cfg = search_threshold(oof["prob_1"].to_numpy(), oof["true_label"].to_numpy())
            m = binary_classification_metrics(
                oof["true_label"],
                (oof["prob_1"] >= thr_cfg["threshold"]).astype(int),
                oof["prob_1"],
            )
            row = {
                "configuration": name,
                "n_features": len(feats),
                "threshold": thr_cfg["threshold"],
                "accuracy": m["accuracy"],
                "roc_auc": m["roc_auc"],
                "fold_mean": float(folds["accuracy"].mean()) if len(folds) else None,
                "fold_std": float(folds["accuracy"].std()) if len(folds) else None,
                **{k: m[k] for k in ("balanced_accuracy", "recall_0", "recall_1") if k in m},
            }
            abl_rows.append(row)
            if m["accuracy"] > best_abl_acc:
                best_abl_acc = float(m["accuracy"])
                best_abl_name = name
                best_abl_feats = list(feats)
                model_oofs["ablation_best"] = oof
                best_p1 = oof["prob_1"].to_numpy()
                best_thr = float(thr_cfg["threshold"])
                oof_acc = best_abl_acc
        atomic_write_csv(output_dir / "ablation" / "feature_ablation.csv", pd.DataFrame(abl_rows))
        selected_features = best_abl_feats
        best_scheme = f"ablation_{best_abl_name}"
        atomic_write_json(
            output_dir / "ablation" / "feature_selection.json",
            {"selected": best_abl_name, "features": selected_features, "oof_accuracy": best_abl_acc},
        )
        gate.complete("feature_ablation", selected=best_abl_name)

    # ---- feature_selection (record) ----
    if gate.should_execute("feature_selection"):
        gate.begin("feature_selection")
        atomic_write_csv(
            output_dir / "features" / "feature_selection_comparison.csv",
            pd.DataFrame(
                [
                    {
                        "method": "ablation_oof_accuracy",
                        "n_features": len(selected_features),
                        "oof_accuracy": oof_acc,
                    }
                ]
            ),
        )
        atomic_write_json(
            output_dir / "features" / "selected_features.json",
            {"features": selected_features},
        )
        gate.complete("feature_selection")

    # ---- M1 ----
    if gate.should_execute("m1_catboost_classifier"):
        gate.begin("m1_catboost_classifier")
        oof_m1, folds_m1, best_iters = oof_catboost_binary(
            df_dev,
            y_dev,
            selected_features,
            m1_params,
            fold_assign,
            output_dir / "checkpoints" / "m1",
            model_name="m1",
            resume_folds=resume,
        )
        atomic_write_csv(output_dir / "oof" / "m1_oof_predictions.csv", oof_m1)
        atomic_write_csv(output_dir / "oof" / "m1_fold_metrics.csv", folds_m1)
        model_oofs["M1"] = oof_m1
        search, thr_cfg = search_threshold(oof_m1["prob_1"].to_numpy(), oof_m1["true_label"].to_numpy())
        if thr_cfg["oof_accuracy"] >= oof_acc:
            oof_acc = float(thr_cfg["oof_accuracy"])
            best_p1 = oof_m1["prob_1"].to_numpy()
            best_thr = float(thr_cfg["threshold"])
            best_scheme = "M1"
        gate.complete("m1_catboost_classifier", oof_accuracy=float(thr_cfg["oof_accuracy"]))

    if gate.should_execute("m1_optuna"):
        gate.begin("m1_optuna")
        m1_params = run_optuna_catboost(
            df_dev,
            y_dev,
            selected_features,
            fold_assign,
            output_dir / "tuning",
            quick=(mode == "quick"),
        )
        # re-OOF with best params
        oof_m1, folds_m1, _ = oof_catboost_binary(
            df_dev,
            y_dev,
            selected_features,
            m1_params,
            fold_assign,
            output_dir / "checkpoints" / "m1_tuned",
            model_name="m1_tuned",
            resume_folds=resume,
        )
        atomic_write_csv(output_dir / "oof" / "m1_tuned_oof_predictions.csv", oof_m1)
        model_oofs["M1"] = oof_m1
        search, thr_cfg = search_threshold(oof_m1["prob_1"].to_numpy(), oof_m1["true_label"].to_numpy())
        if thr_cfg["oof_accuracy"] >= oof_acc:
            oof_acc = float(thr_cfg["oof_accuracy"])
            best_p1 = oof_m1["prob_1"].to_numpy()
            best_thr = float(thr_cfg["threshold"])
            best_scheme = "M1_tuned"
        gate.complete("m1_optuna")

    # ---- M2 ----
    if gate.should_execute("m2_catboost_regressor"):
        gate.begin("m2_catboost_regressor")
        oof_m2, info_m2 = oof_regression_score(
            df_dev,
            y_dev,
            selected_features,
            fold_assign,
            output_dir / "checkpoints" / "m2",
            resume_folds=resume,
        )
        thr_df = pd.DataFrame(
            [
                {
                    "threshold_score": info_m2["best_score_threshold"],
                    "accuracy": info_m2["score_threshold_accuracy"],
                }
            ]
        )
        persist_m2_outputs(oof_m2, info_m2, thr_df, output_dir)
        model_oofs["M2"] = oof_m2
        pred_score_oof = oof_m2["pred_score"].to_numpy()
        if info_m2["score_to_prob_accuracy"] >= oof_acc:
            oof_acc = float(info_m2["score_to_prob_accuracy"])
            best_p1 = oof_m2["prob_1"].to_numpy()
            best_thr = 0.5
            best_scheme = "M2"
        gate.complete("m2_catboost_regressor", **info_m2)

    # ---- M3 ----
    if gate.should_execute("m3_ordinal_score"):
        gate.begin("m3_ordinal_score")
        try:
            oof_m3, dist_m3, met_m3 = oof_ordinal_score(
                df_dev, y_dev, selected_features, fold_assign
            )
            persist_m3(oof_m3, dist_m3, met_m3, output_dir)
            model_oofs["M3"] = oof_m3
            if met_m3.get("accuracy", 0) >= (oof_acc if oof_acc == oof_acc else -1):
                oof_acc = float(met_m3["accuracy"])
                best_p1 = oof_m3["prob_1"].to_numpy()
                best_thr = 0.5
                best_scheme = "M3"
            gate.complete("m3_ordinal_score", accuracy=met_m3.get("accuracy"))
        except Exception as e:
            logger.exception("m3_ordinal_score failed (continue): %s", e)
            gate.partial("m3_ordinal_score", error=str(e))

    # ---- M4-M6 ----
    for stage, fn, key, out_name in [
        ("m4_lightgbm", oof_lightgbm, "M4", "lgbm"),
        ("m5_xgboost", oof_xgboost, "M5", "xgb"),
        ("m6_extra_trees", oof_extra_trees, "M6", "et"),
    ]:
        if gate.should_execute(stage):
            gate.begin(stage)
            try:
                oof_x, folds_x = fn(
                    df_dev,
                    y_dev,
                    selected_features,
                    fold_assign,
                    output_dir / "checkpoints" / out_name,
                    resume_folds=resume,
                )
                persist_oof(
                    oof_x,
                    folds_x,
                    output_dir / "oof" / f"{out_name}_oof_predictions.csv",
                    output_dir / "oof" / f"{out_name}_fold_metrics.csv",
                )
                model_oofs[key] = oof_x
                acc = float(
                    (
                        (oof_x["prob_1"] >= 0.5).astype(int) == oof_x["true_label"]
                    ).mean()
                )
                if acc >= oof_acc:
                    oof_acc = acc
                    best_p1 = oof_x["prob_1"].to_numpy()
                    best_thr = 0.5
                    best_scheme = key
                gate.complete(stage, oof_accuracy=acc)
            except Exception as e:
                logger.warning("%s failed: %s", stage, e)
                gate.partial(stage, error=str(e))

    if gate.should_execute("m7_gam"):
        gate.begin("m7_gam")
        try_gam_skip(output_dir, "disabled_by_config")
        gate.complete("m7_gam", skipped=True)

    if gate.should_execute("m8_tabnet"):
        gate.begin("m8_tabnet")
        try_tabnet_skip(output_dir, "disabled_by_config")
        gate.complete("m8_tabnet", skipped=True)

    if gate.should_execute("cross_task_aux"):
        gate.begin("cross_task_aux")
        atomic_write_json(
            output_dir / "models" / "cross_task_aux_skip.json",
            {"skipped": True, "reason": "high_risk_default_off"},
        )
        gate.complete("cross_task_aux", skipped=True)

    # ---- diversity ----
    if gate.should_execute("diversity") and model_oofs:
        gate.begin("diversity")
        sc = raw_score(df_dev).to_numpy()
        diversity_report(model_oofs, sc, output_dir / "ensemble")
        gate.complete("diversity")

    # ---- classifier-regressor fusion ----
    if gate.should_execute("classifier_regressor_fusion") and "M1" in model_oofs and "M2" in model_oofs:
        gate.begin("classifier_regressor_fusion")
        yv = model_oofs["M1"]["true_label"].to_numpy().astype(int)
        p_cls = model_oofs["M1"]["prob_1"].to_numpy()
        p_reg = model_oofs["M2"].set_index("sample_index").loc[model_oofs["M1"]["sample_index"]]["prob_1"].to_numpy()
        p_ord = None
        if "M3" in model_oofs:
            p_ord = (
                model_oofs["M3"]
                .set_index("sample_index")
                .loc[model_oofs["M1"]["sample_index"]]["prob_1"]
                .to_numpy()
            )
        grid, best_f, p_f = classifier_regressor_fusion(p_cls, p_reg, yv, p_ord)
        oof_f = persist_fusion(
            output_dir / "ensemble",
            grid,
            best_f,
            p_f,
            model_oofs["M1"]["sample_index"].to_numpy(),
            yv,
        )
        model_oofs["fusion"] = oof_f
        if float(best_f.get("accuracy", 0)) >= oof_acc:
            oof_acc = float(best_f["accuracy"])
            best_p1 = p_f
            best_thr = float(best_f.get("threshold", 0.5))
            best_scheme = "clf_reg_fusion"
        gate.complete("classifier_regressor_fusion", **{k: v for k, v in best_f.items() if k != "weights"})

    # ---- soft blend ----
    if gate.should_execute("soft_blend") and len(model_oofs) >= 2:
        gate.begin("soft_blend")
        base = model_oofs["M1"] if "M1" in model_oofs else next(iter(model_oofs.values()))
        yv = base["true_label"].to_numpy().astype(int)
        idx = base["sample_index"]
        mp = {}
        for k, oof in model_oofs.items():
            mp[k] = oof.set_index("sample_index").loc[idx]["prob_1"].to_numpy()
        grid, best_s, p_s = soft_blend_search(mp, yv)
        atomic_write_csv(output_dir / "ensemble" / "soft_blend_search.csv", grid)
        atomic_write_json(output_dir / "ensemble" / "soft_blend_best.json", best_s)
        if float(best_s.get("accuracy", 0)) >= oof_acc:
            oof_acc = float(best_s["accuracy"])
            best_p1 = p_s
            best_thr = float(best_s.get("threshold", 0.5))
            best_scheme = "soft_blend"
        gate.complete("soft_blend", accuracy=best_s.get("accuracy"))

    # ---- stacking ----
    if gate.should_execute("stacking") and len(model_oofs) >= 2:
        gate.begin("stacking")
        base = model_oofs["M1"] if "M1" in model_oofs else next(iter(model_oofs.values()))
        yv = base["true_label"].to_numpy().astype(int)
        idx = base["sample_index"]
        mp = {
            k: oof.set_index("sample_index").loc[idx]["prob_1"].to_numpy()
            for k, oof in model_oofs.items()
        }
        p_st, info_st = stacking_oof(mp, yv, fold_assign, pred_score_oof)
        persist_stacking(
            output_dir / "ensemble",
            p_st,
            yv,
            idx.to_numpy(),
            info_st,
            [{"scheme": "stacking_lr", "accuracy": info_st["oof_accuracy"]}],
        )
        if info_st["oof_accuracy"] >= oof_acc:
            oof_acc = float(info_st["oof_accuracy"])
            best_p1 = p_st
            best_thr = 0.5
            best_scheme = "stacking"
        gate.complete("stacking", **info_st)

    # ---- calibration + threshold ----
    if gate.should_execute("calibration") and best_p1 is not None:
        gate.begin("calibration")
        yv = y_dev.to_numpy().astype(int)
        methods, compare = calibrate_oof(best_p1, yv, fold_assign)
        # keep method that maximizes Acc
        best_name = compare.sort_values("accuracy", ascending=False).iloc[0]["method"]
        cal_p = methods[best_name]
        search, thr_cfg = search_threshold(cal_p, yv)
        persist_calibration(output_dir / "ensemble", compare, thr_cfg, search)
        if thr_cfg["oof_accuracy"] >= oof_acc:
            oof_acc = float(thr_cfg["oof_accuracy"])
            best_p1 = cal_p
            best_thr = float(thr_cfg["threshold"])
            best_scheme = f"{best_scheme}+{best_name}_cal"
        gate.complete("calibration", method=best_name)

    if gate.should_execute("threshold_postprocess") and best_p1 is not None:
        gate.begin("threshold_postprocess")
        search, thr_cfg = search_threshold(best_p1, y_dev.to_numpy().astype(int))
        atomic_write_csv(output_dir / "tuning" / "final_threshold_search.csv", search)
        atomic_write_json(output_dir / "tuning" / "final_threshold.json", thr_cfg)
        best_thr = float(thr_cfg["threshold"])
        oof_acc = float(thr_cfg["oof_accuracy"])
        gate.complete("threshold_postprocess", **thr_cfg)

    # ---- multiseed (light) ----
    if gate.should_execute("multiseed"):
        gate.begin("multiseed")
        rows = []
        for seed in TASK3_STABILITY_SEEDS:
            skf = StratifiedKFold(n_splits=TASK3_N_FOLDS, shuffle=True, random_state=seed)
            accs = []
            for tr, va in skf.split(df_dev, y_dev):
                state_p = fit_cat_preprocessor(df_dev.iloc[tr], selected_features)
                model = CatBoostClassifier(**{**m1_params, "random_seed": seed})
                model.fit(
                    transform_cat(df_dev.iloc[tr], state_p),
                    y_dev.iloc[tr],
                    cat_features=cat_indices(state_p),
                )
                p1 = _prob1(model, transform_cat(df_dev.iloc[va], state_p))
                accs.append(float(((p1 >= best_thr).astype(int) == y_dev.iloc[va].to_numpy()).mean()))
            rows.append({"seed": seed, "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs))})
        (output_dir / "stability").mkdir(parents=True, exist_ok=True)
        atomic_write_csv(output_dir / "stability" / "multiseed_summary.csv", pd.DataFrame(rows))
        atomic_write_csv(output_dir / "oof" / "stability_seeds.csv", pd.DataFrame(rows))
        gate.complete("multiseed")

    # ---- robustness ----
    if gate.should_execute("robustness"):
        gate.begin("robustness")
        lc = learning_curve(df_dev, y_dev, selected_features)
        noise = noise_sensitivity(df_dev, y_dev, selected_features)
        persist_robustness(output_dir / "robustness", lc, noise)
        gate.complete("robustness")

    # ---- final lock ----
    final_cfg = {
        "scheme": best_scheme,
        "features": selected_features,
        "m1_params": m1_params,
        "threshold": best_thr,
        "oof_accuracy": oof_acc,
        "random_seed": RANDOM_SEED,
        "mode": mode,
        "run_id": run_id,
        "configuration_hash": config_hash(
            {"scheme": best_scheme, "features": selected_features, "threshold": best_thr, "params": m1_params}
        ),
    }
    if gate.should_execute("final_lock"):
        gate.begin("final_lock")
        atomic_write_json(output_dir / "final" / "task3_final_config.json", final_cfg)
        atomic_write_json(
            output_dir / "final" / "task3_model_manifest.json",
            {
                "models_in_oof": list(model_oofs.keys()),
                "selected_scheme": best_scheme,
                "packages": {"catboost": True, "lightgbm": "M4" in model_oofs, "xgboost": "M5" in model_oofs},
            },
        )
        atomic_write_json(output_dir / "tuning" / "task3_final_config.json", final_cfg)
        gate.complete("final_lock")

    # ---- final validation (once) ----
    metrics_out: Dict[str, Any] = {}
    if gate.should_execute("final_validation"):
        gate.begin("final_validation")
        state_p = fit_cat_preprocessor(df_dev, selected_features)
        X_tr = transform_cat(df_dev, state_p)
        params = dict(m1_params)
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_dev, cat_features=cat_indices(state_p))
        p_val = _prob1(model, transform_cat(df_val, state_p))
        # If fusion/soft_blend selected, still deploy M1+threshold as locked simple predictor
        # unless scheme is pure M2 — then use regressor path omitted for deploy simplicity:
        # always ship CatBoost clf with locked features/params/threshold for reproducibility.
        y_pred = (p_val >= best_thr).astype(int)
        metrics = binary_classification_metrics(y_val, y_pred, p_val)
        final_acc = float(metrics["accuracy"])
        final_dir = output_dir / "final_validation"
        metrics_out = {
            **final_cfg,
            **metrics,
            "majority_baseline_accuracy": float(maj["majority_baseline_accuracy"]),
            "accuracy_lift_vs_majority": float(metrics["accuracy"] - maj["majority_baseline_accuracy"]),
            "confusion_matrix": make_confusion_matrix_df(y_val, y_pred, [0, 1])[0].tolist(),
            "validation_note": "single evaluation after lock; historical 0.8534 may be non-locked",
        }
        atomic_write_json(final_dir / "metrics.json", metrics_out)
        atomic_write_csv(
            final_dir / "predictions.csv",
            pd.DataFrame(
                {
                    "sample_index": y_val.index.astype(int),
                    "true_label": y_val.values.astype(int),
                    "predicted_label": y_pred,
                    "prob_0": 1.0 - p_val,
                    "prob_1": p_val,
                    "threshold": best_thr,
                }
            ),
        )
        cm, cm_df = make_confusion_matrix_df(y_val, y_pred, [0, 1])
        cm_df.to_csv(final_dir / "confusion_matrix.csv", encoding="utf-8-sig")
        plot_confusion_matrix(cm, ["0", "1"], "Task3 Final CM", final_dir / "confusion_matrix.png")
        plot_probability_distribution(
            y_val, p_val, best_thr, "Task3 Prob", final_dir / "probability_distribution.png"
        )
        plot_roc_curve(y_val, p_val, "Task3 ROC", final_dir / "roc_curve.png")
        fi = extract_feature_importance(model, state_p.final_feature_order)
        atomic_write_csv(final_dir / "feature_importance.csv", fi)
        plot_feature_importance(fi, "Task3 FI", final_dir / "feature_importance.png")
        model.save_model(str(final_dir / "model.cbm"))
        gate.complete("final_validation", accuracy=final_acc)

        # candidates table
        cand_rows = [{"scheme": best_scheme, "oof_accuracy": oof_acc, "final_accuracy": final_acc}]
        for k, oof in model_oofs.items():
            cand_rows.append(
                {
                    "scheme": k,
                    "oof_accuracy": float(
                        ((oof["prob_1"] >= 0.5).astype(int) == oof["true_label"]).mean()
                    ),
                    "final_accuracy": None,
                }
            )
        atomic_write_csv(output_dir / "oof" / "final_candidates.csv", pd.DataFrame(cand_rows))

    # ---- shap ----
    if gate.should_execute("shap"):
        gate.begin("shap")
        try:
            state_p = fit_cat_preprocessor(df_dev, selected_features)
            X_tr = transform_cat(df_dev, state_p)
            model = CatBoostClassifier(**m1_params)
            model_path = output_dir / "final_validation" / "model.cbm"
            if model_path.exists():
                model.load_model(str(model_path))
            else:
                model.fit(X_tr, y_dev, cat_features=cat_indices(state_p))
            info = run_shap_analysis(
                model, X_tr, y_dev, output_dir / "explain", cat_indices(state_p)
            )
            if info.get("available"):
                gate.complete("shap", **{k: v for k, v in info.items() if k != "error"})
            else:
                gate.partial("shap", error=info.get("error", "failed"))
        except Exception as e:
            gate.partial("shap", error=str(e))

    # ---- report ----
    if gate.should_execute("report"):
        gate.begin("report")
        write_paper_reports(
            output_dir,
            {
                "oof_accuracy": oof_acc,
                "final_accuracy": final_acc,
                "threshold": best_thr,
                "scheme": best_scheme,
                "n_features": len(selected_features),
            },
        )
        gate.complete("report")

    # ---- submission ----
    if gate.should_execute("submission") and test_df is not None:
        gate.begin("submission")
        state_all = fit_cat_preprocessor(df, selected_features)
        model_all = CatBoostClassifier(**m1_params)
        model_all.fit(
            transform_cat(df, state_all),
            y_all,
            cat_features=cat_indices(state_all),
        )
        p_test = _prob1(model_all, transform_cat(test_df, state_all))
        atomic_write_csv(
            output_dir / "test_predictions.csv",
            pd.DataFrame(
                {
                    "sample_index": test_df.index.astype(int),
                    "predicted_label": (p_test >= best_thr).astype(int),
                    "prob_1": p_test,
                    "threshold": best_thr,
                }
            ),
        )
        gate.complete("submission")
    elif gate.should_execute("submission"):
        gate.begin("submission")
        atomic_write_json(
            output_dir / "submission_skip.json",
            {"skipped": True, "reason": "no test_df"},
        )
        gate.complete("submission", skipped=True)

    result = {
        "task_name": TASK3_NAME,
        "accuracy": float(final_acc) if final_acc == final_acc else float("nan"),
        "oof_accuracy": float(oof_acc) if oof_acc == oof_acc else float("nan"),
        "threshold": best_thr,
        "selected_configuration": best_scheme,
        "majority_baseline_accuracy": float(maj["majority_baseline_accuracy"]),
        "final_validation": metrics_out,
        "n_features": len(selected_features),
    }
    save_json(result, output_dir / "metrics.json")
    state.data["exit_status"] = "success"
    state.save()
    logger.info(
        "任务三完成 scheme=%s OOF=%.4f Final=%.4f thr=%.4f",
        best_scheme,
        oof_acc if oof_acc == oof_acc else -1,
        final_acc if final_acc == final_acc else -1,
        best_thr,
    )
    return result
