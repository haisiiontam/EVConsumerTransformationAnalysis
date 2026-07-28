"""任务一全量优化主流程（quick / full / resume）。"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.common.baseline import extract_feature_importance
from src.common.metrics import make_confusion_matrix_df, save_json
from src.config import (
    RANDOM_SEED,
    TASK1_LABEL_ORDER,
    TASK1_N_FOLDS,
    TASK1_OFFICIAL_FEATURES,
    TASK1_STABILITY_SEEDS,
    TASK1_TARGET,
    TASK1_TIE_EPS,
)
from src.tasks.task1.audit import run_audit
from src.tasks.task1.catboost_models import (
    confirm_top_catboost,
    default_catboost_params,
    oof_catboost_multiclass,
    optuna_search_catboost,
)
from src.tasks.task1.checkpoints import persist_assembled_oof
from src.tasks.task1.ensemble import (
    apply_bias,
    diversity_table,
    search_class_bias,
    soft_vote_search,
    stacking_nested_oof,
)
from src.tasks.task1.explain import plot_confusion_and_curves, run_shap_analysis, train_final_catboost
from src.tasks.task1.features import (
    add_mechanism_features,
    build_feature_sets,
    features_from_kept_groups,
    select_kept_groups,
    uncertain_feature_report,
)
from src.tasks.task1.io_utils import RunState, atomic_write_csv, atomic_write_json, config_hash, setup_file_logging
from src.tasks.task1.metrics_ext import (
    classification_report_df,
    evaluate_multiclass,
    fold_summary,
    pred_from_proba,
)
from src.tasks.task1.ordinal import oof_ordinal_catboost
from src.tasks.task1.other_models import (
    HAS_LGBM,
    HAS_XGB,
    oof_lgbm,
    oof_ordinal_logistic,
    oof_xgboost,
    optuna_lgbm,
)
from src.tasks.task1.preprocess import cat_indices, fit_cat_preprocessor, transform_cat
from src.tasks.task1.regression import (
    oof_ordinal_regressor,
    search_reg_soft_proba,
    search_reg_thresholds,
)
from src.tasks.task1.report import write_paper_reports
from src.tasks.task1.stage_control import RunOptions, StageGate, completed_from_audit

logger = logging.getLogger(__name__)


def _normalize_y(df: pd.DataFrame) -> pd.Series:
    from src.tasks.task1.features import to_num  # noqa: F401

    s = df[TASK1_TARGET]
    cleaned = s.astype(str).str.strip().str.lower()
    mapping = {"low": "Low", "medium": "Medium", "high": "High"}
    out = cleaned.map(mapping)
    if out.isna().any():
        raise ValueError("任务一标签无法识别或存在缺失")
    return out


def make_folds(y: pd.Series, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_pos = y.reset_index(drop=True)
    return list(skf.split(np.zeros(len(y_pos)), y_pos))


def save_splits(
    output_dir: Path,
    idx_dev: pd.Series,
    idx_val: pd.Series,
    y_dev: pd.Series,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> None:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_index": idx_dev.astype(int)}).to_csv(
        split_dir / "develop_indices.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"sample_index": idx_val.astype(int)}).to_csv(
        split_dir / "final_validation_indices.csv", index=False, encoding="utf-8-sig"
    )
    save_json(
        {
            "test_size": 0.2,
            "random_state": RANDOM_SEED,
            "stratify": True,
            "n_develop": int(len(idx_dev)),
            "n_final_validation": int(len(idx_val)),
            "n_folds": TASK1_N_FOLDS,
            "fold_seed": RANDOM_SEED,
        },
        split_dir / "split_config.json",
    )
    fold_assign = np.full(len(y_dev), -1, dtype=int)
    for fid, (_, va) in enumerate(folds):
        fold_assign[va] = fid
    pd.DataFrame(
        {
            "sample_index": idx_dev.reset_index(drop=True).astype(int),
            "fold": fold_assign,
            "label": y_dev.reset_index(drop=True),
        }
    ).to_csv(split_dir / "fold_assignment_seed42.csv", index=False, encoding="utf-8-sig")


def _oof_bundle(
    y: pd.Series, pred: np.ndarray, proba: np.ndarray, fold_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    y_n = y.reset_index(drop=True)
    m = evaluate_multiclass(y_n, pred, proba)
    m.update(fold_summary(fold_rows, "accuracy"))
    return m


def _load_oof_proba(path: Path, n_expected: int, sample_index: Sequence[int]) -> np.ndarray:
    dfp = pd.read_csv(path)
    if len(dfp) != n_expected:
        raise ValueError(f"OOF rows mismatch: {path} {len(dfp)}!={n_expected}")
    if set(dfp["sample_index"].astype(int)) != set(int(x) for x in sample_index):
        raise ValueError(f"OOF sample_index mismatch: {path}")
    order = {int(s): i for i, s in enumerate(sample_index)}
    out = np.zeros((n_expected, 3), dtype=float)
    for _, r in dfp.iterrows():
        i = order[int(r["sample_index"])]
        out[i] = [r["prob_Low"], r["prob_Medium"], r["prob_High"]]
    return out


def _fold_assign_array(folds: Sequence[Tuple[np.ndarray, np.ndarray]], n: int) -> np.ndarray:
    fa = np.full(n, -1, dtype=int)
    for fid, (_, va) in enumerate(folds):
        fa[va] = fid
    return fa


def run(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.2,
    n_folds: int = TASK1_N_FOLDS,
    test_df: Optional[pd.DataFrame] = None,
    mode: str = "full",
    resume: bool = False,
    start_stage: Optional[str] = None,
    stop_after_stage: Optional[str] = None,
    run_id: Optional[str] = None,
    skip_optuna: bool = False,
    skip_multiseed: bool = False,
    skip_shap: bool = False,
    audit_only: bool = False,
) -> Dict[str, Any]:
    mode = (mode or "full").lower().strip()
    if mode not in {"quick", "full"}:
        raise ValueError("mode 必须是 quick 或 full")

    output_dir = Path(output_dir)
    run_id = run_id or f"task1_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for sub in (
        "splits",
        "baseline",
        "diagnostics",
        "ablation",
        "oof",
        "tuning",
        "ensemble",
        "stability",
        "explain",
        "final_validation",
        "paper",
        "figures",
        "checkpoints",
        "logs",
        "resume_audit",
    ):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    setup_file_logging(output_dir / "logs" / f"task1_{run_id}.log", run_id)
    state = RunState(output_dir / "run_state.json", run_id)
    state.data["exit_status"] = "running"
    state.save()

    logger.info("=" * 60)
    logger.info(
        "任务一开始 | mode=%s resume=%s run_id=%s start=%s stop=%s",
        mode,
        resume,
        run_id,
        start_stage,
        stop_after_stage,
    )

    audit_summary = run_audit(output_dir)
    if audit_only:
        logger.info("audit-only 完成，退出")
        state.data["exit_status"] = "audit_only"
        state.save()
        return {"task_name": "task1", "accuracy": float("nan"), "audit": audit_summary, "mode": mode}

    opts = RunOptions(
        mode=mode,
        resume=resume,
        start_stage=start_stage,
        stop_after_stage=stop_after_stage,
        run_id=run_id,
        skip_optuna=skip_optuna,
        skip_multiseed=skip_multiseed,
        skip_shap=skip_shap,
        audit_only=False,
        completed_stages=completed_from_audit(audit_summary),
    )
    # resume 默认跳过已完成 optuna
    if resume and not skip_optuna:
        if "catboost_optuna" in opts.completed_stages and "lightgbm_optuna" in opts.completed_stages:
            opts.skip_optuna = True
            logger.info("resume: auto --skip-optuna (CSV trials complete)")
    if resume and start_stage is None:
        opts.start_stage = audit_summary.get("first_resume_stage") or "m1_catboost"
        logger.info("resume: auto start-stage=%s", opts.start_stage)
    gate = StageGate(opts, state)

    if TASK1_TARGET not in df.columns:
        raise ValueError(f"缺少目标列 {TASK1_TARGET}")
    missing = [c for c in TASK1_OFFICIAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError("缺少官方特征：" + ", ".join(missing))

    y_all = _normalize_y(df)
    current_stage = "split"
    try:
        return _run_body(
            df=df,
            y_all=y_all,
            output_dir=output_dir,
            test_size=test_size,
            n_folds=n_folds,
            test_df=test_df,
            mode=mode,
            opts=opts,
            gate=gate,
            state=state,
            run_id=run_id,
        )
    except KeyboardInterrupt:
        state.data["exit_status"] = "interrupted"
        # 标记最近 running stage 为 partial
        for st_name, st_info in list(state.data.get("stages", {}).items()):
            if st_info.get("status") == "running":
                state.partial(st_name, error="KeyboardInterrupt")
                break
        state.save()
        logger.error("KeyboardInterrupt — exit_status=interrupted")
        raise
    except Exception:
        for st_name, st_info in list(state.data.get("stages", {}).items()):
            if st_info.get("status") == "running":
                state.fail(st_name, "exception")
                break
        state.data["exit_status"] = "failed"
        state.save()
        logger.exception("任务一异常终止 | exit_status=failed")
        raise


def _run_body(
    *,
    df: pd.DataFrame,
    y_all: pd.Series,
    output_dir: Path,
    test_size: float,
    n_folds: int,
    test_df: Optional[pd.DataFrame],
    mode: str,
    opts: RunOptions,
    gate: StageGate,
    state: RunState,
    run_id: str,
) -> Dict[str, Any]:
    save_json(uncertain_feature_report(df), output_dir / "diagnostics" / "uncertain_features.json")

    # ---- 冻结划分 ----
    split_ok = (
        (output_dir / "splits" / "develop_indices.csv").exists()
        and (output_dir / "splits" / "final_validation_indices.csv").exists()
        and (output_dir / "splits" / "fold_assignment_seed42.csv").exists()
    )
    if opts.resume and split_ok:
        gate.begin("split", expected=["splits/develop_indices.csv"], conf_hash=config_hash({"seed": RANDOM_SEED}))
        logger.info("resume: load frozen splits from disk (do not re-split)")
        idx_dev = pd.Series(
            pd.read_csv(output_dir / "splits" / "develop_indices.csv")["sample_index"].astype(int).tolist()
        )
        idx_val = pd.Series(
            pd.read_csv(output_dir / "splits" / "final_validation_indices.csv")["sample_index"].astype(int).tolist()
        )
        df_dev, df_val = df.loc[idx_dev].copy(), df.loc[idx_val].copy()
        y_dev, y_val = y_all.loc[idx_dev], y_all.loc[idx_val]
        y_dev_np = y_dev.reset_index(drop=True)
        y_val_np = y_val.reset_index(drop=True)
        folds42 = make_folds(y_dev, n_folds, RANDOM_SEED)
        # 校验 fold 与磁盘一致
        fold_disk = pd.read_csv(output_dir / "splits" / "fold_assignment_seed42.csv")
        fa = _fold_assign_array(folds42, len(y_dev))
        if not np.array_equal(fa, fold_disk["fold"].to_numpy()):
            raise RuntimeError("fold_assignment_seed42 与当前 make_folds(seed=42) 不一致，拒绝续跑")
        gate.complete("split", note="loaded_frozen")
    elif gate.should_execute("split"):
        gate.begin("split", expected=["splits/develop_indices.csv"])
        idx_dev, idx_val = train_test_split(
            df.index.to_series(),
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=y_all,
        )
        df_dev, df_val = df.loc[idx_dev].copy(), df.loc[idx_val].copy()
        y_dev, y_val = y_all.loc[idx_dev], y_all.loc[idx_val]
        y_dev_np = y_dev.reset_index(drop=True)
        y_val_np = y_val.reset_index(drop=True)
        folds42 = make_folds(y_dev, n_folds, RANDOM_SEED)
        save_splits(output_dir, idx_dev, idx_val, y_dev, folds42)
        gate.complete("split")
    else:
        raise RuntimeError("无法获得 develop/validation 划分")

    sample_index_dev = idx_dev.reset_index(drop=True).astype(int).to_numpy()
    fold_assign = _fold_assign_array(folds42, len(y_dev))

    feature_sets = build_feature_sets(df)
    save_json({k: v for k, v in feature_sets.items()}, output_dir / "diagnostics" / "feature_sets.json")

    module_scores: Dict[str, Dict[str, Any]] = {}
    model_probas: Dict[str, np.ndarray] = {}
    optuna_db = output_dir / "tuning" / "task1_optuna.db"

    # ---- 基线复现：原稳定逻辑 B / 旧机制C / A，取最优 ----
    baseline_params = default_catboost_params(RANDOM_SEED)
    best_base_pack = None
    m_b_val: Dict[str, Any] = {}
    it_base = 200

    if not gate.should_execute("baseline") and (output_dir / "baseline" / "baseline_oof_predictions.csv").exists():
        logger.info("load baseline artifacts")
        base_metrics = json.loads((output_dir / "baseline" / "baseline_metrics.json").read_text(encoding="utf-8"))
        oof_p = _load_oof_proba(
            output_dir / "baseline" / "baseline_oof_predictions.csv", len(y_dev), sample_index_dev
        )
        best_base_pack = {
            "name": base_metrics.get("selected_configuration", "B_safe_raw"),
            "features": base_metrics.get("features", feature_sets["B_safe_raw"]),
            "oof_p": oof_p,
            "oof_pred": pred_from_proba(oof_p),
            "fold_rows": pd.read_csv(output_dir / "baseline" / "baseline_fold_metrics.csv").to_dict("records"),
            "iters": [it_base],
            "metrics": base_metrics.get("oof_metrics", {"accuracy": base_metrics.get("oof_accuracy", 0.0)}),
        }
        m_b_val = base_metrics.get("final_validation", {})
        model_probas["baseline"] = oof_p
        module_scores["baseline_stable"] = {
            "accuracy": best_base_pack["metrics"]["accuracy"],
            "note": f"loaded {best_base_pack['name']}",
        }
    else:
        gate.begin("baseline")
        logger.info("=== 复现稳定基线 ===")
        from src.config import TASK1_MECHANISM_FEATURES
        from src.tasks.task1.catboost_models import _align_proba

        probe = add_mechanism_features(df_dev.head(3))
        feats_old_c = list(feature_sets["B_safe_raw"]) + [
            f for f in TASK1_MECHANISM_FEATURES if f in probe.columns and f not in feature_sets["B_safe_raw"]
        ]
        baseline_candidates = {
            "A_official": feature_sets["A_official"],
            "B_safe_raw": feature_sets["B_safe_raw"],
            "C_stable_mech": feats_old_c,
        }
        base_rows = []
        best_base = None
        best_base_pack = None
        for name, feats in baseline_candidates.items():
            oof_p, oof_pred, fold_rows, iters = oof_catboost_multiclass(
                df_dev, y_dev, feats, folds42, baseline_params, tag=f"baseline_{name}"
            )
            m = _oof_bundle(y_dev, oof_pred, oof_p, fold_rows)
            base_rows.append({"configuration": name, "n_features": len(feats), **m})
            pack = {
                "name": name,
                "features": feats,
                "oof_p": oof_p,
                "oof_pred": oof_pred,
                "fold_rows": fold_rows,
                "iters": iters,
                "metrics": m,
            }
            if best_base is None or m["accuracy"] > best_base["accuracy"] + TASK1_TIE_EPS:
                best_base, best_base_pack = m, pack
            elif abs(m["accuracy"] - best_base["accuracy"]) <= TASK1_TIE_EPS and m["macro_f1"] > best_base["macro_f1"]:
                best_base, best_base_pack = m, pack
        assert best_base_pack is not None
        pd.DataFrame(base_rows).to_csv(
            output_dir / "baseline" / "baseline_config_compare.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(best_base_pack["fold_rows"]).to_csv(
            output_dir / "baseline" / "baseline_fold_metrics.csv", index=False, encoding="utf-8-sig"
        )
        persist_assembled_oof(
            output_dir / "baseline" / "baseline_oof_predictions.csv",
            sample_index_dev,
            y_dev_np,
            best_base_pack["oof_pred"],
            best_base_pack["oof_p"],
            fold_assign,
        )
        cm, cm_df = make_confusion_matrix_df(y_dev_np, best_base_pack["oof_pred"], TASK1_LABEL_ORDER)
        cm_df.to_csv(output_dir / "baseline" / "baseline_confusion_matrix.csv", encoding="utf-8-sig")
        classification_report_df(y_dev_np, best_base_pack["oof_pred"]).to_csv(
            output_dir / "baseline" / "baseline_classification_report.csv", index=False, encoding="utf-8-sig"
        )
        it_base = max(int(np.median(best_base_pack["iters"])), 200)
        state_b = fit_cat_preprocessor(df_dev, best_base_pack["features"])
        Xtr_b, Xva_b = transform_cat(df_dev, state_b), transform_cat(df_val, state_b)
        model_b = CatBoostClassifier(**{**baseline_params, "iterations": it_base})
        model_b.fit(Xtr_b, y_dev_np, cat_features=cat_indices(state_b))
        p_b_val = _align_proba(model_b, Xva_b)
        pred_b_val = pred_from_proba(p_b_val)
        m_b_val = evaluate_multiclass(y_val_np, pred_b_val, p_b_val)
        save_json(
            {
                "selected_configuration": best_base_pack["name"],
                "oof_accuracy": best_base_pack["metrics"]["accuracy"],
                "oof_metrics": best_base_pack["metrics"],
                "final_validation": m_b_val,
                "n_features": len(best_base_pack["features"]),
                "features": best_base_pack["features"],
            },
            output_dir / "baseline" / "baseline_metrics.json",
        )
        module_scores["baseline_stable"] = {
            "accuracy": best_base_pack["metrics"]["accuracy"],
            "note": f"reproduced {best_base_pack['name']}",
        }
        model_probas["baseline"] = best_base_pack["oof_p"]
        logger.info(
            "基线选定 %s | OOF Acc=%.4f | Final Acc=%.4f",
            best_base_pack["name"],
            best_base_pack["metrics"]["accuracy"],
            m_b_val["accuracy"],
        )
        gate.complete("baseline")

    assert best_base_pack is not None
    from src.tasks.task1.catboost_models import _align_proba  # noqa: F401

    # ---- 特征消融 / 选择 ----
    feat_sel_path = output_dir / "ablation" / "feature_selection.json"
    if not gate.should_execute("ablation") and not gate.should_execute("feature_selection") and feat_sel_path.exists():
        logger.info("load feature_selection.json")
        fs = json.loads(feat_sel_path.read_text(encoding="utf-8"))
        feat_final = list(fs["final_features"])
        kept_groups = fs.get("kept_groups", [])
        rejected_groups = fs.get("rejected_groups", [])
        abl_df = pd.read_csv(output_dir / "ablation" / "feature_ablation.csv")
    else:
        if gate.should_execute("ablation"):
            gate.begin("ablation")
            logger.info("=== 特征消融 ===")
            abl_rows = []
            sets_to_run = dict(feature_sets)
            if mode == "quick":
                keep_keys = ["A_official", "B_safe_raw", "B+all_mech"]
                for k in list(feature_sets):
                    if k.startswith("B+C") and len(keep_keys) < 6:
                        keep_keys.append(k)
                sets_to_run = {k: feature_sets[k] for k in keep_keys if k in feature_sets}
            for name, feats in sets_to_run.items():
                oof_p, oof_pred, fold_rows, _ = oof_catboost_multiclass(
                    df_dev, y_dev, feats, folds42, baseline_params, tag=f"abl_{name}"
                )
                m = _oof_bundle(y_dev, oof_pred, oof_p, fold_rows)
                abl_rows.append({"feature_set": name, "n_features": len(feats), **m})
            abl_df = pd.DataFrame(abl_rows)
            if "oof_accuracy" not in abl_df.columns:
                abl_df["oof_accuracy"] = abl_df["accuracy"]
            if (abl_df["feature_set"] == "B_safe_raw").any():
                base_acc = float(abl_df.loc[abl_df["feature_set"] == "B_safe_raw", "oof_accuracy"].iloc[0])
            else:
                base_acc = float("nan")
            abl_df["lift_vs_B"] = abl_df["oof_accuracy"] - base_acc
            abl_df.to_csv(output_dir / "ablation" / "feature_ablation.csv", index=False, encoding="utf-8-sig")
            gate.complete("ablation")
        else:
            abl_df = pd.read_csv(output_dir / "ablation" / "feature_ablation.csv")

        gate.begin("feature_selection")
        kept_groups, rejected_groups = select_kept_groups(abl_df)
        feats_kept = features_from_kept_groups(df, kept_groups) if kept_groups else list(feature_sets["B_safe_raw"])
        feats_all = feature_sets["B+all_mech"]
        feat_final = feats_kept
        best_feat_acc = -1.0
        for tag, feats in [
            ("B+kept", feats_kept),
            ("B+all_mech", feats_all),
            ("B_safe_raw", feature_sets["B_safe_raw"]),
        ]:
            oof_p, oof_pred, fold_rows, _ = oof_catboost_multiclass(
                df_dev, y_dev, feats, folds42, baseline_params, tag=f"featpick_{tag}"
            )
            acc = float((oof_pred == y_dev_np.to_numpy()).mean())
            if acc > best_feat_acc + 1e-12:
                best_feat_acc = acc
                feat_final = feats
        save_json(
            {"kept_groups": kept_groups, "rejected_groups": rejected_groups, "final_features": feat_final},
            feat_sel_path,
        )
        logger.info("最终特征数=%d | kept_groups=%s", len(feat_final), kept_groups)
        gate.complete("feature_selection")

    # ---- CatBoost Optuna + confirmation ----
    m1_params_path = output_dir / "tuning" / "m1_best_params.json"
    trials_path = output_dir / "tuning" / "optuna_catboost_trials.csv"
    top5_path = output_dir / "tuning" / "optuna_catboost_top5_cv.csv"

    if mode == "full" and gate.should_execute("catboost_optuna"):
        gate.begin("catboost_optuna")
        trials_df, _ = optuna_search_catboost(
            df_dev,
            y_dev,
            feat_final,
            RANDOM_SEED,
            storage_path=optuna_db,
            study_name="task1_catboost",
            existing_trials_csv=trials_path if trials_path.exists() else None,
        )
        trials_df.to_csv(trials_path, index=False, encoding="utf-8-sig")
        gate.complete("catboost_optuna", completed_trials=list(range(len(trials_df))))
    elif trials_path.exists():
        trials_df = pd.read_csv(trials_path)
        logger.info("reuse catboost trials csv n=%d", len(trials_df))
    else:
        trials_df = pd.DataFrame()

    if mode == "full" and gate.should_execute("catboost_confirmation"):
        gate.begin("catboost_confirmation")
        confirm_df = confirm_top_catboost(df_dev, y_dev, feat_final, trials_df, folds42, top_k=5)
        confirm_df.to_csv(top5_path, index=False, encoding="utf-8-sig")
        gate.complete("catboost_confirmation")
    elif top5_path.exists():
        confirm_df = pd.read_csv(top5_path)
    else:
        confirm_df = pd.DataFrame()

    if m1_params_path.exists() and (opts.resume or not gate.should_execute("m1_catboost")):
        m1_params = json.loads(m1_params_path.read_text(encoding="utf-8"))
        logger.info("load m1_best_params.json")
    elif mode == "full" and len(confirm_df):
        best_row = confirm_df.iloc[0]
        m1_params = default_catboost_params(RANDOM_SEED)
        m1_params["depth"] = int(best_row["depth"])
        m1_params["learning_rate"] = float(best_row["learning_rate"])
        m1_params["l2_leaf_reg"] = float(best_row["l2_leaf_reg"])
        m1_params["iterations"] = int(best_row["iterations"])
        m1_params["border_count"] = int(best_row["border_count"])
        m1_params["random_strength"] = float(best_row["random_strength"])
        m1_params["bagging_temperature"] = float(best_row["bagging_temperature"])
    else:
        m1_params = default_catboost_params(RANDOM_SEED)
        if mode == "quick":
            pd.DataFrame([m1_params]).to_csv(
                output_dir / "tuning" / "optuna_catboost_skipped_quick.csv", index=False
            )

    # ---- M1 OOF（checkpoint / 已落盘则直接加载）----
    m1_oof_path = output_dir / "oof" / "m1_oof_predictions.csv"
    gate.begin("m1_catboost")
    iters_m1: List[int] = []
    if m1_oof_path.exists() and opts.resume:
        logger.info("=== M1: load existing OOF ===")
        oof_m1 = _load_oof_proba(m1_oof_path, len(y_dev), sample_index_dev)
        pred_m1 = pred_from_proba(oof_m1)
        fold_m1 = []
        iters_m1 = [int(m1_params.get("iterations", 500))]
    else:
        logger.info("=== M1 CatBoost OOF (best params, 5-fold) ===")
        oof_m1, pred_m1, fold_m1, iters_m1 = oof_catboost_multiclass(
            df_dev,
            y_dev,
            feat_final,
            folds42,
            m1_params,
            tag="M1_catboost",
            checkpoint_dir=output_dir / "checkpoints" / "m1",
            sample_index=sample_index_dev,
            resume_folds=True,
        )
        persist_assembled_oof(m1_oof_path, sample_index_dev, y_dev_np, pred_m1, oof_m1, fold_assign)
        persist_assembled_oof(
            output_dir / "checkpoints" / "m1" / "oof_predictions.csv",
            sample_index_dev,
            y_dev_np,
            pred_m1,
            oof_m1,
            fold_assign,
        )
    m_m1 = _oof_bundle(y_dev, pred_m1, oof_m1, fold_m1)
    module_scores["M1_catboost"] = {"accuracy": m_m1["accuracy"], "note": "best single CatBoost"}
    model_probas["M1_catboost"] = oof_m1
    atomic_write_json(m1_params_path, m1_params)
    gate.complete("m1_catboost", completed_folds=list(range(n_folds)))
    if opts.stop_after_stage == "m1_catboost":
        state.data["exit_status"] = "stopped_after_stage"
        state.save()
        return {"task_name": "task1", "accuracy": float("nan"), "stopped_after": "m1_catboost", "mode": mode}

    # ---- M2 ----
    m2_oof_path = output_dir / "oof" / "m2_oof_predictions.csv"
    gate.begin("m2_ordinal")
    if m2_oof_path.exists() and opts.resume:
        logger.info("=== M2: load existing OOF ===")
        oof_m2 = _load_oof_proba(m2_oof_path, len(y_dev), sample_index_dev)
        pred_m2 = pred_from_proba(oof_m2)
        fold_m2 = []
    else:
        logger.info("=== M2 Ordinal dual-boundary ===")
        oof_m2, pred_m2, fold_m2 = oof_ordinal_catboost(
            df_dev,
            y_dev,
            feat_final,
            folds42,
            checkpoint_dir=output_dir / "checkpoints" / "m2",
            sample_index=sample_index_dev,
            resume_folds=True,
        )
        persist_assembled_oof(
            m2_oof_path, sample_index_dev, y_dev_np, pred_m2, oof_m2, fold_assign
        )
    m_m2 = _oof_bundle(y_dev, pred_m2, oof_m2, fold_m2)
    module_scores["M2_ordinal"] = {"accuracy": m_m2["accuracy"], "note": "dual boundary"}
    model_probas["M2_ordinal"] = oof_m2
    gate.complete("m2_ordinal", completed_folds=list(range(n_folds)))
    if opts.stop_after_stage == "m2_ordinal":
        state.data["exit_status"] = "stopped_after_stage"
        state.save()
        return {"task_name": "task1", "accuracy": float("nan"), "stopped_after": "m2_ordinal", "mode": mode}

    # ---- M3 ----
    m3_oof_path = output_dir / "oof" / "m3_oof_predictions.csv"
    m3_score_path = output_dir / "oof" / "m3_oof_scores.csv"
    m3_thr_path = output_dir / "tuning" / "m3_threshold_temp.json"
    gate.begin("m3_regression")
    if m3_oof_path.exists() and m3_thr_path.exists() and opts.resume:
        logger.info("=== M3: load existing OOF / thresholds ===")
        oof_m3 = _load_oof_proba(m3_oof_path, len(y_dev), sample_index_dev)
        thr_pack = json.loads(m3_thr_path.read_text(encoding="utf-8"))
        thr_best, soft_best_m3 = thr_pack.get("hard", {}), thr_pack.get("soft", {})
        m_m3 = evaluate_multiclass(y_dev_np, pred_from_proba(oof_m3), oof_m3)
    else:
        logger.info("=== M3 Ordinal regression ===")
        oof_score, _ = oof_ordinal_regressor(df_dev, y_dev, feat_final, folds42)
        thr_best, pred_hard = search_reg_thresholds(y_dev_np, oof_score, coarse=(mode == "quick"))
        soft_best_m3, oof_m3 = search_reg_soft_proba(y_dev_np, oof_score, coarse=(mode == "quick"))
        m_m3 = evaluate_multiclass(y_dev_np, pred_from_proba(oof_m3), oof_m3)
        atomic_write_json(m3_thr_path, {"hard": thr_best, "soft": soft_best_m3})
        atomic_write_csv(
            m3_score_path,
            pd.DataFrame({"sample_index": sample_index_dev, "score": oof_score, "fold": fold_assign}),
        )
        persist_assembled_oof(
            m3_oof_path,
            sample_index_dev,
            y_dev_np,
            pred_from_proba(oof_m3),
            oof_m3,
            fold_assign,
        )
    module_scores["M3_reg_soft"] = {
        "accuracy": m_m3["accuracy"],
        "note": f"T={soft_best_m3.get('temperature') if isinstance(soft_best_m3, dict) else soft_best_m3}",
    }
    module_scores["M3_reg_hard"] = {
        "accuracy": thr_best.get("accuracy", float("nan")) if isinstance(thr_best, dict) else float("nan"),
        "note": "threshold mapping",
    }
    model_probas["M3_reg_soft"] = oof_m3
    gate.complete("m3_regression", completed_folds=list(range(n_folds)))

    # ---- LightGBM optuna + OOF ----
    lgb_params = None
    lgb_trials_path = output_dir / "tuning" / "optuna_lgbm_trials.csv"
    if HAS_LGBM:
        if mode == "full" and gate.should_execute("lightgbm_optuna"):
            gate.begin("lightgbm_optuna")
            lgb_trials, lgb_params = optuna_lgbm(
                df_dev,
                y_dev,
                feat_final,
                RANDOM_SEED,
                storage_path=optuna_db,
                study_name="task1_lightgbm",
                existing_trials_csv=lgb_trials_path if lgb_trials_path.exists() else None,
            )
            lgb_trials.to_csv(lgb_trials_path, index=False, encoding="utf-8-sig")
            gate.complete("lightgbm_optuna", completed_trials=list(range(len(lgb_trials))))
        elif lgb_trials_path.exists():
            lgb_trials, lgb_params = optuna_lgbm(
                df_dev,
                y_dev,
                feat_final,
                RANDOM_SEED,
                existing_trials_csv=lgb_trials_path,
            )
            logger.info("reuse lightgbm trials csv")
        gate.begin("lightgbm_model")
        logger.info("=== M4 LightGBM OOF ===")
        oof_m4, pred_m4, fold_m4 = oof_lgbm(
            df_dev,
            y_dev,
            feat_final,
            folds42,
            lgb_params,
            checkpoint_dir=output_dir / "checkpoints" / "lgbm",
            sample_index=sample_index_dev,
            resume_folds=True,
        )
        m_m4 = _oof_bundle(y_dev, pred_m4, oof_m4, fold_m4)
        module_scores["M4_lgbm"] = {"accuracy": m_m4["accuracy"], "note": "lightgbm"}
        model_probas["M4_lgbm"] = oof_m4
        persist_assembled_oof(
            output_dir / "oof" / "lgbm_oof_predictions.csv",
            sample_index_dev,
            y_dev_np,
            pred_m4,
            oof_m4,
            fold_assign,
        )
        gate.complete("lightgbm_model", completed_folds=list(range(n_folds)))
    else:
        logger.warning("跳过 LightGBM（未安装）")

    if HAS_XGB and gate.should_execute("xgboost"):
        gate.begin("xgboost")
        logger.info("=== M5 XGBoost ===")
        oof_m5, pred_m5, fold_m5 = oof_xgboost(
            df_dev,
            y_dev,
            feat_final,
            folds42,
            checkpoint_dir=output_dir / "checkpoints" / "xgb",
            sample_index=sample_index_dev,
            resume_folds=True,
        )
        m_m5 = _oof_bundle(y_dev, pred_m5, oof_m5, fold_m5)
        module_scores["M5_xgb"] = {"accuracy": m_m5["accuracy"], "note": "xgboost fixed"}
        model_probas["M5_xgb"] = oof_m5
        persist_assembled_oof(
            output_dir / "oof" / "xgb_oof_predictions.csv",
            sample_index_dev,
            y_dev_np,
            pred_m5,
            oof_m5,
            fold_assign,
        )
        best_so_far = max(v["accuracy"] for v in module_scores.values())
        if m_m5["accuracy"] + 0.02 < best_so_far:
            logger.info("XGB 落后 >0.02，停止进一步调参，保留对照")
        gate.complete("xgboost")
    elif not HAS_XGB:
        logger.warning("跳过 XGBoost（未安装）")

    gate.begin("logistic")
    logger.info("=== M6 Ordinal Logistic ===")
    oof_m6, pred_m6, fold_m6, coef_df = oof_ordinal_logistic(
        df_dev,
        y_dev,
        feat_final,
        folds42,
        checkpoint_dir=output_dir / "checkpoints" / "m6",
        sample_index=sample_index_dev,
        resume_folds=True,
    )
    m_m6 = _oof_bundle(y_dev, pred_m6, oof_m6, fold_m6)
    module_scores["M6_ord_logit"] = {"accuracy": m_m6["accuracy"], "note": "interpretable baseline"}
    model_probas["M6_ord_logit"] = oof_m6
    coef_df.to_csv(output_dir / "oof" / "m6_coefficients.csv", index=False, encoding="utf-8-sig")
    persist_assembled_oof(
        output_dir / "oof" / "m6_oof_predictions.csv",
        sample_index_dev,
        y_dev_np,
        pred_m6,
        oof_m6,
        fold_assign,
    )
    gate.complete("logistic")

    # ---- 互补性 / 软融合 / stacking / bias ----
    locked_cfg_path = output_dir / "task1_final_config.json"
    soft_path = output_dir / "ensemble" / "soft_vote_best.json"
    bias_path = output_dir / "ensemble" / "class_bias_best.json"
    pick_path = output_dir / "oof" / "final_candidates.csv"
    resume_locked = (
        opts.resume
        and opts.start_stage in {"final_validation", "shap", "report"}
        and locked_cfg_path.exists()
        and soft_path.exists()
        and pick_path.exists()
    )

    if resume_locked:
        logger.info("resume locked config -> skip ensemble/multiseed recompute")
        locked = json.loads(locked_cfg_path.read_text(encoding="utf-8"))
        soft_best = json.loads(soft_path.read_text(encoding="utf-8"))
        bias_best = (
            json.loads(bias_path.read_text(encoding="utf-8"))
            if bias_path.exists()
            else {"bias_Low": 0.0, "bias_Medium": 0.0, "bias_High": 0.0}
        )
        # 用已保存权重重建 soft_p
        weights = soft_best.get("weights", {})
        parts, wsum = [], 0.0
        for n, w in weights.items():
            if n in model_probas and float(w) > 0:
                parts.append(float(w) * model_probas[n])
                wsum += float(w)
        soft_p = sum(parts) / wsum if wsum > 0 else oof_m1
        model_probas["soft_vote"] = soft_p
        use_stack = bool(locked.get("use_stacking", False))
        stack_p = soft_p
        cand_name = "soft_vote"
        if locked.get("class_bias"):
            cand_name = locked["class_bias"].get("applied_on", "soft_vote")
        bias_p = apply_bias(
            model_probas.get(cand_name, soft_p),
            [
                float(bias_best.get("bias_Low", 0.0)),
                float(bias_best.get("bias_Medium", 0.0)),
                float(bias_best.get("bias_High", 0.0)),
            ],
        )
        chosen_name = str(locked.get("chosen_oof_name", "soft_vote"))
        final_candidates = {
            "baseline": model_probas["baseline"],
            "M1_catboost": oof_m1,
            "M2_ordinal": oof_m2,
            "soft_vote": soft_p,
            "class_bias": bias_p,
        }
        chosen_p = final_candidates.get(chosen_name, soft_p)
        chosen_m = evaluate_multiclass(y_dev_np, pred_from_proba(chosen_p), chosen_p)
        stable_ok = bool(locked.get("stable_ok", False))
        chosen_stab = pd.DataFrame()
        base_stab = pd.DataFrame()
        stab_sum = pd.DataFrame()
        stab_df = pd.DataFrame()
        logger.info("OOF 锁定候选（resume）：%s Acc=%.4f", chosen_name, chosen_m["accuracy"])
        for st_name in ("diversity", "soft_blend", "stacking", "class_bias", "multiseed"):
            gate.begin(st_name)
            gate.complete(st_name, note="skipped_locked_resume")
    else:
        gate.begin("diversity")
        logger.info("=== Ensemble ===")
        div_df = diversity_table(model_probas, y_dev_np)
        div_df.to_csv(output_dir / "ensemble" / "diversity_summary.csv", index=False, encoding="utf-8-sig")
        gate.complete("diversity")

        # 融合候选：排除明显弱且无互补价值的模型（仍保留 diversity 表）
        fuse_models = {
            k: v
            for k, v in model_probas.items()
            if evaluate_multiclass(y_dev_np, pred_from_proba(v), v)["accuracy"]
            >= best_base_pack["metrics"]["accuracy"] - 0.02
        }
        gate.begin("soft_blend")
        soft_best, soft_p, soft_grid = soft_vote_search(
            fuse_models,
            y_dev_np,
            max_models=3 if mode == "quick" else 4,
            step=0.1 if mode == "quick" else 0.05,
        )
        soft_grid.to_csv(output_dir / "ensemble" / "soft_vote_grid.csv", index=False, encoding="utf-8-sig")
        module_scores["soft_vote"] = {"accuracy": soft_best["accuracy"], "note": str(soft_best.get("weights"))}
        model_probas["soft_vote"] = soft_p
        save_json(soft_best, output_dir / "ensemble" / "soft_vote_best.json")
        gate.complete("soft_blend")

        gate.begin("stacking")
        stack_p, stack_m = stacking_nested_oof(fuse_models, y_dev_np)
        module_scores["stacking"] = {"accuracy": stack_m["accuracy"], "note": "nested OOF meta LR"}
        save_json(stack_m, output_dir / "ensemble" / "stacking_metrics.json")
        use_stack = stack_m["accuracy"] > soft_best["accuracy"] + 1e-12
        if use_stack:
            model_probas["stacking"] = stack_p
            logger.info("Stacking 优于软融合，纳入候选")
        else:
            logger.info("Stacking 未优于软融合，仅作论文对照")
        gate.complete("stacking")

        gate.begin("class_bias")
        cand_name = max(
            [("M1_catboost", oof_m1), ("soft_vote", soft_p)]
            + ([("stacking", stack_p)] if use_stack else [])
            + ([("M2_ordinal", oof_m2)] if m_m2["accuracy"] >= m_m1["accuracy"] - 0.005 else []),
            key=lambda kv: evaluate_multiclass(y_dev_np, pred_from_proba(kv[1]), kv[1])["accuracy"],
        )[0]
        bias_in = model_probas[cand_name]
        bias_best, bias_p = search_class_bias(bias_in, y_dev_np)
        module_scores["class_bias"] = {"accuracy": bias_best["accuracy"], "note": f"on {cand_name}"}
        save_json(bias_best, output_dir / "ensemble" / "class_bias_best.json")
        gate.complete("class_bias")
        # ---- 选择最终 OOF 方案 ----
        final_candidates = {
            "baseline": model_probas["baseline"],
            "M1_catboost": oof_m1,
            "M2_ordinal": oof_m2,
            "soft_vote": soft_p,
            "class_bias": bias_p,
        }
        if use_stack:
            final_candidates["stacking"] = stack_p
        pick_rows = []
        for name, p in final_candidates.items():
            met = evaluate_multiclass(y_dev_np, pred_from_proba(p), p)
            pick_rows.append({"name": name, **met})
        pick_df = pd.DataFrame(pick_rows).sort_values(
            ["accuracy", "macro_f1", "qwk", "medium_recall", "cross_two_level_errors"],
            ascending=[False, False, False, False, True],
        )
        pick_df.to_csv(output_dir / "oof" / "final_candidates.csv", index=False, encoding="utf-8-sig")
        chosen_name = str(pick_df.iloc[0]["name"])
        chosen_p = final_candidates[chosen_name]
        chosen_m = evaluate_multiclass(y_dev_np, pred_from_proba(chosen_p), chosen_p)
        logger.info("OOF 锁定候选：%s Acc=%.4f", chosen_name, chosen_m["accuracy"])

    # ---- 多 seed 稳定性 ----
    stab_focus = ["baseline", "M1_catboost", "M2_ordinal", "soft_vote"]
    if chosen_name == "stacking":
        stab_focus.append("stacking")
    if chosen_name == "class_bias":
        stab_focus.append("class_bias")
    chosen_stab_key = "soft_vote" if chosen_name == "class_bias" else chosen_name
    stab_rows = []
    if resume_locked:
        logger.info("skip multiseed (locked resume; stable_ok=%s)", stable_ok)
    elif not gate.should_execute("multiseed"):
        logger.info("skip multiseed")
        stable_ok = False
        chosen_stab = pd.DataFrame()
        base_stab = pd.DataFrame()
        stab_df = pd.DataFrame()
        stab_sum = pd.DataFrame()
    else:
        gate.begin("multiseed")
        logger.info("=== 多 seed 稳定性 ===")
        stable_ok = False
        stab_df = pd.DataFrame()
        stab_sum = pd.DataFrame()
    for seed in (
        []
        if resume_locked or not gate.should_execute("multiseed")
        else (TASK1_STABILITY_SEEDS if mode == "full" else [RANDOM_SEED])
    ):
        folds_s = make_folds(y_dev, n_folds, seed)
        # 各模型 OOF
        p_base_s, _, _, _ = oof_catboost_multiclass(
            df_dev, y_dev, best_base_pack["features"], folds_s, {**baseline_params, "random_seed": seed}, tag=f"base_s{seed}"
        )
        p_m1_s, _, _, _ = oof_catboost_multiclass(
            df_dev, y_dev, feat_final, folds_s, {**m1_params, "random_seed": seed}, tag=f"m1_s{seed}"
        )
        p_m2_s, _, _ = oof_ordinal_catboost(
            df_dev, y_dev, feat_final, folds_s, params={"random_seed": seed}, tag=f"m2_s{seed}"
        )
        # soft: 用当前 seed 下可用树模型重搜权重太贵；改用固定权重作用于 m1/m2/base
        weights = soft_best.get("weights", {})
        parts = []
        wsum = 0.0
        mapping = {
            "baseline": p_base_s,
            "M1_catboost": p_m1_s,
            "M2_ordinal": p_m2_s,
            "M3_reg_soft": p_m1_s,  # 回归软概率跨 seed 成本高，退化用 M1 近似
            "M6_ord_logit": p_m2_s,  # 同上，用序数边界近似
        }
        # 若软融合含 lgbm 等，缺失则重归一化
        for name, w in weights.items():
            if name in mapping and w > 0:
                parts.append(w * mapping[name])
                wsum += w
        if wsum <= 0:
            p_soft_s = p_m1_s
        else:
            p_soft_s = sum(parts) / wsum
        if chosen_name == "class_bias":
            p_bias_s = apply_bias(
                p_soft_s,
                [
                    bias_best["bias_Low"],
                    bias_best["bias_Medium"],
                    bias_best["bias_High"],
                ],
            )
        else:
            p_bias_s = p_soft_s
        pack_s = {
            "baseline": p_base_s,
            "M1_catboost": p_m1_s,
            "M2_ordinal": p_m2_s,
            "soft_vote": p_soft_s,
            "class_bias": p_bias_s,
        }
        for name in stab_focus:
            if name not in pack_s:
                continue
            p = pack_s[name]
            met = evaluate_multiclass(y_dev_np, pred_from_proba(p), p)
            # 折级
            fold_ids = np.full(len(y_dev_np), -1)
            for fid, (_, va) in enumerate(folds_s):
                fold_ids[va] = fid
            for fid in range(n_folds):
                mask = fold_ids == fid
                fm = evaluate_multiclass(y_dev_np[mask], pred_from_proba(p[mask]), p[mask])
                stab_rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "fold": fid,
                        "accuracy": fm["accuracy"],
                        "seed_oof_accuracy": met["accuracy"],
                    }
                )
    if gate.should_execute("multiseed") or stab_rows:
        stab_df = pd.DataFrame(stab_rows)
        stab_df.to_csv(output_dir / "stability" / "multiseed_fold_results.csv", index=False, encoding="utf-8-sig")
        summary_rows = []
        for name in stab_focus:
            sub = stab_df[stab_df["model"] == name]
            if sub.empty:
                continue
            seed_means = sub.groupby("seed")["seed_oof_accuracy"].first()
            fold_acc = sub["accuracy"]
            base_fold = stab_df[stab_df["model"] == "baseline"]["accuracy"].to_numpy()
            this_fold = sub["accuracy"].to_numpy()
            wins = int((this_fold > base_fold[: len(this_fold)] + 1e-12).sum()) if len(base_fold) else 0
            summary_rows.append(
                {
                    "model": name,
                    "mean_fold_acc": float(fold_acc.mean()),
                    "std_fold_acc": float(fold_acc.std(ddof=0)),
                    "mean_seed_oof": float(seed_means.mean()),
                    "std_seed_oof": float(seed_means.std(ddof=0)),
                    "worst_seed": int(seed_means.idxmin()) if len(seed_means) else None,
                    "lift_vs_baseline_mean": float(
                        seed_means.mean()
                        - stab_df[stab_df["model"] == "baseline"]
                        .groupby("seed")["seed_oof_accuracy"]
                        .first()
                        .mean()
                    )
                    if (stab_df["model"] == "baseline").any()
                    else float("nan"),
                    "folds_better_than_baseline": wins,
                    "n_folds": int(len(fold_acc)),
                }
            )
        stab_sum = pd.DataFrame(summary_rows)
        stab_sum.to_csv(output_dir / "stability" / "multiseed_summary.csv", index=False, encoding="utf-8-sig")
        models_in_sum = set(stab_sum["model"]) if len(stab_sum) else set()
        stab_model_key = chosen_name if chosen_name in models_in_sum else chosen_stab_key
        chosen_stab = stab_sum[stab_sum["model"] == stab_model_key] if len(stab_sum) else pd.DataFrame()
        base_stab = stab_sum[stab_sum["model"] == "baseline"] if len(stab_sum) else pd.DataFrame()
        stable_ok = False
        if len(chosen_stab) and len(base_stab):
            cs, bs = chosen_stab.iloc[0], base_stab.iloc[0]
            seed_cmp = []
            for seed in stab_df["seed"].unique():
                a = stab_df[(stab_df["seed"] == seed) & (stab_df["model"] == stab_model_key)][
                    "seed_oof_accuracy"
                ].iloc[0]
                b = stab_df[(stab_df["seed"] == seed) & (stab_df["model"] == "baseline")][
                    "seed_oof_accuracy"
                ].iloc[0]
                seed_cmp.append(a >= b - 1e-12)
            need_seeds = 2 if mode == "full" else 1
            stable_ok = (
                cs["mean_seed_oof"] > bs["mean_seed_oof"] + 1e-12
                and sum(seed_cmp) >= min(need_seeds, len(seed_cmp))
            )
        gate.complete("multiseed")
    elif not resume_locked:
        chosen_stab = pd.DataFrame()
        base_stab = pd.DataFrame()
        stable_ok = False

    # ---- 锁定配置 ----
    final_config = {
        "chosen_oof_name": chosen_name,
        "features": feat_final,
        "m1_params": m1_params,
        "soft_vote_weights": soft_best.get("weights"),
        "class_bias": {
            "bias_Low": bias_best.get("bias_Low"),
            "bias_Medium": bias_best.get("bias_Medium"),
            "bias_High": bias_best.get("bias_High"),
            "applied_on": cand_name,
        }
        if chosen_name == "class_bias"
        else None,
        "use_stacking": bool(use_stack and chosen_name == "stacking"),
        "random_seed": RANDOM_SEED,
        "mode": mode,
        "stable_ok": bool(stable_ok),
    }
    save_json(final_config, output_dir / "task1_final_config.json")
    save_json(final_config, output_dir / "tuning" / "task1_final_config.json")

    # ---- 最终验证只一次 ----
    fv_metrics_path = output_dir / "final_validation" / "metrics.json"
    fv_model_path = output_dir / "final_validation" / "model_m1.cbm"
    skip_final_retrain = (
        opts.resume
        and opts.start_stage in {"shap", "report"}
        and fv_metrics_path.exists()
        and (output_dir / "final_validation" / "predictions.csv").exists()
    )
    it_final = max(int(np.median(iters_m1)) if iters_m1 else 200, 200)
    params_final = {**m1_params, "iterations": it_final, "random_seed": RANDOM_SEED}
    model_final = None
    state_final = None
    cm_df = None
    fi = None
    m_val = {}
    metrics_out: Dict[str, Any] = {}

    if skip_final_retrain:
        gate.begin("final_validation")
        logger.info("=== 最终验证：加载已有结果（跳过重训）===")
        metrics_out = json.loads(fv_metrics_path.read_text(encoding="utf-8"))
        m_val = metrics_out.get("final_validation", {})
        m_b_val = metrics_out.get("baseline_final_validation", m_b_val)
        recommend_merge = bool(metrics_out.get("recommend_merge", False))
        stable_ok = bool(metrics_out.get("stable_ok", stable_ok))
        if fv_model_path.exists():
            model_final = CatBoostClassifier()
            model_final.load_model(str(fv_model_path))
            state_final = fit_cat_preprocessor(df_dev, feat_final)
        cm_df = pd.read_csv(output_dir / "final_validation" / "confusion_matrix.csv", index_col=0)
        fi_path = output_dir / "final_validation" / "feature_importance.csv"
        fi = pd.read_csv(fi_path) if fi_path.exists() else pd.DataFrame()
        gate.complete("final_validation", note="loaded_existing")
    else:
        gate.begin("final_validation")
        logger.info("=== 最终验证（一次）===")
        if not m_b_val or "accuracy" not in m_b_val:
            m_b_val = {"accuracy": float(best_base_pack["metrics"]["accuracy"])}

        def _predict_val_cat(features, params):
            st = fit_cat_preprocessor(df_dev, features)
            Xtr, Xva = transform_cat(df_dev, st), transform_cat(df_val, st)
            mdl = CatBoostClassifier(**params)
            mdl.fit(Xtr, y_dev_np, cat_features=cat_indices(st))
            return _align_proba(mdl, Xva), mdl, st

        p_m1_val, model_final, state_final = _predict_val_cat(feat_final, params_final)
        p_base_val, _, _ = _predict_val_cat(
            best_base_pack["features"], {**baseline_params, "iterations": it_base}
        )

        from src.tasks.task1.ordinal import _binary_labels, ordinal_probs_from_boundaries
        from src.tasks.task1.preprocess import fit_tabular_preprocessor, transform_tabular
        from src.tasks.task1.regression import score_to_soft_proba
        from catboost import CatBoostClassifier as CBC
        from catboost import CatBoostRegressor
        from sklearn.linear_model import LogisticRegression

        ge_med, ge_high = _binary_labels(y_dev_np)
        st = fit_cat_preprocessor(df_dev, feat_final)
        Xtr, Xva = transform_cat(df_dev, st), transform_cat(df_val, st)
        cats = cat_indices(st)
        bm = CBC(
            loss_function="Logloss",
            iterations=it_final,
            depth=6,
            learning_rate=0.05,
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        )
        bh = CBC(
            loss_function="Logloss",
            iterations=it_final,
            depth=6,
            learning_rate=0.05,
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        )
        bm.fit(Xtr, ge_med, cat_features=cats)
        bh.fit(Xtr, ge_high, cat_features=cats)
        p_m2_val = ordinal_probs_from_boundaries(
            bm.predict_proba(Xva)[:, 1], bh.predict_proba(Xva)[:, 1]
        )

        y_ord = y_dev_np.map({"Low": 0, "Medium": 1, "High": 2}).astype(float).to_numpy()
        reg = CatBoostRegressor(
            loss_function="RMSE",
            iterations=it_final,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        )
        reg.fit(Xtr, y_ord, cat_features=cats)
        m3_temp = float(soft_best_m3.get("temperature", 0.5)) if isinstance(soft_best_m3, dict) else 0.5
        p_m3_val = score_to_soft_proba(reg.predict(Xva), m3_temp)

        st_tab = fit_tabular_preprocessor(df_dev, feat_final, scale=True)
        Xtr_t = transform_tabular(df_dev, st_tab)
        Xva_t = transform_tabular(df_val, st_tab)
        m6_a = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        m6_b = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        m6_a.fit(Xtr_t, ge_med)
        m6_b.fit(Xtr_t, ge_high)
        p_m6_val = ordinal_probs_from_boundaries(
            m6_a.predict_proba(Xva_t)[:, 1], m6_b.predict_proba(Xva_t)[:, 1]
        )

        def assemble_val(name: str) -> np.ndarray:
            if name == "baseline":
                return p_base_val
            if name == "M1_catboost":
                return p_m1_val
            if name == "M2_ordinal":
                return p_m2_val
            if name == "M3_reg_soft":
                return p_m3_val
            if name == "M6_ord_logit":
                return p_m6_val
            if name == "soft_vote":
                weights = soft_best.get("weights", {})
                mapping = {
                    "baseline": p_base_val,
                    "M1_catboost": p_m1_val,
                    "M2_ordinal": p_m2_val,
                    "M3_reg_soft": p_m3_val,
                    "M6_ord_logit": p_m6_val,
                }
                parts, wsum = [], 0.0
                for n, w in weights.items():
                    if n in mapping and w > 0:
                        parts.append(float(w) * mapping[n])
                        wsum += float(w)
                return sum(parts) / wsum if wsum > 0 else p_m1_val
            if name == "class_bias":
                base_name = cand_name if cand_name != "class_bias" else "M1_catboost"
                p0 = assemble_val(
                    base_name
                    if base_name
                    in {
                        "baseline",
                        "M1_catboost",
                        "M2_ordinal",
                        "M3_reg_soft",
                        "M6_ord_logit",
                        "soft_vote",
                    }
                    else "M1_catboost"
                )
                return apply_bias(
                    p0,
                    [
                        bias_best["bias_Low"],
                        bias_best["bias_Medium"],
                        bias_best["bias_High"],
                    ],
                )
            if name == "stacking":
                return assemble_val("soft_vote")
            return p_m1_val

        p_val = assemble_val(chosen_name)
        pred_val = pred_from_proba(p_val)
        m_val = evaluate_multiclass(y_val_np, pred_val, p_val)

        better_final = bool(m_val["accuracy"] >= m_b_val["accuracy"] - 0.002)
        stable_ok = bool(stable_ok)
        recommend_merge = bool(stable_ok and better_final and chosen_name != "baseline")

        metrics_out = {
            "mode": mode,
            "selected_configuration": chosen_name,
            "oof_accuracy": float(chosen_m["accuracy"]),
            "oof_metrics": chosen_m,
            "final_validation": m_val,
            "baseline_final_validation": m_b_val,
            "baseline_oof_accuracy": float(best_base_pack["metrics"]["accuracy"]),
            "n_features": int(len(feat_final)),
            "features": list(feat_final),
            "recommend_merge": recommend_merge,
            "stable_ok": stable_ok,
            "module_scores": module_scores,
        }
        save_json(metrics_out, output_dir / "final_validation" / "metrics.json")
        save_json(metrics_out, output_dir / "metrics.json")

        pd.DataFrame(
            {
                "sample_index": idx_val.reset_index(drop=True).astype(int),
                "true_label": y_val_np,
                "predicted_label": pred_val,
                "prob_Low": p_val[:, 0],
                "prob_Medium": p_val[:, 1],
                "prob_High": p_val[:, 2],
            }
        ).to_csv(output_dir / "final_validation" / "predictions.csv", index=False, encoding="utf-8-sig")
        cm, cm_df = make_confusion_matrix_df(y_val_np, pred_val, TASK1_LABEL_ORDER)
        cm_df.to_csv(output_dir / "final_validation" / "confusion_matrix.csv", encoding="utf-8-sig")
        plot_confusion_and_curves(y_val_np, pred_val, p_val, output_dir / "figures", "Task1 Final")

        fi = extract_feature_importance(model_final, state_final.final_feature_order)
        fi.to_csv(output_dir / "final_validation" / "feature_importance.csv", index=False, encoding="utf-8-sig")
        model_final.save_model(str(output_dir / "final_validation" / "model_m1.cbm"))
        gate.complete("final_validation")

    # ---- SHAP（独立子进程，原生崩溃不拖垮主流程）----
    if mode == "full" and gate.should_execute("shap") and not opts.skip_shap:
        gate.begin("shap")
        logger.info("=== SHAP ===")
        try:
            import subprocess
            import sys

            model_path = output_dir / "final_validation" / "model_m1.cbm"
            if not model_path.exists() and model_final is not None:
                model_final.save_model(str(model_path))
            feat_json = output_dir / "tuning" / "shap_features.json"
            params_json = output_dir / "tuning" / "shap_params.json"
            atomic_write_json(feat_json, list(feat_final))
            atomic_write_json(params_json, dict(params_final))
            worker = Path(__file__).resolve().parents[2] / "scripts" / "run_task1_shap_worker.py"
            # parents: task1 -> tasks -> src -> repo? Path(__file__)=.../src/tasks/task1/pipeline.py
            # parents[0]=task1, [1]=tasks, [2]=src, [3]=repo
            worker = Path(__file__).resolve().parents[3] / "scripts" / "run_task1_shap_worker.py"
            cmd = [
                sys.executable,
                str(worker),
                "--data",
                str(Path("dataSetB.csv").resolve() if Path("dataSetB.csv").exists() else "dataSetB.csv"),
                "--model",
                str(model_path.resolve()),
                "--features-json",
                str(feat_json.resolve()),
                "--params-json",
                str(params_json.resolve()),
                "--develop-indices",
                str((output_dir / "splits" / "develop_indices.csv").resolve()),
                "--output-dir",
                str((output_dir / "explain").resolve()),
                "--max-samples",
                "200",
            ]
            # 用绝对 data 路径：优先从 output 旁推仓库根
            repo_root = Path(__file__).resolve().parents[3]
            data_path = repo_root / "dataSetB.csv"
            if data_path.exists():
                cmd[cmd.index("--data") + 1] = str(data_path)
            logger.info("SHAP worker: %s", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if proc.returncode == 0:
                logger.info("SHAP worker ok: %s", (proc.stdout or "")[-200:])
                gate.complete("shap")
            else:
                logger.warning(
                    "SHAP worker failed code=%s stderr=%s",
                    proc.returncode,
                    (proc.stderr or "")[-500:],
                )
                gate.partial("shap", error=f"exitcode={proc.returncode}")
        except subprocess.TimeoutExpired:
            logger.warning("SHAP 超时，继续写报告")
            gate.partial("shap", error="timeout")
        except Exception as e:
            logger.warning("SHAP 阶段失败（已跳过）：%s", e)
            gate.partial("shap", error=str(e))
    elif opts.skip_shap:
        logger.info("skip shap")

    # ---- 论文报告 ----
    gate.begin("report")
    formal = [chosen_name, "M1_catboost"]
    paper_only = [k for k in module_scores if k not in formal and k != "baseline_stable"]
    stab_sum_path = output_dir / "stability" / "multiseed_summary.csv"
    if (chosen_stab is None or len(chosen_stab) == 0) and stab_sum_path.exists():
        stab_sum = pd.read_csv(stab_sum_path)
        key = chosen_name if chosen_name in set(stab_sum.get("model", [])) else (
            "soft_vote" if chosen_name == "class_bias" else chosen_name
        )
        chosen_stab = stab_sum[stab_sum["model"] == key] if "model" in stab_sum.columns else pd.DataFrame()
    write_paper_reports(
        output_dir,
        {
            "baseline": {
                "oof_accuracy": best_base_pack["metrics"]["accuracy"],
                "final_accuracy": m_b_val.get("accuracy") if isinstance(m_b_val, dict) else m_b_val,
            },
            "final": {
                "name": chosen_name,
                "oof_accuracy": chosen_m["accuracy"] if isinstance(chosen_m, dict) else chosen_m,
                "final_accuracy": m_val.get("accuracy") if isinstance(m_val, dict) else m_val,
                "multiseed_mean": float(chosen_stab.iloc[0]["mean_seed_oof"]) if len(chosen_stab) else None,
                "multiseed_std": float(chosen_stab.iloc[0]["std_seed_oof"]) if len(chosen_stab) else None,
            },
            "module_scores": module_scores,
            "decision": {
                "recommend_merge": recommend_merge,
                "formal_modules": formal,
                "paper_only_modules": paper_only,
            },
            "ablation_csv": str(output_dir / "ablation" / "feature_ablation.csv"),
        },
    )

    # 兼容根目录别名
    if cm_df is not None:
        cm_df.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    elif (output_dir / "final_validation" / "confusion_matrix.csv").exists():
        pd.read_csv(output_dir / "final_validation" / "confusion_matrix.csv", index_col=0).to_csv(
            output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
        )
    if fi is not None and len(fi):
        fi.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    elif (output_dir / "final_validation" / "feature_importance.csv").exists():
        pd.read_csv(output_dir / "final_validation" / "feature_importance.csv").to_csv(
            output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
        )
    pd.read_csv(output_dir / "final_validation" / "predictions.csv").to_csv(
        output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )

    if test_df is not None:
        logger.info("检测到测试集：用全部有标签数据重训 M1 并导出概率（不用测试标签）。")
        y_full = _normalize_y(df)
        st_all = fit_cat_preprocessor(df, feat_final)
        X_all = transform_cat(df, st_all)
        mdl = CatBoostClassifier(**params_final)
        mdl.fit(X_all, y_full, cat_features=cat_indices(st_all))
        X_te = transform_cat(test_df, st_all)
        p_te = _align_proba(mdl, X_te)
        pred_te = pred_from_proba(p_te)
        pd.DataFrame(
            {
                "predicted_label": pred_te,
                "prob_Low": p_te[:, 0],
                "prob_Medium": p_te[:, 1],
                "prob_High": p_te[:, 2],
            }
        ).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    gate.complete("report")
    state.data["exit_status"] = "success"
    state.save()
    logger.info(
        "任务一完成 | exit_status=success | mode=%s | chosen=%s | OOF=%.4f | Final=%.4f | baseline_final=%.4f | merge=%s",
        mode,
        chosen_name,
        chosen_m["accuracy"],
        m_val["accuracy"],
        m_b_val["accuracy"],
        recommend_merge,
    )
    return {
        "task_name": "task1",
        "accuracy": float(m_val["accuracy"]),
        "macro_f1": float(m_val["macro_f1"]),
        "selected_configuration": chosen_name,
        "oof_accuracy": float(chosen_m["accuracy"]),
        "recommend_merge": recommend_merge,
        "mode": mode,
        "run_id": run_id,
        "final_validation": metrics_out,
    }
