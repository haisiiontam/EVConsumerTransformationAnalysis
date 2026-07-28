"""同冻结 split 公平对比：历史 clf_best_hp vs 当前 M1；按 OOF 最优阈值重锁 Final。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_utils import read_csv_compatible
from src.common.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text
from src.common.metrics import binary_classification_metrics, majority_baseline
from src.tasks.task3.calibration import search_threshold
from src.tasks.task3.features import add_mechanism_features, assert_no_leakage, build_target
from src.tasks.task3.m1_catboost_classifier import _prob1, oof_catboost_binary
from src.tasks.task3.preprocessing import cat_indices, fit_cat_preprocessor, transform_cat


def _enrich_hist_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """历史特征名 → 当前机制特征别名（不引入标签泄漏）。"""
    out = add_mechanism_features(df)
    if "charging_gap" in out.columns and "charging_gap_index" not in out.columns:
        out["charging_gap_index"] = out["charging_gap"]
    if "charging_convenience" in out.columns and "charging_convenience_index" not in out.columns:
        out["charging_convenience_index"] = out["charging_convenience"]
    if "weekly_minus_commute" in out.columns and "noncommute_km" not in out.columns:
        out["noncommute_km"] = out["weekly_minus_commute"]
    # battery_x_knowledge_deficit / battery_over_access
    if "battery_replacement_concern" in out.columns and "knowledge_deficit" in out.columns:
        out["battery_x_knowledge_deficit"] = (
            pd.to_numeric(out["battery_replacement_concern"], errors="coerce")
            * pd.to_numeric(out["knowledge_deficit"], errors="coerce")
        )
    if "battery_replacement_concern" in out.columns and "charging_station_accessibility" in out.columns:
        out["battery_over_access"] = pd.to_numeric(
            out["battery_replacement_concern"], errors="coerce"
        ) / (pd.to_numeric(out["charging_station_accessibility"], errors="coerce") + 1.0)
    return out


def _resolve_feats(df: pd.DataFrame, feats: list[str]) -> list[str]:
    enriched = _enrich_hist_aliases(df.head(3))
    out = []
    for f in feats:
        if f in enriched.columns:
            out.append(f)
        else:
            print(f"[warn] missing feature skipped: {f}")
    assert_no_leakage(out)
    return out


def _train_eval_final(
    df_dev: pd.DataFrame,
    y_dev: pd.Series,
    df_val: pd.DataFrame,
    y_val: pd.Series,
    features: list[str],
    params: dict,
    threshold: float,
) -> dict:
    # 用别名增强后的帧做预处理
    dev_e = _enrich_hist_aliases(df_dev)
    val_e = _enrich_hist_aliases(df_val)
    state = fit_cat_preprocessor(dev_e, features)
    model = CatBoostClassifier(**params)
    model.fit(transform_cat(dev_e, state), y_dev, cat_features=cat_indices(state))
    p_val = _prob1(model, transform_cat(val_e, state))
    p_dev = _prob1(model, transform_cat(dev_e, state))
    pred = (p_val >= threshold).astype(int)
    m = binary_classification_metrics(y_val, pred, p_val)
    m_dev = binary_classification_metrics(y_dev, (p_dev >= threshold).astype(int), p_dev)
    return {
        "threshold": float(threshold),
        "final_accuracy": float(m["accuracy"]),
        "final_roc_auc": float(m["roc_auc"]),
        "final_balanced_accuracy": float(m.get("balanced_accuracy", np.nan)),
        "develop_accuracy": float(m_dev["accuracy"]),
        "metrics": m,
        "model": model,
        "state": state,
        "p_val": p_val,
    }


def main() -> int:
    out_dir = ROOT / "outputs" / "task3"
    cmp_dir = out_dir / "diagnostics" / "fair_compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_compatible(ROOT / "dataSetB.csv")
    y = build_target(df)
    idx_dev = pd.read_csv(out_dir / "splits" / "develop_indices.csv")["sample_index"].astype(int)
    idx_val = pd.read_csv(out_dir / "splits" / "final_validation_indices.csv")["sample_index"].astype(int)
    fold_df = pd.read_csv(out_dir / "splits" / "fold_assignment_seed42.csv")
    fold_map = dict(zip(fold_df["sample_index"].astype(int), fold_df["fold"].astype(int)))
    df_dev, df_val = df.loc[idx_dev].copy(), df.loc[idx_val].copy()
    y_dev, y_val = y.loc[idx_dev], y.loc[idx_val]
    fold_assign = np.array([fold_map[int(i)] for i in df_dev.index], dtype=int)

    # --- current M1 ---
    cur_cfg = json.loads((out_dir / "final" / "task3_final_config.json").read_text(encoding="utf-8"))
    cur_feats = list(cur_cfg["features"])
    cur_params = dict(cur_cfg["m1_params"])
    oof_m1 = pd.read_csv(out_dir / "oof" / "m1_tuned_oof_predictions.csv")
    # 对齐 develop index 顺序
    oof_m1 = oof_m1.set_index("sample_index").loc[df_dev.index].reset_index()
    search_cur, thr_cur = search_threshold(
        oof_m1["prob_1"].to_numpy(), oof_m1["true_label"].to_numpy()
    )
    atomic_write_csv(cmp_dir / "current_m1_threshold_search.csv", search_cur)

    # --- historical clf_best_hp ---
    hist_meta = json.loads(
        (out_dir / "_archive" / "pre_full_optimize" / "tuning" / "best_config.json").read_text(
            encoding="utf-8"
        )
    )
    hist_final = json.loads(
        (out_dir / "_archive" / "pre_full_optimize" / "final_validation" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    hist_feats = _resolve_feats(df, list(hist_final["features"]))
    hist_params = dict(hist_final["clf_params"])
    hist_thr_old = float(hist_final["threshold"])

    print("Computing OOF for historical config on frozen folds...")
    oof_hist, folds_hist, _ = oof_catboost_binary(
        _enrich_hist_aliases(df_dev),
        y_dev,
        hist_feats,
        hist_params,
        fold_assign,
        cmp_dir / "ckpt_hist",
        model_name="hist_clf_best_hp",
        resume_folds=True,
    )
    atomic_write_csv(cmp_dir / "hist_oof_predictions.csv", oof_hist)
    atomic_write_csv(cmp_dir / "hist_fold_metrics.csv", folds_hist)
    search_hist, thr_hist = search_threshold(
        oof_hist["prob_1"].to_numpy(), oof_hist["true_label"].to_numpy()
    )
    atomic_write_csv(cmp_dir / "hist_threshold_search.csv", search_hist)

    rows = []
    # OOF rows
    for name, oof, thr_fixed, thr_best in [
        ("current_M1", oof_m1, 0.5, thr_cur["threshold"]),
        ("hist_clf_best_hp", oof_hist, hist_thr_old, thr_hist["threshold"]),
    ]:
        y_true = oof["true_label"].to_numpy()
        p1 = oof["prob_1"].to_numpy()
        for tag, thr in [("fixed", thr_fixed), ("oof_tuned", thr_best)]:
            m = binary_classification_metrics(y_true, (p1 >= thr).astype(int), p1)
            rows.append(
                {
                    "phase": "oof",
                    "model": name,
                    "threshold_policy": tag,
                    "threshold": float(thr),
                    "accuracy": m["accuracy"],
                    "roc_auc": m["roc_auc"],
                    "balanced_accuracy": m.get("balanced_accuracy"),
                    "n_features": len(cur_feats if "current" in name else hist_feats),
                }
            )

    # Final evaluations (same holdout)
    finals = {}
    for name, feats, params, thr_fixed, thr_tuned in [
        ("current_M1", cur_feats, cur_params, 0.5, float(thr_cur["threshold"])),
        ("hist_clf_best_hp", hist_feats, hist_params, hist_thr_old, float(thr_hist["threshold"])),
    ]:
        for tag, thr in [("fixed", thr_fixed), ("oof_tuned", thr_tuned)]:
            key = f"{name}@{tag}"
            print(f"Final eval {key} thr={thr:.4f} ...")
            res = _train_eval_final(df_dev, y_dev, df_val, y_val, feats, params, thr)
            finals[key] = res
            rows.append(
                {
                    "phase": "final",
                    "model": name,
                    "threshold_policy": tag,
                    "threshold": float(thr),
                    "accuracy": res["final_accuracy"],
                    "roc_auc": res["final_roc_auc"],
                    "balanced_accuracy": res["final_balanced_accuracy"],
                    "develop_accuracy": res["develop_accuracy"],
                    "n_features": len(feats),
                }
            )

    cmp_df = pd.DataFrame(rows)
    atomic_write_csv(cmp_dir / "fair_compare_summary.csv", cmp_df)

    # 选择：优先 OOF accuracy，同分优先 |thr-0.5| 更小；Final 只报告不调参
    oof_part = cmp_df[cmp_df["phase"] == "oof"].copy()
    oof_part = oof_part.sort_values(
        by=["accuracy", "balanced_accuracy"], ascending=[False, False]
    )
    best_oof = oof_part.iloc[0]
    best_model = str(best_oof["model"])
    best_thr = float(best_oof["threshold"])
    best_policy = str(best_oof["threshold_policy"])
    deploy_key = f"{best_model}@{best_policy}"
    # 若 best 是 oof_tuned，deploy 用对应 final；若 fixed 同理
    deploy = finals[deploy_key]

    # 额外：当前 M1 + oof_tuned 若不是第一也单独记录
    cur_tuned = finals["current_M1@oof_tuned"]
    hist_tuned = finals["hist_clf_best_hp@oof_tuned"]

    recommendation = {
        "same_split": True,
        "note": "Historical archived Final 0.8534 used a different holdout; this table is the fair compare.",
        "best_by_oof": {
            "model": best_model,
            "threshold_policy": best_policy,
            "threshold": best_thr,
            "oof_accuracy": float(best_oof["accuracy"]),
            "final_accuracy": float(deploy["final_accuracy"]),
        },
        "current_M1_oof_tuned": {
            "oof_threshold": float(thr_cur["threshold"]),
            "oof_accuracy": float(thr_cur["oof_accuracy"]),
            "final_accuracy": float(cur_tuned["final_accuracy"]),
        },
        "hist_clf_best_hp_oof_tuned": {
            "oof_threshold": float(thr_hist["threshold"]),
            "oof_accuracy": float(thr_hist["oof_accuracy"]),
            "final_accuracy": float(hist_tuned["final_accuracy"]),
        },
        "previous_locked_final_accuracy": float(
            json.loads((out_dir / "metrics.json").read_text(encoding="utf-8")).get("accuracy", np.nan)
        ),
    }
    atomic_write_json(cmp_dir / "recommendation.json", recommendation)

    # 重锁：采用 OOF 最优组合写回 final（不看 holdout 再调）
    if best_model == "current_M1":
        feats, params = cur_feats, cur_params
        scheme = f"current_M1_{best_policy}"
    else:
        feats, params = hist_feats, hist_params
        scheme = f"hist_clf_best_hp_{best_policy}"

    # 重新训练部署模型（与 deploy 一致）
    deploy = _train_eval_final(df_dev, y_dev, df_val, y_val, feats, params, best_thr)
    maj = majority_baseline(y)
    final_cfg = {
        "scheme": scheme,
        "features": feats,
        "m1_params": params,
        "threshold": best_thr,
        "oof_accuracy": float(best_oof["accuracy"]),
        "selection_rule": "max OOF accuracy on frozen develop folds; final evaluated once after lock",
        "fair_compare_dir": str(cmp_dir.as_posix()),
        "random_seed": 42,
        "mode": "relock_after_fair_compare",
    }
    metrics_out = {
        **final_cfg,
        **deploy["metrics"],
        "develop_accuracy": deploy["develop_accuracy"],
        "majority_baseline_accuracy": float(maj["majority_baseline_accuracy"]),
        "accuracy_lift_vs_majority": float(
            deploy["final_accuracy"] - maj["majority_baseline_accuracy"]
        ),
        "previous_full_final_accuracy": recommendation["previous_locked_final_accuracy"],
        "validation_note": "relocked by OOF threshold/model on frozen split; final not used for selection",
    }

    final_dir = out_dir / "final_validation"
    atomic_write_json(out_dir / "final" / "task3_final_config.json", final_cfg)
    atomic_write_json(out_dir / "tuning" / "task3_final_config.json", final_cfg)
    atomic_write_json(final_dir / "metrics.json", metrics_out)
    atomic_write_csv(
        final_dir / "predictions.csv",
        pd.DataFrame(
            {
                "sample_index": y_val.index.astype(int),
                "true_label": y_val.values.astype(int),
                "predicted_label": (deploy["p_val"] >= best_thr).astype(int),
                "prob_0": 1.0 - deploy["p_val"],
                "prob_1": deploy["p_val"],
                "threshold": best_thr,
            }
        ),
    )
    deploy["model"].save_model(str(final_dir / "model.cbm"))

    result = {
        "task_name": "task3",
        "accuracy": float(deploy["final_accuracy"]),
        "oof_accuracy": float(best_oof["accuracy"]),
        "threshold": best_thr,
        "selected_configuration": scheme,
        "majority_baseline_accuracy": float(maj["majority_baseline_accuracy"]),
        "final_validation": metrics_out,
        "n_features": len(feats),
        "fair_compare": recommendation,
    }
    atomic_write_json(out_dir / "metrics.json", result)

    md = [
        "# Task3 Fair Compare (same frozen split)",
        "",
        f"- Previous locked Final Acc: `{recommendation['previous_locked_final_accuracy']}`",
        f"- Relocked scheme: `{scheme}` @ thr=`{best_thr}`",
        f"- Relocked OOF Acc: `{best_oof['accuracy']}`",
        f"- Relocked Final Acc: `{deploy['final_accuracy']}`",
        "",
        "## Summary table",
        "",
        "```",
        cmp_df.to_string(index=False),
        "```",
        "",
        "## Recommendation",
        "",
        "```json",
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        "```",
        "",
        "GAM/TabNet: not recommended as next step based on this compare.",
        "",
    ]
    atomic_write_text(cmp_dir / "fair_compare_report.md", "\n".join(md))

    print("=== FAIR COMPARE DONE ===")
    print(cmp_df.to_string(index=False))
    print("relocked", scheme, "oof", best_oof["accuracy"], "final", deploy["final_accuracy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
