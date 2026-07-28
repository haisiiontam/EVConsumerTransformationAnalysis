from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.data_utils import read_csv_compatible
from src.common.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text
from src.common.metrics import binary_classification_metrics, majority_baseline
from src.config import TASK3_CATBOOST_COMMON, TASK3_TIE_EPS
from src.tasks.task3.calibration import search_threshold
from src.tasks.task3.features import (
    assert_no_leakage,
    build_target,
    feature_sets,
    group_feature_lists,
)
from src.tasks.task3.m1_catboost_classifier import _prob1, oof_catboost_binary
from src.tasks.task3.m2_catboost_regressor import oof_regression_score, persist_m2_outputs
from src.tasks.task3.preprocessing import cat_indices, fit_cat_preprocessor, transform_cat


def _fine_threshold(p1: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """向量化阈值搜索（主目标 Accuracy，避免逐 thr 调 sklearn）。"""
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1, dtype=float)
    thrs = np.round(np.arange(0.35, 0.65 + 1e-12, 0.0025), 10)

    preds = (p1.reshape(1, -1) >= thrs.reshape(-1, 1)).astype(np.int8)
    accs = (preds == y.reshape(1, -1)).mean(axis=1)


    mask0 = y == 0
    if mask0.any():
        recall0 = (preds[:, mask0] == 0).mean(axis=1)
    else:
        recall0 = np.zeros_like(accs)
    thr_dist = np.abs(thrs - 0.5)
    max_acc = accs.max()
    ok = accs >= max_acc - TASK3_TIE_EPS

    order = np.lexsort((thr_dist, -recall0, -accs))

    for i in order:
        if ok[i]:
            best_i = int(i)
            break
    else:
        best_i = int(np.argmax(accs))
    search = pd.DataFrame({"threshold": thrs, "accuracy": accs, "recall_0": recall0})
    return search, {
        "threshold": float(thrs[best_i]),
        "oof_accuracy": float(accs[best_i]),
        "oof_roc_auc": float("nan"),
        "oof_balanced_accuracy": float("nan"),
    }


def _boundary_fusion(
    p_cls: np.ndarray,
    p_reg: np.ndarray,
    y: np.ndarray,
    pred_score: np.ndarray | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    rows = []
    best = {"accuracy": -1.0, "kind": "cls"}
    best_p = p_cls.copy()


    for alpha in np.round(np.arange(0.0, 1.0 + 1e-12, 0.025), 10):
        beta = 1.0 - alpha
        p = alpha * p_cls + beta * p_reg
        search, thr_cfg = _fine_threshold(p, y)
        row = {
            "kind": "linear",
            "alpha": float(alpha),
            "beta": float(beta),
            "w": None,
            "threshold": thr_cfg["threshold"],
            "accuracy": thr_cfg["oof_accuracy"],
        }
        rows.append(row)
        if thr_cfg["oof_accuracy"] > best["accuracy"] + 1e-12:
            best = row
            best_p = p.copy()


    for w in np.round(np.arange(0.0, 1.0 + 1e-12, 0.05), 10):
        unc = 1.0 - 2.0 * np.abs(p_cls - 0.5)
        p = (1.0 - w * unc) * p_cls + (w * unc) * p_reg
        search, thr_cfg = _fine_threshold(p, y)
        row = {
            "kind": "boundary",
            "alpha": None,
            "beta": None,
            "w": float(w),
            "threshold": thr_cfg["threshold"],
            "accuracy": thr_cfg["oof_accuracy"],
        }
        rows.append(row)
        if thr_cfg["oof_accuracy"] > best["accuracy"] + 1e-12:
            best = row
            best_p = p.copy()


    if pred_score is not None:
        dist = np.abs(pred_score - 5.5)

        gate = np.clip(1.0 - dist / 2.5, 0.0, 1.0)
        for w in np.round(np.arange(0.0, 1.0 + 1e-12, 0.05), 10):
            p = (1.0 - w * gate) * p_cls + (w * gate) * p_reg
            search, thr_cfg = _fine_threshold(p, y)
            row = {
                "kind": "score_gate",
                "alpha": None,
                "beta": None,
                "w": float(w),
                "threshold": thr_cfg["threshold"],
                "accuracy": thr_cfg["oof_accuracy"],
            }
            rows.append(row)
            if thr_cfg["oof_accuracy"] > best["accuracy"] + 1e-12:
                best = row
                best_p = p.copy()


    disagree = ((p_cls >= 0.5).astype(int) != (p_reg >= 0.5).astype(int)).astype(float)
    for w in np.round(np.arange(0.0, 1.0 + 1e-12, 0.1), 10):
        p = np.where(disagree > 0, (1 - w) * p_cls + w * p_reg, p_cls)
        search, thr_cfg = _fine_threshold(p, y)
        row = {
            "kind": "disagree_gate",
            "alpha": None,
            "beta": None,
            "w": float(w),
            "threshold": thr_cfg["threshold"],
            "accuracy": thr_cfg["oof_accuracy"],
        }
        rows.append(row)
        if thr_cfg["oof_accuracy"] > best["accuracy"] + 1e-12:
            best = row
            best_p = p.copy()

    return pd.DataFrame(rows), best, best_p


def _soft3(
    p_a: np.ndarray, p_b: np.ndarray, p_c: np.ndarray | None, y: np.ndarray, names: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    rows = []
    best = {"accuracy": -1.0, "weights": {}}
    best_p = p_a.copy()
    step = 0.05
    if p_c is None:
        for wa in np.round(np.arange(0.0, 1.0 + 1e-12, step), 10):
            p = wa * p_a + (1 - wa) * p_b
            search, thr_cfg = _fine_threshold(p, y)
            row = {
                "weights": f"{names[0]}={wa:.2f},{names[1]}={1-wa:.2f}",
                "threshold": thr_cfg["threshold"],
                "accuracy": thr_cfg["oof_accuracy"],
            }
            rows.append(row)
            if thr_cfg["oof_accuracy"] > best["accuracy"]:
                best = {
                    "accuracy": thr_cfg["oof_accuracy"],
                    "threshold": thr_cfg["threshold"],
                    "weights": {names[0]: float(wa), names[1]: float(1 - wa)},
                }
                best_p = p.copy()
    else:
        for wa in np.round(np.arange(0.0, 1.0 + 1e-12, step), 10):
            for wb in np.round(np.arange(0.0, 1.0 - wa + 1e-12, step), 10):
                wc = 1.0 - wa - wb
                p = wa * p_a + wb * p_b + wc * p_c
                search, thr_cfg = _fine_threshold(p, y)
                row = {
                    "weights": f"{names[0]}={wa:.2f},{names[1]}={wb:.2f},{names[2]}={wc:.2f}",
                    "threshold": thr_cfg["threshold"],
                    "accuracy": thr_cfg["oof_accuracy"],
                }
                rows.append(row)
                if thr_cfg["oof_accuracy"] > best["accuracy"]:
                    best = {
                        "accuracy": thr_cfg["oof_accuracy"],
                        "threshold": thr_cfg["threshold"],
                        "weights": {
                            names[0]: float(wa),
                            names[1]: float(wb),
                            names[2]: float(wc),
                        },
                    }
                    best_p = p.copy()
    return pd.DataFrame(rows), best, best_p


def _final_eval(df_dev, y_dev, df_val, y_val, feats, params, thr) -> Dict[str, Any]:
    state = fit_cat_preprocessor(df_dev, feats)
    model = CatBoostClassifier(**params)
    model.fit(transform_cat(df_dev, state), y_dev, cat_features=cat_indices(state))
    p_val = _prob1(model, transform_cat(df_val, state))
    m = binary_classification_metrics(y_val, (p_val >= thr).astype(int), p_val)
    return {"model": model, "state": state, "p_val": p_val, "metrics": m, "accuracy": float(m["accuracy"])}


def main() -> int:
    out_dir = ROOT / "outputs" / "task3"
    boost_dir = out_dir / "diagnostics" / "boost_continue"
    boost_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_compatible(ROOT / "dataSetB.csv")
    y = build_target(df)
    idx_dev = pd.read_csv(out_dir / "splits" / "develop_indices.csv")["sample_index"].astype(int)
    idx_val = pd.read_csv(out_dir / "splits" / "final_validation_indices.csv")["sample_index"].astype(int)
    fold_df = pd.read_csv(out_dir / "splits" / "fold_assignment_seed42.csv")
    fold_map = dict(zip(fold_df["sample_index"].astype(int), fold_df["fold"].astype(int)))
    df_dev, df_val = df.loc[idx_dev].copy(), df.loc[idx_val].copy()
    y_dev, y_val = y.loc[idx_dev], y.loc[idx_val]
    fold_assign = np.array([fold_map[int(i)] for i in df_dev.index], dtype=int)

    sets = feature_sets(df)
    groups = group_feature_lists(sets["B_safe_raw"], sets["mechanism_only_pool"])

    feat_concern = groups["B+B_concern"]
    feat_travel = groups["B+T_travel"]
    feat_union = list(dict.fromkeys(feat_concern + feat_travel))
    feat_histish = list(
        dict.fromkeys(
            groups["B+B_concern"]
            + groups.get("B+T_travel", [])
            + groups.get("B+C_charging", [])
            + groups.get("B+A_eco", [])
            + groups.get("B+E_economy", [])
        )
    )
    for name, feats in [
        ("concern", feat_concern),
        ("travel", feat_travel),
        ("union_ct", feat_union),
        ("histish", feat_histish),
    ]:
        assert_no_leakage(feats)
        print(f"feature set {name}: n={len(feats)}")

    params = json.loads((out_dir / "tuning" / "m1_best_params.json").read_text(encoding="utf-8"))

    params = {**TASK3_CATBOOST_COMMON, **params, "verbose": False, "allow_writing_files": False}

    cand_rows = []
    best_pack: Dict[str, Any] = {"oof_accuracy": -1.0}

    feature_candidates = {
        "B+B_concern": feat_concern,
        "B+T_travel": feat_travel,
        "union_concern_travel": feat_union,
        "histish_groups": feat_histish,
    }


    existing_m1 = out_dir / "oof" / "m1_tuned_oof_predictions.csv"
    existing_feats = json.loads((out_dir / "features" / "selected_features.json").read_text(encoding="utf-8"))[
        "features"
    ]

    for set_name, feats in feature_candidates.items():
        print(f"=== M1 OOF on {set_name} ===")
        ckpt = boost_dir / "ckpt" / f"m1_{set_name}"

        if list(feats) == list(existing_feats) and existing_m1.exists():
            oof_m1 = pd.read_csv(existing_m1)
            print("reuse existing m1_tuned OOF", flush=True)
        else:
            oof_m1, folds_m1, _ = oof_catboost_binary(
                df_dev,
                y_dev,
                feats,
                params,
                fold_assign,
                ckpt,
                model_name=f"m1_{set_name}",
                resume_folds=True,
            )
            atomic_write_csv(boost_dir / f"m1_{set_name}_oof.csv", oof_m1)
            atomic_write_csv(boost_dir / f"m1_{set_name}_folds.csv", folds_m1)

        print(f"=== M2 OOF on {set_name} ===", flush=True)
        oof_m2, info_m2 = oof_regression_score(
            df_dev,
            y_dev,
            feats,
            fold_assign,
            boost_dir / "ckpt" / f"m2_{set_name}",
            resume_folds=True,
        )
        atomic_write_csv(boost_dir / f"m2_{set_name}_oof.csv", oof_m2)
        atomic_write_json(boost_dir / f"m2_{set_name}_info.json", info_m2)

        def _align(oof: pd.DataFrame) -> pd.DataFrame:
            d = oof.copy()
            if "sample_index" not in d.columns:

                d = d.reset_index()
                if "sample_index" not in d.columns and "index" in d.columns:
                    d = d.rename(columns={"index": "sample_index"})
            return d.set_index("sample_index").reindex(df_dev.index)


        m1_a = _align(oof_m1)
        m2_a = _align(oof_m2)
        p_cls = m1_a["prob_1"].to_numpy(dtype=float)
        p_reg = m2_a["prob_1"].to_numpy(dtype=float)
        pred_score = m2_a["pred_score"].to_numpy(dtype=float)
        yv = y_dev.to_numpy().astype(int)

        search_c, thr_c = _fine_threshold(p_cls, yv)
        search_r, thr_r = _fine_threshold(p_reg, yv)
        grid_f, best_f, p_f = _boundary_fusion(p_cls, p_reg, yv, pred_score)
        atomic_write_csv(boost_dir / f"fusion_grid_{set_name}.csv", grid_f)
        atomic_write_json(boost_dir / f"fusion_best_{set_name}.json", best_f)


        p_lgb = None
        lgb_path = out_dir / "oof" / "lgbm_oof_predictions.csv"
        if lgb_path.exists() and set_name == "B+B_concern":
            oof_l = pd.read_csv(lgb_path).set_index("sample_index").loc[df_dev.index]
            p_lgb = oof_l["prob_1"].to_numpy()
            grid_s, best_s, p_s = _soft3(p_cls, p_reg, p_lgb, yv, ["M1", "M2", "M4"])
        else:
            grid_s, best_s, p_s = _soft3(p_cls, p_reg, None, yv, ["M1", "M2"])
        atomic_write_csv(boost_dir / f"soft_grid_{set_name}.csv", grid_s)
        atomic_write_json(boost_dir / f"soft_best_{set_name}.json", best_s)

        local = [
            ("M1", p_cls, thr_c),
            ("M2", p_reg, thr_r),
            ("fusion", p_f, {"threshold": best_f["threshold"], "oof_accuracy": best_f["accuracy"]}),
            ("soft", p_s, {"threshold": best_s["threshold"], "oof_accuracy": best_s["accuracy"]}),
        ]
        for scheme, p, thr_info in local:
            acc = float(thr_info["oof_accuracy"])
            thr = float(thr_info["threshold"])
            row = {
                "feature_set": set_name,
                "n_features": len(feats),
                "scheme": scheme,
                "oof_accuracy": acc,
                "threshold": thr,
                "fusion_meta": best_f if scheme == "fusion" else None,
                "soft_meta": best_s if scheme == "soft" else None,
            }
            cand_rows.append(
                {k: v for k, v in row.items() if k not in ("fusion_meta", "soft_meta")}
                | {
                    "fusion_kind": best_f.get("kind") if scheme == "fusion" else "",
                    "soft_weights": json.dumps(best_s.get("weights", {})) if scheme == "soft" else "",
                }
            )
            if acc > best_pack["oof_accuracy"] + 1e-12:
                best_pack = {
                    "oof_accuracy": acc,
                    "threshold": thr,
                    "feature_set": set_name,
                    "features": feats,
                    "scheme": scheme,
                    "params": params,
                    "p_oof": p,
                    "fusion_best": best_f,
                    "soft_best": best_s,
                }

    cand_df = pd.DataFrame(cand_rows).sort_values("oof_accuracy", ascending=False)
    atomic_write_csv(boost_dir / "boost_candidates.csv", cand_df)
    print(cand_df.head(20).to_string(index=False))


    feats = best_pack["features"]
    thr = float(best_pack["threshold"])
    params = best_pack["params"]
    scheme = f"boost_{best_pack['feature_set']}_{best_pack['scheme']}"


    state = fit_cat_preprocessor(df_dev, feats)
    model = CatBoostClassifier(**params)
    model.fit(transform_cat(df_dev, state), y_dev, cat_features=cat_indices(state))
    p_cls_val = _prob1(model, transform_cat(df_val, state))


    p_final = p_cls_val
    deploy_note = "M1_only"
    if best_pack["scheme"] in {"fusion", "soft", "M2"}:
        from catboost import CatBoostRegressor
        from src.tasks.task3.features import raw_score

        score_dev = raw_score(df_dev).to_numpy().astype(float)
        reg = CatBoostRegressor(
            loss_function="RMSE",
            iterations=int(params.get("iterations", 500)),
            depth=min(int(params.get("depth", 6)), 8),
            learning_rate=float(params.get("learning_rate", 0.05)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 5.0)),
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        reg.fit(transform_cat(df_dev, state), score_dev, cat_features=cat_indices(state))
        pred_score_val = reg.predict(transform_cat(df_val, state))


        m2_info_path = boost_dir / f"m2_{best_pack['feature_set']}_info.json"


        set_key = best_pack["feature_set"]
        m2_info = json.loads((boost_dir / f"m2_{set_key}_info.json").read_text(encoding="utf-8"))
        center = float(m2_info.get("best_score_threshold", 5.5))
        p_reg_val = 1.0 / (1.0 + np.exp(-(pred_score_val - center) / 0.8))

        meta = best_pack["fusion_best"] if best_pack["scheme"] == "fusion" else best_pack["soft_best"]
        if best_pack["scheme"] == "fusion":
            kind = meta.get("kind")
            if kind == "linear":
                a, b = float(meta["alpha"]), float(meta["beta"])
                p_final = a * p_cls_val + b * p_reg_val
            elif kind == "boundary":
                w = float(meta["w"])
                unc = 1.0 - 2.0 * np.abs(p_cls_val - 0.5)
                p_final = (1.0 - w * unc) * p_cls_val + (w * unc) * p_reg_val
            elif kind == "score_gate":
                w = float(meta["w"])
                gate = np.clip(1.0 - np.abs(pred_score_val - 5.5) / 2.5, 0.0, 1.0)
                p_final = (1.0 - w * gate) * p_cls_val + (w * gate) * p_reg_val
            elif kind == "disagree_gate":
                w = float(meta["w"])
                disagree = ((p_cls_val >= 0.5).astype(int) != (p_reg_val >= 0.5).astype(int)).astype(float)
                p_final = np.where(disagree > 0, (1 - w) * p_cls_val + w * p_reg_val, p_cls_val)
            else:
                p_final = p_cls_val
            deploy_note = f"fusion:{kind}"
        elif best_pack["scheme"] == "soft":
            w = meta.get("weights", {})
            wa = float(w.get("M1", 1.0))
            wb = float(w.get("M2", 0.0))

            s = wa + wb
            wa, wb = wa / s, wb / s
            p_final = wa * p_cls_val + wb * p_reg_val
            deploy_note = f"soft:{w}"
        else:
            p_final = p_reg_val
            deploy_note = "M2_only"

    y_pred = (p_final >= thr).astype(int)
    metrics = binary_classification_metrics(y_val, y_pred, p_final)
    final_acc = float(metrics["accuracy"])
    maj = majority_baseline(y)

    prev = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    final_cfg = {
        "scheme": scheme,
        "features": feats,
        "m1_params": params,
        "threshold": thr,
        "oof_accuracy": float(best_pack["oof_accuracy"]),
        "deploy_note": deploy_note,
        "fusion_best": best_pack.get("fusion_best"),
        "soft_best": best_pack.get("soft_best"),
        "selection_rule": "max OOF accuracy among boosted feature sets + M1/M2 fusion",
        "previous_oof_accuracy": prev.get("oof_accuracy"),
        "previous_final_accuracy": prev.get("accuracy"),
        "mode": "boost_continue",
    }
    metrics_out = {
        **final_cfg,
        **metrics,
        "majority_baseline_accuracy": float(maj["majority_baseline_accuracy"]),
        "accuracy_lift_vs_majority": float(final_acc - maj["majority_baseline_accuracy"]),
        "validation_note": "boost continue; final evaluated once after OOF lock",
    }

    atomic_write_json(boost_dir / "boost_best.json", final_cfg)
    atomic_write_json(out_dir / "final" / "task3_final_config.json", final_cfg)
    atomic_write_json(out_dir / "tuning" / "task3_final_config.json", final_cfg)
    atomic_write_json(out_dir / "final_validation" / "metrics.json", metrics_out)
    atomic_write_csv(
        out_dir / "final_validation" / "predictions.csv",
        pd.DataFrame(
            {
                "sample_index": y_val.index.astype(int),
                "true_label": y_val.values.astype(int),
                "predicted_label": y_pred,
                "prob_0": 1.0 - p_final,
                "prob_1": p_final,
                "threshold": thr,
            }
        ),
    )
    model.save_model(str(out_dir / "final_validation" / "model.cbm"))

    atomic_write_json(out_dir / "features" / "selected_features.json", {"features": feats, "source": scheme})

    result = {
        "task_name": "task3",
        "accuracy": final_acc,
        "oof_accuracy": float(best_pack["oof_accuracy"]),
        "threshold": thr,
        "selected_configuration": scheme,
        "majority_baseline_accuracy": float(maj["majority_baseline_accuracy"]),
        "final_validation": metrics_out,
        "n_features": len(feats),
        "boost": {
            "previous_final": prev.get("accuracy"),
            "previous_oof": prev.get("oof_accuracy"),
            "new_final": final_acc,
            "new_oof": float(best_pack["oof_accuracy"]),
            "top_candidates": cand_df.head(10).to_dict(orient="records"),
        },
    }
    atomic_write_json(out_dir / "metrics.json", result)
    atomic_write_text(
        boost_dir / "boost_report.md",
        "\n".join(
            [
                "# Task3 Boost Continue",
                "",
                f"- Previous: OOF `{prev.get('oof_accuracy')}` / Final `{prev.get('accuracy')}`",
                f"- New: OOF `{best_pack['oof_accuracy']}` / Final `{final_acc}`",
                f"- Scheme: `{scheme}` thr=`{thr}` deploy=`{deploy_note}`",
                "",
                "```",
                cand_df.head(15).to_string(index=False),
                "```",
                "",
            ]
        ),
    )
    print("=== BOOST DONE ===")
    print("prev", prev.get("oof_accuracy"), prev.get("accuracy"))
    print("new ", best_pack["oof_accuracy"], final_acc, scheme, deploy_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
