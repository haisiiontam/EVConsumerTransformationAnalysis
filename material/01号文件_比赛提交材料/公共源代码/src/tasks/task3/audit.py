from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text
from src.config import TASK3_STAGES

logger = logging.getLogger(__name__)


ARTIFACT_MAP: List[Tuple[str, str, str]] = [
    ("splits/develop_indices.csv", "split", "required"),
    ("splits/final_validation_indices.csv", "split", "required"),
    ("splits/fold_assignment_seed42.csv", "split", "required"),
    ("baseline/baseline_metrics.json", "baseline", "required"),
    ("baseline/baseline_oof_predictions.csv", "baseline", "required"),
    ("diagnostics/separability.csv", "separability", "optional"),
    ("ablation/feature_ablation.csv", "feature_ablation", "optional"),
    ("features/feature_selection_comparison.csv", "feature_selection", "optional"),
    ("oof/m1_oof_predictions.csv", "m1_catboost_classifier", "required"),
    ("tuning/m1_best_params.json", "m1_optuna", "optional"),
    ("models/m2_regression_oof.csv", "m2_catboost_regressor", "optional"),
    ("models/m3_ordinal_oof.csv", "m3_ordinal_score", "optional"),
    ("oof/lgbm_oof_predictions.csv", "m4_lightgbm", "optional"),
    ("oof/xgb_oof_predictions.csv", "m5_xgboost", "optional"),
    ("oof/et_oof_predictions.csv", "m6_extra_trees", "optional"),
    ("ensemble/diversity_summary.md", "diversity", "optional"),
    ("ensemble/classifier_regressor_best.json", "classifier_regressor_fusion", "optional"),
    ("ensemble/soft_blend_best.json", "soft_blend", "optional"),
    ("ensemble/stacking_best.json", "stacking", "optional"),
    ("final/task3_final_config.json", "final_lock", "optional"),
    ("final_validation/metrics.json", "final_validation", "optional"),
    ("explain/shap_summary.png", "shap", "optional"),
    ("paper/task3_final_conclusion.md", "report", "optional"),
    ("test_predictions.csv", "submission", "optional"),

    ("metrics.json", "legacy_metrics", "legacy"),
    ("tuning/best_config.json", "legacy_tuning", "legacy"),
]


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _check_oof_csv(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "ok": False}
    if not path.exists():
        info["error"] = "missing"
        return info
    try:
        df = pd.read_csv(path)
        info["n_rows"] = int(len(df))
        info["n_unique_index"] = int(df["sample_index"].nunique()) if "sample_index" in df.columns else None
        if "prob_0" in df.columns and "prob_1" in df.columns:
            s = df["prob_0"] + df["prob_1"]
            info["prob_sum_mean"] = float(s.mean())
            info["prob_sum_ok"] = bool(((s - 1.0).abs() < 1e-2).mean() > 0.99)
            info["has_nan"] = bool(df[["prob_0", "prob_1"]].isna().any().any())
        info["ok"] = True
    except Exception as e:
        info["error"] = str(e)
    return info


def archive_legacy_if_needed(output_dir: Path) -> Optional[Path]:
    """若存在历史产物且尚无新 splits，则归档到 _archive/pre_full_optimize。"""
    output_dir = Path(output_dir)
    archive = output_dir / "_archive" / "pre_full_optimize"
    splits = output_dir / "splits" / "develop_indices.csv"
    legacy_metrics = output_dir / "metrics.json"
    if splits.exists():
        return None
    if not legacy_metrics.exists() and not (output_dir / "final_validation").exists():
        return None
    if archive.exists():
        return archive
    archive.mkdir(parents=True, exist_ok=True)
    for child in list(output_dir.iterdir()):
        if child.name in {"_archive", "resume_audit", "logs"}:
            continue
        dest = archive / child.name
        if dest.exists():
            continue
        try:
            shutil.move(str(child), str(dest))
        except Exception as e:
            logger.warning("archive move failed %s: %s", child, e)
    logger.info("archived legacy task3 outputs -> %s", archive)
    return archive


def run_audit(output_dir: Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    audit_dir = output_dir / "resume_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rel, stage, kind in ARTIFACT_MAP:
        p = output_dir / rel

        p_arch = output_dir / "_archive" / "pre_full_optimize" / rel
        exists = p.exists()
        src = "current" if exists else ("archive" if p_arch.exists() else "missing")
        use_p = p if exists else (p_arch if p_arch.exists() else p)
        row = {
            "relative_path": rel,
            "stage": stage,
            "kind": kind,
            "exists": bool(use_p.exists()),
            "location": src,
            "size": int(use_p.stat().st_size) if use_p.exists() else 0,
            "mtime": use_p.stat().st_mtime if use_p.exists() else None,
        }
        if rel.endswith(".csv") and "oof" in rel.lower() or "predictions" in rel:
            chk = _check_oof_csv(use_p) if use_p.exists() else {}
            row.update({f"oof_{k}": v for k, v in chk.items() if k != "path"})
        rows.append(row)

    art_df = pd.DataFrame(rows)
    atomic_write_csv(audit_dir / "artifact_audit.csv", art_df)

    stale = art_df[art_df["location"] == "archive"][["relative_path", "stage", "kind", "size"]]
    atomic_write_csv(audit_dir / "stale_artifacts.csv", stale)

    stages: Dict[str, Any] = {}
    for st in TASK3_STAGES:
        related = art_df[art_df["stage"] == st]
        if related.empty:
            stages[st] = {"status": "pending", "reason": "no mapped artifacts"}
            continue
        cur = related[related["location"] == "current"]
        if not cur.empty and cur["exists"].all():
            stages[st] = {"status": "completed", "n_artifacts": int(len(cur))}
        elif (related["location"] == "archive").any():
            stages[st] = {"status": "stale_archive", "n_artifacts": int(len(related))}
        else:
            stages[st] = {"status": "pending"}


    hist_paths = [
        output_dir / "metrics.json",
        output_dir / "_archive" / "pre_full_optimize" / "metrics.json",
        output_dir / "final_validation" / "metrics.json",
        output_dir / "_archive" / "pre_full_optimize" / "final_validation" / "metrics.json",
        output_dir / "tuning" / "best_config.json",
        output_dir / "_archive" / "pre_full_optimize" / "tuning" / "best_config.json",
    ]
    hist: Dict[str, Any] = {}
    for hp in hist_paths:
        j = _safe_read_json(hp)
        if j:
            hist[str(hp)] = j

    first_resume = "split"
    for st in TASK3_STAGES:
        if st == "audit":
            continue
        if stages.get(st, {}).get("status") != "completed":
            first_resume = st
            break

    summary = {
        "stages": stages,
        "first_resume_stage": first_resume,
        "historical_metrics_sources": list(hist.keys()),
        "note": "历史 Final Acc≈0.8534 若来自反复调参验证集，标注为非完全锁定验证集候选。",
    }
    atomic_write_json(audit_dir / "stage_audit.json", summary)


    oof_acc = fin_acc = thr = None
    for j in hist.values():
        oof_acc = oof_acc or j.get("oof_accuracy")
        fin_acc = fin_acc or j.get("accuracy")
        thr = thr or j.get("threshold")
        fv = j.get("final_validation") or {}
        if isinstance(fv, dict):
            fin_acc = fin_acc or fv.get("accuracy")
            thr = thr or fv.get("threshold")
    md = [
        "# Task3 Current Baseline Summary",
        "",
        "## Historical candidate (from disk, may be non-locked validation)",
        f"- OOF Accuracy: `{oof_acc}`",
        f"- Final Validation Accuracy: `{fin_acc}`",
        f"- Threshold: `{thr}`",
        "",
        "## Audit notes",
        "- `range_anxiety_score` must never enter classification X.",
        "- Prefer frozen splits under `outputs/task3/splits/` once created.",
        "- Stale advanced-run artifacts are listed in `stale_artifacts.csv`.",
        f"- Suggested first resume stage: `{first_resume}`",
        "",
    ]
    atomic_write_text(audit_dir / "current_baseline_summary.md", "\n".join(md))
    logger.info("task3 audit done -> %s first_resume=%s", audit_dir, first_resume)
    return summary
