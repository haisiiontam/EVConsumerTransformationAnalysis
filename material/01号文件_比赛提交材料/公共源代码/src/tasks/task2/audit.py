from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.common.io_utils import atomic_write_csv, atomic_write_json, atomic_write_text
from src.config import (
    TASK2_EXCLUDED_FEATURES,
    TASK2_LEAK_RISK_PROBE_FEATURES,
    TASK2_STAGES,
    TASK2_TARGET,
)

logger = logging.getLogger(__name__)

ARTIFACT_MAP: List[Tuple[str, str, str]] = [
    ("splits/develop_indices.csv", "split", "required"),
    ("splits/final_validation_indices.csv", "split", "required"),
    ("splits/fold_assignment_seed42.csv", "split", "required"),
    ("baseline/baseline_metrics.json", "baseline", "required"),
    ("baseline/baseline_oof_predictions.csv", "baseline", "required"),
    ("label_audit/duplicate_conflicts.csv", "label_audit", "optional"),
    ("separability/separability.csv", "separability", "optional"),
    ("missingness/missingness_report.json", "missingness", "optional"),
    ("empirical_bayes/eb_search.csv", "empirical_bayes", "optional"),
    ("oof/logistic_oof_predictions.csv", "m1_logistic_gam", "optional"),
    ("oof/catboost_b_oof_predictions.csv", "m2_catboost", "optional"),
    ("final/task2_final_config.json", "final_lock", "optional"),
    ("final_validation/metrics.json", "final_validation", "optional"),
    ("paper/task2_final_conclusion.md", "report", "optional"),
    ("metrics.json", "legacy_metrics", "legacy"),
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
        if "prob_0" in df.columns and "prob_1" in df.columns:
            s = df["prob_0"] + df["prob_1"]
            info["prob_sum_ok"] = bool(((s - 1.0).abs() < 1e-2).mean() > 0.99)
        info["ok"] = True
    except Exception as e:
        info["error"] = str(e)
    return info


def archive_legacy_if_needed(output_dir: Path) -> Optional[Path]:
    """若存在 metrics.json 且无 splits/，归档到 _archive/pre_weak_signal。"""
    output_dir = Path(output_dir)
    archive = output_dir / "_archive" / "pre_weak_signal"
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
    logger.info("archived legacy task2 outputs -> %s", archive)
    return archive


def run_audit(output_dir: Path) -> Dict[str, Any]:
    """扫描产物、推断 stage 完成度，写入 resume_audit/。"""
    output_dir = Path(output_dir)
    audit_dir = output_dir / "resume_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rel, stage, kind in ARTIFACT_MAP:
        p = output_dir / rel
        p_arch = output_dir / "_archive" / "pre_weak_signal" / rel
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
        }
        if "oof" in rel.lower() or "predictions" in rel:
            chk = _check_oof_csv(use_p) if use_p.exists() else {}
            row.update({f"oof_{k}": v for k, v in chk.items() if k != "path"})
        rows.append(row)

    art_df = pd.DataFrame(rows)
    atomic_write_csv(audit_dir / "artifact_audit.csv", art_df)

    stages: Dict[str, Any] = {}
    for st in TASK2_STAGES:
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

    first_resume = "split"
    for st in TASK2_STAGES:
        if st == "audit":
            continue
        if stages.get(st, {}).get("status") != "completed":
            first_resume = st
            break

    summary = {
        "stages": stages,
        "first_resume_stage": first_resume,
        "note": "弱信号协议：若模型无显著 lift 则锁定多数类 M。",
    }
    atomic_write_json(audit_dir / "stage_audit.json", summary)


    diag = output_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        diag / "leakage_audit.json",
        {
            "excluded": TASK2_EXCLUDED_FEATURES,
            "leak_risk_probe_only": TASK2_LEAK_RISK_PROBE_FEATURES,
            "target": TASK2_TARGET,
            "rule": "正式模型 X 禁止目标、任务一标签与后验费用字段",
        },
    )
    atomic_write_json(
        diag / "uncertain_features.json",
        {
            "uncertain": [
                {
                    "feature": "range_anxiety_score",
                    "reason": "原始问卷分 vs 任务三二分类标签源，语义重叠但禁止入模",
                }
            ],
            "excluded_from_model": [
                {
                    "feature": "ev_adoption_likelihood",
                    "reason": "任务一目标，泄露风险已验证",
                }
            ],
        },
    )
    feat_rows = (
        [{"feature": c, "status": "excluded"} for c in TASK2_EXCLUDED_FEATURES]
        + [{"feature": "range_anxiety_score", "status": "uncertain_not_in_model"}]
    )
    atomic_write_csv(diag / "feature_audit.csv", pd.DataFrame(feat_rows))

    md = [
        "# Task2 Weak-Signal Baseline Summary",
        "",
        f"- Suggested first resume stage: `{first_resume}`",
        "- Target: `home_charging_available` (~65% class 1)",
        "- Lock rule: no Acc lift > TASK2_TIE_EPS over majority OR max |AUC-0.5| < TASK2_WEAK_SIGNAL_AUC_GAP → M",
        "",
    ]
    atomic_write_text(audit_dir / "current_baseline_summary.md", "\n".join(md))
    logger.info("task2 audit done -> %s first_resume=%s", audit_dir, first_resume)
    return summary
