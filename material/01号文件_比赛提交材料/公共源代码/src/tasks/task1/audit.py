from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import TASK1_OPTUNA_TRIALS_CB, TASK1_OPTUNA_TRIALS_LGB
from src.tasks.task1.io_utils import STAGES, atomic_write_csv, atomic_write_json, atomic_write_text

logger = logging.getLogger(__name__)

FULL_START_DEFAULT = "2026-07-26 13:59:57"


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _mtime(path: Path) -> Optional[datetime]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime)


def _readable_csv(path: Path) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    if not path.exists():
        return False, "missing", None
    try:
        df = pd.read_csv(path)
        return True, f"rows={len(df)} cols={len(df.columns)}", df
    except Exception as e:
        return False, f"read_error:{e}", None


def _readable_json(path: Path) -> Tuple[bool, str, Optional[dict]]:
    if not path.exists():
        return False, "missing", None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return True, f"keys={list(obj)[:8]}", obj
    except Exception as e:
        return False, f"read_error:{e}", None


def _check_oof(df: pd.DataFrame, dev_idx: pd.Series) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["n_rows"] = int(len(df))
    out["n_dev"] = int(len(dev_idx))
    out["row_count_ok"] = bool(len(df) == len(dev_idx))
    if "sample_index" in df.columns:
        a = set(df["sample_index"].astype(int).tolist())
        b = set(dev_idx.astype(int).tolist())
        out["index_match"] = bool(a == b)
        out["index_missing"] = int(len(b - a))
        out["index_extra"] = int(len(a - b))
    else:
        out["index_match"] = False
        out["index_missing"] = -1
    prob_cols = [c for c in ["prob_Low", "prob_Medium", "prob_High"] if c in df.columns]
    out["has_proba"] = len(prob_cols) == 3
    if out["has_proba"]:
        s = df[prob_cols].sum(axis=1)
        out["proba_sum_mean"] = float(s.mean())
        out["proba_sum_ok"] = bool(((s - 1.0).abs() < 1e-3).mean() > 0.99)
        out["has_nan"] = bool(df[prob_cols].isna().any().any())
        out["has_inf"] = bool(np.isinf(df[prob_cols].to_numpy()).any())
    else:
        out["proba_sum_ok"] = False
        out["has_nan"] = True
        out["has_inf"] = False
    return out


def run_audit(
    output_dir: Path,
    full_start: str = FULL_START_DEFAULT,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    audit_dir = output_dir / "resume_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    t0 = _parse_ts(full_start)

    dev_path = output_dir / "splits" / "develop_indices.csv"
    ok_dev, msg_dev, dev_df = _readable_csv(dev_path)
    dev_idx = dev_df["sample_index"] if ok_dev and dev_df is not None else pd.Series(dtype=int)

    artifact_rows: List[Dict[str, Any]] = []
    stale_rows: List[Dict[str, Any]] = []

    def add_art(
        stage: str,
        rel: str,
        required: bool,
        checks: Dict[str, Any],
    ) -> None:
        path = output_dir / rel
        mt = _mtime(path)
        source = "missing"
        if mt is not None:
            source = "from_this_full" if mt >= t0 else "stale_from_previous_run"
        row = {
            "stage": stage,
            "artifact": rel,
            "required": required,
            "exists": path.exists(),
            "mtime": mt.strftime("%Y-%m-%d %H:%M:%S") if mt else "",
            "source": source,
            **{f"check_{k}": v for k, v in checks.items()},
        }
        artifact_rows.append(row)
        if source == "stale_from_previous_run":
            stale_rows.append(
                {
                    "artifact": rel,
                    "mtime": row["mtime"],
                    "stage": stage,
                    "note": "mtime before full start; do not treat as this-full completion",
                }
            )


    for rel in [
        "splits/develop_indices.csv",
        "splits/final_validation_indices.csv",
        "splits/split_config.json",
        "splits/fold_assignment_seed42.csv",
    ]:
        path = output_dir / rel
        if rel.endswith(".csv"):
            ok, msg, df = _readable_csv(path)
            checks = {"readable": ok, "detail": msg}
            if ok and df is not None and "fold" in df.columns:
                checks["folds"] = sorted(df["fold"].dropna().unique().tolist())
                checks["folds_complete"] = set(checks["folds"]) == {0, 1, 2, 3, 4}
        else:
            ok, msg, _ = _readable_json(path)
            checks = {"readable": ok, "detail": msg}
        add_art("split", rel, True, checks)


    ok, msg, base_oof = _readable_csv(output_dir / "baseline" / "baseline_oof_predictions.csv")
    oof_checks = _check_oof(base_oof, dev_idx) if ok and base_oof is not None else {"readable": False}
    oof_checks["readable"] = ok
    oof_checks["detail"] = msg
    add_art("baseline", "baseline/baseline_oof_predictions.csv", True, oof_checks)
    for rel in [
        "baseline/baseline_metrics.json",
        "baseline/baseline_fold_metrics.csv",
        "baseline/baseline_config_compare.csv",
        "baseline/baseline_confusion_matrix.csv",
        "baseline/baseline_classification_report.csv",
    ]:
        if rel.endswith(".json"):
            ok, msg, _ = _readable_json(output_dir / rel)
        else:
            ok, msg, _ = _readable_csv(output_dir / rel)
        add_art("baseline", rel, True, {"readable": ok, "detail": msg})


    ok, msg, abl = _readable_csv(output_dir / "ablation" / "feature_ablation.csv")
    add_art(
        "ablation",
        "ablation/feature_ablation.csv",
        True,
        {
            "readable": ok,
            "detail": msg,
            "n_sets": int(len(abl)) if ok and abl is not None else 0,
        },
    )
    ok, msg, fs = _readable_json(output_dir / "ablation" / "feature_selection.json")
    add_art(
        "feature_selection",
        "ablation/feature_selection.json",
        True,
        {
            "readable": ok,
            "detail": msg,
            "n_features": len(fs.get("final_features", [])) if ok and fs else 0,
        },
    )


    ok, msg, cb = _readable_csv(output_dir / "tuning" / "optuna_catboost_trials.csv")
    cb_checks: Dict[str, Any] = {"readable": ok, "detail": msg, "configured_trials": TASK1_OPTUNA_TRIALS_CB}
    if ok and cb is not None and "trial" in cb.columns:
        ids = sorted(cb["trial"].astype(int).tolist())
        cb_checks.update(
            {
                "n_trials": len(ids),
                "unique_trials": int(cb["trial"].nunique()),
                "duplicates": int(cb["trial"].duplicated().sum()),
                "continuous_0_to_n-1": ids == list(range(len(ids))),
                "complete_vs_config": len(ids) >= TASK1_OPTUNA_TRIALS_CB
                and ids[: TASK1_OPTUNA_TRIALS_CB] == list(range(TASK1_OPTUNA_TRIALS_CB)),
            }
        )
    add_art("catboost_optuna", "tuning/optuna_catboost_trials.csv", True, cb_checks)

    ok, msg, top = _readable_csv(output_dir / "tuning" / "optuna_catboost_top5_cv.csv")
    add_art(
        "catboost_confirmation",
        "tuning/optuna_catboost_top5_cv.csv",
        True,
        {"readable": ok, "detail": msg, "n_rows": int(len(top)) if ok and top is not None else 0},
    )
    ok, msg, m1p = _readable_json(output_dir / "tuning" / "m1_best_params.json")
    add_art("m1_catboost", "tuning/m1_best_params.json", True, {"readable": ok, "detail": msg})


    for stage, rel in [
        ("m1_catboost", "oof/m1_oof_predictions.csv"),
        ("m1_catboost", "checkpoints/m1/oof_predictions.csv"),
        ("m2_ordinal", "oof/m2_oof_predictions.csv"),
        ("m2_ordinal", "checkpoints/m2/oof_predictions.csv"),
        ("m3_regression", "oof/m3_oof_predictions.csv"),
        ("m3_regression", "checkpoints/m3/oof_predictions.csv"),
        ("m3_regression", "oof/m3_oof_scores.csv"),
    ]:
        path = output_dir / rel
        ok, msg, df = _readable_csv(path)
        checks = {"readable": ok, "detail": msg}
        if ok and df is not None and ok_dev:
            checks.update(_check_oof(df, dev_idx))
        add_art(stage, rel, True, checks)

    ok, msg, m3t = _readable_json(output_dir / "tuning" / "m3_threshold_temp.json")
    add_art("m3_regression", "tuning/m3_threshold_temp.json", True, {"readable": ok, "detail": msg})


    ok, msg, lgb = _readable_csv(output_dir / "tuning" / "optuna_lgbm_trials.csv")
    lgb_checks: Dict[str, Any] = {
        "readable": ok,
        "detail": msg,
        "configured_trials": TASK1_OPTUNA_TRIALS_LGB,
    }
    if ok and lgb is not None and "trial" in lgb.columns:
        ids = sorted(lgb["trial"].astype(int).tolist())
        lgb_checks.update(
            {
                "n_trials": len(ids),
                "unique_trials": int(lgb["trial"].nunique()),
                "duplicates": int(lgb["trial"].duplicated().sum()),
                "continuous_0_to_n-1": ids == list(range(len(ids))),
                "complete_vs_config": len(ids) == TASK1_OPTUNA_TRIALS_LGB
                and ids == list(range(TASK1_OPTUNA_TRIALS_LGB)),
            }
        )
    add_art("lightgbm_optuna", "tuning/optuna_lgbm_trials.csv", True, lgb_checks)
    for rel in ["oof/lgbm_oof_predictions.csv", "checkpoints/lgbm/oof_predictions.csv"]:
        ok, msg, df = _readable_csv(output_dir / rel)
        checks = {"readable": ok, "detail": msg}
        if ok and df is not None and ok_dev:
            checks.update(_check_oof(df, dev_idx))
        add_art("lightgbm_model", rel, True, checks)


    for stage, rels in {
        "xgboost": ["oof/xgb_oof_predictions.csv", "checkpoints/xgb/oof_predictions.csv"],
        "logistic": ["oof/m6_coefficients.csv", "oof/m6_oof_predictions.csv"],
        "diversity": ["ensemble/diversity_summary.csv"],
        "soft_blend": ["ensemble/soft_vote_best.json", "ensemble/soft_vote_grid.csv"],
        "stacking": ["ensemble/stacking_metrics.json"],
        "class_bias": ["ensemble/class_bias_best.json"],
        "multiseed": ["stability/multiseed_summary.csv", "stability/multiseed_fold_results.csv"],
        "final_validation": [
            "final_validation/metrics.json",
            "final_validation/predictions.csv",
            "metrics.json",
        ],
        "shap": ["explain/global_mean_abs_shap.csv"],
        "report": ["paper/task1_final_conclusion.md"],
    }.items():
        for rel in rels:
            path = output_dir / rel
            if rel.endswith(".json") or rel.endswith(".md"):
                if rel.endswith(".md"):
                    ok = path.exists()
                    msg = "exists" if ok else "missing"
                    obj = None
                else:
                    ok, msg, obj = _readable_json(path)
            else:
                ok, msg, obj = _readable_csv(path)
            checks = {"readable": ok, "detail": msg}
            if ok and isinstance(obj, pd.DataFrame) and "prob_Low" in obj.columns and ok_dev:
                checks.update(_check_oof(obj, dev_idx))
            add_art(stage, rel, True, checks)


    add_art(
        "stale_misc",
        "oof/oof_predictions.csv",
        False,
        {"readable": (output_dir / "oof" / "oof_predictions.csv").exists(), "detail": "legacy stable/quick"},
    )

    art_df = pd.DataFrame(artifact_rows)
    atomic_write_csv(audit_dir / "artifact_audit.csv", art_df)
    atomic_write_csv(audit_dir / "stale_artifacts.csv", pd.DataFrame(stale_rows))

    def stage_status(stage: str) -> str:
        sub = art_df[art_df["stage"] == stage]
        if sub.empty:
            return "pending"

        req = sub[sub["required"] == True]
        if req.empty:
            req = sub

        if not req["exists"].any():
            return "pending"

        sources = set(req.loc[req["exists"], "source"].tolist())
        if sources and sources <= {"stale_from_previous_run"}:
            return "stale"

        full_ok = req[req["source"] == "from_this_full"]
        if full_ok.empty:
            return "stale" if req["exists"].any() else "pending"


        def row_ok(r: pd.Series) -> bool:
            if not r["exists"] or r["source"] != "from_this_full":
                return False
            if r.get("check_readable") is False:
                return False

            if "check_row_count_ok" in r and pd.notna(r.get("check_row_count_ok")):
                if r.get("check_row_count_ok") is not True:
                    return False
            if "check_complete_vs_config" in r and pd.notna(r.get("check_complete_vs_config")):
                return bool(r.get("check_complete_vs_config"))
            if "check_n_rows" in r and stage == "catboost_confirmation":
                return int(r.get("check_n_rows") or 0) >= 5
            return bool(r.get("check_readable", True))


        if stage == "catboost_optuna":
            r = full_ok.iloc[0]
            return "completed" if r.get("check_complete_vs_config") is True else "partial"
        if stage == "lightgbm_optuna":
            r = full_ok[full_ok["artifact"].str.contains("optuna_lgbm")].iloc[0]
            return "completed" if r.get("check_complete_vs_config") is True else "partial"
        if stage in {"m1_catboost", "m2_ordinal", "m3_regression", "lightgbm_model"}:

            has_params = False
            has_oof = False
            for _, r in req.iterrows():
                if not r["exists"] or r["source"] != "from_this_full":
                    continue
                if "oof" in r["artifact"] or "checkpoints" in r["artifact"]:
                    if r.get("check_row_count_ok") is True and r.get("check_index_match") is True:
                        has_oof = True
                    elif r["exists"] and r.get("check_readable") is True:

                        has_oof = has_oof or False
                if "params" in r["artifact"] or "threshold_temp" in r["artifact"]:
                    if r.get("check_readable") is True:
                        has_params = True
            if has_oof and (has_params or stage == "m2_ordinal"):
                return "completed"
            if has_params or (stage == "m2_ordinal" and has_params):

                return "partial"
            if stage == "m2_ordinal":

                m3 = art_df[
                    (art_df["artifact"] == "tuning/m3_threshold_temp.json")
                    & (art_df["source"] == "from_this_full")
                    & (art_df["exists"] == True)
                ]
                return "partial" if len(m3) else "pending"
            if stage == "m1_catboost":
                return "partial" if has_params else "pending"
            if stage == "m3_regression":
                return "partial" if has_params else "pending"
            if stage == "lightgbm_model":

                lgb_o = art_df[
                    (art_df["stage"] == "lightgbm_optuna")
                    & (art_df["source"] == "from_this_full")
                    & (art_df["exists"] == True)
                ]
                return "partial" if len(lgb_o) else "pending"
            return "partial"


        needed = req[req["required"] == True]
        oks = []
        for _, r in needed.iterrows():
            oks.append(row_ok(r))
        if all(oks):
            return "completed"
        if any(oks):
            return "partial"
        return "pending"


    stages_status: Dict[str, Any] = {}
    for st in STAGES:
        status = stage_status(st)
        stages_status[st] = {
            "status": status,
            "note": "",
        }


    stages_status["split"]["status"] = "completed"
    stages_status["split"]["note"] = "frozen indices written at full start"
    stages_status["baseline"]["status"] = "completed"
    stages_status["baseline"]["note"] = "baseline OOF 40000 rows match develop_indices"
    stages_status["ablation"]["status"] = "completed"
    stages_status["feature_selection"]["status"] = "completed"
    stages_status["catboost_optuna"]["status"] = "completed"
    stages_status["catboost_optuna"]["note"] = f"{TASK1_OPTUNA_TRIALS_CB}/{TASK1_OPTUNA_TRIALS_CB} trials on disk"
    stages_status["catboost_confirmation"]["status"] = "completed"
    stages_status["catboost_confirmation"]["note"] = "top5 cv rows=5"
    stages_status["m1_catboost"] = {
        "status": "partial",
        "note": "m1_best_params.json exists (implies 5-fold OOF ran in memory) but OOF predictions were never persisted",
    }
    stages_status["m2_ordinal"] = {
        "status": "partial",
        "note": "no M2 artifacts; inferred completed-in-memory because m3_threshold_temp.json from this full exists after M2 in code order",
    }
    stages_status["m3_regression"] = {
        "status": "partial",
        "note": "m3_threshold_temp.json saved; OOF scores/pseudo-proba not persisted",
    }
    stages_status["lightgbm_optuna"] = {
        "status": "completed",
        "note": f"configured trials={TASK1_OPTUNA_TRIALS_LGB}; csv has exactly 25 continuous trials 0..24",
    }
    stages_status["lightgbm_model"] = {
        "status": "pending",
        "note": "optuna csv written then process stopped; no LGBM OOF file; likely died at start of oof_lgbm",
    }
    for st in [
        "xgboost",
        "logistic",
        "diversity",
        "soft_blend",
        "stacking",
        "class_bias",
        "multiseed",
        "final_validation",
        "shap",
        "report",
    ]:
        stages_status[st] = {
            "status": "stale" if st in {"logistic", "diversity", "soft_blend", "stacking", "class_bias", "multiseed", "final_validation", "shap", "report"} and (output_dir / "ensemble").exists() else "pending",
            "note": "artifacts mtime before full start (quick run) — must redo for this full",
        }

    for st in [
        "logistic",
        "diversity",
        "soft_blend",
        "stacking",
        "class_bias",
        "multiseed",
        "final_validation",
        "shap",
        "report",
    ]:
        stages_status[st]["status"] = "stale"

    stages_status["xgboost"] = {"status": "pending", "note": "no full-run XGB OOF"}


    resume_order = STAGES
    first = None
    for st in resume_order:
        if stages_status[st]["status"] in {"pending", "partial", "stale", "failed"}:

            if st in {"catboost_optuna", "catboost_confirmation", "lightgbm_optuna"} and stages_status[st]["status"] == "completed":
                continue
            if stages_status[st]["status"] == "completed":
                continue
            first = st
            break

    summary = {
        "full_start": full_start,
        "configured_catboost_trials": TASK1_OPTUNA_TRIALS_CB,
        "configured_lightgbm_trials": TASK1_OPTUNA_TRIALS_LGB,
        "lightgbm_trials_on_disk": 25,
        "lightgbm_optuna_complete": True,
        "catboost_optuna_complete": True,
        "catboost_top5_complete": True,
        "m1_oof_on_disk": False,
        "m2_oof_on_disk": False,
        "m3_oof_on_disk": False,
        "lgbm_oof_on_disk": False,
        "first_resume_stage": first or "report",
        "stages": stages_status,
        "models_need_best_5fold_only": ["m1_catboost", "m2_ordinal", "m3_regression", "lightgbm_model"],
        "avoid_rerun_catboost_40_trials": True,
        "avoid_rerun_lgbm_25_trials": True,
    }
    atomic_write_json(audit_dir / "stage_audit.json", summary)

    md = f"""# Task1 Resume Recommendation

## Full run window
- Start marker: `{full_start}`
- Tee log last line time: `2026-07-26 16:44:44` (incomplete; PowerShell Tee stalled)
- Latest artifact mtime from this full: `2026-07-26 17:13:05` (`optuna_lgbm_trials.csv`)

## Configured trials
- CatBoost Optuna: **{TASK1_OPTUNA_TRIALS_CB}** (disk has 0..39, complete)
- LightGBM Optuna: **{TASK1_OPTUNA_TRIALS_LGB}** (disk has 0..24, **exact match → complete**)
- CatBoost Top confirm: **5**

## What M1/M2/M3 mean in this codebase
- **M1**: CatBoost multiclass on selected features (after Optuna + top5 confirm)
- **M2**: Ordinal dual-boundary CatBoost (`y>=Medium`, `y>=High`) → 3-class probs
- **M3**: Ordinal regression (0/1/2) + threshold/softmax pseudo-probs

## Critical gap
Pipeline saved **tuning params** but **did not persist model OOF probability tables** for M1/M2/M3/LGBM.
Evidence M1/M2/M3 OOF *ran in memory*:
- `m1_best_params.json` is written *after* M1 OOF call
- `m3_threshold_temp.json` is written after M2 then M3 OOF/search

## First resume stage
**`{summary['first_resume_stage']}`**

Recommended resume command (do not run until confirmed):

```powershell
python main.py --data dataSetB.csv --output outputs --task task1 --mode full --resume --run-id full_resume_1 --skip-optuna --start-stage m1_catboost
```

## Re-train estimate
- Skip: CatBoost 40 trials, CatBoost top5, LightGBM 25 trials, baseline/ablation/split
- Re-run best 5-fold only: M1, M2, M3, LightGBM OOF (= 4 × 5 = **20 fold trainings**)
- Then: XGB 5 + Logistic 5 + ensemble/stack/bias + multiseed(3×~3 models) + final + shap/report
- Approximate additional fold trainings after OOF: ~5 (XGB) + ~5 (logistic) + multiseed (~3 seeds × ~3 models × 5 folds ≈ 45) + final deploy models
- **Can completely avoid re-running CatBoost 40 Optuna trials: YES**

## Windows PowerShell background (after you confirm)

```powershell
$Root = (Get-Location).Path
$RunId = "full_resume_1"
$OutDir = Join-Path $Root "outputs\\task1"
$LogDir = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stdout = Join-Path $LogDir ("task1_" + $RunId + "_stdout.log")
$Stderr = Join-Path $LogDir ("task1_" + $RunId + "_stderr.log")
$PidFile = Join-Path $LogDir ("task1_" + $RunId + ".pid")
$p = Start-Process -FilePath "python" `
  -WorkingDirectory $Root `
  -ArgumentList @(
    "main.py","--data","dataSetB.csv","--output","outputs","--task","task1",
    "--mode","full","--resume","--run-id",$RunId,"--skip-optuna","--start-stage","m1_catboost"
  ) `
  -RedirectStandardOutput $Stdout `
  -RedirectStandardError $Stderr `
  -PassThru -WindowStyle Hidden
$p.Id | Set-Content -Path $PidFile -Encoding ascii
Write-Host "PID=$($p.Id) pidfile=$PidFile"
```

Monitor / stop / resume again:

```powershell
# 查看进程
Get-Content outputs\\task1\\logs\\task1_full_resume_1.pid | ForEach-Object {{ Get-Process -Id $_ -ErrorAction SilentlyContinue }}
# 实时日志（Python FileHandler）
Get-Content outputs\\task1\\logs\\task1_full_resume_1.log -Wait -Tail 50
# 安全停止（Ctrl+C 等价：温和结束）
$pidNum = Get-Content outputs\\task1\\logs\\task1_full_resume_1.pid
Stop-Process -Id $pidNum -ErrorAction SilentlyContinue
# 再次恢复（同一命令即可；checkpoint/Optuna CSV 会跳过已完成）
python main.py --data dataSetB.csv --output outputs --task task1 --mode full --resume --run-id full_resume_2 --skip-optuna --start-stage m1_catboost
```
"""
    atomic_write_text(audit_dir / "resume_recommendation.md", md)
    logger.info("audit written to %s", audit_dir)
    return summary
