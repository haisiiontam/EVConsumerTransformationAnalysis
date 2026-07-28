"""自动生成论文材料 Markdown。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def write_paper_reports(output_dir: Path, ctx: Dict[str, Any]) -> None:
    paper = Path(output_dir) / "paper"
    paper.mkdir(parents=True, exist_ok=True)

    baseline = ctx.get("baseline", {})
    final = ctx.get("final", {})
    modules = ctx.get("module_scores", {})
    decision = ctx.get("decision", {})
    ablation_path = ctx.get("ablation_csv")

    (paper / "task1_methodology.md").write_text(
        f"""# Task1 Methodology

## Problem
Predict `ev_adoption_likelihood` ∈ {{Low, Medium, High}} with natural order Low < Medium < High.

## Validation protocol
- Frozen 80/20 stratified split (`random_state=42`)
- Develop-set Stratified 5-fold OOF for all selection
- Final validation evaluated once after locking config
- Multi-seed check: 42 / 52 / 62

## Modeling stack explored
1. CatBoost multiclass (M1) ± Optuna
2. Ordinal dual-boundary CatBoost (M2)
3. Ordinal regression + threshold/softmax (M3)
4. LightGBM / XGBoost heterogeneous trees (M4/M5)
5. Ordinal logistic baseline (M6)
6. Soft voting / nested stacking / class-bias postprocess

## Leakage rules
Excluded targets and uncertain posterior cost/energy fields; `range_anxiety_score` allowed as official feature.
""",
        encoding="utf-8",
    )

    abl_txt = ""
    if ablation_path and Path(ablation_path).exists():
        abl_txt = Path(ablation_path).read_text(encoding="utf-8")[:4000]
    (paper / "task1_feature_ablation.md").write_text(
        f"""# Task1 Feature Ablation

Compared A_official, B_safe_raw, B+C1…C6, B+kept, B+all_mech on identical folds.

Key file: `feature_ablation.csv`

```
{abl_txt}
```
""",
        encoding="utf-8",
    )

    lines = ["# Task1 Model Comparison", "", "| Module | OOF Accuracy | Notes |", "|---|---:|---|"]
    for name, info in modules.items():
        lines.append(
            f"| {name} | {info.get('accuracy', float('nan')):.4f} | {info.get('note', '')} |"
        )
    (paper / "task1_model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (paper / "task1_ordinal_analysis.md").write_text(
        f"""# Task1 Ordinal Analysis

- M2 dual-boundary OOF Acc: {modules.get('M2_ordinal', {}).get('accuracy', 'n/a')}
- M3 regression-soft OOF Acc: {modules.get('M3_reg_soft', {}).get('accuracy', 'n/a')}
- Focus metrics: QWK, Medium Recall, cross-two-level errors (Low↔High)

Ordinal modules are retained in paper ablation even if not selected as the sole final predictor.
""",
        encoding="utf-8",
    )

    (paper / "task1_ensemble_analysis.md").write_text(
        f"""# Task1 Ensemble Analysis

- Soft vote: {modules.get('soft_vote', {})}
- Stacking nested OOF: {modules.get('stacking', {})}
- Class bias: {modules.get('class_bias', {})}

Stacking is kept only if nested OOF beats soft vote. Soft-vote gains as small as +0.0005 are recorded for multi-seed review.
""",
        encoding="utf-8",
    )

    (paper / "task1_explainability.md").write_text(
        """# Task1 Explainability

SHAP analyses (association, not causation) are under `explain/`:
- global mean |SHAP|
- summary / dependence plots
- typical Low/Medium/High waterfall
- Medium misclassification diagnostics when available

Do not claim causal effects from SHAP rankings.
""",
        encoding="utf-8",
    )

    keep = decision.get("recommend_merge", False)
    (paper / "task1_final_conclusion.md").write_text(
        f"""# Task1 Final Conclusion

## Baseline (stable reproduction)
- OOF Accuracy: {baseline.get('oof_accuracy', 'n/a')}
- Final validation Accuracy: {baseline.get('final_accuracy', 'n/a')}

## Locked final model
- Name: {final.get('name', 'n/a')}
- OOF Accuracy: {final.get('oof_accuracy', 'n/a')}
- Final validation Accuracy: {final.get('final_accuracy', 'n/a')}
- Multi-seed mean±std: {final.get('multiseed_mean', 'n/a')} ± {final.get('multiseed_std', 'n/a')}

## Formal vs paper-only modules
- Entered final predictor: {decision.get('formal_modules', [])}
- Paper-only / ablation (no fake gains): {decision.get('paper_only_modules', [])}

## Merge recommendation
{"YES — multi-seed and locked validation support a real gain." if keep else "NO — keep stable task1 as official; advanced modules stay as paper ablation."}
""",
        encoding="utf-8",
    )

    # also mirror to docs/
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    for p in paper.glob("task1_*.md"):
        (docs / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
