from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.common.io_utils import atomic_write_text


def write_paper_reports(output_dir: Path, ctx: Dict[str, Any]) -> None:
    paper = Path(output_dir) / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    oof = ctx.get("oof_accuracy")
    fin = ctx.get("final_accuracy")
    thr = ctx.get("threshold")
    scheme = ctx.get("scheme", "baseline")
    feats_n = ctx.get("n_features")

    docs = {
        "task3_methodology.md": f"""# Task3 Methodology

Binary label: `range_anxiety_score > 5`. Classification features never include the score.

Protocol: stratified 80/20 holdout + develop 5-fold OOF (seed=42). Selection by OOF Accuracy; final validation once.

Locked scheme: `{scheme}` | OOF Acc=`{oof}` | threshold=`{thr}` | n_features=`{feats_n}`
""",
        "task3_feature_engineering.md": """# Feature Engineering

Sets: A_official / B_safe_raw / C_safe_plus_mechanism.
Mechanism groups: T travel, C charging, K cognition, B concern, BT concern×travel, E economy, A eco, D demo.
`monthly_charging_cost` and task1/2 targets excluded.
""",
        "task3_ablation.md": f"""# Ablation

See `ablation/feature_ablation.csv`. Selected features n=`{feats_n}`.
""",
        "task3_classifier_regressor.md": """# Classifier–Regressor Dual Channel

M1 CatBoost classifier + M2 CatBoostRegressor on raw score with thresholding / score→prob mapping.
Boundary-aware fusion lets the regressor focus on uncertain classifier samples.
""",
        "task3_ordinal_model.md": """# Ordinal Score Model

Cumulative boundaries P(score≥k) with monotone correction, plus 10-class distribution → P(score>5).
""",
        "task3_model_comparison.md": """# Model Comparison

See `oof/final_candidates.csv` and `ensemble/` for M1–M6 and blends.
""",
        "task3_ensemble.md": """# Ensemble

Soft blend / stacking / classifier–regressor fusion selected by OOF Accuracy only.
""",
        "task3_stability.md": """# Stability

Seeds 42/52/62 × 5 folds when multiseed enabled. Tiny gains require multi-seed confirmation.
""",
        "task3_robustness.md": """# Robustness

Learning curve and input noise sensitivity under `robustness/`. Exploratory only.
""",
        "task3_explainability.md": """# Explainability

CatBoost native SHAP (association, not causation). See `explain/`.
""",
        "task3_final_conclusion.md": f"""# Final Conclusion

- Scheme: `{scheme}`
- OOF Accuracy: `{oof}`
- Final Validation Accuracy: `{fin}`
- Threshold: `{thr}`
- Merge recommendation depends on multi-seed stability vs baseline.

Modules with no OOF lift remain paper controls only.
""",
    }
    for name, text in docs.items():
        atomic_write_text(paper / name, text)
