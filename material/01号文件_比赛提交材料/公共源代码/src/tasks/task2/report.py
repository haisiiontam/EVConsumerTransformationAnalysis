from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.common.io_utils import atomic_write_text


def write_paper_reports(output_dir: Path, ctx: Dict[str, Any]) -> None:
    paper = Path(output_dir) / "paper"
    paper.mkdir(parents=True, exist_ok=True)

    oof = ctx.get("oof_accuracy")
    fin = ctx.get("final_accuracy")
    scheme = ctx.get("scheme", "M")
    maj = ctx.get("majority_accuracy")
    weak = ctx.get("is_weak_signal_lock", True)
    max_auc_gap = ctx.get("max_auc_gap")
    n_dup = ctx.get("n_duplicate_conflicts")
    eb_alpha = ctx.get("eb_best_alpha")
    perm = ctx.get("permutation") or {}
    boot = ctx.get("bootstrap") or {}


    stats = Path(output_dir) / "statistics"
    if not perm and (stats / "permutation_summary.json").exists():
        perm = json.loads((stats / "permutation_summary.json").read_text(encoding="utf-8"))
    if not boot and (stats / "bootstrap_summary.json").exists():
        boot = json.loads((stats / "bootstrap_summary.json").read_text(encoding="utf-8"))

    docs = {
        "task2_problem_redefinition.md": f"""# Task2 Problem Redefinition

Target: `home_charging_available` (binary). Class 1 prevalence ≈ 0.65.

Research question shifts from “which fancy model wins Accuracy” to:
1) Is there stable predictive signal under leak-safe features?
2) If not, what is the Bayes-optimal decision under 0-1 loss?
3) Which unobserved structural variables should be collected next?

Weak-signal finding: safe observables show limited separability (max |AUC-0.5| ≈ `{max_auc_gap}`).
Missing structural housing/charging variables (parking, garage, tenure) likely dominate the true mechanism.
""",
        "task2_label_audit.md": f"""# Label Audit

Develop-set audits: exact duplicate conflicts (`n={n_dup}`), quantile profile bins, kNN consistency vs global prior.
Labels were **not** modified. See `label_audit/`.

Interpretation: if near-duplicate profiles disagree on labels, or kNN consistency ≤ global prior, local feature geometry lacks class structure.
""",
        "task2_separability.md": f"""# Separability

Univariate AUC / MI on develop safe features. Categorical smoothed rates under `separability/`.
Feature-group ablation (fast logistic OOF) under `features/feature_ablation.csv`.
""",
        "task2_empirical_bayes.md": f"""# Empirical Bayes

Hierarchical EB over city/education/vehicle + binned income/age/distance; backoff 2-way→1-way→global.
Best alpha (OOF search): `{eb_alpha}`.

Even when label predictions remain majority-class, EB probabilities can still rank segments by relative installation potential.
""",
        "task2_clustering.md": """# Clustering Prior

Fold-safe KMeans + Beta-Binomial cluster prior (`clustering/`).
If cluster target-rate gaps are unstable across folds/seeds, keep as descriptive segmentation only—not a submission model.
""",
        "task2_model_comparison.md": f"""# Model Comparison

Baselines B0–B2, logistic, CatBoost, EB, cluster prior, LightGBM/XGBoost/ExtraTrees, optional soft blend.
Locked scheme: `{scheme}` | OOF Acc=`{oof}` | majority=`{maj}`.

Formal vs paper roles:
- Formal submitter: locked scheme (often majority M)
- Probability ranking: EB / best calibrated-ish OOF
- Contrasts: tree ensembles that collapse to majority
""",
        "task2_statistical_tests.md": f"""# Statistical Tests

Permutation (label shuffle, fixed fast logistic OOF):
```json
{json.dumps(perm, ensure_ascii=False, indent=2)}
```

Bootstrap on OOF deltas:
```json
{json.dumps(boot, ensure_ascii=False, indent=2)}
```

See also `statistics/mcnemar_test.json`.
""",
        "task2_uncertainty.md": """# Uncertainty

Constant prior / EB probabilities provide uncertainty-aware ranking even when Accuracy-optimal labels are all-1.
Do not sacrifice ACC2 for prettier confusion matrices.
""",
        "task2_final_conclusion.md": f"""# Final Conclusion

- Locked scheme: `{scheme}`
- Weak-signal lock: `{weak}`
- OOF Accuracy: `{oof}`
- Final holdout Accuracy: `{fin}`
- Majority baseline: `{maj}`
- Permutation supports_signal: `{perm.get("supports_signal")}`
- Bootstrap P(delta Acc > 0): `{boot.get("p_delta_acc_gt_0")}`

Paper claim (when locked to M): after excluding cross-task target leakage, consumer-profile features do not yield stable, generalizable discrimination for home charging availability; under 0-1 loss the majority class is the robust Bayes decision.
""",
        "task2_data_collection_recommendation.md": """# Data Collection Recommendation

Collect housing/charging structure variables not present in current survey:
private parking, garage, housing tenure/type, electrical capacity, property management rules / condo charging policy, installation space.

Do **not** invent numeric lift promises; state them as hypothesized mechanisms for the next data round.
""",
    }
    for name, text in docs.items():
        atomic_write_text(paper / name, text)
