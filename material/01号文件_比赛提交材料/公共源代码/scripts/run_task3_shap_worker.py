\
\
\
\

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataSetB.csv")
    p.add_argument("--output-dir", default="outputs/task3")
    p.add_argument("--max-samples", type=int, default=300)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.common.data_utils import read_csv_compatible
    from src.tasks.task3.explain import run_shap_analysis
    from src.tasks.task3.features import build_target
    from src.tasks.task3.preprocessing import cat_indices, fit_cat_preprocessor, transform_cat

    output_dir = Path(args.output_dir)
    cfg_path = output_dir / "final" / "task3_final_config.json"
    model_path = output_dir / "final_validation" / "model.cbm"
    dev_idx_path = output_dir / "splits" / "develop_indices.csv"
    if not cfg_path.exists() or not model_path.exists() or not dev_idx_path.exists():
        print("missing config/model/splits", cfg_path, model_path, dev_idx_path)
        return 1

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    features = list(cfg["features"])
    df = read_csv_compatible(Path(args.data))
    idx = pd.read_csv(dev_idx_path)["sample_index"].astype(int)
    df_dev = df.loc[idx].copy()
    y_dev = build_target(df_dev)

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    state = fit_cat_preprocessor(df_dev, features)
    X = transform_cat(df_dev, state)
    explain_dir = output_dir / "explain"
    info = run_shap_analysis(
        model,
        X,
        y_dev,
        explain_dir,
        cat_feature_indices=cat_indices(state),
        max_samples=args.max_samples,
    )

    paper = output_dir / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    figs = info.get("figures") or []
    fig_lines = [f"- `{f}`" for f in figs]
    (paper / "task3_explainability.md").write_text(
        "\n".join(
            [
                "# Task3 Explainability",
                "",
                "CatBoost native ShapValues on develop subsample (association, not causation).",
                f"- backend: `{info.get('backend')}`",
                f"- n_samples: `{info.get('n_shap_samples')}`",
                f"- top_feature: `{info.get('top_feature')}`",
                f"- base_value: `{info.get('base_value')}`",
                f"- available: `{info.get('available')}`",
                "",
                "## Figures",
                *fig_lines,
                "",
                "Key reads:",
                "- `shap_summary.png` — global importance (mean |SHAP|)",
                "- `shap_direction.png` — signed mean SHAP (push up vs push down anxiety)",
                "- `shap_summary_beeswarm.png` — distribution + feature value coloring",
                "- `shap_dependence_*.png` — top-feature response curves",
                "- `shap_by_class.png` — importance magnitude by true label",
                "- `waterfall_typical_*.png` — high / low / borderline cases",
                "",
                "Tables: `global_mean_abs_shap.csv`, `shap_by_class.csv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("SHAP_OK", info)
    return 0 if info.get("available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
