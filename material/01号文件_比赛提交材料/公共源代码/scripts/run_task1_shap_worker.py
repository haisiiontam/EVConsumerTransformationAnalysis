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
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="", help="可选 .cbm；缺失则按 params 重训")
    p.add_argument("--features-json", required=True)
    p.add_argument("--params-json", required=True)
    p.add_argument("--develop-indices", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=300)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.common.data_utils import read_csv_compatible
    from src.config import TASK1_TARGET
    from src.tasks.task1.explain import run_shap_analysis, train_final_catboost
    from src.tasks.task1.pipeline import _normalize_y
    from src.tasks.task1.preprocess import cat_indices, fit_cat_preprocessor, transform_cat

    df = read_csv_compatible(Path(args.data))
    idx = pd.read_csv(args.develop_indices)["sample_index"].astype(int)
    df_dev = df.loc[idx].copy()
    y_dev = _normalize_y(df_dev) if TASK1_TARGET in df_dev.columns else _normalize_y(df).loc[idx]
    features = json.loads(Path(args.features_json).read_text(encoding="utf-8"))
    params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))

    explain_dir = Path(args.output_dir)
    explain_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model) if args.model else explain_dir / "m1_for_shap.cbm"

    if model_path.exists():
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        state = fit_cat_preprocessor(df_dev, features)
        X_for_shap = transform_cat(df_dev, state)
        cat_idx = cat_indices(state)
    else:
        model, state, X_for_shap = train_final_catboost(df_dev, y_dev, features, params)
        cat_idx = cat_indices(state)
        model.save_model(str(explain_dir / "m1_for_shap.cbm"))
        print("TRAINED_M1_FOR_SHAP", explain_dir / "m1_for_shap.cbm")

    info = run_shap_analysis(
        model,
        X_for_shap,
        y_dev,
        explain_dir,
        max_samples=args.max_samples,
        cat_feature_indices=cat_idx,
    )
    print("SHAP_OK", info)
    return 0 if info.get("available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
