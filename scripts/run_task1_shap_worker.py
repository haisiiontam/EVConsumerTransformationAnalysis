"""独立 SHAP 子进程：崩溃不影响主训练进程。

优先使用 CatBoost 原生 ShapValues（见 explain.py），避免 shap.TreeExplainer 原生崩溃。
"""

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
    p.add_argument("--model", required=True)
    p.add_argument("--features-json", required=True)
    p.add_argument("--params-json", required=True)
    p.add_argument("--develop-indices", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=200)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.common.data_utils import read_csv_compatible
    from src.config import TASK1_TARGET
    from src.tasks.task1.explain import run_shap_analysis
    from src.tasks.task1.pipeline import _normalize_y
    from src.tasks.task1.preprocess import cat_indices, fit_cat_preprocessor, transform_cat

    df = read_csv_compatible(Path(args.data))
    idx = pd.read_csv(args.develop_indices)["sample_index"].astype(int)
    df_dev = df.loc[idx].copy()
    y_dev = _normalize_y(df_dev) if TASK1_TARGET in df_dev.columns else _normalize_y(df).loc[idx]
    features = json.loads(Path(args.features_json).read_text(encoding="utf-8"))
    # params-json 保留以兼容流水线调用；原生 SHAP 使用已保存模型，无需重训
    _ = json.loads(Path(args.params_json).read_text(encoding="utf-8"))

    model = CatBoostClassifier()
    model.load_model(args.model)
    state = fit_cat_preprocessor(df_dev, features)
    X_for_shap = transform_cat(df_dev, state)
    info = run_shap_analysis(
        model,
        X_for_shap,
        y_dev,
        Path(args.output_dir),
        max_samples=args.max_samples,
        cat_feature_indices=cat_indices(state),
    )
    print("SHAP_OK", info)
    return 0 if info.get("top_feature") or info.get("available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
