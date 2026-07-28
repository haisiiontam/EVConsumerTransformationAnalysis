from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.data_utils import read_csv_compatible
from src.config import RANDOM_SEED, TASK3_SOURCE_SCORE
from src.tasks.task1.pipeline import _normalize_y
from src.tasks.task1.preprocess import (
    cat_indices,
    fit_cat_preprocessor,
    fit_tabular_preprocessor,
    transform_cat,
    transform_tabular,
)
from src.tasks.task1.regression import score_to_soft_proba
from src.tasks.task3.features import build_target as t3_target
from src.tasks.task3.preprocessing import cat_indices as t3_cat_indices
from src.tasks.task3.preprocessing import fit_cat_preprocessor as t3_fit
from src.tasks.task3.preprocessing import transform_cat as t3_transform


def _load_idx(path: Path) -> pd.Index:
    return pd.Index(pd.read_csv(path)["sample_index"].astype(int))


def export_task1(out_dir: Path, data: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "results/task1/task1_final_config.json").read_text(encoding="utf-8"))
    features = list(cfg["features"])
    params = dict(cfg["m1_params"])
    params["verbose"] = False
    params["allow_writing_files"] = False
    weights = dict(cfg.get("soft_vote_weights") or {})

    df = read_csv_compatible(data)
    idx = _load_idx(ROOT / "outputs/task1/splits/develop_indices.csv")
    df_dev = df.loc[idx].copy()
    y = _normalize_y(df_dev)
    y_np = y.to_numpy()
    ge_med = np.array([0 if lab == "Low" else 1 for lab in y_np], dtype=int)
    ge_high = np.array([1 if lab == "High" else 0 for lab in y_np], dtype=int)
    y_ord = np.array([{"Low": 0, "Medium": 1, "High": 2}[lab] for lab in y_np], dtype=float)


    state = fit_cat_preprocessor(df_dev, features)
    X = transform_cat(df_dev, state)
    m1 = CatBoostClassifier(**params)
    m1.fit(X, y, cat_features=cat_indices(state))
    m1.save_model(str(out_dir / "task1_m1_catboost.cbm"))
    joblib.dump(state, out_dir / "task1_cat_prep_state.joblib")


    it_m3 = min(500, int(params.get("iterations", 500)))
    m3 = CatBoostRegressor(
        loss_function="RMSE",
        iterations=it_m3,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
    )
    m3.fit(X, y_ord, cat_features=cat_indices(state))
    m3.save_model(str(out_dir / "task1_m3_regressor.cbm"))


    st_tab = fit_tabular_preprocessor(df_dev, features, scale=True)
    X_tab = transform_tabular(df_dev, st_tab)
    m6_a = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    m6_b = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    m6_a.fit(X_tab, ge_med)
    m6_b.fit(X_tab, ge_high)
    joblib.dump({"ge_med": m6_a, "ge_high": m6_b, "prep": st_tab}, out_dir / "task1_m6_ord_logit.joblib")

    m3_temp = 0.5
    soft_path = ROOT / "outputs/task1/ensemble/m3_soft_best.json"
    if soft_path.exists():
        try:
            m3_temp = float(json.loads(soft_path.read_text(encoding="utf-8")).get("temperature", 0.5))
        except Exception:
            m3_temp = 0.5

    bundle = {
        "scheme": "soft_vote",
        "features": features,
        "label_order": ["Low", "Medium", "High"],
        "soft_vote_weights": weights,
        "m3_temperature": m3_temp,
        "files": {
            "M1_catboost": "task1_m1_catboost.cbm",
            "M1_prep": "task1_cat_prep_state.joblib",
            "M3_reg_soft": "task1_m3_regressor.cbm",
            "M6_ord_logit": "task1_m6_ord_logit.joblib",
        },
        "final_holdout_accuracy": 0.8601,
        "note": "预测时按权重融合 M6/M1/M3 概率；见 infer_task1.py",
    }
    (out_dir / "task1_model_bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("task1 models ok")


def export_task2(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    maj = json.loads((ROOT / "results/task2/diagnostics/majority_baseline.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "results/task2/final/task2_final_config.json").read_text(encoding="utf-8"))
    model = {
        "scheme": "M",
        "type": "constant_majority_classifier",
        "majority_class": int(maj.get("majority_class", 1)),
        "predict_rule": "全部样本预测为 majority_class",
        "final_holdout_accuracy": 0.6498,
        "lock": lock,
        "rationale": "弱信号门控未通过；0-1 损失下多数类为 Bayes 稳健决策",
    }
    (out_dir / "task2_majority_model.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("task2 model ok")


def export_task3(out_dir: Path, data: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "results/task3/final/task3_final_config.json").read_text(encoding="utf-8"))
    features = list(cfg["features"])
    params = dict(cfg["m1_params"])
    params["verbose"] = False
    params["allow_writing_files"] = False
    thr = float(cfg.get("threshold", 0.5))
    weights = (cfg.get("soft_best") or {}).get("weights") or {"M1": 0.8, "M2": 0.15, "M4": 0.05}

    df = read_csv_compatible(data)
    idx = _load_idx(ROOT / "outputs/task3/splits/develop_indices.csv")
    df_dev = df.loc[idx].copy()
    y = t3_target(df_dev)

    state = t3_fit(df_dev, features)
    X = t3_transform(df_dev, state)
    model = CatBoostClassifier(**params)
    model.fit(X, y, cat_features=t3_cat_indices(state))
    model.save_model(str(out_dir / "task3_m1_catboost.cbm"))
    joblib.dump(state, out_dir / "task3_cat_prep_state.joblib")

    score = pd.to_numeric(df_dev[TASK3_SOURCE_SCORE], errors="coerce")
    feat_reg = [c for c in features if c != TASK3_SOURCE_SCORE]
    state_r = t3_fit(df_dev, feat_reg)
    Xr = t3_transform(df_dev, state_r)
    reg = CatBoostRegressor(
        loss_function="RMSE",
        iterations=min(400, int(params.get("iterations", 400))),
        depth=int(params.get("depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    reg.fit(Xr, score.fillna(score.median()), cat_features=t3_cat_indices(state_r))
    reg.save_model(str(out_dir / "task3_m2_regressor.cbm"))
    joblib.dump(state_r, out_dir / "task3_m2_prep_state.joblib")

    bundle = {
        "scheme": cfg.get("scheme", "boost_B+B_concern_soft"),
        "features": features,
        "regression_features": feat_reg,
        "threshold": thr,
        "soft_weights": weights,
        "files": {
            "M1": "task3_m1_catboost.cbm",
            "M1_prep": "task3_cat_prep_state.joblib",
            "M2": "task3_m2_regressor.cbm",
            "M2_prep": "task3_m2_prep_state.joblib",
        },
        "final_holdout_accuracy": 0.8529,
        "deploy_note": "完整 soft 融合需 M1/M2 概率加权后 threshold；仅用 M1+0.5 亦接近。",
    }
    (out_dir / "task3_model_bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("task3 models ok")


def main() -> int:
    data = ROOT / "dataSetB.csv"
    if not data.exists():
        raise FileNotFoundError(data)
    base = ROOT / "delivery_staging" / "models_export"
    export_task1(base / "task1", data)
    export_task2(base / "task2")
    export_task3(base / "task3", data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
