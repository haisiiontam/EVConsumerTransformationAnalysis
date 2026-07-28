"""任务三推理：M1/M2 soft 融合（M4 权重并入归一化）"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


def _add_src_path() -> None:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "00_公共源代码",
        here.parents[2] / "00_公共源代码",
        here.parents[2],
        here.parents[3],
        Path.cwd(),
    ]
    for r in candidates:
        if (r / "src").exists():
            sys.path.insert(0, str(r))
            return
    raise RuntimeError("找不到含 src/ 的代码根目录")


_add_src_path()
from src.common.data_utils import read_csv_compatible
from src.tasks.task3.preprocessing import transform_cat


def score_to_prob(score: np.ndarray, thr: float = 0.5) -> np.ndarray:
    # 将回归分近似映射到 [0,1]：以阈值附近 logistic 形状近似
    x = (score - thr) / 0.15
    return 1.0 / (1.0 + np.exp(-x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    md = Path(args.model_dir)
    bundle = json.loads((md / "task3_model_bundle.json").read_text(encoding="utf-8"))
    thr = float(bundle.get("threshold", 0.5))
    w = dict(bundle.get("soft_weights") or {"M1": 0.8, "M2": 0.15})
    # 部署包未单独导出 M4 时，将其权重并入 M1
    w_m1 = float(w.get("M1", 0.8)) + float(w.get("M4", 0.0))
    w_m2 = float(w.get("M2", 0.15))
    wsum = w_m1 + w_m2

    df = read_csv_compatible(args.data)
    state1 = joblib.load(md / "task3_cat_prep_state.joblib")
    X1 = transform_cat(df, state1)
    m1 = CatBoostClassifier()
    m1.load_model(str(md / "task3_m1_catboost.cbm"))
    p1 = m1.predict_proba(X1)[:, 1]

    state2 = joblib.load(md / "task3_m2_prep_state.joblib")
    X2 = transform_cat(df, state2)
    m2 = CatBoostRegressor()
    m2.load_model(str(md / "task3_m2_regressor.cbm"))
    p2 = score_to_prob(np.asarray(m2.predict(X2), dtype=float), thr=thr)

    p = (w_m1 * p1 + w_m2 * p2) / wsum
    pred = (p >= thr).astype(int)
    out = pd.DataFrame(
        {
            "样本行号": df.index.astype(int),
            "预测类标签": pred,
            "prob_1": p,
            "threshold": thr,
        }
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("wrote", args.out, "n=", len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
