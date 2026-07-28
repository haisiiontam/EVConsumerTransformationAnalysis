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
        here.parents[1] / "公共源代码",
        here.parents[2] / "公共源代码",
        here.parents[2],
        here.parents[3],
        Path.cwd(),
    ]
    for r in candidates:
        if (r / "src").exists():
            sys.path.insert(0, str(r))
            return
    raise RuntimeError("找不到含 src/ 的代码根目录，请从交付包内运行或设置 PYTHONPATH")


_add_src_path()

from src.common.data_utils import read_csv_compatible
from src.tasks.task1.metrics_ext import pred_from_proba
from src.tasks.task1.ordinal import ordinal_probs_from_boundaries
from src.tasks.task1.preprocess import transform_cat, transform_tabular
from src.tasks.task1.regression import score_to_soft_proba


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    md = Path(args.model_dir)
    bundle = json.loads((md / "task1_model_bundle.json").read_text(encoding="utf-8"))
    features = bundle["features"]
    w = bundle["soft_vote_weights"]
    temp = float(bundle.get("m3_temperature", 0.5))

    df = read_csv_compatible(args.data)
    state = joblib.load(md / "task1_cat_prep_state.joblib")
    X = transform_cat(df, state)

    m1 = CatBoostClassifier()
    m1.load_model(str(md / "task1_m1_catboost.cbm"))

    classes = list(m1.classes_)
    p1_raw = m1.predict_proba(X)
    order = ["Low", "Medium", "High"]
    idx = [classes.index(c) for c in order]
    p1 = p1_raw[:, idx]

    m3 = CatBoostRegressor()
    m3.load_model(str(md / "task1_m3_regressor.cbm"))
    p3 = score_to_soft_proba(np.asarray(m3.predict(X), dtype=float), temp)

    pack = joblib.load(md / "task1_m6_ord_logit.joblib")
    Xt = transform_tabular(df, pack["prep"])
    p_ge_m = pack["ge_med"].predict_proba(Xt)[:, 1]
    p_ge_h = pack["ge_high"].predict_proba(Xt)[:, 1]
    p6 = ordinal_probs_from_boundaries(p_ge_m, p_ge_h)

    mapping = {"M1_catboost": p1, "M3_reg_soft": p3, "M6_ord_logit": p6}
    parts, wsum = [], 0.0
    for name, weight in w.items():
        if name in mapping and float(weight) > 0:
            parts.append(float(weight) * mapping[name])
            wsum += float(weight)
    p = sum(parts) / wsum
    pred = pred_from_proba(p)
    out = pd.DataFrame(
        {
            "样本行号": df.index.astype(int),
            "预测类标签": pred,
            "prob_Low": p[:, 0],
            "prob_Medium": p[:, 1],
            "prob_High": p[:, 2],
        }
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("wrote", args.out, "n=", len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
