"""任务二推理：多数类常数分类器"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import pandas as pd


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
    raise RuntimeError("找不到含 src/ 的代码根目录")


_add_src_path()
from src.common.data_utils import read_csv_compatible


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = json.loads((Path(args.model_dir) / "task2_majority_model.json").read_text(encoding="utf-8"))
    maj = int(model["majority_class"])
    df = read_csv_compatible(args.data)
    out = pd.DataFrame({"样本行号": df.index.astype(int), "预测类标签": maj})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("wrote", args.out, "majority=", maj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
