"""组装发给组长的两套交付包：01_比赛提交材料 + 02_论文写作素材。"""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.now().strftime("%Y%m%d")
DELIVERY = ROOT / f"delivery_for_leader_{STAMP}"
COMP = DELIVERY / "01_比赛提交材料"
PAPER = DELIVERY / "02_论文写作素材_按标签查找"
MODELS = ROOT / "delivery_staging" / "models_export"


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[WARN] missing: {src}")
        return
    _ensure(dst.parent)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _write(path: Path, text: str) -> None:
    _ensure(path.parent)
    path.write_text(text.lstrip("\n"), encoding="utf-8")


def format_predictions() -> None:
    mapping = [
        (
            ROOT / "outputs/task1/final_validation/predictions.csv",
            COMP / "任务一_EV采用意愿/03_预测结果CSV/task1_predictions.csv",
            ["sample_index", "true_label", "predicted_label"],
            {"predicted_label": "pred_label"},
        ),
        (
            ROOT / "outputs/task2/final_validation/predictions.csv",
            COMP / "任务二_续航焦虑二分类/03_预测结果CSV/task2_predictions.csv",
            ["sample_index", "true_label", "predicted_label"],
            {"predicted_label": "pred_label"},
        ),
        (
            ROOT / "outputs/task3/final_validation/predictions.csv",
            COMP / "任务三_支付意愿二分类/03_预测结果CSV/task3_predictions.csv",
            ["sample_index", "true_label", "predicted_label"],
            {"predicted_label": "pred_label"},
        ),
    ]
    for src, dst, cols, rename in mapping:
        df = pd.read_csv(src)
        out = df[cols].rename(columns=rename)
        # 赛方格式：样本行号、真值类标签、预测类标签
        out = out.rename(
            columns={
                "sample_index": "样本行号",
                "true_label": "真值类标签",
                "pred_label": "预测类标签",
            }
        )
        _ensure(dst.parent)
        out.to_csv(dst, index=False, encoding="utf-8-sig")
        # 同时保留英文列版本便于程序读取
        eng = df[cols].rename(columns=rename)
        eng.to_csv(dst.with_name(dst.stem + "_en.csv"), index=False, encoding="utf-8-sig")


def copy_models() -> None:
    pairs = [
        (MODELS / "task1", COMP / "任务一_EV采用意愿/02_训练好的模型"),
        (MODELS / "task2", COMP / "任务二_续航焦虑二分类/02_训练好的模型"),
        (MODELS / "task3", COMP / "任务三_支付意愿二分类/02_训练好的模型"),
    ]
    for src, dst in pairs:
        _copy(src, dst)


def copy_source() -> None:
    # 公共源码
    for name in ["src", "scripts", "main.py", "requirements.txt", "pyproject.toml", "README.md"]:
        src = ROOT / name
        if src.exists():
            _copy(src, COMP / "00_公共源代码" / name)

    # 任务级推理脚本（若已生成）
    infer_dir = ROOT / "delivery_staging" / "infer_scripts"
    if infer_dir.exists():
        for task, folder in [
            ("infer_task1.py", "任务一_EV采用意愿/01_源代码"),
            ("infer_task2.py", "任务二_续航焦虑二分类/01_源代码"),
            ("infer_task3.py", "任务三_支付意愿二分类/01_源代码"),
        ]:
            p = infer_dir / task
            if p.exists():
                _copy(p, COMP / folder / task)

    # 各任务指向公共代码的说明
    for task_name in [
        "任务一_EV采用意愿",
        "任务二_续航焦虑二分类",
        "任务三_支付意愿二分类",
    ]:
        _write(
            COMP / task_name / "01_源代码" / "README_源代码说明.md",
            f"""# {task_name} — 源代码说明

完整可复现代码位于交付包根目录：

- `../00_公共源代码/src/` — 三任务实现
- `../00_公共源代码/main.py` — 入口
- `../00_公共源代码/scripts/` — 辅助脚本

本目录另附 `infer_task*.py`（若存在）用于加载本任务「训练好的模型」做推理。

复现训练（需自备 `dataSetB.csv`）：

```bash
python main.py --task all --data dataSetB.csv
```

注意：禁止使用测试集标签训练或调参；本仓库内部验证集由冻结 split 产生。
""",
        )


def write_competition_docs() -> None:
    _write(
        COMP / "00_给组长_比赛提交请先读.md",
        """# 比赛提交材料说明（发给组委会 / 组长汇总用）

## 锁定方案与内部验证 Acc（非官方测试集成绩）

| 任务 | 方案 | Final Acc |
|------|------|-----------|
| 任务一 | soft_vote（M6 0.55 + M1 0.35 + M3 0.10） | **0.8601** |
| 任务二 | M（多数类恒预测 1） | **0.6498** |
| 任务三 | boost_B+B_concern_soft（M1≈0.80, M2≈0.15, M4≈0.05, thr=0.5） | **0.8529** |

加权参考：`0.2*ACC1 + 0.4*ACC2 + 0.4*ACC3` = **0.7731**（内部 holdout，非正式排名）。

## 目录对照赛方要求

| 赛方要求 | 本包位置 |
|----------|----------|
| 模型代码 | `00_公共源代码/` + 各任务 `01_源代码/` |
| 训练好的模型文件 | 各任务 `02_训练好的模型/` |
| 预测结果 CSV（行号/真值/预测） | 各任务 `03_预测结果CSV/` |
| 建模报告（重要性/选型/调参） | 各任务 `04_建模报告.md` |
| Introduction（思路/环境/调用） | `Introduction.md`（可另存为 Introduction.pdf） |

## 重要合规说明

1. 训练集/验证集由选手自行划分；本项目使用冻结 develop / final_validation 划分，**未使用官方测试集标签**训练或选参。
2. 预测 CSV 对应**内部最终验证集**，便于组委会核对流程；官方排名以组委会测试集为准。
3. 任务二为弱信号场景，正式部署为多数类常数分类器（有完整证据链，见论文包）。
""",
    )

    _write(
        COMP / "Introduction.md",
        """# Introduction — EV 消费者转化分析初赛提交说明

## ① 设计思路

本作品针对问卷型表格数据上的三个分类任务，统一采用「冻结划分 + 折内预处理 + OOF 选型 + 最终验证」流程，避免信息泄漏。

- **任务一（EV 采用意愿，Low/Medium/High）**：标签有序。保留 CatBoost 多分类主模型（M1），并引入有序回归软概率（M3）与双阈值 Logistic（M6）作为异构概率源，按验证网格搜索得到 soft_vote 权重（M6:0.55, M1:0.35, M3:0.10）。该融合兼顾可解释线性序数结构与非线性交互，在 Medium 类上更稳健。
- **任务二（续航焦虑二分类）**：系统性可分性/置换检验/多 seed 冲高后，可部署 Acc 增益门控未通过，按 0-1 损失 Bayes 决策锁定**多数类常数分类器**。这不是“不做模型”，而是用证据拒绝虚假提升。
- **任务三（支付意愿二分类）**：在机制特征（concern 比率/缺口）增强特征集上，以 CatBoost 分类（M1）为主、回归分支（M2）为辅做 soft 融合，阈值 0.5。

可行性：全部依赖公开表特征与可复现脚本；与赛题契合在于尊重标签结构（有序/弱信号/阈值二分类），而非盲目堆叠复杂模型。

## ② 环境 & 语言

| 项 | 说明 |
|----|------|
| 系统 | Windows 10 / 亦可 Linux |
| 语言 | Python 3.12（推荐 `.venv-shap`） |
| 核心依赖 | pandas, numpy, scikit-learn, catboost, joblib；可解释性另需 shap, matplotlib |
| 数据 | `dataSetB.csv`（选手自备，不随本包分发时请向数据管理员索取） |
| 随机种子 | 42（见各任务 `*_final_config.json`） |
| 划分 | develop 80% / final_validation 20%，分层抽样，索引冻结于 `outputs/*/splits/` |

安装示例：

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
# 或: uv pip install --python .venv-shap\\Scripts\\python.exe -r requirements.txt
```

## ③ 模型调用方式

各任务模型目录见 `任务*/02_训练好的模型/`，推理脚本见 `任务*/01_源代码/infer_task*.py`。

### 任务一 soft_vote

1. 读取 `task1_model_bundle.json` 获取特征列表与权重。
2. 加载：
   - `task1_m1_catboost.cbm` + `task1_cat_prep_state.joblib`
   - `task1_m3_regressor.cbm`（温度 `m3_temperature`，默认 0.5）
   - `task1_m6_ord_logit.joblib`（含两个 Logistic 与表格预处理）
3. 对每个样本得到三类概率 `p_M1, p_M3, p_M6`，按权重加权归一后 `argmax → {Low,Medium,High}`。

```bash
python 任务一_EV采用意愿/01_源代码/infer_task1.py --data dataSetB.csv --model_dir 任务一_EV采用意愿/02_训练好的模型 --out pred1.csv
```

### 任务二 majority

读取 `task2_majority_model.json` 中 `majority_class`（通常为 1），对所有行输出该常数。

```bash
python 任务二_续航焦虑二分类/01_源代码/infer_task2.py --data dataSetB.csv --model_dir 任务二_续航焦虑二分类/02_训练好的模型 --out pred2.csv
```

### 任务三 soft fusion

1. 加载 `task3_m1_catboost.cbm` 与 prep state → `p1 = P(y=1)`。
2. 加载 `task3_m2_regressor.cbm`：将连续分映射为概率（或按源码与 M1/M4 权重融合；提交包默认权重 M1:0.8, M2:0.15, M4:0.05，部署可简化为 M1+M2）。
3. `pred = 1 if p_soft >= threshold(0.5) else 0`。

```bash
python 任务三_支付意愿二分类/01_源代码/infer_task3.py --data dataSetB.csv --model_dir 任务三_支付意愿二分类/02_训练好的模型 --out pred3.csv
```

### 输出格式建议

推理输出 CSV 至少包含：`样本行号, 预测类标签`；若有真值可附 `真值类标签` 便于自测。
""",
    )

    # Per-task modeling reports
    _write(
        COMP / "任务一_EV采用意愿/04_建模报告.md",
        """# 任务一建模报告：EV 采用意愿（Low / Medium / High）

## 1. 问题与指标

- 目标：`ev_adoption_likelihood` 三分类（有序）。
- 主指标：Accuracy；辅：Macro-F1、QWK、相邻错误率。
- 锁定方案：`soft_vote`，内部 Final Acc = **0.8601**。

## 2. 特征与重要性

- 特征集：原始问卷字段 + 机制交互（收入对数、eco-tech、焦虑-知识比率、充电便利性等），见 `task1_final_config.json` / `task1_model_bundle.json`。
- CatBoost 特征重要性：见公共结果 `feature_importance.csv`（论文包标签 `B01`/`B02`）。
- SHAP（对 M1）：全局 beeswarm、按类、Medium 依赖、waterfall 案例（论文包 `B02_任务一_SHAP图示`）。
  - 说明：可解释性以 **M1 CatBoost** 为代表；提交预测器为 soft_vote。

Top 方向性结论（摘要）：环保-技术亲和、知识/焦虑张力、充电可达性对 Medium↔High 边界影响显著。

## 3. 模型选择依据

| 候选 | 角色 | 结论 |
|------|------|------|
| Baseline / 单模 CatBoost | 对照 | 强，但 Medium 仍难 |
| M2 有序边界 CatBoost | 结构归纳偏置 | 有效但单独不及融合 |
| M3 有序回归+软概率 | 异构概率 | 小权重进入融合 |
| M6 双 Logistic | 可解释线性 | **融合主权重 0.55** |
| soft_vote | 最终 | OOF/Final 最优且稳定 |

选择规则：在 develop 上 OOF 比较 + multiseed 稳定性，再在冻结 final_validation 确认。

## 4. 调参过程

- M1：贝叶斯/网格搜索 iterations≈1109, depth=6, lr≈0.032, l2≈19.9, random_strength / bagging_temperature 同步搜索（见 `m1_params`）。
- soft_vote 权重：在 {M1,M3,M6,baseline} 上网格/离散搜索，锁定 (0.35, 0.10, 0.55, 0)。
- 未使用 final_validation / 测试标签做超参搜索。

## 5. 预测文件

`03_预测结果CSV/task1_predictions.csv`：样本行号、真值、预测（内部验证集）。
""",
    )

    _write(
        COMP / "任务二_续航焦虑二分类/04_建模报告.md",
        """# 任务二建模报告：续航焦虑相关二分类

## 1. 问题与指标

- 目标二分类；主指标 Accuracy（赛方排名用）。
- 锁定方案：`M` 多数类常数分类器，Final Acc = **0.6498**（≈多数类基线）。

## 2. 为何不是复杂模型

完整证据链（论文包 `C01`/`C02`）：

1. 可分性弱：|AUC-0.5| 极小；
2. 置换检验 / Bootstrap：未见稳定可部署增益；
3. ACC 冲高（焦虑消融、缺失指示、细阈值、stacking、跨任务 OOF 辅助特征）：最大 lift ≈ **+0.000075**，门控失败；
4. 故在 0-1 损失下，**多数类为 Bayes 稳健决策**。

## 3. 特征重要性

在探索阶段计算了互信息、消融与缺失率分析（见论文包 CSV），但因无部署信号，正式模型**不依赖特征权重**。报告中保留探索性重要性表供论文讨论“弱信号”而非“高分特征”。

## 4. 模型选择与调参

- 候选含树模型、EB 分层、阈值扫描、stacking 等；
- 选择规则：相对多数类 Acc 提升 > 0.0005 且 AUC 间隙门槛；**全部未过** → 锁定 M；
- 无传统超参调优产物（常数分类器仅需统计 develop 上多数类）。

## 5. 预测文件

`03_预测结果CSV/task2_predictions.csv`：全部预测为 majority_class（通常为 1）。
""",
    )

    _write(
        COMP / "任务三_支付意愿二分类/04_建模报告.md",
        """# 任务三建模报告：支付意愿二分类

## 1. 问题与指标

- 由支付意愿相关连续分阈值化得到二分类（见源码 `build_target`）。
- 锁定方案：`boost_B+B_concern_soft`，Final Acc = **0.8529**（OOF≈0.8619）。

## 2. 特征与重要性

- 特征：通勤/出行、充电可达、知识、电池顾虑、环保/技术、收入等 + **concern_knowledge/tech 比率与缺口**。
- 重要性：`results/task3` 中 CatBoost importance 与 SHAP（论文包 `D01`/`D02`）。
- SHAP 显示知识、环保意识、收入与顾虑机制变量对高支付意愿有清晰方向。

## 3. 模型选择依据

| 模块 | 作用 |
|------|------|
| M1 CatBoost 分类 | 主概率 |
| M2 回归分支 | 保留连续分信息，soft 融入 |
| M4 小权重 | 融合搜索中的补充项 |
| 阈值 0.5 | soft_best 锁定 |

在 boost 特征集与 fair-compare 后选择当前 soft 权重。

## 4. 调参过程

- M1：iterations≈619, depth=6, lr≈0.036, l2≈8.0，及 random_strength / bagging_temperature（见 `task3_final_config.json`）。
- 融合：线性/ soft 权重网格，最佳 soft `{M1:0.8, M2:0.15, M4:0.05}`，threshold=0.5。
- 划分与任务一相同原则：OOF 选型，final_validation 仅报告。

## 5. 预测文件

`03_预测结果CSV/task3_predictions.csv`。
""",
    )


def write_infer_scripts() -> None:
    out = _ensure(ROOT / "delivery_staging" / "infer_scripts")

    _write(
        out / "infer_task1.py",
        r'''"""任务一推理：soft_vote = 0.55*M6 + 0.35*M1 + 0.10*M3"""
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
        here.parents[1] / "00_公共源代码",  # 任务目录旁
        here.parents[2] / "00_公共源代码",  # 比赛提交材料根
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
    # CatBoost classes order
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
''',
    )

    _write(
        out / "infer_task2.py",
        r'''"""任务二推理：多数类常数分类器"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import pandas as pd


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
''',
    )

    _write(
        out / "infer_task3.py",
        r'''"""任务三推理：M1/M2 soft 融合（M4 权重并入归一化）"""
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
''',
    )


def build_paper_pack() -> None:
    _write(
        PAPER / "00_索引_请从这里找图和说明.md",
        """# 论文素材索引（按标签查找）

发给组长写论文用。每个标签对应一个文件夹；**先看本索引，再按标签打开目录**。

## 总成绩

| 标签 | 内容 | 路径 |
|------|------|------|
| **A01** | 总成绩、锁定方案、加权公式 | `A01_总成绩与锁定方案/` |

## 任务一

| 标签 | 内容 | 路径 |
|------|------|------|
| **B01** | 方法、消融、集成、结论文档 | `B01_任务一_方法文档/` |
| **B02** | SHAP / 混淆矩阵 / ROC 等图 | `B02_任务一_图示SHAP与验证/` |
| **B03** | 配置、指标、重要性 CSV | `B03_任务一_配置与数值表/` |

## 任务二

| 标签 | 内容 | 路径 |
|------|------|------|
| **C01** | 弱信号叙事全文（方法/结论/可分性等） | `C01_任务二_弱信号文档/` |
| **C02** | ACC 冲高失败证据、门控、置换等 | `C02_任务二_证据表与诊断/` |
| **C03** | 最终锁定配置与验证指标 | `C03_任务二_锁定配置/` |

## 任务三

| 标签 | 内容 | 路径 |
|------|------|------|
| **D01** | paper_focus、结论、NOTE | `D01_任务三_方法文档/` |
| **D02** | SHAP 与 final_validation 图 | `D02_任务三_图示SHAP与验证/` |
| **D03** | 配置、消融、boost 报告 | `D03_任务三_配置与数值表/` |

## 数据

| 标签 | 内容 | 路径 |
|------|------|------|
| **E01** | 数据概览与目标分布 | `E01_数据概览/` |

## 建议论文结构对应

1. 问题与数据 → **E01** + 各任务 NOTE  
2. 任务一方法与结果 → **B01–B03**（图优先用 B02）  
3. 任务二弱信号论证 → **C01–C02**（必须强调“门控失败→多数类”）  
4. 任务三方法与解释 → **D01–D03**  
5. 总结成绩表 → **A01**
""",
    )

    # A01
    a01 = _ensure(PAPER / "A01_总成绩与锁定方案")
    _copy(ROOT / "results/README.md", a01 / "成绩总表_README.md")
    _write(
        a01 / "LOCKED_SCORES.md",
        """# 锁定成绩（内部 final_validation）

| 任务 | 方案 | Acc |
|------|------|-----|
| 1 | soft_vote | 0.8601 |
| 2 | majority M | 0.6498 |
| 3 | boost_B+B_concern_soft | 0.8529 |

加权：0.2*0.8601 + 0.4*0.6498 + 0.4*0.8529 = **0.7731**
""",
    )

    # B01 docs
    b01 = PAPER / "B01_任务一_方法文档"
    for name in [
        "task1_methodology.md",
        "task1_model_comparison.md",
        "task1_feature_ablation.md",
        "task1_ensemble_analysis.md",
        "task1_ordinal_analysis.md",
        "task1_explainability.md",
        "task1_final_conclusion.md",
        "NOTE.md",
        "README.md",
    ]:
        _copy(ROOT / "docs/task1" / name, b01 / name)

    # B02 figures
    b02 = PAPER / "B02_任务一_图示SHAP与验证"
    _copy(ROOT / "results/task1/explain", b02 / "SHAP")
    _copy(ROOT / "results/task1/figures", b02 / "验证图")
    _write(
        b02 / "图注标签说明.md",
        """# B02 图注建议

- `SHAP/shap_summary_beeswarm.png` — 全局 SHAP 蜂群图（主图）
- `SHAP/shap_by_class.png` — 按 Low/Medium/High 的平均 |SHAP|
- `SHAP/shap_dependence_*_Medium.png` — Medium 类依赖图（讨论边界）
- `SHAP/waterfall_typical_*.png` — 个案瀑布图（典型/边界/误分）
- `验证图/confusion_matrix.png` — 最终验证混淆矩阵
- `验证图/roc_ovr.png` — OvR ROC

注意：SHAP 解释对象为 **M1 CatBoost**；提交预测为 soft_vote。
""",
    )

    # B03 tables
    b03 = PAPER / "B03_任务一_配置与数值表"
    for rel in [
        "task1_final_config.json",
        "metrics.json",
        "final_validation/metrics.json",
        "feature_importance.csv",
        "confusion_matrix.csv",
        "ablation/feature_ablation.csv",
        "ablation/feature_selection.json",
        "ensemble/soft_vote_best.json",
        "stability/multiseed_summary.csv",
        "explain/global_mean_abs_shap.csv",
        "splits/split_config.json",
    ]:
        _copy(ROOT / "results/task1" / rel, b03 / Path(rel).name if "/" not in rel else b03 / rel)

    # C01
    c01 = PAPER / "C01_任务二_弱信号文档"
    for name in [
        "task2_final_conclusion.md",
        "task2_problem_redefinition.md",
        "task2_separability.md",
        "task2_label_audit.md",
        "task2_statistical_tests.md",
        "task2_empirical_bayes.md",
        "task2_model_comparison.md",
        "task2_uncertainty.md",
        "task2_clustering.md",
        "task2_acc_boost.md",
        "task2_data_collection_recommendation.md",
        "NOTE.md",
    ]:
        _copy(ROOT / "docs/task2" / name, c01 / name)

    # C02
    c02 = PAPER / "C02_任务二_证据表与诊断"
    for rel in [
        "diagnostics/majority_baseline.json",
        "diagnostics/acc_boost",
        "diagnostics/leakage_audit.json",
        "separability",
        "statistics",
        "features/feature_ablation.csv",
        "features/feature_ablation_summary.json",
        "baseline/baseline_comparison.csv",
        "label_audit",
        "missingness",
        "empirical_bayes/eb_best_config.json",
        "clustering",
        "ensemble",
    ]:
        _copy(ROOT / "results/task2" / rel, c02 / Path(rel).name if "/" not in rel and (ROOT / "results/task2" / rel).is_file() else c02 / rel)

    # C03
    c03 = PAPER / "C03_任务二_锁定配置"
    for rel in [
        "final/task2_final_config.json",
        "final/candidate_comparison.csv",
        "final/model_manifest.json",
        "final_validation/metrics.json",
        "splits/split_config.json",
    ]:
        _copy(ROOT / "results/task2" / rel, c03 / Path(rel).name)

    # D01
    d01 = PAPER / "D01_任务三_方法文档"
    for name in ["paper_focus.md", "final_conclusion.md", "NOTE.md"]:
        _copy(ROOT / "docs/task3" / name, d01 / name)

    # D02
    d02 = PAPER / "D02_任务三_图示SHAP与验证"
    _copy(ROOT / "results/task3/explain", d02 / "SHAP")
    _copy(ROOT / "results/task3/final_validation", d02 / "验证图")
    _write(
        d02 / "图注标签说明.md",
        """# D02 图注建议

- `SHAP/shap_summary*.png` / `shap_direction.png` / `shap_by_class.png` — 全局解释
- `SHAP/shap_dependence_*.png` — 关键依赖
- `SHAP/waterfall_typical_*.png` — 高低意愿与边界个案
- `验证图/confusion_matrix.png` / `roc_curve.png` / `feature_importance.png` — 验证集表现
""",
    )

    # D03
    d03 = PAPER / "D03_任务三_配置与数值表"
    for rel in [
        "final/task3_final_config.json",
        "metrics.json",
        "final_validation/metrics.json",
        "final_validation/confusion_matrix.csv",
        "ablation/feature_ablation.csv",
        "ablation/feature_selection.json",
        "diagnostics/boost_continue",
        "diagnostics/fair_compare",
        "explain/global_mean_abs_shap.csv",
        "splits/split_config.json",
    ]:
        src = ROOT / "results/task3" / rel
        _copy(src, d03 / (Path(rel).name if src.is_file() else rel))

    # E01
    e01 = PAPER / "E01_数据概览"
    _copy(ROOT / "results/overview", e01 / "overview")


def zip_delivery() -> Path:
    zip_path = ROOT / f"delivery_for_leader_{STAMP}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in DELIVERY.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(DELIVERY.parent))
    return zip_path


def main() -> int:
    if DELIVERY.exists():
        shutil.rmtree(DELIVERY)
    _ensure(COMP)
    _ensure(PAPER)

    write_infer_scripts()
    format_predictions()
    copy_models()
    copy_source()
    # copy infer into task folders after copy_source created dirs
    infer_dir = ROOT / "delivery_staging" / "infer_scripts"
    for task, folder in [
        ("infer_task1.py", "任务一_EV采用意愿/01_源代码"),
        ("infer_task2.py", "任务二_续航焦虑二分类/01_源代码"),
        ("infer_task3.py", "任务三_支付意愿二分类/01_源代码"),
    ]:
        _copy(infer_dir / task, COMP / folder / task)

    write_competition_docs()
    build_paper_pack()

    _write(
        DELIVERY / "00_给组长_请先读.txt",
        f"""发给组长的两套材料（{STAMP}）

1) 01_比赛提交材料/
   - 对应初赛：代码、模型文件、预测 CSV、建模报告、Introduction.md
   - 先看：01_比赛提交材料/00_给组长_比赛提交请先读.md

2) 02_论文写作素材_按标签查找/
   - 先看：02_论文写作素材_按标签查找/00_索引_请从这里找图和说明.md
   - 用标签 A01/B01/B02/... 在索引表定位图和文档

打包文件：delivery_for_leader_{STAMP}.zip（与本文件夹同级或同目录）
""",
    )

    z = zip_delivery()
    print("DELIVERY=", DELIVERY)
    print("ZIP=", z)
    print("ZIP_SIZE_MB=", round(z.stat().st_size / 1024 / 1024, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
