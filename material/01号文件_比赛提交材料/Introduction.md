# Introduction — EV 消费者转化分析初赛提交说明

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
.venv\Scripts\activate
pip install -r requirements.txt
# 或: uv pip install --python .venv-shap\Scripts\python.exe -r requirements.txt
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
python 任务一_EV采用意愿/01_源代码/infer_task1.py --data B题数据集.csv --model_dir 任务一_EV采用意愿/02_训练好的模型 --out pred1.csv
```

### 任务二 majority

读取 `task2_majority_model.json` 中 `majority_class`（通常为 1），对所有行输出该常数。

```bash
python 任务二_续航焦虑二分类/01_源代码/infer_task2.py --data B题数据集.csv --model_dir 任务二_续航焦虑二分类/02_训练好的模型 --out pred2.csv
```

### 任务三 soft fusion

1. 加载 `task3_m1_catboost.cbm` 与 prep state → `p1 = P(y=1)`。
2. 加载 `task3_m2_regressor.cbm`：将连续分映射为概率（或按源码与 M1/M4 权重融合；提交包默认权重 M1:0.8, M2:0.15, M4:0.05，部署可简化为 M1+M2）。
3. `pred = 1 if p_soft >= threshold(0.5) else 0`。

```bash
python 任务三_支付意愿二分类/01_源代码/infer_task3.py --data B题数据集.csv --model_dir 任务三_支付意愿二分类/02_训练好的模型 --out pred3.csv
```

### 输出格式建议

推理输出 CSV 至少包含：`样本行号, 预测类标签`；若有真值可附 `真值类标签` 便于自测。
