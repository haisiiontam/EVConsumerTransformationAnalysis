# README — 电动汽车消费者转型分析（三任务终版）

## 1. 项目目的

2026 第五届「创新杯」大学生大数据挑战赛 · 初赛 B  
**从燃油到纯电——电动汽车消费者转型分析**

| 任务 | 目标 | 锁定方案 | Final Acc |
|------|------|----------|-----------|
| 任务一 | `ev_adoption_likelihood`（Low/Medium/High） | `soft_vote` | **0.8607** |
| 任务二 | `home_charging_available`（0/1） | `M` 多数类 | **0.6498** |
| 任务三 | `range_anxiety_score > 5` → 1 | `boost_B+B_concern_soft` | **0.8520** |

内部加权分（非官方）：`0.2*ACC1 + 0.4*ACC2 + 0.4*ACC3` = **0.77286**  
详见 [`results/README.md`](results/README.md) 与 [`docs/README.md`](docs/README.md)。

## 2. 环境

Python 3.10+（SHAP 推荐 3.12 独立环境 `.venv-shap`）。

```bash
pip install -r requirements.txt
# 可选：uv 创建 .venv-shap 后安装 catboost shap lightgbm xgboost matplotlib
```

## 3. 运行

```bash
# 任务一 / 三（支持断点）
python main.py --data dataSetB.csv --task task1 --mode full --resume
python main.py --data dataSetB.csv --task task3 --mode full --resume

# 任务二（弱信号诊断；先 quick 再 full）
python main.py --data dataSetB.csv --task task2 --mode quick
python main.py --data dataSetB.csv --task task2 --mode full --resume
```

常用参数：`--start-stage`、`--stop-after-stage`、`--run-id`、`--skip-optuna`、`--skip-multiseed`、`--skip-shap`、`--skip-deep`、`--audit-only`。

本地完整跑次写入 **`outputs/`（gitignore）**；仓库发布精简结果在 **`results/`（跟踪）**。

## 4. 仓库结构

```text
├── main.py
├── requirements.txt
├── README.md
├── docs/                 # 论文与任务说明（入库）
│   ├── task1/
│   ├── task2/
│   └── task3/
├── results/              # 锁定指标、关键表、SHAP 图（入库）
├── scripts/              # SHAP worker、诊断脚本
└── src/
    ├── config.py
    ├── common/           # 指标、断点、StageGate、可视化
    └── tasks/
        ├── task1/        # 全量优化包
        ├── task2/        # 弱信号诊断包
        └── task3/        # 分类-回归融合包
```

## 5. 三任务要点（一句话）

- **任务一**：有序结构 + 异构模型 + OOF 软投票；SHAP 用 CatBoost 原生值。  
- **任务二**：证明弱信号后采用多数类稳健决策，并给出补采数据建议。  
- **任务三**：机制特征消融 + 分类/回归双通道软融合；解释见 SHAP 套件。

## 6. 重要约定

- 最终 holdout **不**用于选模 / 选阈值 / 选融合权重。  
- 任务二禁止直接使用 `ev_adoption_likelihood` 真标签。  
- 任务三输入不含 `range_anxiety_score`（仅作标签源）。  
- 不把 `outputs/` 大文件与 `.cbm` 提交进库。
