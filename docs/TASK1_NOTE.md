# 任务一优化简记（全量增强版）

## 正式结论（`--mode full`，已完成）

| 项 | 值 |
|---|---|
| 稳定基线 | `B_safe_raw`，OOF ≈ **0.8560**，Final ≈ **0.8592** |
| 锁定方案 | **`soft_vote`**（M6 0.55 + M1 0.35 + M3 0.10） |
| 增强 OOF | **0.8587** |
| 增强 Final | **0.8601** |
| 多 seed | mean ≈ 0.8563（相对基线稳定） |
| **合并建议** | **是（YES）** |

权威产物目录：`outputs/task1/`（见其中 `README.md`）。  
论文材料：`outputs/task1/paper/` 与仓库 `docs/task1_*.md`。

稳定版快照：`backups/task1_stable_50662a4.py`。增强代码：`src/tasks/task1/`（含断点续跑 / 每折 checkpoint / Optuna CSV 复用）。

## 运行

```bash
# 冒烟
python main.py --data dataSetB.csv --output outputs --task task1 --mode quick

# 论文全量
python main.py --data dataSetB.csv --output outputs --task task1 --mode full

# 断点续跑（跳过已完成 Optuna）
python main.py --data dataSetB.csv --output outputs --task task1 --mode full --resume --skip-optuna --start-stage m1_catboost
```

禁止特征：`ev_adoption_likelihood`、`home_charging_available`、能耗/充电费用后验字段（见 `uncertain_features.json`）。

## 说明

- SHAP：本机 Python 3.14 上易原生崩溃，已改为独立子进程；失败不阻断主结果。可单独跑 `scripts/run_task1_shap_worker.py`。
- 旧 quick 产物已迁到 `outputs/task1/_archive/stale_quick/`，勿与本次 full 混用。
