# 任务三优化简记（全量增强版）

## 标签

`range_anxiety_score > 5 -> 1`，否则 `0`。  
**分类输入严禁** `range_anxiety_score` 及其派生字段。

## 历史候选 vs 同 split 公平对比（2026-07-27）

| 项 | 历史归档（旧 holdout） | 同冻结 split 重放 |
|---|---|---|
| 当前 M1 OOF / Final | — | **0.86185 / 0.8528** @0.5 |
| 历史 clf_best_hp OOF / Final | 0.8615 / 0.8534 @0.495 | 0.861425 / **0.8530** @0.495 |

正式锁定仍按 **OOF**：`current_M1_fixed`（Final 0.8528）。  
详见 `outputs/task3/diagnostics/fair_compare/`。不建议为此上 GAM/TabNet。

## 续训（M1/M2 软融合，2026-07-27）

| 项 | 续训前 | 续训后 |
|---|---|---|
| OOF | 0.86185 | **0.861925** |
| Final | 0.8528 | **0.8529** |
| 方案 | M1@0.5 | `boost_B+B_concern_soft`（M1 0.80 / M2 0.15 / M4 0.05）@0.5 |

详见 `outputs/task3/diagnostics/boost_continue/`。增益很小；GAM/TabNet 仍非优先。

## 代码

- 包：`src/tasks/task3/`
- 稳定备份：`backups/task3_stable_c6808df.py`
- 续跑框架：`src/common/{io_utils,stage_control,fold_checkpoints}.py`

## 运行

```bash
# 仅审计
python main.py --data dataSetB.csv --output outputs --task task3 --audit-only

# 冒烟（至 baseline）
python main.py --data dataSetB.csv --output outputs --task task3 --mode quick --stop-after-stage baseline

# 全量（确认后再跑）
python main.py --data dataSetB.csv --output outputs --task task3 --mode full

# 断点续跑
python main.py --data dataSetB.csv --output outputs --task task3 --mode full --resume --skip-optuna
```
