# Published Results Index

本目录是**入库发布用精简结果**。单任务 ACC 已由 `outputs/*/final_validation/predictions.csv` 重算核对。

## 锁定成绩（Final holdout, n=10000）

| 任务 | 锁定方案 | Final Acc | 正确数 |
|------|----------|-----------|--------|
| 任务一 | `soft_vote`（M6 0.55 + M1 0.35 + M3 0.10） | **0.8607** | 8607 |
| 任务二 | `M`（多数类 1） | **0.6498** | 6498 |
| 任务三 | `boost_B+B_concern_soft` | **0.8520** | 8520 |

加权：`0.2*ACC1 + 0.4*ACC2 + 0.4*ACC3` = **0.77286**（内部验证，非正式排名）。

## metrics 文件

- `overview/overall_summary.json` / `locked_scores.json`
- `task1/metrics.json` + `task1/final_validation/metrics.json`
- `task2/metrics.json` + `task2/final_validation/metrics.json`
- `task3/metrics.json` + `task3/final_validation/metrics.json`
