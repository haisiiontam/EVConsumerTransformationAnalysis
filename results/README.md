# Published Results Index

本目录是**入库发布用精简结果**，与本地可丢弃的 `outputs/` 运行缓存分离。

## 锁定成绩（Final holdout）

| 任务 | 锁定方案 | Final Acc | 说明 |
|------|----------|-----------|------|
| 任务一 | `soft_vote`（M6 0.55 + M1 0.35 + M3 0.10） | **0.8601** | 有序三分类 + 异构软投票 |
| 任务二 | `M`（多数类 1） | **0.6498** | 弱信号证伪后的 Bayes 稳健决策 |
| 任务三 | `boost_B+B_concern_soft` | **0.8529** | 分类–回归软融合；OOF≈0.8619 |

加权参考：`0.2*ACC1 + 0.4*ACC2 + 0.4*ACC3` ≈ **0.7331**（内部验证，非官方成绩）。

## 目录

- `overview/`：全表概览与目标分布
- `task1/`：配置、指标、消融、稳定性、SHAP 图、验证图
- `task2/`：弱信号证据链（可分性、标签审计、置换/Bootstrap、EB、聚类对照）
- `task3/`：锁定配置、消融、boost/fair-compare 报告、SHAP 套件

## 不入库（仅本地 `outputs/`）

checkpoints、日志、完整 OOF 大表、`.cbm` 模型、`_archive` 历史跑次。可用代码 + 冻结 `splits/split_config.json` 复现。

## 论文说明

权威叙事见 `docs/task1/`、`docs/task2/`、`docs/task3/`。
