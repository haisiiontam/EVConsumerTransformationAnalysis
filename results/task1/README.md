# Task1 输出目录说明（full 运行）

> `outputs/` 已被 `.gitignore` 忽略，本目录不会进 Git。  
> 代码与论文摘要同步在仓库 `docs/`、`src/tasks/task1/`。

## 本次 full 结论（权威）

| 项 | 值 |
|---|---|
| 方案 | `soft_vote`（权重见 `ensemble/soft_vote_best.json`） |
| OOF Acc | 0.858725 |
| Final Acc | **0.8601** |
| Baseline Final | 0.8592 |
| recommend_merge | **true** |
| 运行状态 | `run_state.json` → `exit_status=success` |

## 推荐阅读顺序

1. `metrics.json` / `final_validation/metrics.json` — 最终分数  
2. `task1_final_config.json` — 锁定配置与特征  
3. `paper/task1_final_conclusion.md` — 文字结论  
4. `resume_audit/resume_recommendation.md` — 断点审计记录  
5. `ensemble/`、`oof/`、`stability/` — 融合与稳定性细节  

## 目录结构

```
task1/
  README.md                 ← 本说明
  metrics.json              ← 根目录快捷入口（与 final_validation 同步）
  task1_final_config.json
  run_state.json            ← 阶段续跑状态
  validation_predictions.csv
  confusion_matrix.csv
  feature_importance.csv

  splits/                   ← 冻结划分（勿改）
  baseline/                 ← 稳定基线复现
  ablation/                 ← 特征消融与最终特征
  diagnostics/              ← 特征集 / uncertain 报告
  tuning/                   ← Optuna trials、best params、阈值
  oof/                      ← 各模型完整 OOF 概率（融合用）
  checkpoints/              ← 每折原子 checkpoint
  ensemble/                 ← soft_vote / stacking / bias
  stability/                ← 多 seed
  final_validation/         ← 锁定一次最终验证（权威）
  figures/                  ← 最终验证图
  paper/                    ← 论文材料
  resume_audit/             ← 断点审计
  logs/                     ← 当前成功 run 日志
  explain/                  ← SHAP（本机未生成，见 README.txt）
  _archive/                 ← 旧 quick / 失败日志，勿当正式结果
```

## 不要当作本次 full 结果的内容

- `_archive/stale_quick/`：mtime 早于 full 开始时间的旧 quick 产物  
- `_archive/logs_old/`：中断/失败的 resume 日志  

## 续跑命令（参考）

```powershell
python main.py --data dataSetB.csv --output outputs --task task1 --mode full --resume --skip-optuna --run-id <id>
```
