# 任务二 ACC 冲高实验结论

实验目录：`outputs/task2/diagnostics/acc_boost/`（精简副本：`results/task2/diagnostics/acc_boost/`）

## 做了什么（提示词合法边界内）

1. **跨任务安全 OOF**：任务一采用倾向概率 + 任务三焦虑概率/评分预测（绝不用 `ev_adoption_likelihood` 真值作输入）
2. **`range_anxiety_score` 消融**（uncertain 问卷字段）
3. **缺失指示特征**
4. **细阈值** 0.30–0.80 step=0.001（优化 Accuracy）
5. **二层 stacking**（Logistic/CatBoost ± aux）
6. **门控**：OOF lift ≥ 0.0005 + 多 seed + Bootstrap 才允许改锁

## 结果

| 项目 | 数值 |
|------|------|
| 多数类 Acc | 0.64975 |
| 最佳候选 | `B_safe_missing_logistic@thr0.550` |
| 最佳 Acc | 0.649825 |
| 相对多数类 lift | **+0.000075**（噪声级） |
| 门控 | **未通过** → **保持正式方案 M** |

跨任务 aux、焦虑分、stacking **均未**带来稳定 Acc 增益；AUC 仍≈0.50–0.508。

## 结论

在当前可观测字段下，ACC2 的信息上限贴近视类先验。继续堆模型无法诚实抬分；正式提交维持多数类 Bayes 决策（提示词情况 C）。
