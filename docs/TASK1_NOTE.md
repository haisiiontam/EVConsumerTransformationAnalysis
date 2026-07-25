# 任务一优化简记

## 可分性结论

目标：`ev_adoption_likelihood`（Low / Medium / High）

标签分布约：High 59.3%，Medium 24.2%，Low 16.5%；多数类基准 ≈ **0.593**。

| 特征 | 互信息 MI（约） | 说明 |
|------|----------------|------|
| ev_knowledge_score | 0.35 | 最强（官方未列，但很有用） |
| range_anxiety_score | 0.32 | 强（官方建议） |
| technology_affinity_score | 0.31 | 强（官方建议） |
| environmental_awareness_score | 0.31 | 强（官方建议） |
| annual_income | 0.10 | 中等（官方建议） |
| fuel_expense_per_month / education_level | ~0 | 官方建议中几乎无信号 |

## OOF 对比（开发集）

| 配置 | 特征 | OOF Accuracy |
|------|------|--------------|
| A | 官方 6 特征 | ~0.78 |
| B | 安全扩展 | ~0.85+ |
| C | 安全扩展 + 机制特征 | 通常最优或接近最优 |

禁止特征：`ev_adoption_likelihood`、`home_charging_available`、能耗/充电费用后验字段。

## 运行

```bash
python main.py --data dataSetB.csv --output outputs --task task1
```
