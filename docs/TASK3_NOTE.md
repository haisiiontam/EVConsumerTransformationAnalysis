# 任务三优化简记

## 可分性结论

对标签 `range_anxiety_score > 5`（高焦虑=1）：

| 特征 | 单变量 \|AUC−0.5\| | 说明 |
|------|-------------------|------|
| environmental_awareness_score | ~0.34 | 强信号（负相关） |
| ev_knowledge_score | ~0.33 | 强信号（官方特征） |
| technology_affinity_score | ~0.31 | 强信号（负相关） |
| annual_income | ~0.21 | 中强信号 |
| 官方其余 4 个通勤/充电特征 | ~0.00 | 几乎无信号 |

标签接近平衡（约 50.8% / 49.2%），多数类基准 ≈ 0.51。

## 模型对比（开发集 OOF）

| 配置 | 特征 | OOF Accuracy | OOF AUC |
|------|------|--------------|---------|
| A | 官方 5 特征 | ~0.77 | ~0.80 |
| B | 安全扩展 18 特征 | ~0.86 | ~0.91 |
| C | 安全扩展 + 机制特征 | ~0.86 | ~0.90 |

最终通常选定 **B 或 C**（Accuracy 主指标）。`range_anxiety_score` / 任务一/二目标不进特征。

## 运行

```bash
python main.py --data dataSetB.csv --output outputs --task task3
```
