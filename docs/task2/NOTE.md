# 任务二说明（弱信号诊断与稳健决策）

## 锁定结论

- **正式方案：`M`（全局多数类 = 1）**
- Final Acc = **0.6498**（= 类别 1 比例）
- 原因：在严格排除跨任务标签后，现有消费者画像**无稳定可泛化的判别信号**

## 证据链（详见同目录各 md + `results/task2/`）

1. 单变量 max \|AUC−0.5\| ≈ 0.009  
2. 精确重复画像冲突 = 0；kNN 一致率 **低于** 全局先验  
3. Logistic / CatBoost / LGBM / XGB / ExtraTrees / EB / 聚类先验 Acc 均未稳定超过多数类  
4. 置换检验：Acc lift 不显著；AUC 仅 ≈0.506，不构成可部署信号（`supports_signal=false`）  
5. Bootstrap：Acc 提升 95% CI 含 0  

## 代码入口

```bash
python main.py --data dataSetB.csv --task task2 --mode quick
python main.py --data dataSetB.csv --task task2 --mode full --resume
```

包路径：`src/tasks/task2/`（非旧单体 `task2.py`）。

## 论文贡献点

- 可预测性评估优先于堆模型  
- 层次经验贝叶斯与分群先验作**对照/排序**，非强行 Acc 创新  
- 明确未观测结构变量：停车位、车库、产权、物业许可、配电容量等  
