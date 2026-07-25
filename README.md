# 电动汽车消费者转型分析 · 三任务基线 + 任务二优化

## 1. 项目目的

本项目面向 **2026 年第五届「创新杯」大学生大数据挑战赛 · 初赛 B**：

> 从燃油到纯电——电动汽车消费者转型分析

- **任务一 / 任务三**：保留朴素基线（官方建议特征 + 单次分层划分 + 单一 CatBoost）
- **任务二**：升级为诊断 + 安全扩展特征 + 机制特征 + OOF + 概率融合 + 阈值搜索，且**最终 20% 验证集不参与模型选择**

## 2. 三个任务定义

| 任务 | 目标 | 类型 | 说明 |
|------|------|------|------|
| 任务一 | `ev_adoption_likelihood` | 三分类 Low / Medium / High | 电动汽车采用意愿 |
| 任务二 | `home_charging_available` | 二分类 0 / 1 | 家庭充电条件（不是“未来安装意愿”） |
| 任务三 | 由 `range_anxiety_score` 构造 | 二分类：`> 5` → 1，否则 0 | 里程焦虑敏感度；该评分不进特征 |

## 3. 环境安装

需要 Python 3.10+。

```bash
pip install -r requirements.txt
```

## 4. 数据字段要求

列名使用 snake_case。任务二官方基础字段必须存在：

`city_type`, `nearest_charging_station_km`, `education_level`, `annual_income`, `vehicle_age_years`, `home_charging_available`

任务一 / 任务三所需字段见 `src/config.py`。

## 5. 运行命令

```bash
# 三个任务全部运行
python main.py --data dataSetB.csv --output outputs --task all

# 只跑任务二优化
python main.py --data dataSetB.csv --output outputs --task task2

# 可选独立测试集（仅任务二在调参完成后预测）
python main.py --data dataSetB.csv --test data/test.csv --output outputs --task task2
```

常用参数：

- `--task`：`all` / `task1` / `task2` / `task3`
- `--test-size`：最终验证集比例（默认 0.2）
- `--task2-folds`：默认 5
- `--task2-alpha-step`：默认 0.05
- `--task2-threshold-min/max/step`：默认 0.30 / 0.80 / 0.005

## 6. 项目结构

```text
project/
├── main.py
├── requirements.txt
├── README.md
└── src/
    ├── config.py
    ├── common/
    │   ├── data_utils.py
    │   ├── preprocessing.py
    │   ├── metrics.py
    │   ├── visualization.py
    │   └── baseline.py
    └── tasks/
        ├── task1.py
        ├── task2.py
        └── task3.py
```

## 7. 任务二优化版说明

### 7.1 基线为什么失效

朴素任务二在类别不平衡下容易退化成“几乎全预测为 1”。此时 Accuracy 可能接近多数类比例（约 0.65），但几乎识别不了 0 类。因此不能只看 Accuracy，必须同时报告：

- 多数类基准 Accuracy
- Balanced Accuracy
- 0/1 类 Recall
- 混淆矩阵与预测类别计数

### 7.2 多数类基准

```text
majority_baseline_accuracy = 数量最多的类别样本数 / 总样本数
```

若模型 Accuracy 不高于该基准，说明尚未形成有效分类提升。

### 7.3 什么是 OOF

OOF（Out-of-Fold）概率：开发集内做分层 K 折，每条样本的概率由**未见过该样本**的模型产生。`alpha` 与 `threshold` 只在 OOF 上搜索，最终 20% 验证集只评估一次。

### 7.4 普通 CatBoost vs Balanced CatBoost

- **普通模型**：不额外调整类别权重
- **Balanced 模型**：`auto_class_weights="Balanced"`，提升对少数类的关注

二者不一定谁更好，需用 OOF 比较。

### 7.5 五个容易混淆的概念

| 概念 | 含义 |
|------|------|
| 特征重要性 | 模型内部对特征贡献的度量，**不是**手工预测权重 |
| 类别权重 | Balanced 模型对 0/1 类的样本权重 |
| 模型融合权重 `alpha` | `p_final = alpha * p_plain + (1-alpha) * p_balanced` |
| 官方任务评分权重 | 综合分里 ACC2 占 40%，**不乘到单条概率上** |
| 分类阈值 `threshold` | `p_final >= threshold` 则预测为 1 |

### 7.6 机制特征含义

- `charging_gap_index`：距离远且便利度低 → 公共充电缺口大
- `charging_convenience_index`：公共充电便利程度
- `commute_ratio`：通勤在周出行中的占比
- `fuel_income_pressure`：年燃油支出相对年收入压力
- `replacement_pressure`：车龄 × 月燃油支出

排除字段：`home_charging_available`、`ev_adoption_likelihood`、`range_anxiety_score`、`monthly_energy_consumption_kwh`、`monthly_charging_cost`。

### 7.7 配置对比（A–E + M）

| 配置 | 特征 | 模型 | 阈值 |
|------|------|------|------|
| A | 官方五特征 | plain | 0.5 |
| B | 安全扩展原始特征 | plain | 0.5 |
| C | 安全扩展 + 机制/代理特征 | plain | 0.5 |
| D | 安全扩展 + 机制/代理特征 | balanced | 0.5 |
| E | 同上 | plain+balanced 融合 | OOF 搜索 |
| M | 无 | 多数类基线 | — |

### 7.8 若 AUC 真的≈0.5（弱信号协议）

流水线会在**开发集**上输出：

- `diagnostics/separability.csv`：单变量可分性
- `diagnostics/leak_risk_probe.csv`：对被排除字段的探查（不进模型）
- `diagnostics/WEAK_SIGNAL_REPORT.md`：结论与建议

当安全特征与泄露探查都接近随机，且模型 OOF Accuracy 无法显著超过多数类时，**自动选用配置 M**。  
这意味着：再调阈值/融合也抬不高官方 Accuracy，竞赛应把精力转向任务一/三。

### 7.9 任务二输出

```text
outputs/task2/
├── diagnostics/
│   ├── majority_baseline.json
│   ├── separability.csv
│   ├── leak_risk_probe.csv
│   ├── weak_signal_report.json
│   ├── WEAK_SIGNAL_REPORT.md
│   ├── feature_quality.csv
│   ├── target_distribution.csv
│   ├── categorical_target_rates.csv
│   └── numeric_group_statistics.csv
├── oof/
│   ├── plain_oof_predictions.csv
│   ├── balanced_oof_predictions.csv
│   ├── fold_metrics.csv
│   └── model_comparison.csv
├── tuning/
│   ├── blend_threshold_search.csv
│   └── best_blend_config.json
├── final_validation/
│   ├── metrics.json
│   ├── predictions.csv
│   ├── confusion_matrix.csv / .png
│   ├── probability_distribution.png
│   ├── roc_curve.png
│   ├── feature_importance_*.csv / .png
│   └── train_vs_validation.csv
└── metrics.json
```

## 8. 指标与总成绩

内部验证加权分（**validation weighted score**，非最终比赛成绩）：

```text
weighted_score = 100 × (0.2 × ACC1 + 0.4 × ACC2 + 0.4 × ACC3)
```

## 9. 任务三优化版（摘要）

可分性分析显示：`environmental_awareness_score`、`ev_knowledge_score`、`technology_affinity_score`、`annual_income` 对「高里程焦虑」有强信号；官方五特征中仅 `ev_knowledge_score` 很强。

优化流程（`src/tasks/task3.py`）：

1. 开发集可分性诊断  
2. 配置 A/B/C：官方五特征 / 安全扩展 / 扩展+机制特征  
3. 分层 5 折 OOF + 阈值搜索（最终 20% 不参与调参）  
4. 选定配置后在完整开发集重训，验证集只评估一次  

任务二详细结论见 [`docs/TASK2_SUMMARY.md`](docs/TASK2_SUMMARY.md)。

## 10. 当前版本限制

- 任务一仍为朴素基线
- 任务二为弱信号，采用多数类策略稳住 ACC2
- 最终验证集不参与模型/阈值选择
- 不做 target encoding、不做大规模超参搜索
