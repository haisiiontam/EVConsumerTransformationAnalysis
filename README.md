# 电动汽车消费者转型分析 · 三任务朴素基线

## 1. 项目目的

本项目面向 **2026 年第五届「创新杯」大学生大数据挑战赛 · 初赛 B**：

> 从燃油到纯电——电动汽车消费者转型分析

目标是建立一套**结构清楚、无明显数据泄露、可完整运行**的朴素基线：三个分类任务分别用 CatBoost 训练，仅使用赛题建议特征，单次分层训练/验证划分，主指标为 Accuracy。

## 2. 三个任务定义

| 任务 | 目标 | 类型 | 说明 |
|------|------|------|------|
| 任务一 | `ev_adoption_likelihood` | 三分类 Low / Medium / High | 电动汽车采用意愿 |
| 任务二 | `home_charging_available` | 二分类 0 / 1 | 家庭充电条件预测（第一版按现有字段建模，不解释为「未来安装意愿」） |
| 任务三 | 由 `range_anxiety_score` 构造 | 二分类：`> 5` → 1，`<= 5` → 0 | 里程焦虑敏感度；**不把该评分放入特征** |

各任务官方建议特征见 `src/config.py`。

## 3. 环境安装

需要 Python 3.10+。

```bash
pip install -r requirements.txt
```

依赖：`pandas`、`numpy`、`scikit-learn`、`catboost`、`matplotlib`。

## 4. 数据字段要求

CSV 至少包含以下字段（列名需与赛题英文列名一致，允许前后空格，程序会自动清理）：

**任务一特征：** `annual_income`, `education_level`, `environmental_awareness_score`, `technology_affinity_score`, `range_anxiety_score`, `fuel_expense_per_month`  
**任务一目标：** `ev_adoption_likelihood`

**任务二特征：** `city_type`, `nearest_charging_station_km`, `education_level`, `annual_income`, `vehicle_age_years`  
**任务二目标：** `home_charging_available`

**任务三特征：** `daily_commute_km`, `weekly_travel_distance_km`, `charging_station_accessibility`, `ev_knowledge_score`, `battery_replacement_concern`  
**任务三评分源：** `range_anxiety_score`（仅用于构造标签，不作为任务三特征）

## 5. 运行命令

```bash
python main.py --data data/train.csv --output outputs
```

参数说明：

- `--data`：原始 CSV 路径（必填）
- `--output`：输出目录（默认 `outputs`）
- `--test-size`：验证集比例（默认 `0.2`）

## 6. 输出目录说明

```text
outputs/
├── data_overview.csv
├── missing_report.csv
├── target_distribution.csv
├── overall_summary.csv
├── task1/
│   ├── metrics.json
│   ├── validation_predictions.csv
│   ├── confusion_matrix.csv / .png
│   ├── feature_importance.csv / .png
│   └── model.cbm
├── task2/  （同上）
└── task3/  （同上）
```

## 7. 指标解释

- 每个任务主指标为验证集 **Accuracy**。
- 同时输出 classification report、confusion matrix、特征重要性。
- 内部验证加权分（**validation weighted score**，非最终比赛成绩）：

```text
weighted_score = 100 × (0.2 × ACC1 + 0.4 × ACC2 + 0.4 × ACC3)
```

结果写入 `outputs/overall_summary.csv`。

## 8. 当前版本限制

当前版本是朴素基线，只使用赛题建议特征、单次分层训练验证划分和单一 CatBoost 模型。
尚未加入交叉验证、模型融合、复杂特征工程、阈值优化和超参数搜索。
