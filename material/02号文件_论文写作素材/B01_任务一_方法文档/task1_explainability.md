# Task1 Explainability

SHAP 解释对象为 **M1 CatBoost**（关联非因果）。正式提交器是 `soft_vote`（M6+M1+M3），正文需写明这一点。

产物目录：`outputs/task1/explain/`（发布副本：`results/task1/explain/`）

## 图件清单

| 文件 | 用途 |
|------|------|
| `shap_summary.png` | 全局 mean\|SHAP\| |
| `shap_by_class.png` | Low/Medium/High 分班级重要性对比 |
| `shap_summary_beeswarm.png` | 多分类蜂群图 |
| `shap_dependence_*_Medium.png` | Top 特征对 Medium 的依赖（中类最难） |
| `shap_dependence_eco_tech_mean_{Low,High}.png` | Top1 对两端类的依赖 |
| `waterfall_typical_{Low,Medium,High}.png` | 各类典型样本 |
| `waterfall_typical_Medium_borderline.png` | Medium 边界样本 |
| `waterfall_typical_Medium_confident.png` | 高置信 Medium |
| `waterfall_typical_Medium_as_{Low,High}.png` | Medium 误判个案（论文亮点） |
| `global_mean_abs_shap.csv` | 数值表（含分班级列） |

## 写作注意

- 不要把 SHAP 排序写成因果。
- Medium 误判瀑布用于讲清“中类混淆机制”，与消融/有序分支叙事对齐。
- 后端：CatBoost 原生 `ShapValues`（避免 Windows 上 TreeExplainer 崩溃）。
