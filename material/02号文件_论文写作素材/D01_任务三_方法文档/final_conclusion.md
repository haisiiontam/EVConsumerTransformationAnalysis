# 任务三最终结论（与锁定方案对齐）

## 正式提交

- **方案**：`boost_B+B_concern_soft`
- **特征**：B + battery-concern 机制组（22 维）
- **软权重**：M1 CatBoost 分类 ≈0.80，M2 回归分支 ≈0.15，M4 LightGBM ≈0.05
- **阈值**：0.5  
- **OOF Acc ≈ 0.8619**，**Final Acc = 0.8520**

## 不采用

- 扩大到全机制特征集（未更好）  
- 复杂 stacking（未优于软融合）  
- 仅用历史 `clf_best_hp` 偷看验证集的口径（坚持 OOF 锁定）  
- GAM / TabNet（默认关闭）

## 材料

- 配置：`results/task3/final/task3_final_config.json`
- Boost 报告：`results/task3/diagnostics/boost_continue/boost_report.md`
- SHAP：`results/task3/explain/`
- 写作重点：`docs/task3/paper_focus.md`、`docs/task3/NOTE.md`
