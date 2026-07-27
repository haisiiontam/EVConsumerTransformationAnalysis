Task1 SHAP artifacts (CatBoost native ShapValues; association ≠ causation).
Backend: catboost_native. Explains M1 CatBoost (final submitter is soft_vote).
n_samples=300 (stratified). top_feature=eco_tech_mean.

Core figures:
- shap_summary.png / shap_summary_beeswarm.png / shap_by_class.png
- shap_dependence_*_Medium/Low/High.png
- waterfall_typical_{Low,Medium,High}.png
- waterfall_typical_Medium_{borderline,confident}.png
- waterfall_typical_Medium_as_{Low,High}.png  (misclassification cases)
- global_mean_abs_shap.csv
