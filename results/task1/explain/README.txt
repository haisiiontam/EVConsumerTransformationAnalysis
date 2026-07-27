Task1 SHAP artifacts (CatBoost native ShapValues; 200 develop samples).
Backend: catboost_native (avoids shap.TreeExplainer Windows native crash).
Env used: .venv-shap (Python 3.12 via uv).

Files:
- shap_summary.png              global mean(|SHAP|) bar
- shap_summary_beeswarm.png    optional (if shap package plot succeeds)
- shap_dependence_<feat>.png   dependence for top feature (Medium class)
- waterfall_typical_*.png      typical Low/Medium/High sample contributions
- global_mean_abs_shap.csv     numeric importance table
