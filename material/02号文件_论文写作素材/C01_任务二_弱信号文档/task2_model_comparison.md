# Model Comparison

Baselines B0–B2, logistic, CatBoost, EB, cluster prior, LightGBM/XGBoost/ExtraTrees, optional soft blend.
Locked scheme: `M` | OOF Acc=`0.64975` | majority=`0.64976`.

Formal vs paper roles:
- Formal submitter: locked scheme (often majority M)
- Probability ranking: EB / best calibrated-ish OOF
- Contrasts: tree ensembles that collapse to majority
