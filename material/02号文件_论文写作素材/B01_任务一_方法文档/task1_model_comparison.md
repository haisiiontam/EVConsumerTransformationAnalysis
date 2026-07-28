# Task1 Model Comparison

| Module | OOF Accuracy | Notes |
|---|---:|---|
| baseline_stable | 0.8560 | loaded B_safe_raw |
| M1_catboost | 0.8572 | best single CatBoost |
| M2_ordinal | 0.8554 | dual boundary |
| M3_reg_soft | 0.8568 | T=0.1 |
| M3_reg_hard | 0.8568 | threshold mapping |
| M4_lgbm | 0.8543 | lightgbm |
| M6_ord_logit | 0.8581 | interpretable baseline |
