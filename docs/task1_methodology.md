# Task1 Methodology

## Problem
Predict `ev_adoption_likelihood` ∈ {Low, Medium, High} with natural order Low < Medium < High.

## Validation protocol
- Frozen 80/20 stratified split (`random_state=42`)
- Develop-set Stratified 5-fold OOF for all selection
- Final validation evaluated once after locking config
- Multi-seed check: 42 / 52 / 62

## Modeling stack explored
1. CatBoost multiclass (M1) ± Optuna
2. Ordinal dual-boundary CatBoost (M2)
3. Ordinal regression + threshold/softmax (M3)
4. LightGBM / XGBoost heterogeneous trees (M4/M5)
5. Ordinal logistic baseline (M6)
6. Soft voting / nested stacking / class-bias postprocess

## Leakage rules
Excluded targets and uncertain posterior cost/energy fields; `range_anxiety_score` allowed as official feature.
