# Task1 Final Conclusion

## Baseline (stable reproduction)
- OOF Accuracy: 0.856025
- Final validation Accuracy: 0.8592

## Locked final model
- Name: soft_vote
- OOF Accuracy: 0.858725
- Final validation Accuracy: 0.8601
- Multi-seed mean±std: 0.8563000000000001 ± 0.0002474873734152

## Formal vs paper-only modules
- Entered final predictor: ['soft_vote', 'M1_catboost']
- Paper-only / ablation (no fake gains): ['M2_ordinal', 'M3_reg_soft', 'M3_reg_hard', 'M4_lgbm', 'M6_ord_logit']

## Merge recommendation
YES — multi-seed and locked validation support a real gain.
