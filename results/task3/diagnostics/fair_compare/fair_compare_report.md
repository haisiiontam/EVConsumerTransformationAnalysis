# Task3 Fair Compare (same frozen split)

- Previous locked Final Acc: `0.8528`
- Relocked scheme: `current_M1_fixed` @ thr=`0.5`
- Relocked OOF Acc: `0.86185`
- Relocked Final Acc: `0.8528`

## Summary table

```
phase            model threshold_policy  threshold  accuracy  roc_auc  balanced_accuracy  n_features  develop_accuracy
  oof       current_M1            fixed      0.500  0.861850 0.895505           0.861168          22               NaN
  oof       current_M1        oof_tuned      0.500  0.861850 0.895505           0.861168          22               NaN
  oof hist_clf_best_hp            fixed      0.495  0.861425 0.886125           0.860765          29               NaN
  oof hist_clf_best_hp        oof_tuned      0.495  0.861425 0.886125           0.860765          29               NaN
final       current_M1            fixed      0.500  0.852800 0.900834           0.852106          22          0.862650
final       current_M1        oof_tuned      0.500  0.852800 0.900834           0.852106          22          0.862650
final hist_clf_best_hp            fixed      0.495  0.853000 0.900402           0.852315          29          0.866225
final hist_clf_best_hp        oof_tuned      0.495  0.853000 0.900402           0.852315          29          0.866225
```

## Recommendation

```json
{
  "same_split": true,
  "note": "Historical archived Final 0.8534 used a different holdout; this table is the fair compare.",
  "best_by_oof": {
    "model": "current_M1",
    "threshold_policy": "fixed",
    "threshold": 0.5,
    "oof_accuracy": 0.86185,
    "final_accuracy": 0.8528
  },
  "current_M1_oof_tuned": {
    "oof_threshold": 0.5,
    "oof_accuracy": 0.86185,
    "final_accuracy": 0.8528
  },
  "hist_clf_best_hp_oof_tuned": {
    "oof_threshold": 0.495,
    "oof_accuracy": 0.861425,
    "final_accuracy": 0.853
  },
  "previous_locked_final_accuracy": 0.8528
}
```

GAM/TabNet: not recommended as next step based on this compare.
