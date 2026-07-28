# Statistical Tests

Permutation (label shuffle, fixed fast logistic OOF):
```json
{
  "n_perm": 200,
  "model": "logistic_l2_oof",
  "observed_accuracy": 0.64975,
  "observed_auc": 0.5065010591291494,
  "observed_lift_vs_majority": 0.0,
  "majority_accuracy": 0.64975,
  "perm_auc_mean": 0.49885882309645807,
  "perm_auc_std": 0.004238642967748722,
  "p_value_auc_ge_observed": 0.035,
  "p_value_lift_ge_observed": 1.0,
  "supports_signal": false,
  "note": "Association test under label permutation; not causation."
}
```

Bootstrap on OOF deltas:
```json
{
  "n_boot": 1000,
  "majority_class": 1,
  "majority_accuracy": 0.64976,
  "delta_acc_mean": 6.604999999999583e-05,
  "delta_acc_ci95": [
    -0.004735000000000045,
    0.004765000000000019
  ],
  "p_delta_acc_gt_0": 0.512,
  "auc_minus_0_5_mean": 0.006456011125499211,
  "auc_minus_0_5_ci95": [
    0.0006488278546351534,
    0.011961102929832986
  ],
  "p_auc_gap_gt_0": 0.986
}
```

See also `statistics/mcnemar_test.json`.
