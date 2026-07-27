# Task2 Final Conclusion

- Locked scheme: **M** (majority class = 1)
- Weak-signal lock: **true**
- OOF Accuracy: **0.64975**
- Final holdout Accuracy: **0.6498**
- Majority baseline: **0.64976**
- Permutation `supports_signal`: **false** (Acc lift p=1.0; AUC≈0.5065 not actionable)
- Bootstrap P(ΔAcc > 0) ≈ **0.51**; Acc CI contains 0

Paper claim: after excluding cross-task target leakage, consumer-profile features do not yield stable Accuracy gains for home charging availability; under 0-1 loss the majority class is the robust Bayes decision. Publish EB/cluster probabilities only as exploratory ranking, not as formal ACC2 submitters.
