# Task2 Problem Redefinition

Target: `home_charging_available` (binary). Class 1 prevalence ≈ 0.65.

Research question shifts from “which fancy model wins Accuracy” to:
1) Is there stable predictive signal under leak-safe features?
2) If not, what is the Bayes-optimal decision under 0-1 loss?
3) Which unobserved structural variables should be collected next?

Weak-signal finding: safe observables show limited separability (max |AUC-0.5| ≈ `None`).
Missing structural housing/charging variables (parking, garage, tenure) likely dominate the true mechanism.
