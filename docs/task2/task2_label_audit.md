# Label Audit

Develop-set audits: exact duplicate conflicts (`n=None`), quantile profile bins, kNN consistency vs global prior.
Labels were **not** modified. See `label_audit/`.

Interpretation: if near-duplicate profiles disagree on labels, or kNN consistency ≤ global prior, local feature geometry lacks class structure.
