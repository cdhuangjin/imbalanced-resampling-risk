# V5.5/V5.6 residual-stability and regime-switch report

Decision: **NO-GO**
Input complete: `True`; effect rows: `432`; behavior rows: `9`.

RSC multiplies the frozen residual-AUC adaptation weight by a label-free target-prior stability weight; RASS keeps V5.1 in the accepted regime and switches to RSC above residual AUC 0.60.
Negative deltas mean the candidate policy improves the reference metric.

## Gate criteria

- Candidate policy `platt_target_rass` utility at target priors 0.20 and 0.50: `3` methods pass.
- Mixed-shift practical safety: `3` methods pass.
- Mixed-shift non-inferiority to V5.1 shrinkage: `0` methods pass.

A NO-GO result is retained as a boundary and does not license post-hoc threshold changes.
