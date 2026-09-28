# V5.1 Residual-Shift Shrinkage Follow-up

Date: 2026-09-16  
Status: frozen after V5 Gate NO-GO  
Scope: experiments only; post-hoc follow-up, not a replacement for V5

## Motivation and boundary

The V5 six-task Gate was complete and the detector separated mixed from pure label-shift cells, but the binary policy did not pass the strict mixed-shift safety criterion. V5.1 tests whether the failure is caused by a discontinuous decision rather than by the residual-shift signal itself.

This is a new follow-up protocol declared after inspecting V5. It must be reported as a post-hoc conditional result. It does not alter the V5 `NO-GO` decision and does not claim a new estimator.

## Frozen policy

Let `p_s` be source-Platt probability, `p_c` be V4 consensus-corrected probability, and `a` be the V5 residual domain AUC. Define

`w(a) = clip((0.75 - a) / (0.75 - 0.55), 0, 1)`

and

`p_shrink = (1-w(a)) p_s + w(a) p_c`.

The lower and upper residual-AUC knots (`0.55`, `0.75`) are frozen before running V5.1. They are not fitted using target labels. The comparison includes source-Platt, always-consensus, binary V5 gate, shrinkage, and oracle reference.

## Matrix

- Same six Gate tasks, two learners, three training methods, and seeds `0,1,2` as V5.
- Same ten scenarios: iid; three pure label-shift priors; two covariate-shift severities; four joint-shift cells.
- Expected fits: 108.
- Five policy variants × six metrics × ten scenarios × 108 fits = 32,400 raw metric rows for the six-task Gate; the 24-task confirmation therefore contains 216,000 raw metric rows.

## Gate criteria

1. Complete hash-linked output with zero duplicate keys and finite metrics/diagnostics.
2. At target priors `0.20` and `0.50`, shrinkage improves both Brier and logloss over source-Platt with task-cluster bootstrap CI upper bounds below zero in at least two of three training-method strata.
3. Under the pooled covariate/joint-shift family, shrinkage has no practically harmful increase over source-Platt: Brier CI upper bound ≤ `0.01` and logloss CI upper bound ≤ `0.10` in at least two of three training-method strata.
4. Shrinkage improves or matches the binary V5 gate on the pooled pure-label-shift primary endpoints in at least two of three methods, judged by mean delta rather than a new significance claim.

If any condition fails, the result is a bounded negative/conditional result and no 24-task expansion is unlocked.

## Provenance

V5 raw results remain immutable at `residual_shift_gate/results/gate_20260916/`. V5.1 writes a separate output directory and records the V5 input protocol hash, V5 input data-manifest hash, shrinkage knots, and source-code hashes.
