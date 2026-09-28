# Confidence-guarded selective resampling Gate

Date: 2026-09-17  
Status: frozen for Gate execution  
Scope: experiments only

## Research question

Can a validation-row bootstrap confidence guard select a non-baseline training treatment only when its estimated minority-risk benefit is statistically compatible with no majority-logloss or Brier harm, while preserving the `none` fallback?

This is a selective policy experiment, not a new sampler. It is a refinement of the previously tested risk-constrained selector and must not overwrite the V3 decision.

## Frozen selection rule

The training candidates are `none`, `random_q05`, `smote_q05_k5` and `support_smote_q05_k5`. For each candidate, the validation predictions are compared to `none` using 400 deterministic row-bootstrap replicates at declared target priors `.05`, `.20` and `.50`.

A candidate is feasible only if, at every declared prior:

- the 97.5% upper bound of candidate-minus-none majority logloss is at most `+0.02`;
- the 97.5% upper bound of candidate-minus-none Brier is at most `+0.005`;
- the 97.5% upper bound of candidate-minus-none minority logloss is at most `0`.

Among feasible candidates, choose the lowest mean validation minority logloss; ties use the frozen candidate order. If none is feasible, select `none`.

## Gate matrix

- Tasks: `ecoli`, `yeast_me2`, `abalone_19`, `mammography`, `oil`, `spectrometer`.
- Learners: `dt_leaf01`, `cat_iter500`.
- Seeds: `0,1,2`.
- Scenarios: the existing adaptive-resampling Gate matrix (`iid` and three declared label-shift scenarios).
- Expected fit count: `36`.
- Expected raw rows: `36 × 5 policies × 4 scenarios × 6 metrics = 4,320`.

## Gate decision

Reuse the frozen adaptive-policy Gate: complete finite output, at least four of six task clusters with non-baseline selection rate at least `.5`, adaptive versus fixed SMOTE passes majority-logloss and Brier screens, and adaptive versus `none` has a negative minority-logloss CI with at least four task clusters showing at least 5% relative minority improvement.

Any `NO-GO` stops this exact branch and forbids threshold, bootstrap count, candidate order or task changes without a new protocol.

## Provenance

The branch monkey-patches the existing runner only in a separate wrapper. The existing V3 output is immutable. The wrapper records base-runner, selector and data-manifest hashes in `confidence_guarded_protocol.json`.
