# Utility-first selective treatment Gate

Date: 2026-09-17  
Status: frozen for Gate execution  
Scope: experiments only

## Research question

Can a validation-only 5:1 decision-utility selector choose among `none`, random q05, ordinary SMOTE q05/k5 and support-filtered SMOTE q05/k5 so that held-out operational utility improves without unacceptable majority-logloss or Brier harm?

This is an empirical deployment-risk audit. It is not a new sampler, a new threshold theory or a new utility optimizer.

## Selection rule

For each candidate, choose the lowest threshold on the validation-only grid `{0.01, 0.02, ..., 0.99}` that minimizes `5*FN + FP`; ties use the lowest threshold. Compute validation utility as `-(5*FN + FP)/N`.

Use 400 deterministic validation-row bootstrap replicates for candidate-minus-`none` differences. A candidate is feasible only when:

- the 97.5% upper bound of majority-logloss difference is at most `+0.02` at all declared source/target priors;
- the 97.5% upper bound of Brier difference is at most `+0.005` at all declared priors;
- the 2.5% lower bound of validation utility difference is strictly above `0`.

Among feasible candidates, choose the largest mean validation utility; ties use the frozen candidate order. If none is feasible, select `none`.

## Matrix

- Tasks: `ecoli`, `yeast_me2`, `abalone_19`, `mammography`, `oil`, `spectrometer`.
- Learners: `dt_leaf01`, `cat_iter500`.
- Seeds: `0,1,2`.
- Scenarios: existing adaptive-resampling Gate (`iid` and three declared label-shift scenarios).
- Expected fits: `36`.
- Policies: four fixed candidates plus `utility_guarded`.
- Metrics: six existing metrics plus `decision_utility_c5`; expected rows `36 × 5 × 4 × 7 = 5,040`.

## Gate criteria

The Gate is `GO` only if output is complete and finite, at least two of six task clusters have non-baseline selection rate at least `.5`, utility `utility_guarded - none` has a task-cluster 95% CI lower bound above `0`, utility `utility_guarded - smote_q05_k5` has lower bound at least `-0.005`, and majority-logloss/Brier upper bounds versus `none` are within the frozen margins.

Otherwise it is `NO-GO`; no threshold grid, cost ratio, bootstrap count, candidate list or task set may be changed after inspection.

## Provenance and stopping

The runner is separate from V3 and V11 outputs. Validation labels are used only for threshold selection; test labels are used only for final evaluation. The Gate does not unlock a 24-task confirmation unless all criteria pass.
