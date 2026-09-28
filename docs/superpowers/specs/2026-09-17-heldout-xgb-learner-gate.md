# Held-out XGBoost learner Gate — protocol v0.1

Date: 2026-09-17  
Iteration: `AIEP-20260917-XGB-GATE-01`  
Parent evidence: fixed iid main experiment and V5.1/V5.6 shift-policy branches

## Observation

The current follow-up evidence evaluates the fixed sampling comparison mainly
with one decision-tree configuration and one CatBoost configuration. The
repository also retains the mother-study XGBoost implementation, but XGBoost
has not yet been evaluated in the current fixed-factor pipeline. This leaves a
learner-family generalization risk that cannot be closed by more DT/CatBoost
seeds.

## Research question

Under the existing iid task and split protocol, does the separation between
ranking response and class-conditional probability risk observed for
resampling remain visible in a held-out boosted-tree learner family?

## Hypothesis and rivals

- Candidate hypothesis: with the learner family fixed to XGBoost, the three
  sampling treatments can still produce different effects on AUC, minority
  logloss, majority logloss, and Brier score; the signs need not match either
  DT or CatBoost.
- Rival 1 (learner-capacity explanation): any separation is specific to the
  current DT/CatBoost capacity choices and disappears for XGBoost.
- Rival 2 (sampling-intensity/implementation explanation): the result is caused
  by the q05/k5 sampling operation rather than a learner-family interaction.
- Null prediction: the XGBoost treatment contrasts are near zero or have
  uncertainty intervals spanning zero across the Gate tasks.

## Frozen design

| Factor | Frozen value |
|---|---|
| Gate tasks | `ecoli`, `yeast_me2`, `abalone_19`, `mammography`, `oil`, `spectrometer` |
| learner | `xgb_gate` only; official XGBoost path, `n_estimators=1000`, `learning_rate=0.1`, validation early stopping |
| treatments | `none`, `random_q05`, `smote_q05_k5` |
| seeds | `0, 1, 2, 3, 4` |
| environment | iid only; no target-prior or covariate perturbation in this Gate |
| split | existing stratified train/validation/test split function, seed-specific and disjoint |
| primary effects | treatment minus `none` for AUC, minority logloss, majority logloss, Brier |
| secondary effects | AP, total logloss, fixed/tuned F1, runtime and fit completeness |
| independent unit | task; seed means are collapsed within task before Gate inference |
| uncertainty | 10,000 task-level paired bootstrap replicates, seed mean within task |
| test labels | used only for final metric computation; never for sampling, calibration, or selection |
| stopping | Gate only; no 24-task expansion is unlocked automatically by a positive mean |

## Gate criteria

1. All `90` fits complete and emit the expected `90 × 8 = 720` metric rows.
2. Every raw row has non-empty data/split/config hashes and finite metrics.
3. No test label is used before final metric computation.
4. Report task-level paired effects and 95% bootstrap CIs for all primary
   effects; do not use seed-level rows as independent task replicates.
5. `GO` requires the protocol and integrity checks to pass and at least one
   predeclared primary effect to be conditionally interpretable without a
   catastrophic task-level failure. `NO-GO` is retained if the effect is
   absent, unstable, or safety-relevant risk increases are not bounded.
6. Regardless of Gate status, an independent reanalysis must reproduce the
   effect table and Gate summary byte-for-byte.

The Gate is a generalization test, not a claim that XGBoost is state of the art
or that a positive result proves a universal sampler recommendation.

## Planned artifacts

- output: `code/outputs/heldout_xgb_gate_20260917/`
- raw results, summary, status and manifests from `factor_experiment.py`
- `analysis/effects.csv`
- `analysis/gate_summary.json`
- `analysis/gate_report.md`
- independent copy: `analysis_recheck_20260917/`
- evidence update: `innovation_ledger.md`

## Stop/next decision

- `GO`: keep XGBoost as a conditional held-out learner result and consider a
  separately frozen 24-task confirmation only after reviewer-risk review.
- `NO-GO`: retain the learner-specific boundary and do not tune XGBoost or
  expand the task set to search for a favorable subset.
- `INCONCLUSIVE`: fix only integrity/software errors under a new protocol;
  never change the effect criteria after seeing the metrics.
