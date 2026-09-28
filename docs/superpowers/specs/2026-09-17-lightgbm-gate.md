# LightGBM learner-family Gate — protocol v0.1

## Purpose

V6–V8 show that minority-risk improvements can coexist with probability-risk
and utility costs for XGBoost and Balanced Random Forest. The remaining common
tabular boosting family, LightGBM, is added as a held-out learner-family test.
The purpose is to close a reviewer concern about family coverage, not to claim
that adding a third booster creates a new algorithm.

## Frozen hypothesis

Under the same six Gate tasks, split indices, seeds, q05 treatments, validation
threshold protocol, and decision costs, LightGBM without external resampling
will either (a) improve or preserve decision utility and class-conditional risk
relative to matched XGBoost, or (b) expose another learner-family boundary.

## Frozen design

- Learner: `LGBMClassifier(n_estimators=500, learning_rate=0.05,
  num_leaves=31, max_depth=-1, max_features=None, n_jobs=1,
  verbosity=-1, deterministic=True, force_col_wise=True)`.
- Validation early stopping: 10 rounds; validation labels are never used for
  preprocessing or test-time evaluation.
- Tasks: the same six Gate tasks as V6–V8; iid only; seeds `0..4`.
- Treatments: `none`, `random_q05`, `smote_q05_k5`.
- Primary learner-family comparison: `lgbm_none - xgb_none`.
- Decision utility: `-(5*FN + FP)/N`; threshold chosen on validation-only
  0.01 grid with lowest-threshold tie breaking.
- Inference: task-level paired bootstrap, 10,000 replicates, seed `20260917`.
- Safety margins: Brier increase `<=0.01`; majority-logloss increase `<=0.10`.

## Gate rule

Require complete LightGBM and matched XGBoost coverage. The Gate is `GO` only
if LightGBM-none shows a positive decision-utility CI or a conditional
minority-logloss improvement with utility no worse than `-0.02`, and no primary
safety margin fails on any task. Otherwise retain the learner-family boundary
and do not run 24-task confirmation for this exact configuration.

## Provenance

Use the existing local `imblearn` Zenodo cache at
`<local path omitted>`, and record LightGBM version, data
manifest, protocol hash, and comparator input path. Source archive:
`https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz`.

