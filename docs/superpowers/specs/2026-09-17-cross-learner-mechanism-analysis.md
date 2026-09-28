# Cross-learner mechanism synthesis — protocol v0.1

## Purpose

The V6–V8 learner-family Gates show repeated class-conditional versus overall
risk trade-offs, but each Gate alone is too small to establish whether the
pattern is learner-specific. This analysis reuses completed, provenance-checked
Gate outputs and does not add a new model, task, seed, or endpoint after seeing
results.

## Frozen question

Across the six common Gate tasks, does q05 resampling consistently change
minority logloss more than ranking AUC across DT, CatBoost, XGBoost, Balanced
Random Forest, and LightGBM configurations?

## Frozen input and units

- Inputs: the completed V1 Gate output, V6 XGBoost cost-sensitive output, V8
  BRF output, and V9 LightGBM output.
- Tasks: the six common Gate tasks only.
- Learner configurations: four DT capacities, two CatBoost configurations,
  XGBoost, BRF, and LightGBM; all are retained as separate configurations.
- Treatments: `random_q05` and `smote_q05_k5` versus `none`.
- Metrics: AUC, minority logloss, majority logloss, and Brier.
- Seed aggregation: mean within each task × learner × treatment cell.
- Inference unit: first average learner configurations within each task, then
  paired bootstrap over the six tasks (10,000 replicates, seed `20260917`).
  Learner count is not treated as independent n.

## Mechanism decision rule

Report `SUPPORTS_CONDITIONAL_MECHANISM` only if at least one treatment has a
minority-logloss CI entirely below zero, its AUC CI overlaps zero, and the
majority/Brier effects are reported with their worst task rather than hidden.
Otherwise report `BOUNDARY_OR_INCONCLUSIVE`. This is a descriptive synthesis,
not a causal claim and not a universal sampler recommendation.

## Reproducibility

The analyzer must validate each input's task, learner, method, metric,
provenance hashes, and expected seed coverage. It writes the exact input paths,
input manifest hashes, aggregated effects, family-level cells, and an
independent reanalysis whose outputs are byte-identical.

