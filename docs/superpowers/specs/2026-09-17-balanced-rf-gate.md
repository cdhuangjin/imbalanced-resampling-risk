# Balanced Random Forest learner-family Gate — protocol v0.1

## Purpose

V6 showed that the XGBoost learner-family result is conditional, and V7 showed
that one prevalence-balancing weight rule can improve minority logloss while
harming cost-sensitive utility. The next information-gain step is to test a
specialized imbalance ensemble rather than further tuning XGBoost.

CLIMB (NeurIPS 2025) reports a unified comparison of 29 class-imbalanced
learning algorithms on 73 tabular tasks and emphasizes that ensemble methods
and data quality matter while naive rebalancing is not universally reliable.
Balanced Random Forest (BRF) is a direct, established specialized baseline
that balances each tree's bootstrap sample. This branch tests learner-family
generalization; it is not presented as a new BRF algorithm.

## Hypothesis card

- **Observation:** XGBoost resampling gains minority logloss but can incur
  task-dependent Brier/majority-risk costs.
- **Research question:** Does a specialized imbalance ensemble provide a safer
  decision-utility/risk profile on the same tasks and splits?
- **Candidate hypothesis:** BRF without external resampling will have a better
  or non-inferior cost-sensitive decision utility than XGBoost without
  treatment, with no practical Brier or majority-logloss harm.
- **Rivals:** BRF's internal balancing may over-focus on minority points;
  external SMOTE may compound or undo its benefit; the apparent difference may
  be learner-specific rather than a general imbalance treatment effect.
- **Prediction:** `brf_none - xgb_none` will improve or preserve
  `decision_utility_c5` and minority logloss, while remaining within the
  frozen Brier and majority-logloss margins.
- **Null prediction:** BRF is not safer or more useful than the XGBoost baseline,
  or gains are offset by a worst-task safety failure.
- **Primary endpoints:** task-level paired effects on decision utility,
  minority logloss, majority logloss, and Brier.
- **Secondary endpoints:** AUC, AP, overall logloss, F1, treatment interaction,
  worst task, and runtime.
- **Failure boundary:** no learner-family utility signal or any candidate safety
  failure stops this Gate from entering 24-task confirmation.
- **Claim type:** conditional descriptive/associational learner-family result.

## Frozen design

| factor | frozen choice |
|---|---|
| candidate learner | `BalancedRandomForestClassifier(n_estimators=300, replacement=True, sampling_strategy='all', max_features='sqrt', n_jobs=1)` |
| comparator learner | existing frozen XGBoost `xgb_gate`; use its existing cost-sensitive-run `none` rows for matched metrics |
| Gate tasks | `ecoli`, `yeast_me2`, `abalone_19`, `mammography`, `oil`, `spectrometer` |
| confirmation tasks | existing 24-task library, unchanged |
| scenarios | iid only |
| seeds | `0,1,2,3,4` |
| candidate treatments | `none`, `random_q05`, `smote_q05_k5` |
| decision utility | `-(5*FN + 1*FP)/N`, with validation-only 0.01 threshold grid and lowest-threshold tie rule |
| inference | task-level paired bootstrap, 10,000 replicates, seed `20260917` |
| practical safety margins | Brier increase `<=0.01`; majority-logloss increase `<=0.10` |

BRF's internal balancing is part of the learner and is held fixed. External
sampling is applied only to the source training split. The validation/test
split and all threshold decisions are unchanged from the prior XGBoost branch.

## Gate rule

The Gate requires exact BRF completeness and a matched XGBoost comparator. The
primary candidate comparison is `brf_none - xgb_none`; it must show either:

1. a positive task-bootstrap CI for decision utility, or
2. minority-logloss improvement with utility no worse than `-0.02`.

The candidate comparison must also remain within both practical safety margins
on every task. Treatment interactions are reported but cannot rescue a failed
learner-family comparison.

## Data and provenance

Use the existing local `imblearn` Zenodo cache at
`<local path omitted>`; do not download a duplicate. Record
dataset manifest, package versions, protocol hash, data hash, and comparator
input path. Source archive:
`https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz`.

## Execution order

1. Add failing BRF runner/analyzer tests.
2. Implement the frozen BRF runner and run targeted tests.
3. Run smoke, then the six-task Gate.
4. Analyze BRF-versus-XGBoost effects and independently reanalyze.
5. Only if the Gate passes, run the frozen 24-task confirmation.

