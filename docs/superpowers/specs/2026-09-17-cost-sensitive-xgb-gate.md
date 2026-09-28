# Cost-sensitive versus resampling XGBoost Gate — protocol v0.1

## Purpose

V6 confirmed that the held-out XGBoost learner preserves the separation between
ranking and class-conditional risk, but the 24-task confirmation did not support
a universal resampling claim. The next unresolved reviewer risk is narrower:
the existing comparison does not put a train-objective intervention and a
data-level intervention on the same learner, split, seed, and decision-utility
scale.

Recent benchmark evidence is used only to motivate the comparison, not as a
claim that a particular method is state of the art. CLIMB reports a unified
benchmark of 29 class-imbalanced-learning algorithms on 73 tabular datasets and
highlights limitations of naive rebalancing; the original cost-sensitive
random-forest work explicitly compares cost-sensitive and sampling strategies;
Imbalance-XGBoost provides a direct precedent for weighted losses in XGBoost.

## Hypothesis card

- **Observation:** q05 random over-sampling and SMOTE improve minority logloss
  in the V6 XGBoost Gate, while their AUC effects are near zero and their
  majority/Brier costs are task-dependent.
- **Research question:** Under the same XGBoost learner and split, does a
  training-only cost-sensitive objective provide a better decision-utility /
  risk trade-off than q05 resampling?
- **Candidate hypothesis:** class-weighted XGBoost will improve validation-only
  cost-sensitive test utility with no larger worst-task majority-logloss or
  Brier harm than q05 SMOTE.
- **Rivals:** (1) resampling changes the empirical feature geometry and can
  outperform weighting; (2) treatment differences are calibration-only and do
  not improve ranking; (3) any apparent gain is caused by threshold selection
  rather than the training objective.
- **Prediction:** the cost-weighted method should show a positive paired effect
  on `decision_utility_c5` and a non-positive or practically bounded effect on
  majority logloss/Brier. AUC is exploratory and is not expected to move
  reliably.
- **Null prediction:** cost weighting is not better than no treatment or SMOTE
  on decision utility, or its risk cost exceeds the frozen safety margins.
- **Primary endpoints:** task-level paired effects on `decision_utility_c5`,
  `loss_minority`, `loss_majority`, and `brier`.
- **Secondary endpoints:** AUC, AP, overall logloss, F1 at 0.5 and F1 at the
  validation-F1 threshold, plus worst-task effects and runtime.
- **Failure boundary:** any integrity failure, any safety-margin failure for
  the candidate, or no candidate utility signal in the Gate stops this branch
  from entering confirmation. A positive mean alone is insufficient.
- **Claim type:** conditional descriptive/associational experiment; no causal
  or universal superiority claim.

## Frozen design

| factor | frozen choice |
|---|---|
| learner | XGBoost `n_estimators=1000`, `learning_rate=0.1`, `tree_method=hist`, `n_jobs=1`, validation early stopping 10 |
| Gate tasks | `ecoli`, `yeast_me2`, `abalone_19`, `mammography`, `oil`, `spectrometer` |
| confirmation tasks | the existing 24-task library, unchanged from V6 |
| scenarios | iid only |
| seeds | `0,1,2,3,4` |
| treatments | `none`, `random_q05`, `smote_q05_k5`, `cost_weight_q05` |
| cost weighting | training labels only; negative weight `1`; positive weight `q(1-p)/(p(1-q))` with `q=0.5` and source-train prior `p`; no test labels |
| decision utility | `-(5*FN + 1*FP)/N` on the untouched test set; larger is better |
| utility threshold | validation-only grid `0.00,0.01,...,0.99`, minimizing validation cost; ties select the smallest threshold |
| standard threshold | the existing validation-only F1 threshold, retained for the standard F1 endpoints |
| primary inference | task-level paired bootstrap, 10,000 replicates, seed `20260917` |
| practical safety margins | Brier increase `<=0.01`; majority-logloss increase `<=0.10` on every task |

The cost-weighted branch never resamples rows. The only changed training
quantity is the per-row weight, calculated from the training labels after the
frozen split. Validation and test labels are used only for threshold selection
on validation and final evaluation, respectively.

## Gate and confirmation rules

The Gate requires:

1. exact fit/row completeness, finite metrics, and non-empty provenance hashes;
2. candidate (`cost_weight_q05`) utility signal whose task-bootstrap CI excludes
   zero in the predicted positive direction, or a predeclared conditional
   minority-risk signal together with utility within `0.02` of no treatment;
3. no candidate safety failure under the Brier or majority-logloss margins;
4. independent reanalysis with byte-identical effect, summary, and report files.

If the Gate is `GO`, the 24-task confirmation is run unchanged. Confirmation
does not upgrade a conditional result to a universal claim; it reports the
candidate's mean, CI, worst task, and treatment-by-task heterogeneity.

## Data and provenance

Use the existing local `imblearn` Zenodo cache at
`<local path omitted>`; do not download a duplicate. Record the
dataset manifest, package versions, protocol hash, and data hash in the output
directory. The source archive is
`https://zenodo.org/record/61452/files/benchmark-imbalanced-learn.tar.gz`.

## Execution order

1. Add the runner/analyzer tests and observe a failing test.
2. Implement the cost-weight path and run targeted tests.
3. Run the Gate smoke, then the six-task Gate.
4. Analyze the Gate and only if it passes run the frozen 24-task confirmation.
5. Reanalyze from a clean process, compare bytes, run the project regression
   suite, and append the result and boundary to `innovation_ledger.md`.

